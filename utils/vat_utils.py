import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

@contextlib.contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
            
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VATPerturbation(nn.Module):

    def __init__(self, xi=5.0, eps=1.0, ip=1):
        """VAT Perturbation
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATPerturbation, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):
        with torch.no_grad():
            pred = F.softmax(model(x), dim=1)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model(x + self.xi * d)
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()
            r_adv = d * self.eps

        return r_adv

def normalize_eps(x):
    x_flat = x.view(len(x), -1)
    mag = torch.sqrt((x_flat * x_flat).sum(dim=1))
    return x / (mag[:, None, None, None]+1e-12)

def normalized_noise(x, requires_grad=False, scale=1.0):
    eps = torch.randn(x.shape, dtype=torch.float, device=x.device)
    eps = normalize_eps(eps) * scale
    if requires_grad:
        eps = eps.clone().detach().requires_grad_(True)
    return eps

def vat_direction(model, x, cons_loss_fn='kld'):
    """
    Compute the VAT perturbation direction vector

    :param x: input image as a `(N, C, H, W)` tensor
    :return: VAT direction as a `(N, C, H, W)` tensor
    """
    # Put the network used to get the VAT direction in eval mode and get the predicted
    # logits and probabilities for the batch of samples x
    #model.eval()
    with torch.no_grad():
        y_pred_logits = model(x).detach()
    y_pred_prob = F.softmax(y_pred_logits, dim=1)

    # Initial noise offset vector with requires_grad=True
    noise_scale = 1.0e-6 * x.shape[2] * x.shape[3] / 1000
    eps = normalized_noise(x, requires_grad=True, scale=noise_scale)

    # Predict logits and probs for sample perturbed by eps
    eps_pred_logits = model(x.detach() + eps)
    eps_pred_prob = F.softmax(eps_pred_logits, dim=1)

    # Choose our loss function
    if cons_loss_fn == 'var':
        delta = (eps_pred_prob - y_pred_prob)
        loss = (delta * delta).sum()
    elif cons_loss_fn == 'bce':
        loss = network_architectures.robust_binary_crossentropy(eps_pred_prob, y_pred_prob).sum()
    elif cons_loss_fn == 'kld':
        loss = F.kl_div(F.log_softmax(eps_pred_logits, dim=1), y_pred_prob, reduce=False).sum()
    elif cons_loss_fn == 'logits_var':
        delta = (eps_pred_logits - y_pred_logits)
        loss = (delta * delta).sum()
    else:
        raise ValueError('Unknown consistency loss function {}'.format(cons_loss_fn))

    # Differentiate the loss w.r.t. the perturbation
    eps_adv = torch.autograd.grad(
        outputs=loss, inputs=eps,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]

    # Normalize the adversarial perturbation
    return normalize_eps(eps_adv)


def vat_perburbation(model, x, adaptive_vat_radius=False, vat_radius=0.5):
    eps_adv_nrm = vat_direction(model, x)

    if adaptive_vat_radius:
        # We view semantic segmentation as predicting the class of a pixel
        # given a patch centred on that pixel.
        # The most similar patch in terms of pixel content to a patch P
        # is a patch Q whose central pixel is an immediate neighbour
        # of the central pixel P.
        # We therefore use the image Jacobian (gradient w.r.t. x and y) to
        # get a sense of the distance between neighbouring patches
        # so we can scale the vat radius according to the image content.

        # delta in vertical and horizontal directions
        delta_v = x[:, :, 2:, :] - x[:, :, :-2, :]
        delta_h = x[:, :, :, 2:] - x[:, :, :, :-2]

        # delta_h and delta_v are the difference between pixels where the step size is 2, rather than 1
        # so divide by 2 to get the magnitude of the jacobian

        delta_v = delta_v.view(len(delta_v), -1)
        delta_h = delta_h.view(len(delta_h), -1)
        adv_radius = vat_radius * torch.sqrt((delta_v**2).sum(dim=1) + (delta_h**2).sum(dim=1))[:, none, none, none] * 0.5
    else:
        scale = math.sqrt(float(x.shape[1] * x.shape[2] * x.shape[3]))
        adv_radius = vat_radius * scale

    return (eps_adv_nrm * adv_radius).detach()
