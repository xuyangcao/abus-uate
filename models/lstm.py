import math
import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, in_channel=4, ngf=32):
        super(LSTM, self).__init__()

        self.conv_i1 = nn.Conv2d(in_channel, ngf, kernel_size=3, padding=1, bias=False)
        self.bn_i = nn.BatchNorm2d(ngf)
        self.relu_i = nn.ReLU(inplace=True)
        self.conv_i2 = nn.Conv2d(ngf, in_channel // 2, kernel_size=3, padding=1, bias=False)

        self.conv_g1 = nn.Conv2d(in_channel, ngf, kernel_size=3, padding=1, bias=False)
        self.bn_g = nn.BatchNorm2d(ngf)
        self.relu_g = nn.ReLU(inplace=True)
        self.conv_g2 = nn.Conv2d(ngf, in_channel // 2, kernel_size=3, padding=1, bias=False)

        self.conv_f1 = nn.Conv2d(in_channel, ngf, kernel_size=3, padding=1, bias=False)
        self.bn_f = nn.BatchNorm2d(ngf)
        self.relu_f = nn.ReLU(inplace=True)
        self.conv_f2 = nn.Conv2d(ngf, in_channel // 2, kernel_size=3, padding=1, bias=False)

        self.conv_o1 = nn.Conv2d(in_channel, ngf, kernel_size=3, padding=1, bias=False)
        self.bn_o = nn.BatchNorm2d(ngf)
        self.relu_o = nn.ReLU(inplace=True)
        self.conv_o2 = nn.Conv2d(ngf, in_channel // 2, kernel_size=3, padding=1, bias=False)

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)

    def forward(self, x_t, cell_t_1, hide_t_1):
        x = torch.cat((x_t, hide_t_1), dim=1)

        ft = self.relu_f(self.bn_f(self.conv_f1(x)))
        ft = torch.sigmoid(self.conv_f2(ft))
        
        it = self.relu_i(self.bn_i(self.conv_i1(x)))
        it = torch.sigmoid(self.conv_i2(it))

        gt = self.relu_g(self.bn_f(self.conv_g1(x)))
        gt = torch.sigmoid(self.conv_g2(gt))

        ot = self.relu_o(self.bn_o(self.conv_o1(x)))
        ot = torch.sigmoid(self.conv_o2(ot))

        cell_t = ft * cell_t_1 + it * gt
        hide_t = ot * torch.tanh(cell_t)

        return cell_t, hide_t
