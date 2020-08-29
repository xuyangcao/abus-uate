import numpy as np
import matplotlib.pyplot as plt

# method 1
base_lr = 3e-5
max_iter = 30000
power = 0.5
iter_idx = np.linspace(0, max_iter, max_iter)
lr = base_lr * (1 - iter_idx/max_iter)**power

# method 2
step = 8000
scale = 0.5 
power_1 = iter_idx // step
lr_1 = base_lr * scale**power_1

# method 4
step = 5000
scale = 0.65 
power_1 = iter_idx // step
lr_3 = base_lr * scale**power_1

# method 3
lr_2 = 1.5e-5 * (np.cos(np.pi * iter_idx / 40000) + 1)


fig = plt.figure()
ax = fig.add_subplot(111) 
ax.plot(iter_idx, lr)
ax.plot(iter_idx, lr_1)
ax.plot(iter_idx, lr_2)
ax.plot(iter_idx, lr_3)
ax.set_yscale('log')
plt.show()
