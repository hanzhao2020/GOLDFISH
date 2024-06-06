import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


data_init = np.load('slice_data_init.npz')

s0_x_pts_init = data_init['s0x']
s0_y_pts_init = data_init['s0y']
s1_x_pts_init = data_init['s1x']
s1_y_pts_init = data_init['s1y']
xpts_init = data_init['xpts']
ypts_init = data_init['ypts']



data = np.load('slice_data.npz')

s0_x_pts = data['s0x']
s0_y_pts = data['s0y']
s1_x_pts = data['s1x']
s1_y_pts = data['s1y']
xpts = data['xpts']
ypts = data['ypts']


plt.figure()
plt.plot(s0_x_pts_init, s0_y_pts_init, '--', color='tab:blue', linewidth=3, label='Patch 1 initial')
plt.plot(s1_x_pts_init, s1_y_pts_init, '--', color='tab:orange', linewidth=3, label='Patch 3 initial')
plt.plot(s0_x_pts, s0_y_pts, linewidth=3, color='tab:blue', label='Patch 1 optimized')
plt.plot(s1_x_pts, s1_y_pts, linewidth=3, color='tab:orange',label='Patch 3 optimized')
plt.plot(xpts, ypts, '-.', color='dimgray', linewidth=2, label='Exact quarter circle')
plt.xlabel(r'$x$',fontsize=14)
plt.ylabel(r'$y$',fontsize=14)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
ax.tick_params(axis="x", direction='in', length=5, pad=8)
ax.tick_params(axis="y", direction='in', length=5, pad=8)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.show()