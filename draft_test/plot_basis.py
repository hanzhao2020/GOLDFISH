import numpy as np
import matplotlib.pyplot as plt
from tIGAr.BSplines import *

p = 2
knots = np.array([0,0,0,1,1,1])
# knots = [0,0,0.125, 0.25 , 0.375, 0.5  , 0.625, 0.75 , 0.875,1,1]
# knots = np.array([0., 0., 0., 0.125, 0.25 , 0.375, 0.5  , 0.625, 0.75 , 0.875, 1., 1., 1.])
# knots = np.array([0., 0., 0., 0., 0.125, 0.25 , 0.375, 0.5  , 0.625, 0.75 , 0.875, 1., 1., 1., 1.])
# knots = np.array([0., 0., 0., 0.16666667, 0.33333333, 0.5, 0.66666667, 0.83333333, 1., 1. ,1.])
bspline = BSpline1(p, knots)

num_evals = 201
eval_pts = np.linspace(0,1,num_evals)
num_basis = len(knots) - (p+1)
basis_funcs = np.zeros((num_basis, num_evals))

for pt_ind in range(num_evals):
    u = eval_pts[pt_ind]
    knot_span = bspline.getKnotSpan(u)
    basis_val = bspline.basisFuncs(knot_span, u)
    nodes = bspline.getNodes(u)
    for node_ind in range(len(nodes)):
        basis_funcs[nodes[node_ind], pt_ind] = basis_val[node_ind]


plt.figure()
for basis_ind in range(num_basis):
    plt.plot(eval_pts, basis_funcs[basis_ind], color='blue', linewidth=4)
plt.xlim([0,1])
plt.ylim([0,1])
# plt.xticks(ticks=knots[p:-p-1],color='w')
plt.xticks(ticks=[0-0.05, 1+0.05],color='w')
plt.yticks(ticks=[0-0.02,1], fontsize=20, color='w')
plt.axis('off')
plt.show()