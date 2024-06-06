import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def residual(x):
    res_vec = np.zeros(x.size)
    pts_size = int(x.size/2)
    for i in range(pts_size):
        res_vec[i*2] = np.sin(2*np.pi*(i+1)*x[i]) - \
                 np.cos(3*np.pi*(i+1)*x[i+pts_size])
        res_vec[i*2+1] = np.cos(2*np.pi*(i+1)*x[i]) - \
                     np.sin(3*np.pi*(i+1)*x[i+pts_size])
    return res_vec

def drdx(x):
    deriv_mat = np.zeros((x.size, x.size))
    pts_size = int(x.size/2)
    for i in range(pts_size):
        col_ind0 = i
        col_ind1 = i+pts_size
        deriv_mat[i*2,col_ind0] = 2*np.pi*(i+1)*np.cos(2*np.pi*(i+1)*x[i])
        deriv_mat[i*2,col_ind1] = 3*np.pi*(i+1)*np.sin(3*np.pi*(i+1)*x[i+pts_size])
        deriv_mat[i*2+1,col_ind0] = -2*np.pi*(i+1)*np.sin(2*np.pi*(i+1)*x[i])
        deriv_mat[i*2+1,col_ind1] = -3*np.pi*(i+1)*np.cos(3*np.pi*(i+1)*x[i+pts_size])
    return deriv_mat


x0 = np.array([2.,3.,4.,5.])
x_root0 = fsolve(residual, x0=x0, fprime=drdx)
print("x_root from fsolve:", x_root0)


def newton_solver(x0, rtol=1e-6, max_iter=100):
    x_root = x0.copy()
    x_root[0] = 3.
    iter_ind = 0
    while True:
        r = residual(x_root)
        r_norm = np.linalg.norm(r)

        if iter_ind == 0:
            init_r_norm = r_norm

        rel_r_norm = r_norm/init_r_norm
        print(f"Newton solver iteration: {iter_ind}, rel err: {rel_r_norm}")

        if iter_ind > max_iter:
            print(f"Max number of iterations {max_iter} exceeded ...")
            break

        if rel_r_norm < rtol:
            print(f"Newton solver converged with {iter_ind} iterations")
            break

        drdxi = drdx(x_root)

        drdxi[0,:] = np.zeros(x0.size)
        drdxi[:,0] = np.zeros(x0.size)
        drdxi[0,0] = 1.

        b_vec = -r.copy()
        b_vec[0] = 0.


        dx = np.linalg.solve(drdxi, b_vec)
        x_root += dx
        iter_ind += 1

    return x_root


x_root1 = newton_solver(x0)
print("x_root from newton solver:", x_root1)