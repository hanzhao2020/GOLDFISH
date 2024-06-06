import numpy as np
import matplotlib.pyplot as plt

# xi0 = np.array([0.96156606, 0.62372103, 0.23822087, 0., 0.67961264, 0.59202079,
#                0.00812203, 0.46740998, 0.55746699, 0.57634286, 0.31362242, 1.])

rho = 1e3
# xi0 = np.random.random(100)
# xi1 = np.concatenate([np.array([0,1]), xi0])

xi1 = np.concatenate([np.zeros(1), np.random.random(1000), np.ones(1)])
# xi1 = np.array([0.51,0.52,0.53,0.55])
xi = np.concatenate([xi1]*1000)
nt = len(xi)

max_xi = np.max(xi)
min_xi = np.min(xi)

# # KS functional
# max_xi_agg = 1/rho*np.log(1./nt*np.sum(np.exp(rho*(xi))))
# min_xi_agg = 1.-1/rho*np.log(1./nt*np.sum(np.exp(rho*(-xi+1.))))

# # p norm
# max_xi_agg = (1/nt*np.sum(xi**rho))**(1/rho)
# min_xi_agg = 1-(1/nt*np.sum((-xi+1)**rho))**(1/rho)

# Induced power
max_xi_agg = (np.sum((xi)**(rho+1)))/(np.sum((xi)**(rho)))
min_xi_agg = 1.-(np.sum((-xi+1)**(rho+1)))/(np.sum((-xi+1.)**(rho)))

max_xi_rel_err = abs(max_xi_agg-max_xi)
min_xi_rel_err = abs(min_xi_agg-min_xi)


print("true max: {:10.8f}, agg max: {:10.8f}, err: {:10.8f}"
      .format(max_xi, max_xi_agg, max_xi_rel_err))
print("true min: {:10.8f}, agg min: {:10.8f}, err: {:10.8f}"
      .format(min_xi, min_xi_agg, min_xi_rel_err))


x = 0.9
h = 1e-12
xm = x-h
xp = x+h
a0 = np.log(np.exp(2*xm))
a1 = np.log(np.exp(2*xp))
da = (a1-a0)/(2*h)