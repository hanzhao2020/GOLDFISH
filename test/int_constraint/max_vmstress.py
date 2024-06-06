import numpy as np
import matplotlib.pyplot as plt

rho = 1e3
vm1 = np.array([0, 1.1e3, 1.5e4, 1.6e7, 1.9e8, 2.9e10, 4.9e10, 5.6e7, 7.9e8])
vm = np.concatenate([vm1]*1000)
nt = len(vm)
scale_fac = 1e11

max_vm = np.max(vm)
min_vm = np.min(vm)

# # KS functional
# max_vm_agg = scale_fac*1/rho*np.log(1./nt*np.sum(np.exp(rho*(vm/scale_fac))))

# # p norm
# max_vm_agg = scale_fac*(1/nt*np.sum((vm/scale_fac)**rho))**(1/rho)

# Induced power
max_vm_agg = scale_fac*(np.sum((vm/scale_fac)**(rho+1)))/(np.sum((vm/scale_fac)**(rho)))

max_vm_rel_err = abs(max_vm_agg-max_vm)/max_vm

print("true max: {:10.8f}, agg max: {:10.8f}, err: {:10.8f}"
      .format(max_vm, max_vm_agg, max_vm_rel_err))