from GOLDFISH.nonmatching_opt_ffd import *

class EqIntXiExop(object):
    """
    Custom explicit operation.
    """
    def __init__(self, nonmatching_opt, rho=1e3, method='induced power'):
        self.nonmatching_opt = nonmatching_opt
        self.method = method
        self.rho = rho
        self.xi_size = self.nonmatching_opt.xi_size

    def max_xi(self, xi):
        if self.method == 'KS':
            max_xi_agg = 1/self.rho*np.log(1./self.xi_size*np.sum(np.exp(self.rho*(xi))))
        elif self.method == 'pnorm':
            max_xi_agg = (1/self.xi_size*np.sum(xi**self.rho))**(1/self.rho)
        elif self.method == 'induced power':
            max_xi_agg = (np.sum((xi)**(self.rho+1)))/(np.sum((xi)**(self.rho)))
        else:
            raise ValueError("Undefined method for max xi")
        return max_xi_agg

    def derivative(self, xi):
        if self.method == 'KS':
            pass
        elif self.method == 'pnorm':
            pass
        elif self.method == 'induced power':
            dximax_dxi = np.zeros(self.xi_size)
            xi_max_num = (np.sum((xi)**(self.rho+1)))
            xi_max_den = (np.sum((xi)**(self.rho)))
            for i in range(self.xi_size):
                dxi_max_num = (self.rho+1)*xi[i]**self.rho
                dxi_max_den = self.rho*xi[i]**(self.rho-1)
                dximax_dxi[i] = (dxi_max_num*xi_max_den - 
                                 xi_max_num*dxi_max_den)/(xi_max_den**2)
        else:
            raise ValueError("Undefined method for max xi")
        return dximax_dxi


if __name__ == '__main__':
    pass