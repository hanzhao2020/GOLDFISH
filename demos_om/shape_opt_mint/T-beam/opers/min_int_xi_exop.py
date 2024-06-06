from GOLDFISH.nonmatching_opt_ffd import *

class MinIntXiExop(object):
    """
    Custom explicit operation.
    """
    def __init__(self, nonmatching_opt, rho=1e3, method='induced power'):
        self.nonmatching_opt = nonmatching_opt
        self.method = method
        self.rho = rho
        self.xi_size = self.nonmatching_opt.xi_size

    def min_xi(self, xi):
        if self.method == 'KS':
            min_xi_agg = 1-1/self.rho*np.log(1./self.xi_size*np.sum(np.exp(self.rho*(-xi+1.))))
        elif self.method == 'pnorm':
            min_xi_agg = 1-(1/self.xi_size*np.sum((-xi+1.)**self.rho))**(1/self.rho)
        elif self.method == 'induced power':
            min_xi_agg = 1-(np.sum((-xi+1)**(self.rho+1)))/(np.sum((-xi+1)**(self.rho)))
        return min_xi_agg

    def derivative(self, xi):
        if self.method == 'KS':
            pass
        elif self.method == 'pnorm':
            pass
        elif self.method == 'induced power':
            dximin_dxi = np.zeros(self.xi_size)
            xi_min_num = (np.sum((-xi+1.)**(self.rho+1)))
            xi_min_den = (np.sum((-xi+1.)**(self.rho)))
            for i in range(self.xi_size):
                dxi_min_num = -(self.rho+1)*(-xi[i]+1)**self.rho
                dxi_min_den = -self.rho*(-xi[i]+1)**(self.rho-1)
                dximin_dxi[i] = -(dxi_min_num*xi_min_den - 
                                 xi_min_num*dxi_min_den)/(xi_min_den**2)
        return dximin_dxi


if __name__ == '__main__':
    pass