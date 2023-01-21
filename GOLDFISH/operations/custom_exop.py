from GOLDFISH.nonmatching_opt_ffd import *

class CustomExOperation(object):
    """
    Custom explicit operation.
    """
    def __init__(self, nonmatching_opt, func_symb, func_deriv_symb):
        self.nonmatching_opt = nonmatching_opt
        self.func_symb = func_symb
        self.func_deriv_symb = func_deriv_symb

        self.num_splines = self.nonmatching_opt.num_splines
        self.opt_field = self.nonmatching_opt.opt_field

    def func(self):
        func_val = 0
        for s_ind in range(self.num_splines):
            func_val += assemble(self.wint_forms[s_ind])
        return func_val

    def func_deriv(self, extract=True, scalar=False, array=True):
        func_deriv_list = []
        for s_ind in range(self.num_splines):
            func_deriv_sub = assemble(self.func_deriv_symb[s_ind])
            func_deriv_list += [v2p(func_deriv_sub),]
        if extract:
            if scalar:
                func_deriv_nest = self.nonmatching_opt.\
                extract_nonmatching_vec(func_deriv_list, scalar=True)
            else:
                func_deriv_nest = self.nonmatching_opt.\
                extract_nonmatching_vec(func_deriv_list, scalar=False)
        else:
            func_deriv_nest = create_nest_PETScVec(func_deriv_list, 
                              comm=self.nonmatching_opt.comm)
        if array:
            return get_petsc_vec_array(func_deriv_nest,
                                       comm=self.nonmatching_opt.comm)
        else:
            return func_deriv_nest

if __name__ == '__main__':
    pass