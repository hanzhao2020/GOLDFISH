from GOLDFISH.nonmatching_opt_ffd import *

class IntXiEdgeIntExop(object):
    """
    Explicit operation to compute internal energy of non-matching 
    structure, derivatives of compliacne w.r.t. displacements 
    and control points both in IGA DoFs.
    """
    def __init__(self, nonmatching_opt):
        self.nonmatching_opt = nonmatching_opt

        self.num_splines = self.nonmatching_opt.num_splines
        self.splines = self.nonmatching_opt.splines
        # self.opt_field = self.nonmatching_opt.opt_field
        self.opt_shape = self.nonmatching_opt.opt_shape
        self.shopt_surf_inds = self.nonmatching_opt.shopt_surf_inds
        self.opt_thickness = self.nonmatching_opt.opt_thickness

        self.cpiga2xi = self.nonmatching_opt.cpiga2xi
        self.preprocessor = self.nonmatching_opt.preprocessor
        self.int_xi_edge_forms = []
        self.xi_edge_func_inds = []
        for i, diff_int_ind in enumerate(self.cpiga2xi.diff_int_inds):
            int_type = self.preprocessor.intersections_type[diff_int_ind]
            if int_type[0] == 'surf-edge' or int_type[0] == 'edge-surf':
                edge_indicator = self.preprocessor.diff_int_edge_cons[i]
                side = int(edge_indicator[edge_indicator.index('-')-1])
                para_dir = int(edge_indicator[edge_indicator.index('-')+1])
                edge_val = int(edge_indicator[edge_indicator.index('.')+1])

                xi_func = self.nonmatching_opt.xi_funcs[i*2+side][para_dir]
                temp_form = (xi_func-Constant(edge_val))**2*dx

                self.int_xi_edge_forms += [temp_form,]
                self.xi_edge_func_inds += [i*2+side]

        self.int_xi_edge_deriv_forms = []
        for i in range(len(self.nonmatching_opt.xi_funcs)):
            if i in self.xi_edge_func_inds:
                local_ind = self.xi_edge_func_inds.index(i)
                self.int_xi_edge_deriv_forms += [derivative(self.int_xi_edge_forms[local_ind], 
                                                 self.nonmatching_opt.xi_funcs[i])]
            else:
                dx_temp = dx(domain=self.nonmatching_opt.xi_funcs[i].ufl_domain())
                temp_form = Constant(0.)*dx_temp
                self.int_xi_edge_deriv_forms += [derivative(temp_form,
                                                 self.nonmatching_opt.xi_funcs[i])]

    def int_edge_int(self):
        int_edge_int_val = 0
        for i in range(len(self.int_xi_edge_forms)):
            int_edge_int_val += assemble(self.int_xi_edge_forms[i])
        return int_edge_int_val

    def dint_xi_edge_int_dxi(self, array=True):
        dint_edge_int_dxi_list = []
        for i in range(len(self.nonmatching_opt.xi_funcs)):
            dint_edge_int_dxi_assemble = assemble(self.int_xi_edge_deriv_forms[i])
            dint_edge_int_dxi_list += [v2p(dint_edge_int_dxi_assemble)]

        dint_edge_int_dxi_nest = create_nest_PETScVec(dint_edge_int_dxi_list, 
                        comm=self.nonmatching_opt.comm)

        if array:
            return get_petsc_vec_array(dint_edge_int_dxi_nest, comm=self.nonmatching_opt.comm)
        else:
            return dint_edge_int_dxi_nest




if __name__ == '__main__':
    # from GOLDFISH.tests.test_tbeam import nonmatching_opt
    # from GOLDFISH.tests.test_slr import nonmatching_opt
    from GOLDFISH.tests.test_dRdt import nonmatching_opt

    int_xi_edge_op = IntXiEdgeIntExop(nonmatching_opt)

    # vec0 = np.ones(int_xi_edge_op.nonmatching_opt.vec_iga_dof)
    # vec1 = np.ones(int_xi_edge_op.nonmatching_opt.vec_iga_dof)
    # vec_disp = np.concatenate([vec0, vec1])
    # int_xi_edge_op.nonmatching_opt.update_uIGA(vec_disp)

    wint = int_xi_edge_op.int_edge_int()
    dwint_duiga = int_xi_edge_op.dint_xi_edge_int_dxi()