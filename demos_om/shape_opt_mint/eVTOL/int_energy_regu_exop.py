from GOLDFISH.nonmatching_opt_ffd import *

class IntEnergyReguExOperation(object):
    """
    Explicit operation to compute internal energy of non-matching 
    structure, derivatives of compliacne w.r.t. displacements 
    and control points both in IGA DoFs.
    """
    def __init__(self, nonmatching_opt, regu_para, regu_xi_edge=False):
        self.nonmatching_opt = nonmatching_opt
        self.num_splines = self.nonmatching_opt.num_splines
        self.splines = self.nonmatching_opt.splines
        # self.opt_field = self.nonmatching_opt.opt_field
        self.opt_shape = self.nonmatching_opt.opt_shape
        self.shopt_surf_inds = self.nonmatching_opt.shopt_surf_inds
        self.opt_thickness = self.nonmatching_opt.opt_thickness

        self.regu_para = regu_para
        self.regu_xi_edge = regu_xi_edge
        if self.regu_xi_edge:
            print("************ Regularization parameter", float(regu_para))

        self.wint_regu_forms = []
        self.wint_forms = []
        self.dwintdu_forms = []
        for s_ind in range(self.num_splines):
            X = self.splines[s_ind].F
            u = self.splines[s_ind].rationalize(
                self.nonmatching_opt.spline_funcs[s_ind])
            x = X + u
            wint = surfaceEnergyDensitySVK(self.splines[s_ind],
                   X, x, self.nonmatching_opt.E[s_ind], 
                   self.nonmatching_opt.nu[s_ind],
                   self.nonmatching_opt.h_th[s_ind])*self.splines[s_ind].dx
            self.wint_forms += [wint]
            # if self.regu_para > 0:
            #     regu_para_full = self.regu_para*self.nonmatching_opt.E[s_ind]*\
            #                      self.init_h_th**3/(12*self.nonmatching_opt.ha_phy_linear[s_ind]*\
            #                      (1-self.nonmatching_opt.nu[s_ind]**2))
            #     regu_term = 0
            #     for field in [0]: #self.opt_field:
            #         grad_diff = self.splines[s_ind].grad(self.splines[s_ind].cpFuncs[field])\
            #                 -self.splines[s_ind].grad(self.nonmatching_opt.init_cpfuncs_list[s_ind][field])
            #         regu_term += regu_para_full*inner(grad_diff, grad_diff)*self.splines[s_ind].dx
            #     self.wint_regu_forms += [wint + regu_term]
            # else:
            #     self.wint_regu_forms += [wint]
            dwintdu = derivative(wint, 
                      self.nonmatching_opt.spline_funcs[s_ind])
            self.dwintdu_forms += [dwintdu]

        if self.opt_shape:
            self.opt_field = self.nonmatching_opt.opt_field
            self.dwintdcp_forms = [[] for i in range(len(self.opt_field))]
            for i, field in enumerate(self.opt_field):
                for j, s_ind in enumerate(self.shopt_surf_inds[i]):
                    dwintdcp = derivative(self.wint_forms[s_ind],
                               self.splines[s_ind].cpFuncs[field])
                    self.dwintdcp_forms[i] += [dwintdcp]


        if self.opt_thickness:
            self.dwintdh_th_forms = []
            for s_ind in range(self.num_splines):
                dwintdh_th = derivative(self.wint_forms[s_ind],
                                        self.nonmatching_opt.h_th[s_ind])
                self.dwintdh_th_forms += [dwintdh_th]

        if self.regu_xi_edge:
            self.int_xi_edge()

    def Wint(self):
        wint_val = 0
        for s_ind in range(self.num_splines):
            wint_val += assemble(self.wint_forms[s_ind])
        print("Internal energy without regularization:", wint_val)

        if self.regu_xi_edge:
            for i in range(len(self.int_xi_edge_forms)):
                wint_val += assemble(self.int_xi_edge_forms[i])

        return wint_val

    def dWintduIGA(self, array=True, apply_bcs=True):
        # wint_woregu = 0
        # for i in range(len(self.wint_forms)):
        #     wint_woregu += assemble(self.wint_forms[i])
        dwintdu_iga_list = []
        for s_ind in range(self.num_splines):
            dwintdu_assmble = assemble(self.dwintdu_forms[s_ind])
            dwintdu_iga_list += [v2p(FE2IGA(self.splines[s_ind],
                                  dwintdu_assmble, apply_bcs)),]
        dwintdu_iga_nest = create_nest_PETScVec(dwintdu_iga_list,
                            comm=self.nonmatching_opt.comm)
        if array:
            return get_petsc_vec_array(dwintdu_iga_nest,
                                       comm=self.nonmatching_opt.comm)
        else:
            return dwintdu_iga_nest

    def dWintdCPIGA(self, field, array=True):
        dwintdcp_fe_list = []
        field_ind = self.opt_field.index(field)
        for j, s_ind in enumerate(self.shopt_surf_inds[field_ind]):
            dwintdcp_fe_assemble = assemble(
                                   self.dwintdcp_forms[field_ind][j])
            dwintdcp_fe_list += [v2p(dwintdcp_fe_assemble)]
        dwintdcp_iga_nest = self.nonmatching_opt.extract_nonmatching_vec(
                            dwintdcp_fe_list, 
                            ind_list=self.shopt_surf_inds[field_ind],
                            scalar=True)
        if array:
            return get_petsc_vec_array(dwintdcp_iga_nest,
                                       comm=self.nonmatching_opt.comm)
        else:
            return dwintdcp_iga_nest

    def dWintdh_th(self, array=True):
        dwintdh_th_list = []
        for s_ind in range(self.num_splines):
            dwintdh_th_assemble = assemble(self.dwintdh_th_forms[s_ind])
            dwintdh_th_list += [v2p(dwintdh_th_assemble),]            
        if self.nonmatching_opt.var_thickness:
            dwintdh_th_nest = self.nonmatching_opt.\
                extract_nonmatching_vec(dwintdh_th_list, scalar=True)
        else:
            dwintdh_th_nest = create_nest_PETScVec(dwintdh_th_list,
                              comm=self.nonmatching_opt.comm)
        if array:
            return get_petsc_vec_array(dwintdh_th_nest,
                                       comm=self.nonmatching_opt.comm)
        else:
            return dwintdh_th_nest

    def dWintdxi(self, array=True):
        dwintdxi_list = []
        for i in range(len(self.nonmatching_opt.xi_funcs)):
            dwintdxi_assemble = assemble(self.int_xi_edge_deriv_forms[i])
            dwintdxi_list += [v2p(dwintdxi_assemble)]

        dwintdxi_nest = create_nest_PETScVec(dwintdxi_list, 
                        comm=self.nonmatching_opt.comm)

        if array:
            return get_petsc_vec_array(dwintdxi_nest, comm=self.nonmatching_opt.comm)
        else:
            return dwintdxi_nest


            
    def int_xi_edge(self):
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

                self.int_xi_edge_forms += [self.regu_para*temp_form,]
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



if __name__ == '__main__':
    # from GOLDFISH.tests.test_tbeam import nonmatching_opt
    # from GOLDFISH.tests.test_slr import nonmatching_opt
    from GOLDFISH.tests.test_dRdt import nonmatching_opt

    wint_op = IntEnergyExOperation(nonmatching_opt)

    vec0 = np.ones(wint_op.nonmatching_opt.vec_iga_dof)
    vec1 = np.ones(wint_op.nonmatching_opt.vec_iga_dof)
    vec_disp = np.concatenate([vec0, vec1])
    wint_op.nonmatching_opt.update_uIGA(vec_disp)

    wint = wint_op.Wint()
    dwint_duiga = wint_op.dWintduIGA()
    dwint_dcpiga = wint_op.dWintdCPIGA(1)
    dwintdh_th = wint_op.dWintdh_th()