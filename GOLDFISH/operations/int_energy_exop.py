from GOLDFISH.nonmatching_opt_ffd import *

class IntEnergyExOperation(object):
    """
    Explicit operation to compute internal energy of non-matching 
    structure, derivatives of compliacne w.r.t. displacements 
    and control points both in IGA DoFs.
    """
    def __init__(self, nonmatching_opt):
        self.nonmatching_opt = nonmatching_opt
        self.num_splines = self.nonmatching_opt.num_splines
        self.splines = self.nonmatching_opt.splines
        self.opt_field = self.nonmatching_opt.opt_field

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
            dwintdu = derivative(wint, 
                      self.nonmatching_opt.spline_funcs[s_ind])
            self.dwintdu_forms += [dwintdu]

        self.dwintdcp_forms = [[] for i in range(len(self.opt_field))]
        for i, field in enumerate(self.opt_field):
            for s_ind in range(self.num_splines):
                dwintdcp = derivative(self.wint_forms[s_ind],
                           self.splines[s_ind].cpFuncs[field])
                self.dwintdcp_forms[i] += [dwintdcp]

    def Wint(self):
        wint_val = 0
        for s_ind in range(self.num_splines):
            wint_val += assemble(self.wint_forms[s_ind])
        return wint_val

    def dWintduIGA(self, array=True, apply_bcs=True):
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
        for s_ind in range(self.num_splines):
            dwintdcp_fe_assemble = assemble(
                                   self.dwintdcp_forms[field_ind][s_ind])
            dwintdcp_fe_list += [v2p(dwintdcp_fe_assemble)]
        dwintdcp_iga_nest = self.nonmatching_opt.extract_nonmatching_vec(
                            dwintdcp_fe_list, scalar=True)
        if array:
            return get_petsc_vec_array(dwintdcp_iga_nest,
                                       comm=self.nonmatching_opt.comm)
        else:
            return dwintdcp_iga_nest

if __name__ == '__main__':
    from GOLDFISH.tests.test_tbeam import nonmatching_opt
    # from GOLDFISH.tests.test_slr import nonmatching_opt

    wint_op = IntEnergyExOperation(nonmatching_opt)

    vec0 = np.ones(wint_op.nonmatching_opt.vec_iga_dof)
    vec1 = np.ones(wint_op.nonmatching_opt.vec_iga_dof)
    vec_disp = np.concatenate([vec0, vec1])
    wint_op.nonmatching_opt.update_uIGA(vec_disp)

    wint = wint_op.Wint()
    dwint_duiga = wint_op.dWintduIGA()
    dwint_dcpiga = wint_op.dWintdCPIGA(1)