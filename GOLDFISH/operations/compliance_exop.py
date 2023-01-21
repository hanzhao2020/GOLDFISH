from GOLDFISH.nonmatching_opt_ffd import *

class ComplianceExOperation(object):
    """
    Explicit operation to compute compliance of non-matching structure,
    derivatives of compliacne w.r.t. displacements and control points
    both in IGA DoFs.
    """
    def __init__(self, nonmatching_opt, force):
        self.nonmatching_opt = nonmatching_opt
        self.num_splines = self.nonmatching_opt.num_splines
        self.splines = self.nonmatching_opt.splines
        self.opt_field = self.nonmatching_opt.opt_field
        self.force = force

        self.c_forms = []
        self.dcpldu_forms = []
        for s_ind in range(self.num_splines):
            cpl_sub = inner(self.force, 
                    self.nonmatching_opt.spline_funcs[s_ind])\
                    *self.splines[s_ind].dx
            self.c_forms += [cpl_sub]
            dcpldu = derivative(cpl_sub, 
                     self.nonmatching_opt.spline_funcs[s_ind])
            self.dcpldu_forms += [dcpldu]

        self.dcpldcp_forms = [[] for i in range(len(self.opt_field))]
        for i, field in enumerate(self.opt_field):
            for s_ind in range(self.num_splines):
                dcpldcp = derivative(self.c_forms[s_ind],
                           self.splines[s_ind].cpFuncs[field])
                self.dcpldcp_forms[i] += [dcpldcp]

    def cpl(self):
        cpl_val = 0
        for s_ind in range(self.num_splines):
            cpl_val += assemble(self.c_forms[s_ind])
        return cpl_val

    def dcplduIGA(self, array=True, apply_bcs=True):
        dcpldu_iga_list = []
        for s_ind in range(self.num_splines):
            dcpldu_assmble = assemble(self.dcpldu_forms[s_ind])
            dcpldu_iga_list += [v2p(FE2IGA(self.splines[s_ind],
                                  dcpldu_assmble, apply_bcs)),]
        dcpldu_iga_nest = create_nest_PETScVec(dcpldu_iga_list,
                          comm=self.nonmatching_opt.comm)
        if array:
            return get_petsc_vec_array(dcpldu_iga_nest,
                                       comm=self.nonmatching_opt.comm)
        else:
            return dcpldu_iga_nest

    def dcpldCPIGA(self, field, array=True):
        dcpldcp_fe_list = []
        field_ind = self.opt_field.index(field)
        for s_ind in range(self.num_splines):
            dcpldcp_fe_assemble = assemble(
                                   self.dcpldcp_forms[field_ind][s_ind])
            dcpldcp_fe_list += [v2p(dcpldcp_fe_assemble)]
        dcpldcp_iga_nest = self.nonmatching_opt.extract_nonmatching_vec(
                           dcpldcp_fe_list, scalar=True)
        if array:
            return get_petsc_vec_array(dcpldcp_iga_nest,
                                       comm=self.nonmatching_opt.comm)
        else:
            return dcpldcp_iga_nest

if __name__ == '__main__':
    from GOLDFISH.tests.test_tbeam import nonmatching_opt
    # from GOLDFISH.tests.test_slr import nonmatching_opt

    force = as_vector([Constant(0.), Constant(0.), Constant(1.)])
    cpl_op = ComplianceExOperation(nonmatching_opt, force)

    vec0 = np.ones(cpl_op.nonmatching_opt.vec_iga_dof)
    vec1 = np.ones(cpl_op.nonmatching_opt.vec_iga_dof)
    vec_disp = np.concatenate([vec0, vec1])
    cpl_op.nonmatching_opt.update_uIGA(vec_disp)

    cpl_val = cpl_op.cpl()
    dcpl_duiga = cpl_op.dcplduIGA()
    dcpl_dcpiga = cpl_op.dcpldCPIGA(0)