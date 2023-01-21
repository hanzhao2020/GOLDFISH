from GOLDFISH.nonmatching_opt_ffd import *

class VolumeExOperation(object):
    """
    Explicit operation to compute internal energy of non-matching 
    structure, derivatives of compliacne w.r.t. displacements 
    and control points both in IGA DoFs.
    """
    def __init__(self, nonmatching_opt, thickness_deriv=False):
        self.nonmatching_opt = nonmatching_opt
        self.num_splines = self.nonmatching_opt.num_splines
        self.splines = self.nonmatching_opt.splines
        self.opt_field = self.nonmatching_opt.opt_field

        self.vol_forms = []
        for s_ind in range(self.num_splines):
            vol = self.nonmatching_opt.h_th[s_ind]*self.splines[s_ind].dx
            self.vol_forms += [vol]


        if thickness_deriv is True:
            self.dvoldu_forms = []
            for s_ind in range(self.num_splines):
                dvoldh_th = derivative(self.vol_forms[s_ind], 
                                       self.nonmatching_opt.h_th[s_ind])
                self.dvoldu_forms += [dvoldh_th]
        else:
            self.dvoldu_forms = None

        self.dvoldcp_forms = [[] for i in range(len(self.opt_field))]
        for i, field in enumerate(self.opt_field):
            for s_ind in range(self.num_splines):
                dvoldcp = derivative(self.vol_forms[s_ind],
                          self.splines[s_ind].cpFuncs[field])
                self.dvoldcp_forms[i] += [dvoldcp]

    def volume(self):
        vol_val = 0
        for s_ind in range(self.num_splines):
            vol_val += assemble(self.vol_forms[s_ind])
        return vol_val

    def dvoldh_th(self, extract=False, array=True):
        dvoldh_th_list = []
        for s_ind in range(self.num_splines):
            dwintdu_assmble = assemble(self.dvoldu_forms[s_ind])
            dvoldh_th_list += [v2p(dwintdu_assmble),]
        if extract:
            dvoldh_th_nest = self.nonmatching_opt.\
                extract_nonmatching_vec(dvoldh_th_list, scalar=True)
        else:
            dvoldh_th_nest = create_nest_PETScVec(dvoldh_th_list,
                             comm=self.nonmatching_opt.comm)
        if array:
            return get_petsc_vec_array(dvoldh_th_nest,
                                       comm=self.nonmatching_opt.comm)
        else:
            return dvoldh_th_nest

    def dvoldCPIGA(self, field, array=True):
        dvoldcp_fe_list = []
        field_ind = self.opt_field.index(field)
        for s_ind in range(self.num_splines):
            dvoldcp_fe_assemble = assemble(
                                  self.dvoldcp_forms[field_ind][s_ind])
            dvoldcp_fe_list += [v2p(dvoldcp_fe_assemble)]
        dvoldcp_iga_nest = self.nonmatching_opt.extract_nonmatching_vec(
                           dvoldcp_fe_list, scalar=True)
        if array:
            return get_petsc_vec_array(dvoldcp_iga_nest,
                                       comm=self.nonmatching_opt.comm)
        else:
            return dvoldcp_iga_nest

if __name__ == '__main__':
    from GOLDFISH.tests.test_tbeam import nonmatching_opt
    # from GOLDFISH.tests.test_slr import nonmatching_opt

    wint_op = VolumeExOperation(nonmatching_opt)

    vec0 = np.ones(wint_op.nonmatching_opt.vec_iga_dof)
    vec1 = np.ones(wint_op.nonmatching_opt.vec_iga_dof)
    vec_disp = np.concatenate([vec0, vec1])
    wint_op.nonmatching_opt.update_uIGA(vec_disp)

    wint = wint_op.volume()
    # dwint_duiga = wint_op.dvoldh_th()
    dwint_dcpiga = wint_op.dvoldCPIGA(1)