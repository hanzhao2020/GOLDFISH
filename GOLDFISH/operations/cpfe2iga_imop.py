from GOLDFISH.nonmatching_opt_ffd import *
from scipy.sparse import coo_matrix, csr_matrix
from scipy.linalg import block_diag

class CPFE2IGAImOperation(object):
    """
    Implicit operation to solve shells' control points 
    in IGA DoFs from FE DoFs.
    """
    def __init__(self, nonmatching_opt):
        """
        ``nonmatching_opt`` can be a instance of NonMatchingOpt
        or NonMatchingOptFFD.
        """
        self.nonmatching_opt = nonmatching_opt
        self.shopt_surf_inds = self.nonmatching_opt.shopt_surf_inds
        self.num_splines = self.nonmatching_opt.num_splines
        self.opt_field = self.nonmatching_opt.opt_field
        self.comm = self.nonmatching_opt.comm

        self.dRdcp_iga_list = [[] for field in self.opt_field]
        self.dRdcp_iga_csr_list = [[] for field in self.opt_field]
        self.dRdcp_fe_list = [[] for field in self.opt_field]
        self.dRdcp_fe_csr_list = [[] for field in self.opt_field]
        self.dRdcp_iga_coo = [None for field in self.opt_field]
        self.dRdcp_fe_coo = [None for field in self.opt_field]
        self.sub_vec_iga_sizes = [None for field in self.opt_field]
        # self.sub_vec_fe_sizes = [None for field in self.opt_field]
        for field_ind, field in enumerate(self.opt_field):
            for local_ind, s_ind in enumerate(self.shopt_surf_inds[field_ind]):
                Mc = m2p(self.nonmatching_opt.splines[s_ind].M_control)
                dRdcp_iga_sub = Mc.transposeMatMult(Mc)
                McT = Mc.copy()
                McT.transpose()
                dRdcp_fe_sub = -McT
                self.dRdcp_iga_list[field_ind] += [dRdcp_iga_sub]
                self.dRdcp_iga_csr_list[field_ind] += [csr_matrix(dRdcp_iga_sub.
                                            getValuesCSR()[::-1],
                                            shape=dRdcp_iga_sub.size)]
                self.dRdcp_fe_list[field_ind] += [dRdcp_fe_sub]
                self.dRdcp_fe_csr_list[field_ind] += [csr_matrix(dRdcp_fe_sub.
                                           getValuesCSR()[::-1],
                                           shape=dRdcp_fe_sub.size)]

            dRdcp_iga_dense_list = [csr_mat.todense() for csr_mat
                                    in self.dRdcp_iga_csr_list[field_ind]]
            dRdcp_fe_dense_list = [csr_mat.todense() for csr_mat
                                   in self.dRdcp_fe_csr_list[field_ind]]
            self.dRdcp_iga_coo[field_ind] = coo_matrix(block_diag(*dRdcp_iga_dense_list))
            self.dRdcp_fe_coo[field_ind] = coo_matrix(block_diag(*dRdcp_fe_dense_list))

            self.sub_vec_iga_sizes[field_ind] = [vec.sizes[1] for vec in
                                      self.nonmatching_opt.cpdes_iga_list[field_ind]]
            # self.sub_vec_fe_sizes[field_ind] = [vec.sizes[1] for vec in
            #                          self.nonmatching_opt.cpdes_fe_list[field_ind]]

        self.cp_fe_vecs = [vec.copy() for vec in self.nonmatching_opt.cpdes_fe_nest]
        self.cp_iga_vecs = [vec.copy() for vec in self.nonmatching_opt.cpdes_iga_nest]
        self.dcp_iga_vec = [vec.copy() for vec in self.nonmatching_opt.cpdes_iga_nest]
        self.dcp_fe_vec = [vec.copy() for vec in self.nonmatching_opt.cpdes_fe_nest]
        self.dres_vec = [vec.copy() for vec in self.nonmatching_opt.cpdes_iga_nest]

    def apply_nonlinear(self):
        res_array_list = []
        for i, field in enumerate(self.opt_field):
            res_sub_array_list = []
            for local_ind, s_ind in enumerate(self.shopt_surf_inds[i]):
                Mc = m2p(self.nonmatching_opt.splines[s_ind].M_control)
                cp_iga_sub = self.cp_iga_vecs[i].getNestSubVecs()[local_ind]
                cp_fe_sub = self.cp_fe_vecs[i].getNestSubVecs()[local_ind]
                Mccp_iga_sub = A_x(Mc, cp_iga_sub)
                res_sub_fe = Mccp_iga_sub - cp_fe_sub
                res_sub_fe.assemble()
                res_sub_iga = AT_x(Mc, res_sub_fe)
                res_sub_array_list += [get_petsc_vec_array(res_sub_iga)]
            res_array_list += [np.concatenate(res_sub_array_list)]
        return res_array_list

    def solve_nonlinear(self):
        cp_iga_array_list = []
        for i, field in enumerate(self.opt_field):
            cp_iga_sub_array_list = []
            for local_ind, s_ind in enumerate(self.shopt_surf_inds[i]):
                Mc = m2p(self.nonmatching_opt.splines[s_ind].M_control)
                cp_iga_sub = self.cp_iga_vecs[i].getNestSubVecs()[local_ind]
                cp_fe_sub = self.cp_fe_vecs[i].getNestSubVecs()[local_ind]
                McTMc = self.dRdcp_iga_list[i][local_ind]
                McTcp_fe = AT_x(Mc, cp_fe_sub)
                solve(PETScMatrix(McTMc), PETScVector(cp_iga_sub),
                      PETScVector(McTcp_fe), 'mumps')
                # cp_iga_sub.assemble()
                cp_iga_sub_array_list += [get_petsc_vec_array(cp_iga_sub)]
            cp_iga_array_list += [np.concatenate(cp_iga_sub_array_list)]
        return cp_iga_array_list

    def apply_linear_fwd(self, d_inputs_array_list=None, 
                         d_outputs_array_list=None, 
                         d_residuals_array_list=None):
        # PETSc implementation
        if d_residuals_array_list is not None:
            if d_outputs_array_list is not None:
                for i, field in enumerate(self.opt_field):
                    update_nest_vec(d_outputs_array_list[i], self.dcp_iga_vec[i])
                    dres_array_out_list = []
                    for local_ind, s_ind in enumerate(self.shopt_surf_inds[i]):
                        cp_iga_sub = self.dcp_iga_vec[i].getNestSubVecs()[local_ind]
                        dres_sub_out = A_x(self.dRdcp_iga_list[i][local_ind], 
                                           cp_iga_sub)
                        dres_sub_array_out = get_petsc_vec_array(
                                             dres_sub_out, self.comm)
                        dres_array_out_list += [dres_sub_array_out]
                    d_residuals_array_list[i][:] += np.concatenate(
                                                    dres_array_out_list)
            if d_inputs_array_list is not None:
                for i, field in enumerate(self.opt_field):
                    update_nest_vec(d_inputs_array_list[i], self.dcp_fe_vec[i])
                    dres_array_in_list = []
                    for local_ind, s_ind in enumerate(self.shopt_surf_inds[field_ind]):
                        cp_fe_sub = self.dcp_fe_vec[i].getNestSubVecs()[local_ind]
                        dres_sub_in = A_x(self.dRdcp_fe_list[i][local_ind], 
                                          cp_fe_sub)
                        dres_sub_array_in = get_petsc_vec_array(dres_sub_in, 
                                            self.comm)
                        dres_array_in_list += [dres_sub_array_in]
                    d_residuals_array_list[i][:] += np.concatenate(
                                                    dres_array_in_list)
        # # Numpy implementation
        # if d_residuals_array_list is not None:
        #     if d_outputs_array_list is not None:
        #         for i, field in enumerate(self.opt_field):
        #             dres_array_out = self.dRdcp_iga_coo\
        #                              *d_outputs_array_list[i]
        #             d_residuals_array_list[i][:] += dres_array_out
        #     if d_inputs_array_list is not None:
        #         for i, field in enumerate(self.opt_field):
        #             dres_array_in = self.dRdcp_fe_coo\
        #                             *d_inputs_array_list[i]
        #             d_residuals_array_list[i][:] += dres_array_in
        return d_residuals_array_list

    def apply_linear_rev(self, d_inputs_array_list=None, 
                         d_outputs_array_list=None,
                         d_residuals_array_list=None):
        # PETSc implementation
        if d_residuals_array_list is not None:
            for i, field in enumerate(self.opt_field):
                update_nest_vec(d_residuals_array_list[i], self.dres_vec[i])
            if d_outputs_array_list is not None:
                for i, field in enumerate(self.opt_field):
                    dcp_iga_array_list = []
                    for local_ind, s_ind in enumerate(self.shopt_surf_inds[field_ind]):
                        dres_sub = self.dres_vec[i].getNestSubVecs()[local_ind]
                        dcp_iga_sub = self.dcp_iga_vec[i].getNestSubVecs()[local_ind]
                        AT_x_b(self.dRdcp_iga_list[i][local_ind], dres_sub, 
                               dcp_iga_sub)
                        dcp_iga_sub_array = get_petsc_vec_array(dcp_iga_sub, 
                                                                self.comm)
                        dcp_iga_array_list += [dcp_iga_sub_array]
                    d_outputs_array_list[i][:] += np.concatenate(
                                                  dcp_iga_array_list)
            if d_inputs_array_list is not None:
                for i, field in enumerate(self.opt_field):
                    dcp_fe_array_list = []
                    for local_ind, s_ind in enumerate(self.shopt_surf_inds[i]):
                        dres_sub = self.dres_vec[i].getNestSubVecs()[local_ind]
                        dcp_fe_sub = self.dcp_fe_vec[i].getNestSubVecs()[local_ind]
                        AT_x_b(self.dRdcp_fe_list[i][local_ind], dres_sub, 
                               dcp_fe_sub)
                        dcp_fe_sub_array = get_petsc_vec_array(dcp_fe_sub, 
                                                               self.comm)
                        dcp_fe_array_list += [dcp_fe_sub_array]
                    d_inputs_array_list[i][:] += np.concatenate(
                                                 dcp_fe_array_list)

        # # Numpy implementation
        # if d_residuals_array_list is not None:
        #     if d_outputs_array_list is not None:
        #         for i, field in enumerate(self.opt_field):
        #             dcp_iga_array = self.dRdcp_iga_coo.T\
        #                             *d_residuals_array_list[i]
        #             d_outputs_array_list[i][:] += dcp_iga_array
        #     if d_inputs_array_list is not None:
        #         for i, field in enumerate(self.opt_field):
        #             dcp_fe_array = self.dRdcp_fe_coo.T\
        #                            *d_residuals_array_list[i]
        #             d_inputs_array_list[i][:] += dcp_fe_array
        return d_inputs_array_list, d_outputs_array_list

    def solve_linear_fwd(self, d_outputs_array_list, d_residuals_array_list):
        for i, field in enumerate(self.opt_field):
            d_outputs_sub_array_list = []
            for local_ind, s_ind in enumerate(self.shopt_surf_inds[i]):
                dcp_iga_sub = self.dcp_iga_vec[i].getNestSubVecs()[local_ind]
                dres_sub = self.dres_vec[i].getNestSubVecs()[local_ind]
                dres_sub_array = d_residuals_array_list[i]\
                    [int(np.sum(self.sub_vec_iga_sizes[i][:local_ind])):
                     int(np.sum(self.sub_vec_iga_sizes[i][:local_ind+1]))]
                dres_sub.setValues(range(dres_sub.sizes[1]),
                                    dres_sub_array)
                dres_sub.assemble()
                solve(PETScMatrix(self.dRdcp_iga_list[i][local_ind]),
                      PETScVector(dcp_iga_sub),
                      PETScVector(dres_sub))
                d_outputs_sub_array_list += [get_petsc_vec_array(
                                             dcp_iga_sub),]
            d_outputs_array_list[i][:] = np.concatenate(
                                         d_outputs_sub_array_list)
        return d_outputs_array_list

    def solve_linear_rev(self, d_outputs_array_list, d_residuals_array_list):
        for i, field in enumerate(self.opt_field):
            d_residuals_sub_array_list = []
            for local_ind, s_ind in enumerate(self.shopt_surf_inds[i]):
                dcp_iga_sub = self.dcp_iga_vec[i].getNestSubVecs()[local_ind]
                dcp_iga_sub_array = d_outputs_array_list[i]\
                    [int(np.sum(self.sub_vec_iga_sizes[i][:local_ind])):
                     int(np.sum(self.sub_vec_iga_sizes[i][:local_ind+1]))]
                dcp_iga_sub.setValues(range(dcp_iga_sub.sizes[1]),
                                      dcp_iga_sub_array)
                dcp_iga_sub.assemble()
                dres_sub = self.dres_vec[i].getNestSubVecs()[local_ind]
                solve(PETScMatrix(self.dRdcp_iga_list[i][local_ind]),
                      PETScVector(dres_sub),
                      PETScVector(dcp_iga_sub), 'mumps')
                d_residuals_sub_array_list += [get_petsc_vec_array(
                                               dres_sub),]
            d_residuals_array_list[i][:] = np.concatenate(
                                           d_residuals_sub_array_list)
        return d_residuals_array_list


if __name__ == '__main__':
    from GOLDFISH.tests.test_tbeam import nonmatching_opt
    # from GOLDFISH.tests.test_slr import nonmatching_opt

    nonmatching_opt.set_shopt_FFD_surf_inds(opt_field=[0,1,2], shopt_surf_inds=[0,1])
    
    ffd_block_num_el = [4,4,1]
    p = 3
    # Create FFD block in igakit format
    cp_ffd_lims = nonmatching_opt.cpsurf_des_lims
    for field in [2]:
        cp_range = cp_ffd_lims[field][1] - cp_ffd_lims[field][0]
        cp_ffd_lims[field][0] = cp_ffd_lims[field][0] - 0.2*cp_range
        cp_ffd_lims[field][1] = cp_ffd_lims[field][1] + 0.2*cp_range
    FFD_block = create_3D_block(ffd_block_num_el, p, cp_ffd_lims)
    nonmatching_opt.set_shopt_FFD(FFD_block.knots, FFD_block.control)
    nonmatching_opt.set_shopt_align_CPFFD(align_dir=[[1],[2],[0]])
    nonmatching_opt.set_shopt_regu_CPFFD()


    cpfe2iga = CPFE2IGAImOperation(nonmatching_opt)
    cpfe2iga.apply_nonlinear()
    cpfe2iga.solve_nonlinear()

    d_outputs_array_list = [np.ones(nonmatching_opt.vec_scalar_iga_dof)
                            for i in range(len(nonmatching_opt.opt_field))]
    d_residuals_array_list = [np.ones(nonmatching_opt.vec_scalar_iga_dof)
                              for i in range(len(nonmatching_opt.opt_field))]

    cpfe2iga.solve_linear_fwd(d_outputs_array_list, 
                              d_residuals_array_list)
    cpfe2iga.solve_linear_rev(d_outputs_array_list, 
                              d_residuals_array_list)