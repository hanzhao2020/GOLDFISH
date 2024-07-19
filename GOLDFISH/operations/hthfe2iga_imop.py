from GOLDFISH.nonmatching_opt_ffd import *
from scipy.sparse import coo_matrix, csr_matrix
from scipy.linalg import block_diag

class HthFE2IGAImOperation(object):
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
        self.num_splines = self.nonmatching_opt.num_splines
        self.comm = self.nonmatching_opt.comm

        self.h_th_fe_vec = self.nonmatching_opt.vec_scalar_fe_nest.copy()
        self.h_th_iga_vec = self.nonmatching_opt.vec_scalar_iga_nest.copy()

        self.dRdh_th_iga_list = []
        self.dRdh_th_iga_csr_list = []
        self.dRdh_th_fe_list = []
        self.dRdh_th_fe_csr_list = []
        for s_ind in range(self.num_splines):
            Mc = m2p(self.nonmatching_opt.splines[s_ind].M_control)
            dRdh_th_iga_sub = Mc.transposeMatMult(Mc)
            McT = Mc.copy()
            McT.transpose()
            dRdh_th_fe_sub = -McT
            self.dRdh_th_iga_list += [dRdh_th_iga_sub]
            self.dRdh_th_iga_csr_list += [csr_matrix(dRdh_th_iga_sub.
                                        getValuesCSR()[::-1],
                                        shape=dRdh_th_iga_sub.size)]
            self.dRdh_th_fe_list += [dRdh_th_fe_sub]
            self.dRdh_th_fe_csr_list += [csr_matrix(dRdh_th_fe_sub.
                                       getValuesCSR()[::-1],
                                       shape=dRdh_th_fe_sub.size)]

        dRdh_th_iga_dense_list = [csr_mat.todense() for csr_mat
                                in self.dRdh_th_iga_csr_list]
        dRdh_th_fe_dense_list = [csr_mat.todense() for csr_mat
                               in self.dRdh_th_fe_csr_list]
        self.dRdh_th_iga_coo = coo_matrix(block_diag(*dRdh_th_iga_dense_list))
        self.dRdh_th_fe_coo = coo_matrix(block_diag(*dRdh_th_fe_dense_list))

        self.sub_vec_iga_sizes = [vec.sizes[1] for vec in
            self.nonmatching_opt.vec_scalar_iga_nest.getNestSubVecs()]
        self.sub_vec_fe_sizes = [vec.sizes[1] for vec in
            self.nonmatching_opt.vec_scalar_fe_nest.getNestSubVecs()]

        self.dh_th_iga_vec = self.nonmatching_opt.vec_scalar_iga_nest.copy()
        self.dh_th_fe_vec = self.nonmatching_opt.vec_scalar_fe_nest.copy()
        self.dres_vec = self.nonmatching_opt.vec_scalar_iga_nest.copy()

    def apply_nonlinear(self):
        res_sub_array_list = []
        for s_ind in range(self.num_splines):
            Mc = m2p(self.nonmatching_opt.splines[s_ind].M_control)
            h_th_iga_sub = self.h_th_iga_vec.getNestSubVecs()[s_ind]
            h_th_fe_sub = self.h_th_fe_vec.getNestSubVecs()[s_ind]
            Mch_th_iga_sub = A_x(Mc, h_th_iga_sub)
            res_sub_fe = Mch_th_iga_sub - h_th_fe_sub
            res_sub_fe.assemble()
            res_sub_iga = AT_x(Mc, res_sub_fe)
            res_sub_array_list += [get_petsc_vec_array(res_sub_iga)]
        res_array = np.concatenate(res_sub_array_list)
        return res_array

    def solve_nonlinear(self):
        h_th_iga_sub_array_list = []
        for s_ind in range(self.num_splines):
            Mc = m2p(self.nonmatching_opt.splines[s_ind].M_control)
            h_th_iga_sub = self.h_th_iga_vec.getNestSubVecs()[s_ind]
            h_th_fe_sub = self.h_th_fe_vec.getNestSubVecs()[s_ind]
            McTMc = self.dRdh_th_iga_list[s_ind]
            McTcp_fe = AT_x(Mc, h_th_fe_sub)
            solve(PETScMatrix(McTMc), PETScVector(h_th_iga_sub),
                  PETScVector(McTcp_fe), 'mumps')
            # h_th_iga_sub.assemble()
            h_th_iga_sub_array_list += [get_petsc_vec_array(h_th_iga_sub)]
        h_th_iga_array = np.concatenate(h_th_iga_sub_array_list)
        return h_th_iga_array

    def apply_linear_fwd(self, d_inputs_array=None, 
                         d_outputs_array=None, 
                         d_residuals_array=None):
        # PETSc implementation
        if d_residuals_array is not None:
            if d_outputs_array is not None:
                update_nest_vec(d_outputs_array, self.dh_th_iga_vec)
                dres_array_out_list = []
                for s_ind in range(self.num_splines):
                    h_th_iga_sub = self.dh_th_iga_vec.getNestSubVecs()[s_ind]
                    dres_sub_out = A_x(self.dRdh_th_iga_list[s_ind], 
                                       h_th_iga_sub)
                    dres_sub_array_out = get_petsc_vec_array(
                                         dres_sub_out, self.comm)
                    dres_array_out_list += [dres_sub_array_out]
                d_residuals_array[:] += np.concatenate(dres_array_out_list)
            if d_inputs_array is not None:
                update_nest_vec(d_inputs_array, self.dh_th_fe_vec)
                dres_array_in_list = []
                for s_ind in range(self.num_splines):
                    h_th_fe_sub = self.dh_th_fe_vec.getNestSubVecs()[s_ind]
                    dres_sub_in = A_x(self.dRdh_th_fe_list[s_ind], 
                                        h_th_fe_sub)
                    dres_sub_array_in = get_petsc_vec_array(dres_sub_in, 
                                        self.comm)
                    dres_array_in_list += [dres_sub_array_in]
                d_residuals_array[:] += np.concatenate(dres_array_in_list)
        # # Numpy implementation
        # if d_residuals_array is not None:
        #     if d_outputs_array_list is not None:
        #         for i, field in enumerate(self.opt_field):
        #             dres_array_out = self.dRdh_th_iga_coo\
        #                              *d_outputs_array_list[i]
        #             d_residuals_array[i][:] += dres_array_out
        #     if d_inputs_array_list is not None:
        #         for i, field in enumerate(self.opt_field):
        #             dres_array_in = self.dRdh_th_fe_coo\
        #                             *d_inputs_array_list[i]
        #             d_residuals_array[i][:] += dres_array_in
        return d_residuals_array

    def apply_linear_rev(self, d_inputs_array=None, 
                         d_outputs_array=None,
                         d_residuals_array=None):
        # PETSc implementation
        if d_residuals_array is not None:
            update_nest_vec(d_residuals_array, self.dres_vec)
            if d_outputs_array is not None:
                dh_th_iga_array_list = []
                for s_ind in range(self.num_splines):
                    dres_sub = self.dres_vec.getNestSubVecs()[s_ind]
                    dh_th_iga_sub = self.dh_th_iga_vec.getNestSubVecs()[s_ind]
                    AT_x_b(self.dRdh_th_iga_list[s_ind], dres_sub, 
                           dh_th_iga_sub)
                    dh_th_iga_sub_array = get_petsc_vec_array(dh_th_iga_sub, 
                                                            self.comm)
                    dh_th_iga_array_list += [dh_th_iga_sub_array]
                d_outputs_array[:] += np.concatenate(dh_th_iga_array_list)
            if d_inputs_array is not None:
                dcp_fe_array_list = []
                for s_ind in range(self.num_splines):
                    dres_sub = self.dres_vec.getNestSubVecs()[s_ind]
                    dh_th_fe_sub = self.dh_th_fe_vec.getNestSubVecs()[s_ind]
                    AT_x_b(self.dRdh_th_fe_list[s_ind], dres_sub, 
                           dh_th_fe_sub)
                    dh_th_fe_sub_array = get_petsc_vec_array(dh_th_fe_sub, 
                                                             self.comm)
                    dcp_fe_array_list += [dh_th_fe_sub_array]
                d_inputs_array[:] += np.concatenate(dcp_fe_array_list)

        # # Numpy implementation
        # if d_residuals_array is not None:
        #     if d_outputs_array is not None:
        #         for i, field in enumerate(self.opt_field):
        #             dcp_iga_array = self.dRdh_th_iga_coo.T\
        #                             *d_residuals_array[i]
        #             d_outputs_array[i][:] += dcp_iga_array
        #     if d_inputs_array is not None:
        #         for i, field in enumerate(self.opt_field):
        #             dcp_fe_array = self.dRdh_th_fe_coo.T\
        #                            *d_residuals_array[i]
        #             d_inputs_array[i][:] += dcp_fe_array
        return d_inputs_array, d_outputs_array

    def solve_linear_fwd(self, d_outputs_array, d_residuals_array):
        d_outputs_sub_array_list = []
        for s_ind in range(self.num_splines):
            dh_th_iga_sub = self.dh_th_iga_vec.getNestSubVecs()[s_ind]
            dres_sub = self.dres_vec.getNestSubVecs()[s_ind]
            dres_sub_array = d_residuals_array\
                [int(np.sum(self.sub_vec_iga_sizes[:s_ind])):
                 int(np.sum(self.sub_vec_iga_sizes[:s_ind+1]))]
            # print(dres_sub.sizes[1], "--", len(dres_sub_array))
            dres_sub.setValues(range(dres_sub.sizes[1]),
                                dres_sub_array)
            dres_sub.assemble()
            solve(PETScMatrix(self.dRdh_th_iga_list[s_ind]),
                  PETScVector(dh_th_iga_sub),
                  PETScVector(dres_sub))
            d_outputs_sub_array_list += [get_petsc_vec_array(
                                         dh_th_iga_sub),]
        d_outputs_array[:] = np.concatenate(d_outputs_sub_array_list)
        return d_outputs_array

    def solve_linear_rev(self, d_outputs_array, d_residuals_array):
        d_residuals_sub_array_list = []
        for s_ind in range(self.num_splines):
            dh_th_iga_sub = self.dh_th_iga_vec.getNestSubVecs()[s_ind]
            dh_th_iga_sub_array = d_outputs_array\
                [int(np.sum(self.sub_vec_iga_sizes[:s_ind])):
                 int(np.sum(self.sub_vec_iga_sizes[:s_ind+1]))]
            dh_th_iga_sub.setValues(range(dh_th_iga_sub.sizes[1]),
                                  dh_th_iga_sub_array)
            dh_th_iga_sub.assemble()
            dres_sub = self.dres_vec.getNestSubVecs()[s_ind]
            solve(PETScMatrix(self.dRdh_th_iga_list[s_ind]),
                  PETScVector(dres_sub),
                  PETScVector(dh_th_iga_sub), 'mumps')
            d_residuals_sub_array_list += [get_petsc_vec_array(
                                           dres_sub),]
        d_residuals_array[:] = np.concatenate(d_residuals_sub_array_list)
        return d_residuals_array


if __name__ == '__main__':
    from GOLDFISH.tests.test_tbeam import nonmatching_opt
    # from GOLDFISH.tests.test_slr import nonmatching_opt

    # Create FFD block
    num_el = [4,1,1]
    p = 3
    cp_ffd_lims = nonmatching_opt.cpsurf_lims
    for field in range(nonmatching_opt.nsd):
        cp_range = cp_ffd_lims[field][1] - cp_ffd_lims[field][0]
        cp_ffd_lims[field][0] = cp_ffd_lims[field][0] - 0.2*cp_range
        cp_ffd_lims[field][1] = cp_ffd_lims[field][1] + 0.2*cp_range

    FFD_block = create_3D_block(num_el, p, cp_ffd_lims)

    nonmatching_opt.set_FFD(FFD_block.knots, FFD_block.control)
    nonmatching_opt.set_regu_CPFFD([None, 0, 1], [None, 1, 1])
    nonmatching_opt.set_pin_CPFFD(2, [0], 1, [0])
    nonmatching_opt.set_align_CPFFD(1)

    h_th_fe2iga = HthFE2IGAImOperation(nonmatching_opt)
    h_th_fe2iga.apply_nonlinear()
    h_th_fe2iga.solve_nonlinear()

    # d_outputs_array_list = [np.ones(nonmatching_opt.vec_scalar_iga_dof)
    #                         for i in range(len(nonmatching_opt.opt_field))]
    # d_residuals_array_list = [np.ones(nonmatching_opt.vec_scalar_iga_dof)
    #                           for i in range(len(nonmatching_opt.opt_field))]


    d_outputs_array = np.ones(nonmatching_opt.vec_scalar_iga_dof)
    d_residuals_array = np.ones(nonmatching_opt.vec_scalar_iga_dof)

    h_th_fe2iga.solve_linear_fwd(d_outputs_array, 
                              d_residuals_array)
    h_th_fe2iga.solve_linear_rev(d_outputs_array, 
                              d_residuals_array)