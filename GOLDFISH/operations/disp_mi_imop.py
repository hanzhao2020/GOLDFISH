from GOLDFISH.nonmatching_opt_ffd import *

class DispMintImOpeartion(object):
    """
    Implicit operation to solve non-matching shell structures 
    displacements using nonlinear solver.
    """
    def __init__(self, nonmatching_opt, save_files=False):
        # print("Running init ...")
        self.nonmatching_opt = nonmatching_opt
        self.save_files = save_files
        self.comm = self.nonmatching_opt.comm
        self.opt_field = self.nonmatching_opt.opt_field

        self.dres_iga = self.nonmatching_opt.vec_iga_nest.copy()
        self.du_iga = self.nonmatching_opt.vec_iga_nest.copy()
        # self.dcp_iga = self.nonmatching_opt.vec_scalar_iga_nest.copy()
        self.dcps_iga = [vec.copy() for vec in self.nonmatching_opt.cpdes_iga_nest]
        self.dxi_vec = self.nonmatching_opt.xi_nest.copy()
        # print("Finish runing init ...")

    def apply_nonlinear(self):
        # print("Running apply_nonlinear...")
        res_iga = self.nonmatching_opt.RIGA()
        res_iga_array = get_petsc_vec_array(res_iga, self.comm)
        # print("Finish Running apply_nonlinear...")
        return res_iga_array

    def solve_nonlinear(self, max_it=30, rtol=1e-3):
        # print("Running solve_nonlinear ...")
        _, u_iga = self.nonmatching_opt.\
                   solve_nonlinear_nonmatching_problem(
                   max_it=max_it, zero_mortar_funcs=True, 
                   rtol=rtol, iga_dofs=True)
        # _, u_iga = self.nonmatching_opt.\
        #            solve_linear_nonmatching_problem(iga_dofs=True)
        u_iga_array = get_petsc_vec_array(u_iga, self.comm)

        # print("Finish Running solve_nonlinear ...")
        return u_iga_array

    def linearize(self):
        # print("Running linearize ...")
        self.dRdu_iga = self.nonmatching_opt.dRIGAduIGA()
        self.dRigadcpiga_list = []
        for i, field in enumerate(self.opt_field):
            # print('*'*50, field)
            self.dRigadcpiga_list += [self.nonmatching_opt.
                                      dRIGAdCPIGA(field),]
        self.dRigadxi = self.nonmatching_opt.dRIGAdxi()
        return self.dRdu_iga, self.dRigadcpiga_list, self.dRigadxi
        # print("Finish Running linearize ...")

    def apply_linear_fwd(self, d_inputs_array_list=None, 
                         d_outputs_array=None, d_residuals_array=None):
        """
        ``d_inputs_array_list`` is the list of control points in IGA DoFs
        ``d_outputs_array`` is the displacement in IGA DoFs
        ``d_residuals_array`` is the residual in IGA DoFs
        """
        # print("Running apply_linear fwd ...")
        if d_residuals_array is not None:
            if d_outputs_array is not None:
                update_nest_vec(d_outputs_array, self.du_iga)
                A_x_b(self.dRdu_iga, self.du_iga, self.dres_iga)
                dres_iga_array = get_petsc_vec_array(self.dres_iga, 
                                                     self.comm)
                d_residuals_array[:] += dres_iga_array
            if d_inputs_array_list is not None:
                for i, field in enumerate(self.opt_field):
                    update_nest_vec(d_inputs_array_list[i], self.dcps_iga[i])
                    A_x_b(self.dRigadcpiga_list[i], self.dcps_iga[i], 
                          self.dres_iga)
                    dres_iga_array = get_petsc_vec_array(self.dres_iga, 
                                                         self.comm)
                    d_residuals_array[:] += dres_iga_array
                update_nest_vec(d_inputs_array_list[-1], self.dxi_vec)
                A_x_b(self.dRigadxi, self.dxi_vec, self.dres_iga)
                dres_iga_array = get_petsc_vec_array(self.dres_iga, 
                                                     self.comm)
                d_residuals_array[:] += dres_iga_array
        return d_residuals_array

    def apply_linear_rev(self, d_inputs_array_list=None, 
                         d_outputs_array=None, d_residuals_array=None):
        """
        ``d_inputs_array_list`` is the list of control points in IGA DoFs
        ``d_outputs_array`` is the displacement in IGA DoFs
        ``d_residuals_array`` is the residual in IGA DoFs
        """
        # print("Running apply_linear rev ...")
        if d_residuals_array is not None:
            update_nest_vec(d_residuals_array, self.dres_iga)
            if d_outputs_array is not None:
                AT_x_b(self.dRdu_iga, self.dres_iga, self.du_iga)
                du_iga_array = get_petsc_vec_array(self.du_iga, self.comm)
                d_outputs_array[:] += du_iga_array
            if d_inputs_array_list is not None:
                for i, field in enumerate(self.opt_field):
                    AT_x_b(self.dRigadcpiga_list[i], self.dres_iga, 
                           self.dcps_iga[i])
                    dcp_iga_array = get_petsc_vec_array(self.dcps_iga[i], 
                                                        self.comm)
                    d_inputs_array_list[i][:] += dcp_iga_array
                AT_x_b(self.dRigadxi, self.dres_iga, self.dxi_vec)
                dxi_array = get_petsc_vec_array(self.dxi_vec, comm=self.comm)
                d_inputs_array_list[-1][:] += dxi_array
        return d_inputs_array_list, d_outputs_array

    def solve_linear_fwd(self, d_outputs_array, d_residuals_array):
        # print("Running solve_linear fwd ...")
        dRdu_IGA_temp = self.dRdu_iga.copy()
        update_nest_vec(d_residuals_array, self.dres_iga)
        d_outputs_array[:] = solve_Ax_b(dRdu_IGA_temp, self.dres_iga, 
                             array=True, comm=self.comm)
        return d_outputs_array

    def solve_linear_rev(self, d_outputs_array, d_residuals_array):
        # print("Running solve_linear rev ...")
        dRdu_IGA_temp = self.dRdu_iga.copy()
        update_nest_vec(d_outputs_array, self.du_iga)
        d_residuals_array[:] = solve_ATx_b(dRdu_IGA_temp, self.du_iga, 
                               array=True, comm=self.comm)
        return d_residuals_array


if __name__ == '__main__':
    # from GOLDFISH.tests.test_tbeam import nonmatching_opt
    from GOLDFISH.tests.test_slr import nonmatching_opt

    disp_op = DispMintImOpeartion(nonmatching_opt)

    disp_op.apply_nonlinear()
    disp_op.solve_nonlinear()
    disp_op.linearize()

    # t0 = nonmatching_opt.dRIGAdxi_sub(0, 0)
    indices = [1,3,5,7]
    t = nonmatching_opt.dRIGAdxi(indices)