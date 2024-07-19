from GOLDFISH.nonmatching_opt_ffd import *

class DispImOpeartion(object):
    """
    Implicit operation to solve non-matching shell structures 
    displacements using nonlinear solver.
    """
    def __init__(self, nonmatching_opt):
        self.nonmatching_opt = nonmatching_opt
        self.comm = self.nonmatching_opt.comm

        self.opt_shape = self.nonmatching_opt.opt_shape        
        self.opt_field = self.nonmatching_opt.opt_field
        self.opt_thickness = self.nonmatching_opt.opt_thickness
        self.var_thickness = self.nonmatching_opt.var_thickness
        self.use_aero_pressure = self.nonmatching_opt.use_aero_pressure
        
        self.dres_iga = self.nonmatching_opt.vec_iga_nest.copy()
        self.du_iga = self.nonmatching_opt.vec_iga_nest.copy()

        if self.opt_shape:
            self.dcp_iga = self.nonmatching_opt.vec_scalar_iga_nest.copy()

        if self.opt_thickness:
            if self.var_thickness:
                self.dh_th = self.nonmatching_opt.vec_scalar_iga_nest.copy()
            else:
                self.dh_th = self.nonmatching_opt.h_th_nest.copy()

        if self.use_aero_pressure:
            self.dPaero = self.nonmatching_opt.linear_spline_vec_iga_nest.copy()

    def apply_nonlinear(self):
        res_iga = self.nonmatching_opt.RIGA()
        res_iga_array = get_petsc_vec_array(res_iga, self.comm)
        return res_iga_array

    def solve_nonlinear(self, max_it=30, rtol=1e-3):
        _, u_iga = self.nonmatching_opt.\
                   solve_nonlinear_nonmatching_problem(
                   max_it=max_it, zero_mortar_funcs=True, 
                   rtol=rtol, iga_dofs=True)
        u_iga_array = get_petsc_vec_array(u_iga, self.comm)
        return u_iga_array

    def linearize(self):
        self.dRdu_iga = self.nonmatching_opt.dRIGAduIGA()
        if self.opt_shape:
            self.dRigadcpiga_list = []
            for i, field in enumerate(self.opt_field):
                self.dRigadcpiga_list += [self.nonmatching_opt.
                                          dRIGAdCPIGA(field),]
        if self.opt_thickness:
            self.dRigadh_th = self.nonmatching_opt.dRIGAdh_th()
        if self.use_aero_pressure:
            self.dRigadPaero = self.nonmatching_opt.dRIGAdPaero()

    def apply_linear_fwd(self, d_inputs_array_list=None, 
                         d_outputs_array=None, d_residuals_array=None):
        """
        ``d_inputs_array_list`` is the list of control points in IGA DoFs
        ``d_outputs_array`` is the displacement in IGA DoFs
        ``d_residuals_array`` is the residual in IGA DoFs
        """
        if d_residuals_array is not None:
            if d_outputs_array is not None:
                update_nest_vec(d_outputs_array, self.du_iga)
                A_x_b(self.dRdu_iga, self.du_iga, self.dres_iga)
                dres_iga_array = get_petsc_vec_array(self.dres_iga, 
                                                     self.comm)
                d_residuals_array[:] += dres_iga_array
            if d_inputs_array_list is not None:
                if self.opt_shape:
                    for i, field in enumerate(self.opt_field):
                        update_nest_vec(d_inputs_array_list[i], self.dcp_iga)
                        A_x_b(self.dRigadcpiga_list[i], self.dcp_iga, 
                              self.dres_iga)
                        dres_iga_array = get_petsc_vec_array(self.dres_iga, 
                                                             self.comm)
                        d_residuals_array[:] += dres_iga_array
                if self.opt_thickness:
                    # Assume thickness vector is always on the last
                    # index of ``d_inputs_array_list``
                    update_nest_vec(d_inputs_array_list[len(self.opt_field)], 
                                    self.dh_th)
                    A_x_b(self.dRigadh_th, self.dh_th, self.dres_iga)
                    dres_iga_array = get_petsc_vec_array(self.dres_iga,
                                                         self.comm)
                    d_residuals_array[:] += dres_iga_array
                if self.use_aero_pressure:
                    update_nest_vec(d_inputs_array_list[len(self.opt_field)+1],
                                    self.dPaero)
                    A_x_b(self.dRigadPaero, self.dPaero, self.dres_iga)
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
        if d_residuals_array is not None:
            update_nest_vec(d_residuals_array, self.dres_iga)
            if d_outputs_array is not None:
                AT_x_b(self.dRdu_iga, self.dres_iga, self.du_iga)
                du_iga_array = get_petsc_vec_array(self.du_iga, self.comm)
                d_outputs_array[:] += du_iga_array
            if d_inputs_array_list is not None:
                if self.opt_shape:
                    for i, field in enumerate(self.opt_field):
                        AT_x_b(self.dRigadcpiga_list[i], self.dres_iga, 
                               self.dcp_iga)
                        dcp_iga_array = get_petsc_vec_array(self.dcp_iga, 
                                                            self.comm)
                        d_inputs_array_list[i][:] += dcp_iga_array
                if self.opt_thickness:
                    AT_x_b(self.dRigadh_th, self.dres_iga, self.dh_th)
                    dh_th_array = get_petsc_vec_array(self.dh_th, self.comm)
                    d_inputs_array_list[len(self.opt_field)][:] += dh_th_array
                if self.use_aero_pressure:
                    AT_x_b(self.dRigadPaero, self.dres_iga, self.dPaero)
                    dPaero_array = get_petsc_vec_array(self.dPaero, self.comm)
                    d_inputs_array_list[len(self.opt_field)+1][:] += dPaero_array
        return d_inputs_array_list, d_outputs_array

    def solve_linear_fwd(self, d_outputs_array, d_residuals_array):
        dRdu_IGA_temp = self.dRdu_iga.copy()
        update_nest_vec(d_residuals_array, self.dres_iga)
        d_outputs_array[:] = solve_Ax_b(dRdu_IGA_temp, self.dres_iga, 
                             array=True, comm=self.comm)
        return d_outputs_array

    def solve_linear_rev(self, d_outputs_array, d_residuals_array):
        dRdu_IGA_temp = self.dRdu_iga.copy()
        update_nest_vec(d_outputs_array, self.du_iga)
        d_residuals_array[:] = solve_ATx_b(dRdu_IGA_temp, self.du_iga, 
                               array=True, comm=self.comm)
        return d_residuals_array


if __name__ == '__main__':
    # from GOLDFISH.tests.test_tbeam import nonmatching_opt
    from GOLDFISH.tests.test_dRdt import nonmatching_opt

    disp_op = DispImOpeartion(nonmatching_opt)

    disp_op.apply_nonlinear()
    disp_op.solve_nonlinear()
    disp_op.linearize()