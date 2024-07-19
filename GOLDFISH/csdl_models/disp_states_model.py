from GOLDFISH.nonmatching_opt_ffd import *
from GOLDFISH.operations.disp_imop import *

import csdl_alpha as csdl

class DispStatesModel(csdl.experimental.CustomImplicitOperation):

    def __init__(self, nonmatching_opt, input_cp_iga_name_pre='CP_IGA',
                 input_h_th_name='h_th', input_Paero_name='aero_pressure',
                 output_u_name='u'):
        super().__init__()
        csdl.check_parameter(nonmatching_opt, 'nonmatching_opt')
        csdl.check_parameter(input_cp_iga_name_pre, 'input_cp_iga_name_pre')
        csdl.check_parameter(input_h_th_name, 'input_h_th_name')
        csdl.check_parameter(input_Paero_name, 'input_Paero_name')
        csdl.check_parameter(output_u_name, 'output_u_name')

        self.nonmatching_opt = nonmatching_opt
        self.input_cp_iga_name_pre = input_cp_iga_name_pre
        self.input_h_th_name = input_h_th_name
        self.input_Paero_name = input_Paero_name
        self.output_u_name = output_u_name
        self.save_files = False
        self.nonlinear_solver_rtol = 1e-3
        self.nonlinear_solver_max_it = 30

        self.major_iter_ind = 0
        self.func_eval_ind = 0
        self.func_eval_major_ind = []

        self.disp_state_imop = DispImOpeartion(self.nonmatching_opt)
        self.opt_field = self.nonmatching_opt.opt_field
        self.opt_shape = self.nonmatching_opt.opt_shape
        self.use_aero_pressure = self.nonmatching_opt.use_aero_pressure
        self.opt_thickness = self.nonmatching_opt.opt_thickness
        self.var_thickness = self.nonmatching_opt.var_thickness

        self.output_shape = self.nonmatching_opt.vec_iga_dof

        if self.opt_shape:
            self.input_cp_shape = self.nonmatching_opt.vec_scalar_iga_dof
            self.init_cp_iga = self.nonmatching_opt.get_init_CPIGA()
            self.input_cp_iga_name_list = []
            for i, field in enumerate(self.opt_field):
                self.input_cp_iga_name_list += \
                    [self.input_cp_iga_name_pre+str(field)]
        if self.opt_thickness:
            if self.var_thickness:
                self.input_h_th_shape = self.nonmatching_opt.vec_scalar_iga_dof
                self.init_h_th = np.ones(self.nonmatching_opt.vec_scalar_iga_dof)*0.1
            else:
                self.input_h_th_shape = self.nonmatching_opt.h_th_dof
                self.init_h_th = self.nonmatching_opt.init_h_th
        if self.use_aero_pressure:
            self.inpout_Paero_shape = \
                self.nonmatching_opt.linear_spline_vec_iga_dof

    def evaluate(self, inputs: csdl.VariableGroup):
        u = self.create_output(self.output_u_name, shape=(self.output_shape,))
        u.add_name(self.output_u_name)
        # output = csdl.VariableGroup()
        # output.u = u
        # self.declare_derivative_parameters(self.output_u_name, 
        #                                    self.output_u_name,
        #                                    dependent=True)

        if self.opt_shape:
            for i, field in enumerate(self.opt_field):
                self.declare_input(self.input_cp_iga_name_list[i],
                                   inputs.cp_iga[i])
                                   # val=self.init_cp_iga[:,field])
            # for i, field in enumerate(self.opt_field):
            #     self.declare_derivative_parameters(self.output_u_name,
            #                           self.input_cp_iga_name_list[i],
            #                           dependent=True)
        if self.opt_thickness:
            self.declare_input(self.input_h_th_name, inputs.h_th)
                           # shape=self.input_h_th_shape,)
                           # val=self.init_h_th
            # self.declare_derivative_parameters(self.output_u_name, 
            #     self.input_h_th_name, dependent=True)
        if self.use_aero_pressure:
            self.declare_input(self.input_Paero_name, inputs.Paero)
                               # shape=self.inpout_Paero_shape)
            # self.declare_derivative_parameters(self.output_u_name, 
            #     self.input_Paero_name, dependent=True)

        self.declare_derivative_parameters(self.output_u_name, '*', dependent=True)
        
        return u

    def update_inputs_outpus(self, input_vals, output_vals):
        if self.opt_shape:
            for i, field in enumerate(self.opt_field):
                self.nonmatching_opt.update_CPIGA(
                    input_vals[self.input_cp_iga_name_list[i]], field)
        if self.opt_thickness:
            if self.var_thickness:
                self.nonmatching_opt.update_h_th_IGA(
                                     input_vals[self.input_h_th_name])
            else:
                self.nonmatching_opt.update_h_th(input_vals[self.input_h_th_name])
        if self.use_aero_pressure:
            self.nonmatching_opt.update_Paero(input_vals[self.input_Paero_name])
        self.nonmatching_opt.update_uIGA(output_vals[self.output_u_name])

    def solve_residual_equations(self, input_vals, output_vals):
        # self.update_inputs_outpus(input_vals, output_vals)
        output_vals[self.output_u_name] = self.disp_state_imop.solve_nonlinear(
                                      self.nonlinear_solver_max_it,
                                      self.nonlinear_solver_rtol)
        self.update_inputs_outpus(input_vals, output_vals)

        self.func_eval_ind += 1

        # if self.save_files:
        #     self.nonmatching_opt.save_files(thickness=self.opt_thickness)

    # def compute_derivatives(self, inputs, outputs, derivatives):
    #     self.update_inputs_outpus(inputs, outputs)
    #     self.disp_state_imop.linearize()

    #     if self.save_files:
    #         self.func_eval_major_ind += [self.func_eval_ind-1]
    #         print("**** Saving pvd files, ind: {:6d} ****"
    #               .format(self.major_iter_ind))
    #         self.nonmatching_opt.save_files(
    #             thickness=self.opt_thickness)
    #         self.major_iter_ind += 1

    def compute_jacvec_product(self, input_vals, output_vals, d_inputs, 
                               d_outputs, d_residuals, mode):
        self.update_inputs_outpus(input_vals, output_vals)
        self.disp_state_imop.linearize()
        self.major_iter_ind += 1

        d_inputs_array_list = []
        if self.opt_shape:
            for i, field in enumerate(self.opt_field):
                if self.input_cp_iga_name_list[i] in d_inputs:
                    d_inputs_array_list += \
                        [d_inputs[self.input_cp_iga_name_list[i]]]
        if self.opt_thickness:
            if self.input_h_th_name in d_inputs:
                d_inputs_array_list += [d_inputs[self.input_h_th_name]]
        if self.use_aero_pressure:
            if self.input_Paero_name in d_inputs:
                d_inputs_array_list += [d_inputs[self.input_Paero_name]]
        if len(d_inputs_array_list) == 0:
            d_inputs_array_list = None

        d_outputs_array = None
        if self.output_u_name in d_outputs:
            d_outputs_array = d_outputs[self.output_u_name]
        d_residuals_array = None
        if self.output_u_name in d_residuals:
            d_residuals_array = d_residuals[self.output_u_name]
            
        if mode == 'fwd':
            self.disp_state_imop.apply_linear_fwd(d_inputs_array_list, 
                d_outputs_array, d_residuals_array)
        elif mode == 'rev':
            self.disp_state_imop.apply_linear_rev(d_inputs_array_list, 
                d_outputs_array, d_residuals_array)

    def apply_inverse_jacobian(self, input_vals, output_vals, 
                               d_outputs, d_residuals, mode):
        self.update_inputs_outpus(input_vals, output_vals)
        self.disp_state_imop.linearize()
        d_outputs_array = d_outputs[self.output_u_name]
        d_residuals_array = d_residuals[self.output_u_name]
        if mode == 'fwd':
            self.disp_state_imop.solve_linear_fwd(d_outputs_array,
                                                  d_residuals_array)
        if mode == 'rev': 
            self.disp_state_imop.solve_linear_rev(d_outputs_array,
                                                  d_residuals_array)

if __name__ == "__main__":
    # from csdl_om import Simulator
    from python_csdl_backend import Simulator
    from GOLDFISH.tests.test_dRdt import nonmatching_opt
    # from GOLDFISH.tests.test_tbeam import nonmatching_opt
    # from GOLDFISH.tests.test_slr import nonmatching_opt

    # ffd_block_num_el = [4,4,1]
    # p = 3
    # # Create FFD block in igakit format
    # cp_ffd_lims = nonmatching_opt.cpsurf_lims
    # for field in [2]:
    #     cp_range = cp_ffd_lims[field][1] - cp_ffd_lims[field][0]
    #     cp_ffd_lims[field][0] = cp_ffd_lims[field][0] - 0.2*cp_range
    #     cp_ffd_lims[field][1] = cp_ffd_lims[field][1] + 0.2*cp_range
    # FFD_block = create_3D_block(ffd_block_num_el, p, cp_ffd_lims)
    # nonmatching_opt.set_FFD(FFD_block.knots, FFD_block.control)

    # m = DispStatesModel(nonmatching_opt=nonmatching_opt)
    # m.init_parameters()
    # sim = Simulator(m, analytics=False)
    # sim.run()
    # print("Check partials:")
    # sim.check_partials(compact_print=True)

    nonmatching_opt.get_init_CPIGA()

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    inputs = csdl.VariableGroup()
    inputs.cp_iga = []
    for i, field in enumerate(nonmatching_opt.opt_field):
        inputs.cp_iga += [csdl.Variable(value=nonmatching_opt.init_cp_iga[i], 
                                        name='cp_iga'+str(field))]

    inputs.h_th = csdl.Variable(value=nonmatching_opt.init_h_th, name='h_th')

    # inputs.h_th = csdl.Variable(value=0.0, name='h_th')

    m = DispStatesModel(nonmatching_opt=nonmatching_opt)

    u = m.evaluate(inputs)
    # u = outputs.u

    print(u.value)

    # from csdl_alpha.src.operations.derivative.utils import verify_derivatives_inline
    # print("Checking derivatives ....")
    # verify_derivatives_inline([u], inputs.cp_iga, 
    #                           step_size=1e-6, raise_on_error=False)
    recorder.stop()