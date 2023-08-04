from GOLDFISH.nonmatching_opt_ffd import *
from GOLDFISH.operations.disp_imop import *
import openmdao.api as om
from openmdao.api import Problem

class DispStatesComp(om.ImplicitComponent):

    def initialize(self):
        self.options.declare('nonmatching_opt')
        self.options.declare('input_cp_iga_name_pre', default='CP_IGA')
        self.options.declare('input_h_th_name', default='thickness')
        self.options.declare('output_u_name', default='displacements')

    def init_paramters(self, save_files=False, nonlinear_solver_rtol=1e-3,
                       nonlinear_solver_max_it=30):
        self.nonmatching_opt = self.options['nonmatching_opt']
        self.input_cp_iga_name_pre = self.options['input_cp_iga_name_pre']
        self.input_h_th_name = self.options['input_h_th_name']
        self.output_u_name = self.options['output_u_name']
        self.save_files = save_files
        self.nonlinear_solver_max_it = nonlinear_solver_max_it
        self.nonlinear_solver_rtol = nonlinear_solver_rtol
        
        self.major_iter_ind = 0
        self.func_eval_ind = 0
        self.func_eval_major_ind = []

        self.disp_state_imop = DispImOpeartion(self.nonmatching_opt)
        self.opt_field = self.nonmatching_opt.opt_field
        self.opt_shape = self.nonmatching_opt.opt_shape
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


    def setup(self):
        self.add_output(self.output_u_name, shape=self.output_shape)
        self.declare_partials(self.output_u_name, self.output_u_name)

        if self.opt_shape:
            for i, field in enumerate(self.opt_field):
                self.add_input(self.input_cp_iga_name_list[i],
                               shape=self.input_cp_shape,
                               val=self.init_cp_iga[:,field])
                self.declare_partials(self.output_u_name,
                                      self.input_cp_iga_name_list[i])
        if self.opt_thickness:
            self.add_input(self.input_h_th_name, shape=self.input_h_th_shape,
                           val=self.init_h_th)
            self.declare_partials(self.output_u_name, self.input_h_th_name)

    def update_inputs_outpus(self, inputs, outputs):
        if self.opt_shape:
            for i, field in enumerate(self.opt_field):
                self.nonmatching_opt.update_CPIGA(
                    inputs[self.input_cp_iga_name_list[i]], field)
        if self.opt_thickness:
            if self.var_thickness:
                self.nonmatching_opt.update_h_th_IGA(
                                     inputs[self.input_h_th_name])
            else:
                self.nonmatching_opt.update_h_th(inputs[self.input_h_th_name])
        self.nonmatching_opt.update_uIGA(outputs[self.output_u_name])

    def apply_nonlinear(self, inputs, outputs, residuals):
        self.update_inputs_outpus(inputs, outputs)
        residuals[self.output_u_name] = self.disp_state_imop.apply_nonlinear()

    def solve_nonlinear(self, inputs, outputs):
        self.update_inputs_outpus(inputs, outputs)
        outputs[self.output_u_name] = self.disp_state_imop.solve_nonlinear(
                                      self.nonlinear_solver_max_it,
                                      self.nonlinear_solver_rtol)
        self.func_eval_ind += 1
        # if self.save_files:
        #     self.nonmatching_opt.save_files(
        #         thickness=self.opt_thickness)

    def linearize(self, inputs, outputs, partials):
        self.update_inputs_outpus(inputs, outputs)
        self.disp_state_imop.linearize()

        if self.save_files:
            self.func_eval_major_ind += [self.func_eval_ind-1]
            # print("**** Saving pvd files, ind: {:6d} ****"
            #       .format(self.major_iter_ind))
            self.nonmatching_opt.save_files(
                thickness=self.opt_thickness)
            self.major_iter_ind += 1

    def apply_linear(self, inputs, outputs, d_inputs, 
                     d_outputs, d_residuals, mode):
        self.update_inputs_outpus(inputs, outputs)
        d_inputs_array_list = []
        if self.opt_shape:
            for i, field in enumerate(self.opt_field):
                if self.input_cp_iga_name_list[i] in d_inputs:
                    d_inputs_array_list += \
                        [d_inputs[self.input_cp_iga_name_list[i]]]
        if self.opt_thickness:
            if self.input_h_th_name in d_inputs:
                d_inputs_array_list += [d_inputs[self.input_h_th_name]]
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

    def solve_linear(self, d_outputs, d_residuals, mode):
        d_outputs_array = d_outputs[self.output_u_name]
        d_residuals_array = d_residuals[self.output_u_name]
        if mode == 'fwd':
            self.disp_state_imop.solve_linear_fwd(d_outputs_array,
                                                  d_residuals_array)
        if mode == 'rev': 
            self.disp_state_imop.solve_linear_rev(d_outputs_array,
                                                  d_residuals_array)

if __name__ == "__main__":
    from GOLDFISH.tests.test_tbeam import nonmatching_opt
    # from GOLDFISH.tests.test_slr import nonmatching_opt
    # from GOLDFISH.tests.test_dRdt import nonmatching_opt

    prob = Problem()
    comp = DispStatesComp(nonmatching_opt=nonmatching_opt)
    comp.init_paramters()
    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print('check_partials:')
    prob.check_partials(compact_print=True)