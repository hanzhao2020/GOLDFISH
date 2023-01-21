from GOLDFISH.nonmatching_opt_ffd import *
from GOLDFISH.operations.disp_imop import *
import openmdao.api as om
from openmdao.api import Problem

class DispStatesComp(om.ImplicitComponent):

    def initialize(self):
        self.options.declare('nonmatching_opt')
        self.options.declare('input_cp_iga_name_pre', default='CP_IGA')
        self.options.declare('output_u_name', default='displacements')

    def init_paramters(self, save_files=False):
        self.nonmatching_opt = self.options['nonmatching_opt']
        self.input_cp_iga_name_pre = self.options['input_cp_iga_name_pre']
        self.output_u_name = self.options['output_u_name']
        self.save_files = save_files

        self.disp_state_imop = DispImOpeartion(self.nonmatching_opt)
        self.opt_field = self.nonmatching_opt.opt_field
        self.input_shape = self.nonmatching_opt.vec_scalar_iga_dof
        self.output_shape = self.nonmatching_opt.vec_iga_dof
        self.init_cp_iga = self.nonmatching_opt.get_init_CPIGA()

        self.input_cp_iga_name_list = []
        for i, field in enumerate(self.opt_field):
            self.input_cp_iga_name_list += \
                [self.input_cp_iga_name_pre+str(field)]

    def setup(self):
        self.add_output(self.output_u_name, shape=self.output_shape)
        self.declare_partials(self.output_u_name, self.output_u_name)
        for i, field in enumerate(self.opt_field):
            self.add_input(self.input_cp_iga_name_list[i],
                           shape=self.input_shape,
                           val=self.init_cp_iga[:,field])
            self.declare_partials(self.output_u_name,
                                  self.input_cp_iga_name_list[i])

    def update_inputs_outpus(self, inputs, outputs):
        for i, field in enumerate(self.opt_field):
            self.nonmatching_opt.update_CPIGA(
                inputs[self.input_cp_iga_name_list[i]], field)
        self.nonmatching_opt.update_uIGA(outputs[self.output_u_name])

    def apply_nonlinear(self, inputs, outputs, residuals):
        self.update_inputs_outpus(inputs, outputs)
        residuals[self.output_u_name] = self.disp_state_imop.apply_nonlinear()

    def solve_nonlinear(self, inputs, outputs):
        self.update_inputs_outpus(inputs, outputs)
        outputs[self.output_u_name] = self.disp_state_imop.solve_nonlinear()
        if self.save_files:
            self.nonmatching_opt.save_files(thickness=False)

    def linearize(self, inputs, outputs, partials):
        self.update_inputs_outpus(inputs, outputs)
        self.disp_state_imop.linearize()

    def apply_linear(self, inputs, outputs, d_inputs, 
                     d_outputs, d_residuals, mode):
        self.update_inputs_outpus(inputs, outputs)
        d_inputs_array_list = []
        for i, field in enumerate(self.opt_field):
            if self.input_cp_iga_name_list[i] in d_inputs:
                d_inputs_array_list += \
                    [d_inputs[self.input_cp_iga_name_list[i]]]
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

    prob = Problem()
    comp = DispStatesComp(nonmatching_opt=nonmatching_opt)
    comp.init_paramters()
    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print('check_partials:')
    prob.check_partials(compact_print=True)