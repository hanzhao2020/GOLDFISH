import sys
sys.path.append("../opers/")
# sys.path.append("../")

from GOLDFISH.nonmatching_opt_ffd import *
from cpiga2xi_imop import *
import openmdao.api as om
from openmdao.api import Problem

class CPIGA2XiComp(om.ImplicitComponent):

    def initialize(self):
        self.options.declare('nonmatching_opt')
        self.options.declare('input_cp_iga_name_pre', default='CP_IGA')
        self.options.declare('output_xi_name', default='int_para_coord')

    def init_parameters(self):
        self.nonmatching_opt = self.options['nonmatching_opt']
        self.opt_field = self.nonmatching_opt.opt_field
        self.input_cp_iga_name_pre = self.options['input_cp_iga_name_pre']
        self.output_xi_name = self.options['output_xi_name']

        self.cpiga2xi_imop = CPIGA2XiImOperation(self.nonmatching_opt)

        self.input_cp_shapes = []
        for field_ind, field in enumerate(self.opt_field):        
            self.input_cp_shapes += [len(self.nonmatching_opt.cpdes_iga_dofs_full[field_ind])]
        self.init_cp_iga = self.nonmatching_opt.init_cp_iga
        self.output_shape = self.cpiga2xi_imop.cpiga2xi.xi_size_global

        self.input_cp_iga_name_list = []
        for i, field in enumerate(self.opt_field):
            self.input_cp_iga_name_list += \
                [self.input_cp_iga_name_pre+str(field)]

    def setup(self):
        for i, field in enumerate(self.opt_field):
            self.add_input(self.input_cp_iga_name_list[i],
                           shape=self.input_cp_shapes[i],
                           val=self.init_cp_iga[i])
        self.add_output(self.output_xi_name, shape=self.output_shape)
        for i, field in enumerate(self.opt_field):
            self.declare_partials(self.output_xi_name,
                                  self.input_cp_iga_name_list[i])
        self.declare_partials(self.output_xi_name, self.output_xi_name)

    def update_inputs(self, inputs):
        for i, field in enumerate(self.opt_field):
            self.cpiga2xi_imop.cpiga2xi.update_CPs(
                inputs[self.input_cp_iga_name_list[i]], field)

    def apply_nonlinear(self, inputs, outputs, residuals):
        self.update_inputs(inputs)
        self.cpiga2xi_imop.cpiga2xi.update_occ_surfs()
        xi_flat = outputs[self.output_xi_name]
        residuals[self.output_xi_name] = self.cpiga2xi_imop.apply_nonlinear(xi_flat)

    def solve_nonlinear(self, inputs, outputs):
        self.update_inputs(inputs)
        self.cpiga2xi_imop.cpiga2xi.update_occ_surfs()
        xi_flat_init = self.cpiga2xi_imop.cpiga2xi.xi_flat_global
        outputs[self.output_xi_name] = self.cpiga2xi_imop.solve_nonlinear(xi_flat_init)

    def linearize(self, inputs, outputs, partials):
        self.update_inputs(inputs)
        self.cpiga2xi_imop.cpiga2xi.update_occ_surfs()
        # print("input_cp_iga_name_list[0]\n", 
        #       inputs[self.input_cp_iga_name_list[0]])
        xi_flat = outputs[self.output_xi_name]
        self.cpiga2xi_imop.linearize(xi_flat)

    def apply_linear(self, inputs, outputs, d_inputs, 
                     d_outputs, d_residuals, mode):
        self.update_inputs(inputs)
        self.cpiga2xi_imop.cpiga2xi.update_occ_surfs()
        d_inputs_array_list = []
        for i, field in enumerate(self.opt_field):
            if self.input_cp_iga_name_list[i] in d_inputs:
                d_inputs_array_list += \
                    [d_inputs[self.input_cp_iga_name_list[i]]]
        if len(d_inputs_array_list) == 0:
            d_inputs_array_list = None

        d_outputs_array = None
        if self.output_xi_name in d_outputs:
            d_outputs_array = d_outputs[self.output_xi_name]
        d_residuals_array = None
        if self.output_xi_name in d_residuals:
            d_residuals_array = d_residuals[self.output_xi_name]
            
        if mode == 'fwd':
            self.cpiga2xi_imop.apply_linear_fwd(d_inputs_array_list, 
                d_outputs_array, d_residuals_array)
        elif mode == 'rev':
            self.cpiga2xi_imop.apply_linear_rev(d_inputs_array_list, 
                d_outputs_array, d_residuals_array)

    def solve_linear(self, d_outputs, d_residuals, mode):
        d_outputs_array = d_outputs[self.output_xi_name]
        d_residuals_array = d_residuals[self.output_xi_name]
        if mode == 'fwd':
            self.cpiga2xi_imop.solve_linear_fwd(d_outputs_array,
                                                  d_residuals_array)
        if mode == 'rev': 
            self.cpiga2xi_imop.solve_linear_rev(d_outputs_array,
                                                  d_residuals_array)

if __name__ == "__main__":
    from GOLDFISH.tests.test_tbeam_mint import preprocessor, nonmatching_opt

    #################################
    nonmatching_opt.set_xi_diff_info(preprocessor)
    #################################

    prob = Problem()
    comp = CPIGA2XiComp(nonmatching_opt=nonmatching_opt)
    comp.init_parameters()
    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print('check_partials:')
    prob.check_partials(compact_print=True)