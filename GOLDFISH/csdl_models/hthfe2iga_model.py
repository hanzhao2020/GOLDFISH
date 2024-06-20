from GOLDFISH.nonmatching_opt_ffd import *
from GOLDFISH.operations.hthfe2iga_imop import *

import csdl
from csdl import Model, CustomImplicitOperation
from csdl_om import Simulator

class HthFE2IGAModel(Model):

    def initialize(self):
        self.parameters.declare('nonmatching_opt')
        self.parameters.declare('input_h_th_fe_name', default='thickness_FE')
        self.parameters.declare('output_h_th_iga_name', default='thickness_IGA')

    def init_parameters(self):
        self.nonmatching_opt = self.parameters['nonmatching_opt']
        self.input_h_th_fe_name = self.parameters['input_h_th_fe_name']
        self.output_h_th_iga_name = self.parameters['output_h_th_iga_name']

        self.op = HthFE2IGAOperation(
                  nonmatching_opt=self.nonmatching_opt,
                  input_h_th_fe_name=self.input_h_th_fe_name, 
                  output_h_th_iga_name=self.output_h_th_iga_name)
        self.op.init_parameters()

    def define(self):
        h_th_fe = self.declare_variable(self.op.input_h_th_fe_name,
                  shape=(self.op.input_shape),
                  val=self.nonmatching_opt.init_h_th_fe)
        h_th_iga = csdl.custom(h_th_fe, op=self.op)
        self.register_output(self.op.output_h_th_iga_name, h_th_iga)


class HthFE2IGAOperation(CustomImplicitOperation):

    def initialize(self):
        self.parameters.declare('nonmatching_opt')
        self.parameters.declare('input_h_th_fe_name', default='thickness_FE')
        self.parameters.declare('output_h_th_iga_name', default='thickness_IGA')

    def init_parameters(self):
        self.nonmatching_opt = self.parameters['nonmatching_opt']
        self.input_h_th_fe_name = self.parameters['input_h_th_fe_name']
        self.output_h_th_iga_name = self.parameters['output_h_th_iga_name']

        self.h_th_fe2iga_imop = HthFE2IGAImOperation(self.nonmatching_opt)
        self.input_shape = self.nonmatching_opt.vec_scalar_fe_dof
        self.output_shape = self.nonmatching_opt.vec_scalar_iga_dof
        self.h_th_fe_vec = self.h_th_fe2iga_imop.h_th_fe_vec
        self.h_th_iga_vec = self.h_th_fe2iga_imop.h_th_iga_vec

    def define(self):
        self.add_input(self.input_h_th_fe_name,
                       shape=self.input_shape)
        self.add_output(self.output_h_th_iga_name,
                        shape=self.output_shape)
        self.declare_derivatives(self.output_h_th_iga_name,
                              self.input_h_th_fe_name,)
                            #   val=self.h_th_fe2iga_imop.dRdh_th_fe_coo.data,
                            #   rows=self.h_th_fe2iga_imop.dRdh_th_fe_coo.row,
                            #   cols=self.h_th_fe2iga_imop.dRdh_th_fe_coo.col)
        self.declare_derivatives(self.output_h_th_iga_name,
                              self.output_h_th_iga_name,)
                            #   val=self.h_th_fe2iga_imop.dRdh_th_iga_coo.data,
                            #   rows=self.h_th_fe2iga_imop.dRdh_th_iga_coo.row,
                            #   cols=self.h_th_fe2iga_imop.dRdh_th_iga_coo.col)

    def update_cp_vecs(self, inputs, outputs):
        update_nest_vec(inputs[self.input_h_th_fe_name],
                        self.h_th_fe_vec)
        update_nest_vec(outputs[self.output_h_th_iga_name],
                        self.h_th_iga_vec)

    def evaluate_residuals(self, inputs, outputs, residuals):
        self.update_cp_vecs(inputs, outputs)
        res_array = self.h_th_fe2iga_imop.apply_nonlinear()
        residuals[self.output_h_th_iga_name] = res_array

    def solve_residual_equations(self, inputs, outputs):
        self.update_cp_vecs(inputs, outputs)
        cp_iga_arary = self.h_th_fe2iga_imop.solve_nonlinear()
        outputs[self.output_h_th_iga_name] = cp_iga_arary

    def compute_jacvec_product(self, inputs, outputs, d_inputs, 
                               d_outputs, d_residuals, mode):
        self.update_cp_vecs(inputs, outputs)
        d_inputs_array = None
        if self.input_h_th_fe_name in d_inputs:
            d_inputs_array = d_inputs[self.input_h_th_fe_name]
        d_outputs_array = None
        if self.output_h_th_iga_name in d_outputs:
            d_outputs_array = d_outputs[self.output_h_th_iga_name]
        d_residuals_array = None
        if self.output_h_th_iga_name in d_residuals:
            d_residuals_array = d_residuals[self.output_h_th_iga_name]            
        if mode == 'fwd':
            self.h_th_fe2iga_imop.apply_linear_fwd(d_inputs_array, 
                d_outputs_array, d_residuals_array)
        elif mode == 'rev':
            self.h_th_fe2iga_imop.apply_linear_rev(d_inputs_array, 
                d_outputs_array, d_residuals_array)

    def apply_inverse_jacobian(self, d_outputs, d_residuals, mode):
        d_outputs_array = d_outputs[self.output_h_th_iga_name]
        d_residuals_array = d_residuals[self.output_h_th_iga_name]
        if mode == 'fwd':
            self.h_th_fe2iga_imop.solve_linear_fwd(d_outputs_array,
                                                d_residuals_array)
        if mode == 'rev':
            self.h_th_fe2iga_imop.solve_linear_rev(d_outputs_array,
                                                d_residuals_array)

if __name__ == "__main__":
    from GOLDFISH.tests.test_tbeam import nonmatching_opt
    # from GOLDFISH.tests.test_slr import nonmatching_opt
    # from GOLDFISH.tests.test_dRdt import nonmatching_opt

    m = HthFE2IGAModel(nonmatching_opt=nonmatching_opt)
    m.init_parameters()
    sim = Simulator(m)
    sim.run()
    sim.check_partials(compact_print=True)