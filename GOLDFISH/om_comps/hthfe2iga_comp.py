from GOLDFISH.nonmatching_opt_ffd import *
from GOLDFISH.operations.hthfe2iga_imop import *
import openmdao.api as om
from openmdao.api import Problem

class HthFE2IGAComp(om.ImplicitComponent):

    def initialize(self):
        self.options.declare('nonmatching_opt')
        self.options.declare('input_h_th_fe_name', default='thickness_FE')
        self.options.declare('output_h_th_iga_name', default='thickness_IGA')

    def init_paramters(self):
        self.nonmatching_opt = self.options['nonmatching_opt']
        self.input_h_th_fe_name = self.options['input_h_th_fe_name']
        self.output_h_th_iga_name = self.options['output_h_th_iga_name']

        self.h_th_fe2iga_imop = HthFE2IGAImOperation(self.nonmatching_opt)
        self.input_shape = self.nonmatching_opt.vec_scalar_fe_dof
        self.output_shape = self.nonmatching_opt.vec_scalar_iga_dof
        self.h_th_fe_vec = self.h_th_fe2iga_imop.h_th_fe_vec
        self.h_th_iga_vec = self.h_th_fe2iga_imop.h_th_iga_vec

    def setup(self):
        self.add_input(self.input_h_th_fe_name,
                       shape=self.input_shape,
                       val=self.nonmatching_opt.init_h_th_fe)
        self.add_output(self.output_h_th_iga_name,
                        shape=self.output_shape)
        self.declare_partials(self.output_h_th_iga_name,
                              self.input_h_th_fe_name,
                              val=self.h_th_fe2iga_imop.dRdh_th_fe_coo.data,
                              rows=self.h_th_fe2iga_imop.dRdh_th_fe_coo.row,
                              cols=self.h_th_fe2iga_imop.dRdh_th_fe_coo.col)
        self.declare_partials(self.output_h_th_iga_name,
                              self.output_h_th_iga_name,
                              val=self.h_th_fe2iga_imop.dRdh_th_iga_coo.data,
                              rows=self.h_th_fe2iga_imop.dRdh_th_iga_coo.row,
                              cols=self.h_th_fe2iga_imop.dRdh_th_iga_coo.col)

    def update_cp_vecs(self, inputs, outputs):
        update_nest_vec(inputs[self.input_h_th_fe_name],
                        self.h_th_fe_vec)
        update_nest_vec(outputs[self.output_h_th_iga_name],
                        self.h_th_iga_vec)

    def apply_nonlinear(self, inputs, outputs, residuals):
        self.update_cp_vecs(inputs, outputs)
        res_array = self.h_th_fe2iga_imop.apply_nonlinear()
        residuals[self.output_h_th_iga_name] = res_array

    def solve_nonlinear(self, inputs, outputs):
        self.update_cp_vecs(inputs, outputs)
        cp_iga_arary = self.h_th_fe2iga_imop.solve_nonlinear()
        outputs[self.output_h_th_iga_name] = cp_iga_arary

    def solve_linear(self, d_outputs, d_residuals, mode):
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

    prob = Problem()
    comp = HthFE2IGAComp(nonmatching_opt=nonmatching_opt)
    comp.init_paramters()
    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print('check_partials:')
    prob.check_partials(compact_print=True)