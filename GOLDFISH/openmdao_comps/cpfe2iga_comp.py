from GOLDFISH.nonmatching_opt_ffd import *
from GOLDFISH.operations.cpfe2iga_imop import *
import openmdao.api as om
from openmdao.api import Problem

class CPFE2IGAComp(om.ImplicitComponent):

    def initialize(self):
        self.options.declare('nonmatching_opt')
        self.options.declare('input_cp_fe_name_pre', default='CP_FE')
        self.options.declare('output_cp_iga_name_pre', default='CP_IGA')

    def init_paramters(self):
        self.nonmatching_opt = self.options['nonmatching_opt']
        self.input_cp_fe_name_pre = self.options['input_cp_fe_name_pre']
        self.output_cp_iga_name_pre = self.options['output_cp_iga_name_pre']

        self.cpfe2iga_imop = CPFE2IGAImOperation(self.nonmatching_opt)
        self.opt_field = self.nonmatching_opt.opt_field
        self.input_shape = self.nonmatching_opt.vec_scalar_fe_dof
        self.output_shape = self.nonmatching_opt.vec_scalar_iga_dof
        self.cp_fe_vecs = self.cpfe2iga_imop.cp_fe_vecs
        self.cp_iga_vecs = self.cpfe2iga_imop.cp_iga_vecs

        self.input_cp_fe_name_list = []
        self.output_cp_iga_name_list = []
        for i, field in enumerate(self.opt_field):
            self.input_cp_fe_name_list += \
                [self.input_cp_fe_name_pre+str(field)]
            self.output_cp_iga_name_list += \
                [self.output_cp_iga_name_pre+str(field)]

    def setup(self):
        for i, field in enumerate(self.opt_field):
            init_cp_fe_array = get_petsc_vec_array(
                               self.nonmatching_opt.cp_funcs_nest[field], 
                               comm=self.nonmatching_opt.comm)
            self.add_input(self.input_cp_fe_name_list[i],
                           shape=self.input_shape,
                           val=init_cp_fe_array)
            self.add_output(self.output_cp_iga_name_list[i],
                            shape=self.output_shape)
            self.declare_partials(self.output_cp_iga_name_list[i],
                                  self.input_cp_fe_name_list[i],
                                  val=self.cpfe2iga_imop.dRdcp_fe_coo.data,
                                  rows=self.cpfe2iga_imop.dRdcp_fe_coo.row,
                                  cols=self.cpfe2iga_imop.dRdcp_fe_coo.col)
            self.declare_partials(self.output_cp_iga_name_list[i],
                                  self.output_cp_iga_name_list[i],
                                  val=self.cpfe2iga_imop.dRdcp_iga_coo.data,
                                  rows=self.cpfe2iga_imop.dRdcp_iga_coo.row,
                                  cols=self.cpfe2iga_imop.dRdcp_iga_coo.col)

    def update_cp_vecs(self, inputs, outputs):
        for i, field in enumerate(self.opt_field):
            update_nest_vec(inputs[self.input_cp_fe_name_list[i]],
                            self.cp_fe_vecs[i])
            update_nest_vec(outputs[self.output_cp_iga_name_list[i]],
                            self.cp_iga_vecs[i])

    def apply_nonlinear(self, inputs, outputs, residuals):
        self.update_cp_vecs(inputs, outputs)
        res_array_list = self.cpfe2iga_imop.apply_nonlinear()
        for i, field in enumerate(self.opt_field):
            residuals[self.output_cp_iga_name_list[i]] = res_array_list[i]

    def solve_nonlinear(self, inputs, outputs):
        self.update_cp_vecs(inputs, outputs)
        cp_iga_arary_list = self.cpfe2iga_imop.solve_nonlinear()
        for i, field in enumerate(self.opt_field):
            outputs[self.output_cp_iga_name_list[i]] = cp_iga_arary_list[i]

    def solve_linear(self, d_outputs, d_residuals, mode):
        d_outputs_array_list = []
        d_residuals_array_list = []
        for i, field in enumerate(self.opt_field):
            d_outputs_array_list += [d_outputs[
                                     self.output_cp_iga_name_list[i]]]
            d_residuals_array_list += [d_residuals[
                                       self.output_cp_iga_name_list[i]]]
        if mode == 'fwd':
            self.cpfe2iga_imop.solve_linear_fwd(d_outputs_array_list,
                                                d_residuals_array_list)
        if mode == 'rev':
            self.cpfe2iga_imop.solve_linear_rev(d_outputs_array_list,
                                                d_residuals_array_list)

if __name__ == "__main__":
    from GOLDFISH.tests.test_tbeam import nonmatching_opt
    # from GOLDFISH.tests.test_slr import nonmatching_opt

    prob = Problem()
    comp = CPFE2IGAComp(nonmatching_opt=nonmatching_opt)
    comp.init_paramters()
    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print('check_partials:')
    prob.check_partials(compact_print=True)