from GOLDFISH.nonmatching_opt_ffd import *
from GOLDFISH.operations.cpfe2iga_imop import *

from csdl import Model, CustomImplicitOperation
import csdl
from csdl_om import Simulator


class CPFE2IGAModel(Model):

    def initialize(self):
        self.parameters.declare('nonmatching_opt')
        self.parameters.declare('input_cp_fe_name_pre', default='CP_FE')
        self.parameters.declare('output_cp_iga_name_pre', default='CP_IGA')

    def init_paramters(self):
        self.nonmatching_opt = self.parameters['nonmatching_opt']
        self.input_cp_fe_name_pre = self.parameters['input_cp_fe_name_pre']
        self.output_cp_iga_name_pre = self.parameters['output_cp_iga_name_pre']
        self.op = CPFE2IGAOperation(
                        nonmatching_opt=self.nonmatching_opt,
                        input_cp_fe_name_pre=self.input_cp_fe_name_pre,
                        output_cp_iga_name_pre=self.output_cp_iga_name_pre)
        self.op.init_paramters()

    def define(self):
        cp_fe_list = [None for i in range(len(self.op.opt_field))]
        for i, field in enumerate(self.op.opt_field):
            init_cp_fe_array = get_petsc_vec_array(
                               self.nonmatching_opt.cp_funcs_nest[field], 
                               comm=self.nonmatching_opt.comm)
            cp_fe_list[i] = self.declare_variable(
                            self.op.input_cp_fe_name_list[i],
                            shape=(self.op.input_shape),
                            val=init_cp_fe_array)
        
        cp_iga_list = csdl.custom(*cp_fe_list, op=self.op)
        if not isinstance(cp_iga_list, (list, tuple)):
            cp_iga_list = [cp_iga_list]

        for i, field in enumerate(self.op.opt_field):
            self.register_output(self.op.output_cp_iga_name_list[i],
            cp_iga_list[i])


class CPFE2IGAOperation(CustomImplicitOperation):

    def initialize(self):
        self.parameters.declare('nonmatching_opt')
        self.parameters.declare('input_cp_fe_name_pre', default='CP_FE')
        self.parameters.declare('output_cp_iga_name_pre', default='CP_IGA')

    def init_paramters(self):
        self.nonmatching_opt = self.parameters['nonmatching_opt']
        self.input_cp_fe_name_pre = self.parameters['input_cp_fe_name_pre']
        self.output_cp_iga_name_pre = self.parameters['output_cp_iga_name_pre']

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

    def define(self):
        for i, field in enumerate(self.opt_field):
            self.add_input(self.input_cp_fe_name_list[i],
                           shape=self.input_shape,)
            self.add_output(self.output_cp_iga_name_list[i],
                            shape=self.output_shape)
            self.declare_derivatives(self.output_cp_iga_name_list[i],
                                  self.input_cp_fe_name_list[i],
                                  val=self.cpfe2iga_imop.dRdcp_fe_coo.data,
                                  rows=self.cpfe2iga_imop.dRdcp_fe_coo.row,
                                  cols=self.cpfe2iga_imop.dRdcp_fe_coo.col)
            self.declare_derivatives(self.output_cp_iga_name_list[i],
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

    def evaluate_residuals(self, inputs, outputs, residuals):
        self.update_cp_vecs(inputs, outputs)
        res_array_list = self.cpfe2iga_imop.apply_nonlinear()
        for i, field in enumerate(self.opt_field):
            residuals[self.output_cp_iga_name_list[i]] = res_array_list[i]

    def solve_residual_equations(self, inputs, outputs):
        self.update_cp_vecs(inputs, outputs)
        cp_iga_arary_list = self.cpfe2iga_imop.solve_nonlinear()
        for i, field in enumerate(self.opt_field):
            outputs[self.output_cp_iga_name_list[i]] = cp_iga_arary_list[i]

    def compute_derivatives(self, inputs, outputs, derivatives):
        self.update_cp_vecs(inputs, outputs)
        for i, field in enumerate(self.opt_field):
            derivatives[self.output_cp_iga_name_list[i],
                        self.input_cp_fe_name_list[i]] = \
                        self.cpfe2iga_imop.dRdcp_fe_coo.data
            derivatives[self.output_cp_iga_name_list[i],
                        self.output_cp_iga_name_list[i]] = \
                        self.cpfe2iga_imop.dRdcp_iga_coo.data

    # def compute_jacvec_product(self, inputs, outputs, d_inputs, 
    #                            d_outputs, d_residuals, mode):
    #     self.update_cp_vecs(inputs, outputs)
    #     d_inputs_array_list = []
    #     d_outputs_array_list = []
    #     d_residuals_array_list = []
    #     for i, field in enumerate(self.opt_field):
    #         if self.input_cp_fe_name_list[i] in d_inputs:
    #             d_inputs_array_list += \
    #                 [d_inputs[self.input_cp_fe_name_list[i]]]
    #         if self.output_cp_iga_name_list[i] in d_outputs:
    #             d_outputs_array_list += \
    #                 [d_outputs[self.output_cp_iga_name_list[i]]]
    #         if self.output_cp_iga_name_list[i] in d_residuals:
    #             d_residuals_array_list += \
    #                 [d_residuals[self.output_cp_iga_name_list[i]]]
    #     if len(d_inputs_array_list) == 0:
    #         d_inputs_array_list = None
    #     if len(d_outputs_array_list) == 0:
    #         d_outputs_array_list = None
    #     if len(d_residuals_array_list) == 0:
    #         d_residuals_array_list = None

    #     if mode == 'fwd':
    #         self.cpfe2iga_imop.apply_linear_fwd(d_inputs_array_list, 
    #             d_outputs_array_list, d_residuals_array_list)
    #     elif mode == 'rev':
    #         self.cpfe2iga_imop.apply_linear_rev(d_inputs_array_list, 
    #             d_outputs_array_list, d_residuals_array_list)

    def apply_inverse_jacobian(self, d_outputs, d_residuals, mode):
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

    ffd_block_num_el = [4,4,1]
    p = 3
    # Create FFD block in igakit format
    cp_ffd_lims = nonmatching_opt.cpsurf_lims
    for field in [2]:
        cp_range = cp_ffd_lims[field][1] - cp_ffd_lims[field][0]
        cp_ffd_lims[field][0] = cp_ffd_lims[field][0] - 0.2*cp_range
        cp_ffd_lims[field][1] = cp_ffd_lims[field][1] + 0.2*cp_range
    FFD_block = create_3D_block(ffd_block_num_el, p, cp_ffd_lims)
    nonmatching_opt.set_FFD(FFD_block.knots, FFD_block.control)

    m = CPFE2IGAModel(nonmatching_opt=nonmatching_opt)
    m.init_paramters()
    sim = Simulator(m)
    sim.run()
    print("Check partials:")
    sim.check_partials(compact_print=True)