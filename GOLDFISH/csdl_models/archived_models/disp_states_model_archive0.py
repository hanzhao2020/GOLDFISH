from GOLDFISH.nonmatching_opt_ffd import *
from GOLDFISH.operations.disp_imop import *

from csdl import Model, CustomImplicitOperation
import csdl
from csdl_om import Simulator

class DispStatesModel(Model):

    def initialize(self):
        self.parameters.declare('nonmatching_opt')
        self.parameters.declare('input_cp_iga_name_pre', default='CP_IGA')
        self.parameters.declare('output_u_name', default='displacements')

    def init_parameters(self, save_files=False):
        self.nonmatching_opt = self.parameters['nonmatching_opt']
        self.input_cp_iga_name_pre = self.parameters['input_cp_iga_name_pre']
        self.output_u_name = self.parameters['output_u_name']
        self.op = DispStatesOperation(
                  nonmatching_opt=self.nonmatching_opt,
                  input_cp_iga_name_pre=self.input_cp_iga_name_pre,
                  output_u_name=self.output_u_name)
        self.op.init_parameters(save_files=save_files)

    def define(self):
        cp_iga_list = [None for i in range(len(self.op.opt_field))]
        for i, field in enumerate(self.op.opt_field):
            cp_iga_list[i] = self.declare_variable(
                            self.op.input_cp_iga_name_list[i],
                            shape=(self.op.input_shape),
                            val=self.op.init_cp_iga[:,field])
        
        disp = csdl.custom(*cp_iga_list, op=self.op)
        self.register_output(self.op.output_u_name, disp)


class DispStatesOperation(CustomImplicitOperation):

    def initialize(self):
        self.parameters.declare('nonmatching_opt')
        self.parameters.declare('input_cp_iga_name_pre', default='CP_IGA')
        self.parameters.declare('output_u_name', default='displacements')

    def init_parameters(self, save_files=False):
        self.nonmatching_opt = self.parameters['nonmatching_opt']
        self.input_cp_iga_name_pre = self.parameters['input_cp_iga_name_pre']
        self.output_u_name = self.parameters['output_u_name']
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

    def define(self):        
        for i, field in enumerate(self.opt_field):
            self.add_input(self.input_cp_iga_name_list[i],
                           shape=self.input_shape,)
                           # val=self.init_cp_iga[:,field])
        self.add_output(self.output_u_name, shape=self.output_shape)
        for i, field in enumerate(self.opt_field):
            self.declare_derivatives(self.output_u_name,
                                  self.input_cp_iga_name_list[i])
        self.declare_derivatives(self.output_u_name, self.output_u_name)

    def update_inputs_outpus(self, inputs, outputs):
        for i, field in enumerate(self.opt_field):
            self.nonmatching_opt.update_CPIGA(
                inputs[self.input_cp_iga_name_list[i]], field)
        self.nonmatching_opt.update_uIGA(outputs[self.output_u_name])

    def evaluate_residuals(self, inputs, outputs, residuals):
        self.update_inputs_outpus(inputs, outputs)
        residuals[self.output_u_name] = self.disp_state_imop.apply_nonlinear()

    def solve_residual_equations(self, inputs, outputs):
        self.update_inputs_outpus(inputs, outputs)
        outputs[self.output_u_name] = self.disp_state_imop.solve_nonlinear()

        if self.save_files:
            self.nonmatching_opt.save_files(thickness=False)

    def compute_derivatives(self, inputs, outputs, derivatives):
        self.update_inputs_outpus(inputs, outputs)
        self.disp_state_imop.linearize()

    def compute_jacvec_product(self, inputs, outputs, d_inputs, 
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

    def apply_inverse_jacobian(self, d_outputs, d_residuals, mode):
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

    m = DispStatesModel(nonmatching_opt=nonmatching_opt)
    m.init_parameters()
    sim = Simulator(m)
    sim.run()
    print("Check partials:")
    sim.check_partials(compact_print=True)