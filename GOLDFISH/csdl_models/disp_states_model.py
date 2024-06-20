from GOLDFISH.nonmatching_opt_ffd import *
from GOLDFISH.operations.disp_imop import *

import csdl
from csdl import Model, CustomImplicitOperation

import psutil
def memory_usage_psutil():
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0]/float(1024**2)
    return mem

# print("Inspection disp extraction 1: Memory usage: {:8.2f} MB.\n"\
#               .format(memory_usage_psutil()))

class DispStatesModel(Model):

    def initialize(self):
        self.parameters.declare('nonmatching_opt')
        self.parameters.declare('input_cp_iga_name_pre', default='CP_IGA')
        self.parameters.declare('input_h_th_name', default='thickness')
        self.parameters.declare('input_Paero_name', default='aero_pressure')
        self.parameters.declare('output_u_name', default='displacements')

    def init_parameters(self, save_files=False, nonlinear_solver_rtol=1e-3,
                       nonlinear_solver_max_it=30):
        self.nonmatching_opt = self.parameters['nonmatching_opt']
        self.input_cp_iga_name_pre = self.parameters['input_cp_iga_name_pre']
        self.input_h_th_name = self.parameters['input_h_th_name']
        self.input_Paero_name = self.parameters['input_Paero_name']
        self.output_u_name = self.parameters['output_u_name']
        self.op = DispStatesOperation(
                  nonmatching_opt=self.nonmatching_opt,
                  input_cp_iga_name_pre=self.input_cp_iga_name_pre,
                  input_h_th_name=self.input_h_th_name,
                  input_Paero_name=self.input_Paero_name,
                  output_u_name=self.output_u_name)
        self.op.init_parameters(save_files=save_files,
                               nonlinear_solver_rtol=nonlinear_solver_rtol,
                               nonlinear_solver_max_it=nonlinear_solver_max_it)

    def define(self):
        input_list = []
        if self.op.opt_shape:
            cp_iga_list = [None for i in range(len(self.op.opt_field))]
            for i, field in enumerate(self.op.opt_field):
                cp_iga_list[i] = self.declare_variable(
                                 self.op.input_cp_iga_name_list[i],
                                 shape=(self.op.input_cp_shape),
                                 val=self.op.init_cp_iga[:,field])
            input_list += cp_iga_list
        if self.op.opt_thickness:
            h_th = self.declare_variable(self.op.input_h_th_name,
                   shape=(self.op.input_h_th_shape),
                   val=self.op.init_h_th)
            input_list += [h_th]
        if self.nonmatching_opt.use_aero_pressure:
            Paero = self.declare_variable(self.op.input_Paero_name,
                    shape=(self.op.inpout_Paero_shape),
                    val=np.zeros(self.op.inpout_Paero_shape))
            input_list += [Paero]
        
        disp = csdl.custom(*input_list, op=self.op)
        self.register_output(self.op.output_u_name, disp)


class DispStatesOperation(CustomImplicitOperation):

    def initialize(self):
        self.parameters.declare('nonmatching_opt')
        self.parameters.declare('input_cp_iga_name_pre', default='CP_IGA')
        self.parameters.declare('input_h_th_name', default='thickness')
        self.parameters.declare('input_Paero_name', default='aero_pressure')
        self.parameters.declare('output_u_name', default='displacements')

    def init_parameters(self, save_files, nonlinear_solver_rtol,
                       nonlinear_solver_max_it):
        self.nonmatching_opt = self.parameters['nonmatching_opt']
        self.input_cp_iga_name_pre = self.parameters['input_cp_iga_name_pre']
        self.input_h_th_name = self.parameters['input_h_th_name']
        self.input_Paero_name = self.parameters['input_Paero_name']
        self.output_u_name = self.parameters['output_u_name']
        self.save_files = save_files
        self.nonlinear_solver_rtol = nonlinear_solver_rtol
        self.nonlinear_solver_max_it = nonlinear_solver_max_it

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

    def define(self):
        self.add_output(self.output_u_name, shape=self.output_shape)
        self.declare_derivatives(self.output_u_name, self.output_u_name)

        if self.opt_shape:
            for i, field in enumerate(self.opt_field):
                self.add_input(self.input_cp_iga_name_list[i],
                               shape=self.input_cp_shape,)
                               # val=self.init_cp_iga[:,field])
            for i, field in enumerate(self.opt_field):
                self.declare_derivatives(self.output_u_name,
                                         self.input_cp_iga_name_list[i])
        if self.opt_thickness:
            self.add_input(self.input_h_th_name, shape=self.input_h_th_shape,)
                           # val=self.init_h_th
            self.declare_derivatives(self.output_u_name, self.input_h_th_name)
        if self.use_aero_pressure:
            self.add_input(self.input_Paero_name, shape=self.inpout_Paero_shape)
            self.declare_derivatives(self.output_u_name, self.input_Paero_name)

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
        if self.use_aero_pressure:
            self.nonmatching_opt.update_Paero(inputs[self.input_Paero_name])
        self.nonmatching_opt.update_uIGA(outputs[self.output_u_name])

    def evaluate_residuals(self, inputs, outputs, residuals):
        self.update_inputs_outpus(inputs, outputs)
        residuals[self.output_u_name] = self.disp_state_imop.apply_nonlinear()

    def solve_residual_equations(self, inputs, outputs):

        print("Inspection disp nonlinear solve 0: Memory usage: {:8.2f} MB.\n"\
              .format(memory_usage_psutil()))

        self.update_inputs_outpus(inputs, outputs)
        outputs[self.output_u_name] = self.disp_state_imop.solve_nonlinear(
                                      self.nonlinear_solver_max_it,
                                      self.nonlinear_solver_rtol)

        print("Inspection disp nonlinear solve 1: Memory usage: {:8.2f} MB.\n"\
              .format(memory_usage_psutil()))

        self.func_eval_ind += 1
        # if self.save_files:
        #     self.nonmatching_opt.save_files(thickness=self.opt_thickness)

    def compute_derivatives(self, inputs, outputs, derivatives):
        self.update_inputs_outpus(inputs, outputs)
        self.disp_state_imop.linearize()

        if self.save_files:
            self.func_eval_major_ind += [self.func_eval_ind-1]
            print("**** Saving pvd files, ind: {:6d} ****"
                  .format(self.major_iter_ind))
            self.nonmatching_opt.save_files(
                thickness=self.opt_thickness)
            self.major_iter_ind += 1

    def compute_jacvec_product(self, inputs, outputs, d_inputs, 
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

    m = DispStatesModel(nonmatching_opt=nonmatching_opt)
    m.init_parameters()
    sim = Simulator(m, analytics=False)
    sim.run()
    print("Check partials:")
    sim.check_partials(compact_print=True)