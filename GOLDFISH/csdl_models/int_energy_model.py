from GOLDFISH.nonmatching_opt_ffd import *
from GOLDFISH.operations.int_energy_exop import *

import csdl_alpha as csdl

class IntEnergyModel(csdl.CustomExplicitOperation):

    def __init__(self, nonmatching_opt, input_cp_iga_name_pre='CP_IGA', 
                 input_h_th_name='h_th', input_u_name='u', 
                 output_wint_name='w_int'):
        super().__init__()
        csdl.check_parameter(nonmatching_opt, 'nonmatching_opt')
        csdl.check_parameter(input_cp_iga_name_pre, 'input_cp_iga_name_pre')
        csdl.check_parameter(input_h_th_name, 'input_h_th_name')
        csdl.check_parameter(input_u_name, 'input_u_name')
        csdl.check_parameter(output_wint_name, 'output_wint_name')

        self.nonmatching_opt = nonmatching_opt
        self.input_cp_iga_name_pre = input_cp_iga_name_pre
        self.input_h_th_name = input_h_th_name
        self.input_u_name = input_u_name
        self.output_wint_name = output_wint_name

        self.wint_exop = IntEnergyExOperation(self.nonmatching_opt, wint_regu=None)

        self.opt_field = self.nonmatching_opt.opt_field
        self.opt_shape = self.nonmatching_opt.opt_shape
        self.opt_thickness = self.nonmatching_opt.opt_thickness
        self.var_thickness = self.nonmatching_opt.var_thickness

        self.input_u_shape = self.nonmatching_opt.vec_iga_dof
        self.init_disp_array = get_petsc_vec_array(
                               self.nonmatching_opt.u_iga_nest)
        # self.init_disp_array = np.ones(self.nonmatching_opt.vec_iga_dof)
        
        if self.opt_shape:
            self.input_cpiga_shape = self.nonmatching_opt.vec_scalar_iga_dof
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

    def evaluate(self, inputs: csdl.VariableGroup):
        w_int = self.create_output(self.output_wint_name, (1,))
        w_int.add_name(self.output_wint_name)
        # output = csdl.VariableGroup()
        # output.w_int = w_int

        self.declare_input(self.input_u_name, inputs.u)

        if self.opt_shape:
            for i, field in enumerate(self.opt_field):
                self.declare_input(self.input_cp_iga_name_list[i],
                                   inputs.cp_iga[i],)
                self.declare_derivative_parameters(self.output_wint_name,
                                      self.input_cp_iga_name_list[i],
                                      dependent=True)
        if self.opt_thickness:
            self.declare_input(self.input_h_th_name,
                           inputs.h_th,)
            self.declare_derivative_parameters(self.output_wint_name,
                                     self.input_h_th_name,
                                     dependent=True)
        return w_int

    def update_inputs(self, input_vals):
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
        self.nonmatching_opt.update_uIGA(input_vals[self.input_u_name])

    def compute(self, input_vals, output_vals):
        self.update_inputs(input_vals)
        output_vals[self.output_wint_name] = self.wint_exop.Wint()

    def compute_derivatives(self, input_vals, output_vals, derivatives):
        self.update_inputs(input_vals)
        dwintdu_IGA = self.wint_exop.dWintduIGA(array=True, apply_bcs=True)
        derivatives[self.output_wint_name, self.input_u_name] = dwintdu_IGA
        if self.opt_shape:
            for i, field in enumerate(self.opt_field):
                dwintdcp_IGA = self.wint_exop.dWintdCPIGA(field, array=True)
                derivatives[self.output_wint_name, 
                         self.input_cp_iga_name_list[i]] = dwintdcp_IGA
        if self.opt_thickness:
            dwintdh_th_vec = self.wint_exop.dWintdh_th(array=True)
            derivatives[self.output_wint_name, self.input_h_th_name] = \
                dwintdh_th_vec

if __name__ == "__main__":
    # from GOLDFISH.tests.test_tbeam import nonmatching_opt
    # from GOLDFISH.tests.test_slr import nonmatching_opt
    from GOLDFISH.tests.test_dRdt import nonmatching_opt

    nonmatching_opt.get_init_CPIGA()

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    inputs = csdl.VariableGroup()
    inputs.cp_iga = []
    for i, field in enumerate(nonmatching_opt.opt_field):
        inputs.cp_iga += [csdl.Variable(value=nonmatching_opt.init_cp_iga[i], 
                                        name='cp_iga'+str(field))]

    inputs.h_th = csdl.Variable(value=nonmatching_opt.init_h_th, name='h_th')
    inputs.u = csdl.Variable(value=np.random.random(nonmatching_opt.vec_iga_dof),#get_petsc_vec_array(nonmatching_opt.u_iga_nest),
                            name='u')

    m = IntEnergyModel(nonmatching_opt=nonmatching_opt)
    w_int = m.evaluate(inputs)
    # w_int = outputs.w_int

    print(w_int.value)

    from csdl_alpha.src.operations.derivative.utils import verify_derivatives_inline
    verify_derivatives_inline([w_int], inputs.cp_iga+[inputs.h_th, inputs.u], 
                              step_size=1e-6, raise_on_error=False)
    recorder.stop()