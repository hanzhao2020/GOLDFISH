from GOLDFISH.nonmatching_opt_ffd import *
from GOLDFISH.operations.volume_exop import *

import csdl_alpha as csdl

class VolumeModel(csdl.CustomExplicitOperation):

    def __init__(self, nonmatching_opt, input_cp_iga_name_pre='CP_IGA', 
                 input_h_th_name='h_th', output_vol_name='volume'):
        super().__init__()
        csdl.check_parameter(nonmatching_opt, 'nonmatching_opt')
        csdl.check_parameter(input_cp_iga_name_pre, 'input_cp_iga_name_pre')
        csdl.check_parameter(input_h_th_name, 'input_h_th_name')
        csdl.check_parameter(output_vol_name, 'output_vol_name')

        self.nonmatching_opt = nonmatching_opt
        self.input_cp_iga_name_pre = input_cp_iga_name_pre
        self.input_h_th_name = input_h_th_name
        self.output_vol_name = output_vol_name

        self.vol_exop = VolumeExOperation(self.nonmatching_opt)

        self.opt_field = self.nonmatching_opt.opt_field
        self.opt_shape = self.nonmatching_opt.opt_shape
        self.opt_thickness = self.nonmatching_opt.opt_thickness
        self.var_thickness = self.nonmatching_opt.var_thickness

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
        vol = self.create_output(self.output_vol_name, (1,))
        vol.add_name(self.output_vol_name)
        # output = csdl.VariableGroup()
        # output.vol = vol

        if self.opt_shape:
            for i, field in enumerate(self.opt_field):
                self.declare_input(self.input_cp_iga_name_list[i],
                                   inputs.cp_iga[i],)
                               # shape=self.input_cpiga_shape,)
                               # val=self.init_cp_iga[:,field])
                self.declare_derivative_parameters(self.output_vol_name,
                                      self.input_cp_iga_name_list[i],
                                      dependent=True)
        if self.opt_thickness:
            self.declare_input(self.input_h_th_name,
                           inputs.h_th,)
                           # shape=self.input_h_th_shape)
            self.declare_derivative_parameters(self.output_vol_name,
                                     self.input_h_th_name,
                                     dependent=True)
        return vol

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

    def compute(self, input_vals, output_vals):
        self.update_inputs(input_vals)
        output_vals[self.output_vol_name] = self.vol_exop.volume()

    def compute_derivatives(self, input_vals, output_vals, derivatives):
        self.update_inputs(input_vals)
        if self.opt_shape:
            for i, field in enumerate(self.opt_field):
                dvoldcp_IGA = self.vol_exop.dvoldCPIGA(field, array=True)
                derivatives[self.output_vol_name, 
                         self.input_cp_iga_name_list[i]] = dvoldcp_IGA
        if self.opt_thickness:
            dvoldh_th_vec = self.vol_exop.dvoldh_th(array=True)
            derivatives[self.output_vol_name, self.input_h_th_name] = \
                dvoldh_th_vec

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

    m = VolumeModel(nonmatching_opt=nonmatching_opt)
    vol = m.evaluate(inputs)
    # vol = outputs.vol

    print(vol.value)

    from csdl_alpha.src.operations.derivative.utils import verify_derivatives_inline
    verify_derivatives_inline([vol], inputs.cp_iga, 
                              step_size=1e-6, raise_on_error=False)
    recorder.stop()