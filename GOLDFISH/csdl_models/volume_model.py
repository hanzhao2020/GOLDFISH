from GOLDFISH.nonmatching_opt_ffd import *
from GOLDFISH.operations.volume_exop import *

import csdl
from csdl import Model, CustomExplicitOperation
from csdl_om import Simulator

class VolumeModel(Model):

    def initialize(self):
        self.parameters.declare('nonmatching_opt')
        self.parameters.declare('input_cp_iga_name_pre', default='CP_IGA')
        self.parameters.declare('output_vol_name', default='volume')

    def init_paramters(self):
        self.nonmatching_opt = self.parameters['nonmatching_opt']
        self.input_cp_iga_name_pre = self.parameters['input_cp_iga_name_pre']
        self.output_vol_name = self.parameters['output_vol_name']
        self.op = VolumeOperation(
                  nonmatching_opt=self.nonmatching_opt,
                  input_cp_iga_name_pre=self.input_cp_iga_name_pre, 
                  output_vol_name=self.output_vol_name)
        self.op.init_paramters()

    def define(self):
        cp_iga_list = [None for i in range(len(self.op.opt_field))]
        for i, field in enumerate(self.op.opt_field):
            cp_iga_list[i] = self.declare_variable(self.op.input_cp_iga_name_list[i],
                             shape=(self.op.input_cpiga_shape),
                             val=self.nonmatching_opt.init_cp_iga[:,field])
        vol = csdl.custom(*cp_iga_list, op=self.op)
        self.register_output(self.op.output_vol_name, vol)


class VolumeOperation(CustomExplicitOperation):

    def initialize(self):
        self.parameters.declare('nonmatching_opt')
        self.parameters.declare('input_cp_iga_name_pre', default='CP_IGA')
        self.parameters.declare('output_vol_name', default='volume')

    def init_paramters(self):
        self.nonmatching_opt = self.parameters['nonmatching_opt']
        self.input_cp_iga_name_pre = self.parameters['input_cp_iga_name_pre']
        self.output_vol_name = self.parameters['output_vol_name']

        self.vol_exop = VolumeExOperation(self.nonmatching_opt)

        self.opt_field = self.nonmatching_opt.opt_field
        self.input_cpiga_shape = self.nonmatching_opt.vec_scalar_iga_dof
        self.init_cp_iga = self.nonmatching_opt.get_init_CPIGA()

        self.input_cp_iga_name_list = []
        for i, field in enumerate(self.opt_field):
            self.input_cp_iga_name_list += \
                [self.input_cp_iga_name_pre+str(field)]

    def define(self):
        self.add_output(self.output_vol_name)
        for i, field in enumerate(self.opt_field):
            self.add_input(self.input_cp_iga_name_list[i],
                           shape=self.input_cpiga_shape,)
                           # val=self.init_cp_iga[:,field])
            self.declare_derivatives(self.output_vol_name,
                                  self.input_cp_iga_name_list[i])

    def update_inputs(self, inputs):
        for i, field in enumerate(self.opt_field):
            self.nonmatching_opt.update_CPIGA(
                inputs[self.input_cp_iga_name_list[i]], field)

    def compute(self, inputs, outputs):
        self.update_inputs(inputs)
        outputs[self.output_vol_name] = self.vol_exop.volume()

    def compute_derivatives(self, inputs, derivatives):
        self.update_inputs(inputs)
        for i, field in enumerate(self.opt_field):
            dvoldcp_IGA = self.vol_exop.dvoldCPIGA(field, array=True)
            derivatives[self.output_vol_name, 
                        self.input_cp_iga_name_list[i]] = dvoldcp_IGA

if __name__ == "__main__":
    from GOLDFISH.tests.test_tbeam import nonmatching_opt
    # from GOLDFISH.tests.test_slr import nonmatching_opt

    m = VolumeModel(nonmatching_opt=nonmatching_opt)
    m.init_paramters()
    sim = Simulator(m)
    sim.run()
    sim.check_partials(compact_print=True)