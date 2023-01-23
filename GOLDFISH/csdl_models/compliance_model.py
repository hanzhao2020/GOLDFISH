from GOLDFISH.nonmatching_opt_ffd import *
from GOLDFISH.operations.compliance_exop import *

import csdl
from csdl import Model, CustomExplicitOperation
from csdl_om import Simulator

class ComplianceModel(Model):

    def initialize(self):
        self.parameters.declare('nonmatching_opt')
        self.parameters.declare('forces')
        self.parameters.declare('input_cp_iga_name_pre', default='CP_IGA')
        self.parameters.declare('input_u_name', default='displacements')
        self.parameters.declare('output_c_name', default='w_int')

    def init_paramters(self):
        self.nonmatching_opt = self.parameters['nonmatching_opt']
        self.forces = self.parameters['forces']
        self.input_cp_iga_name_pre = self.parameters['input_cp_iga_name_pre']
        self.input_u_name = self.parameters['input_u_name']
        self.output_c_name = self.parameters['output_c_name']
        self.op = ComplianceOperation(
                  nonmatching_opt=self.nonmatching_opt,
                  forces=self.forces,
                  input_cp_iga_name_pre=self.input_cp_iga_name_pre, 
                  input_u_name=self.input_u_name,
                  output_c_name=self.output_c_name)
        self.op.init_paramters()

    def define(self):
        cp_iga_list = [None for i in range(len(self.op.opt_field))]
        for i, field in enumerate(self.op.opt_field):
            cp_iga_list[i] = self.declare_variable(self.op.input_cp_iga_name_list[i],
                             shape=(self.op.input_cpiga_shape),
                             val=self.nonmatching_opt.init_cp_iga[:,field])
        u = self.declare_variable(self.op.input_u_name,
                                  shape=(self.op.input_u_shape),
                                  val=self.op.init_disp_array)
        wint = csdl.custom(u, *cp_iga_list, op=self.op)
        self.register_output(self.op.output_c_name, wint)


class ComplianceOperation(CustomExplicitOperation):

    def initialize(self):
        self.parameters.declare('nonmatching_opt')
        self.parameters.declare('forces')
        self.parameters.declare('input_cp_iga_name_pre', default='CP_IGA')
        self.parameters.declare('input_u_name', default='displacement')
        self.parameters.declare('output_c_name', default='w_int')

    def init_paramters(self):
        self.nonmatching_opt = self.parameters['nonmatching_opt']
        self.forces = self.parameters['forces']
        self.input_cp_iga_name_pre = self.parameters['input_cp_iga_name_pre']
        self.input_u_name = self.parameters['input_u_name']
        self.output_c_name = self.parameters['output_c_name']

        self.cpl_exop = ComplianceExOperation(self.nonmatching_opt,
                                              self.forces)

        self.opt_field = self.nonmatching_opt.opt_field
        self.input_u_shape = self.nonmatching_opt.vec_iga_dof
        self.input_cpiga_shape = self.nonmatching_opt.vec_scalar_iga_dof
        self.init_disp_array = get_petsc_vec_array(
                               self.nonmatching_opt.u_iga_nest)
        self.init_cp_iga = self.nonmatching_opt.get_init_CPIGA()

        self.input_cp_iga_name_list = []
        for i, field in enumerate(self.opt_field):
            self.input_cp_iga_name_list += \
                [self.input_cp_iga_name_pre+str(field)]

    def define(self):
        self.add_output(self.output_c_name)
        self.add_input(self.input_u_name, shape=self.input_u_shape,)
                       # val=self.init_disp_array)
        for i, field in enumerate(self.opt_field):
            self.add_input(self.input_cp_iga_name_list[i],
                           shape=self.input_cpiga_shape,)
                           # val=self.init_cp_iga[:,field])
            self.declare_derivatives(self.output_c_name,
                                  self.input_cp_iga_name_list[i])
        self.declare_derivatives(self.output_c_name, self.input_u_name)

    def update_inputs(self, inputs):
        for i, field in enumerate(self.opt_field):
            self.nonmatching_opt.update_CPIGA(
                inputs[self.input_cp_iga_name_list[i]], field)
        self.nonmatching_opt.update_uIGA(inputs[self.input_u_name])

    def compute(self, inputs, outputs):
        self.update_inputs(inputs)
        outputs[self.output_c_name] = self.cpl_exop.cpl()

    def compute_derivatives(self, inputs, derivatives):
        self.update_inputs(inputs)
        dwintdu_IGA = self.cpl_exop.dcplduIGA(array=True, apply_bcs=True)
        derivatives[self.output_c_name, self.input_u_name] = dwintdu_IGA
        for i, field in enumerate(self.opt_field):
            dwintdcp_IGA = self.cpl_exop.dcpldCPIGA(field, array=True)
            derivatives[self.output_c_name, 
                     self.input_cp_iga_name_list[i]] = dwintdcp_IGA

if __name__ == "__main__":
    from GOLDFISH.tests.test_tbeam import nonmatching_opt
    # from GOLDFISH.tests.test_slr import nonmatching_opt

    force = as_vector([Constant(0.), Constant(0.), Constant(1.)])
    forces = [force for i in range(nonmatching_opt.num_splines)]
    m = ComplianceModel(nonmatching_opt=nonmatching_opt,
                        forces=forces)
    m.init_paramters()
    sim = Simulator(m)
    sim.run()
    sim.check_partials(compact_print=True)