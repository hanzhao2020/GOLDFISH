from GOLDFISH.nonmatching_opt_ffd import *
from GOLDFISH.operations.compliance_exop import *

import openmdao.api as om
from openmdao.api import Problem

class ComplianceComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('nonmatching_opt')
        self.options.declare('forces')
        self.options.declare('input_cp_iga_name_pre', default='CP_IGA')
        self.options.declare('input_u_name', default='displacements')
        self.options.declare('output_c_name', default='w_int')

    def init_parameters(self, c_regu=None):
        self.nonmatching_opt = self.options['nonmatching_opt']
        self.forces = self.options['forces']
        self.input_cp_iga_name_pre = self.options['input_cp_iga_name_pre']
        self.input_u_name = self.options['input_u_name']
        self.output_c_name = self.options['output_c_name']

        self.c_exop = ComplianceExOperation(self.nonmatching_opt,
                                            self.forces, c_regu)

        self.opt_field = self.nonmatching_opt.opt_field
        self.opt_shape = self.nonmatching_opt.opt_shape

        self.input_u_shape = self.nonmatching_opt.vec_iga_dof
        self.init_disp_array = get_petsc_vec_array(
                               self.nonmatching_opt.u_iga_nest)
        self.init_disp_array = np.ones(self.nonmatching_opt.vec_iga_dof)
        
        if self.opt_shape:
            self.input_cpiga_shape = self.nonmatching_opt.vec_scalar_iga_dof
            self.init_cp_iga = self.nonmatching_opt.get_init_CPIGA()
            self.input_cp_iga_name_list = []
            for i, field in enumerate(self.opt_field):
                self.input_cp_iga_name_list += \
                    [self.input_cp_iga_name_pre+str(field)]

    def setup(self):
        self.add_output(self.output_c_name)
        self.add_input(self.input_u_name, shape=self.input_u_shape,
                       val=self.init_disp_array)
        self.declare_partials(self.output_c_name, self.input_u_name)

        if self.opt_shape:
            for i, field in enumerate(self.opt_field):
                self.add_input(self.input_cp_iga_name_list[i],
                               shape=self.input_cpiga_shape,
                               val=self.init_cp_iga[:,field])
                self.declare_partials(self.output_c_name,
                                      self.input_cp_iga_name_list[i])

    def update_inputs(self, inputs):
        if self.opt_shape:
            for i, field in enumerate(self.opt_field):
                self.nonmatching_opt.update_CPIGA(
                    inputs[self.input_cp_iga_name_list[i]], field)
        self.nonmatching_opt.update_uIGA(inputs[self.input_u_name])

    def compute(self, inputs, outputs):
        self.update_inputs(inputs)
        outputs[self.output_c_name] = self.c_exop.cpl()

    def compute_partials(self, inputs, partials):
        self.update_inputs(inputs)
        dcpldu_IGA = self.c_exop.dcplduIGA(array=True, apply_bcs=True)
        partials[self.output_c_name, self.input_u_name] = dcpldu_IGA
        if self.opt_shape:
            for i, field in enumerate(self.opt_field):
                dcpldcp_IGA = self.c_exop.dcpldCPIGA(field, array=True)
                partials[self.output_c_name, 
                         self.input_cp_iga_name_list[i]] = dcpldcp_IGA


if __name__ == "__main__":
    from GOLDFISH.tests.test_tbeam import nonmatching_opt
    # from GOLDFISH.tests.test_slr import nonmatching_opt

    force = as_vector([Constant(0.), Constant(0.), Constant(1.)])
    forces = [force for i in range(nonmatching_opt.num_splines)]
    prob = Problem()
    comp = ComplianceComp(nonmatching_opt=nonmatching_opt,
                          forces=forces)
    comp.init_parameters()
    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print('check_partials:')
    prob.check_partials(compact_print=True)