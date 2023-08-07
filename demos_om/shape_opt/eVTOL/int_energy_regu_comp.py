from GOLDFISH.nonmatching_opt_ffd import *
from int_energy_regu_exop import *

import openmdao.api as om
from openmdao.api import Problem

class IntEnergyReguComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('nonmatching_opt')
        self.options.declare('input_cp_iga_name_pre', default='CP_IGA')
        self.options.declare('input_h_th_name', default='thickness')
        self.options.declare('input_u_name', default='displacements')
        self.options.declare('output_wint_name', default='w_int')
        self.options.declare('regu_para', default=1e-1)

    def init_paramters(self):
        self.nonmatching_opt = self.options['nonmatching_opt']
        self.input_cp_iga_name_pre = self.options['input_cp_iga_name_pre']
        self.input_h_th_name = self.options['input_h_th_name']
        self.input_u_name = self.options['input_u_name']
        self.output_wint_name = self.options['output_wint_name']
        self.regu_para = self.options['regu_para']

        self.wint_exop = IntEnergyReguExOperation(self.nonmatching_opt, self.regu_para)

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

    def setup(self):
        self.add_output(self.output_wint_name)
        self.add_input(self.input_u_name, shape=self.input_u_shape,
                       val=self.init_disp_array)
        self.declare_partials(self.output_wint_name, self.input_u_name)

        if self.opt_shape:
            for i, field in enumerate(self.opt_field):
                self.add_input(self.input_cp_iga_name_list[i],
                               shape=self.input_cpiga_shape,
                               val=self.init_cp_iga[:,field])
                self.declare_partials(self.output_wint_name,
                                      self.input_cp_iga_name_list[i])
        if self.opt_thickness:
            self.add_input(self.input_h_th_name, shape=self.input_h_th_shape,
                           val=self.init_h_th)
            self.declare_partials(self.output_wint_name, self.input_h_th_name)

    def update_inputs(self, inputs):
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
            # self.nonmatching_opt.update_h_th(inputs[self.input_h_th_name])
        self.nonmatching_opt.update_uIGA(inputs[self.input_u_name])

    def compute(self, inputs, outputs):
        self.update_inputs(inputs)
        outputs[self.output_wint_name] = self.wint_exop.Wint()

    def compute_partials(self, inputs, partials):
        self.update_inputs(inputs)
        dwintdu_IGA = self.wint_exop.dWintduIGA(array=True, apply_bcs=True)
        partials[self.output_wint_name, self.input_u_name] = dwintdu_IGA
        if self.opt_shape:
            for i, field in enumerate(self.opt_field):
                dwintdcp_IGA = self.wint_exop.dWintdCPIGA(field, array=True)
                partials[self.output_wint_name, 
                         self.input_cp_iga_name_list[i]] = dwintdcp_IGA
        if self.opt_thickness:
            dwintdh_th_vec = self.wint_exop.dWintdh_th(array=True)
            partials[self.output_wint_name, self.input_h_th_name] = \
                dwintdh_th_vec

if __name__ == "__main__":
    # from GOLDFISH.tests.test_tbeam import nonmatching_opt
    # from GOLDFISH.tests.test_slr import nonmatching_opt
    from GOLDFISH.tests.test_dRdt import nonmatching_opt

    prob = Problem()
    comp = IntEnergyComp(nonmatching_opt=nonmatching_opt)
    comp.init_paramters()
    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print('check_partials:')
    prob.check_partials(compact_print=True)