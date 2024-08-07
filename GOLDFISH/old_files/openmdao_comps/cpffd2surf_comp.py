from GOLDFISH.nonmatching_opt_ffd import *

import openmdao.api as om
from openmdao.api import Problem

class CPFFD2SurfComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('nonmatching_opt_ffd')
        self.options.declare('input_cpffd_name_pre', default='CP_FFD')
        self.options.declare('output_cpsurf_name_pre', default='CP_FE')

    def init_parameters(self):
        self.nonmatching_opt_ffd = self.options['nonmatching_opt_ffd']
        self.input_cpffd_name_pre = self.options['input_cpffd_name_pre']
        self.output_cpsurf_name_pre = self.options['output_cpsurf_name_pre']

        self.deriv = self.nonmatching_opt_ffd.shopt_dcpsurf_fedcpffd
        self.opt_field = self.nonmatching_opt_ffd.opt_field
        self.nsd = self.nonmatching_opt_ffd.nsd
        self.knotsffd = self.nonmatching_opt_ffd.shopt_knotsffd
        self.input_shape = self.nonmatching_opt_ffd.shopt_cpffd_size
        self.output_shape = self.nonmatching_opt_ffd.cpsurf_fe_list.shape[0]

        self.input_cpffd_name_list = []
        self.output_cpsurf_name_list = []
        for i, field in enumerate(self.opt_field):
            self.input_cpffd_name_list += \
                [self.input_cpffd_name_pre+str(field)]
            self.output_cpsurf_name_list += \
                [self.output_cpsurf_name_pre+str(field)]

    def setup(self):
        for i, field in enumerate(self.opt_field):
            self.add_input(self.input_cpffd_name_list[i],
                           shape=self.input_shape,
                           val=self.nonmatching_opt_ffd.\
                           shopt_cpffd_flat[:,field])
            self.add_output(self.output_cpsurf_name_list[i],
                            shape=self.output_shape)
            self.declare_partials(self.output_cpsurf_name_list[i],
                                  self.input_cpffd_name_list[i],
                                  val=self.deriv.data,
                                  rows=self.deriv.row,
                                  cols=self.deriv.col)

    def compute(self, inputs, outputs):
        for i, field in enumerate(self.opt_field):
            outputs[self.output_cpsurf_name_list[i]] = \
                self.deriv*inputs[self.input_cpffd_name_list[i]]


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
    nonmatching_opt.set_shopt_FFD(FFD_block.knots, FFD_block.control)

    prob = Problem()
    comp = CPFFD2SurfComp(nonmatching_opt_ffd=nonmatching_opt)
    comp.init_parameters()
    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print('check_partials:')
    prob.check_partials(compact_print=True)