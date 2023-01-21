from GOLDFISH.nonmatching_opt_ffd import *
import openmdao.api as om
from openmdao.api import Problem


class CPFFDReguComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('nonmatching_opt_ffd')
        self.options.declare('input_cpffd_name_pre', default='CP_FFD')
        self.options.declare('output_cpregu_name_pre', default='CP_FFD_regu')

    def init_paramters(self):
        self.nonmatching_opt_ffd = self.options['nonmatching_opt_ffd']
        self.input_cpffd_name_pre = self.options['input_cpffd_name_pre']
        self.output_cpregu_name_pre = self.options['output_cpregu_name_pre']

        self.opt_field = self.nonmatching_opt_ffd.opt_field
        self.input_shape = self.nonmatching_opt_ffd.cpffd_size
        self.output_shapes = self.nonmatching_opt_ffd.cpregu_sizes
        self.derivs = self.nonmatching_opt_ffd.dcpregudcpffd_list

        self.input_cpffd_name_list = []
        self.output_cpregu_name_list = []
        for i, field in enumerate(self.opt_field):
            self.input_cpffd_name_list += \
                [self.input_cpffd_name_pre+str(field)]
            self.output_cpregu_name_list += \
                [self.output_cpregu_name_pre+str(field)]

    def setup(self):
        for i, field in enumerate(self.opt_field):
            self.add_input(self.input_cpffd_name_list[i],
                           shape=self.input_shape,
                           val=self.nonmatching_opt_ffd.cpffd_flat[:,field])
            self.add_output(self.output_cpregu_name_list[i],
                            shape=self.output_shapes[i])
            self.declare_partials(self.output_cpregu_name_list[i],
                                  self.input_cpffd_name_list[i],
                                  val=self.derivs[i].data,
                                  rows=self.derivs[i].row,
                                  cols=self.derivs[i].col)

    def compute(self, inputs, outputs):
        for i, field in enumerate(self.opt_field):
            outputs[self.output_cpregu_name_list[i]] = \
                self.derivs[i]*inputs[self.input_cpffd_name_list[i]]


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
    nonmatching_opt.set_regu_CPFFD(regu_dir=[None, None, None],
                                   regu_side=[None, None, None])

    prob = Problem()
    comp = CPFFDReguComp(nonmatching_opt_ffd=nonmatching_opt)
    comp.init_paramters()
    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print('check_partials:')
    prob.check_partials(compact_print=True)