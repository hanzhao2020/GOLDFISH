from GOLDFISH.nonmatching_opt_ffd import *
import openmdao.api as om
from openmdao.api import Problem

class HthFFDAlignComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('nonmatching_opt_ffd')
        self.options.declare('input_h_th_name', default='thickness_FFD')
        self.options.declare('output_h_th_align_name', 
                             default='thickness_FFD_align')

    def init_parameters(self):
        self.nonmatching_opt_ffd = self.options['nonmatching_opt_ffd']
        self.input_h_th_name = self.options['input_h_th_name']
        self.output_h_th_align_name = self.options['output_h_th_align_name']

        if self.nonmatching_opt_ffd.thopt_multiffd:
            self.deriv = self.nonmatching_opt_ffd.thopt_dcpaligndcpmultiffd
            self.init_h_th_ffd = self.nonmatching_opt_ffd.get_init_h_th_multiFFD()
        else:
            self.deriv = self.nonmatching_opt_ffd.thopt_dcpaligndcpffd
            self.init_h_th_ffd = self.nonmatching_opt_ffd.get_init_h_th_FFD()

        self.input_shape = self.deriv.shape[1]
        self.output_shape = self.deriv.shape[0]

    def setup(self):
        self.add_input(self.input_h_th_name,
                       shape=self.input_shape,
                       val=self.init_h_th_ffd)
        self.add_output(self.output_h_th_align_name,
                        shape=self.output_shape)
        self.declare_partials(self.output_h_th_align_name,
                              self.input_h_th_name,
                              val=self.deriv.data,
                              rows=self.deriv.row,
                              cols=self.deriv.col)

    def compute(self, inputs, outputs):
        outputs[self.output_h_th_align_name] = \
            self.deriv*inputs[self.input_h_th_name]


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
    nonmatching_opt.set_thopt_FFD(FFD_block.knots, FFD_block.control)
    nonmatching_opt.set_thopt_align_CPFFD(thopt_align_dir=1)

    prob = Problem()
    comp = HthFFDAlignComp(nonmatching_opt_ffd=nonmatching_opt)
    comp.init_parameters()
    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print('check_partials:')
    prob.check_partials(compact_print=True)