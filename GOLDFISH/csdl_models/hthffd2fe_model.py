from GOLDFISH.nonmatching_opt_ffd import *
from GOLDFISH.operations.volume_exop import *

import csdl
from csdl import Model, CustomExplicitOperation
from csdl_om import Simulator

from scipy.linalg import block_diag
from scipy.sparse import coo_matrix

class HthFFD2FEModel(Model):

    def initialize(self):
        self.parameters.declare('nonmatching_opt_ffd')
        self.parameters.declare('input_h_th_ffd_name', default='thickness_FFD')
        self.parameters.declare('output_h_th_fe_name', default='thickness_FE')

    def init_paramters(self):
        self.nonmatching_opt_ffd = self.parameters['nonmatching_opt_ffd']
        self.input_h_th_ffd_name = self.parameters['input_h_th_ffd_name']
        self.output_h_th_fe_name = self.parameters['output_h_th_fe_name']

        self.op = HthFFD2FEOperation(
                  nonmatching_opt_ffd=self.nonmatching_opt_ffd,
                  input_h_th_ffd_name=self.input_h_th_ffd_name,
                  output_h_th_fe_name=self.output_h_th_fe_name)
        self.op.init_paramters()

    def define(self):
        h_th_ffd = self.declare_variable(self.op.input_h_th_ffd_name,
                   shape=(self.op.input_shape),
                   val=self.op.init_h_th_ffd)
        h_th_fe = csdl.custom(h_th_ffd, op=self.op)
        self.register_output(self.op.output_h_th_fe_name, h_th_fe)


class HthFFD2FEOperation(CustomExplicitOperation):

    def initialize(self):
        self.parameters.declare('nonmatching_opt_ffd')
        self.parameters.declare('input_h_th_ffd_name', default='thickness_FFD')
        self.parameters.declare('output_h_th_fe_name', default='thickness_FE')

    def init_paramters(self):
        self.nonmatching_opt_ffd = self.parameters['nonmatching_opt_ffd']
        self.input_h_th_ffd_name = self.parameters['input_h_th_ffd_name']
        self.output_h_th_fe_name = self.parameters['output_h_th_fe_name']

        if self.nonmatching_opt_ffd.thopt_multiffd:
            self.init_h_th_ffd = self.nonmatching_opt_ffd.get_init_h_th_multiFFD()
            self.deriv_mat = self.nonmatching_opt_ffd.thopt_dcpsurf_fedcpmultiffd
        else:
            self.init_h_th_ffd = self.nonmatching_opt_ffd.get_init_h_th_FFD()
            self.deriv_mat = self.nonmatching_opt_ffd.thopt_dcpsurf_fedcpffd
        
        self.input_shape = self.deriv_mat.shape[1]
        self.output_shape = self.deriv_mat.shape[0]

    def define(self):
        self.add_input(self.input_h_th_ffd_name,
                       shape=self.input_shape)
        self.add_output(self.output_h_th_fe_name,
                        shape=self.output_shape)
        self.declare_derivatives(self.output_h_th_fe_name,
                                 self.input_h_th_ffd_name,
                                 val=self.deriv_mat.data,
                                 rows=self.deriv_mat.row,
                                 cols=self.deriv_mat.col)

    def compute(self, inputs, outputs):
        outputs[self.output_h_th_fe_name] = \
            self.deriv_mat*inputs[self.input_h_th_ffd_name]

if __name__ == "__main__":
    # from GOLDFISH.tests.test_tbeam import nonmatching_opt
    # from GOLDFISH.tests.test_slr import nonmatching_opt
    # from GOLDFISH.tests.test_dRdt import nonmatching_opt

    # m = HthFFD2FEModel(nonmatching_opt=nonmatching_opt)
    # m.init_paramters()
    # sim = Simulator(m)
    # sim.run()
    # sim.check_partials(compact_print=True)

    # # Test for multi thickness FFD blocks
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
    # nonmatching_opt.set_FFD(FFD_block.knots, FFD_block.control)
    nonmatching_opt.set_thopt_multiFFD_surf_inds([[1],[0]])
    nonmatching_opt.set_thopt_multiFFD([FFD_block.knots]*2, [FFD_block.control]*2)
    a0 = nonmatching_opt.set_thopt_align_CP_multiFFD([[2],[0]])

    m = HthFFD2FEModel(nonmatching_opt_ffd=nonmatching_opt)
    m.init_paramters()
    sim = Simulator(m)
    sim.run()
    sim.check_partials(compact_print=True)
