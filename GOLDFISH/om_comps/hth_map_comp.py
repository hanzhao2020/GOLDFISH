from scipy.linalg import block_diag
from scipy.sparse import coo_matrix
from GOLDFISH.nonmatching_opt import *
import openmdao.api as om
from openmdao.api import Problem

class HthMapComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('nonmatching_opt')
        self.options.declare('order', default=0)
        self.options.declare('input_h_th_name_design', default='thickness')
        self.options.declare('output_h_th_name_full', default='thickness_full')

    def init_paramters(self):
        self.nonmatching_opt = self.options['nonmatching_opt']
        self.order = self.options['order']
        self.input_h_th_name_design = self.options['input_h_th_name_design']
        self.output_h_th_name_full = self.options['output_h_th_name_full']
        self.num_splines = self.nonmatching_opt.num_splines

        # Only consider constant thickness for now
        if self.order == 0:
            self.input_shape = self.num_splines
            self.h_th_sizes = self.nonmatching_opt.h_th_sizes
            self.init_val = [np.average(h_th_sub_array) for h_th_sub_array
                             in self.nonmatching_opt.init_h_th_list]
        else:
            raise ValueError("Order {:2d} is not supported yet".format(order))

        self.output_shape = self.nonmatching_opt.h_th_dof
        self.deriv_mat = self.get_derivative()

    def setup(self):
        self.add_input(self.input_h_th_name_design, shape=self.input_shape,
                       val=self.init_val)
        self.add_output(self.output_h_th_name_full, shape=self.output_shape)
        self.declare_partials(self.output_h_th_name_full,
                              self.input_h_th_name_design,
                              val=self.deriv_mat.data,
                              rows=self.deriv_mat.row,
                              cols=self.deriv_mat.col)

    def compute(self, inputs, outputs):
        outputs[self.output_h_th_name_full] = \
            self.deriv_mat*inputs[self.input_h_th_name_design]


    def get_derivative(self, coo=True):
        diag_vecs = []
        for s_ind in range(self.num_splines):
            diag_vecs += [np.ones((self.h_th_sizes[s_ind],1))]
        deriv_mat = block_diag(*diag_vecs)
        if coo:
            return coo_matrix(deriv_mat)
        else:
            return deriv_mat

if __name__ == '__main__':
    # from GOLDFISH.tests.test_tbeam import nonmatching_opt
    # from GOLDFISH.tests.test_slr import nonmatching_opt
    from GOLDFISH.tests.test_dRdt import nonmatching_opt

    prob = Problem()
    comp = HthMapComp(nonmatching_opt=nonmatching_opt)
    comp.init_paramters()
    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print('check_partials:')
    prob.check_partials(compact_print=True)

