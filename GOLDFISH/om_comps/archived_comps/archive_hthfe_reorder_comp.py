from GOLDFISH.nonmatching_opt_ffd import *
import openmdao.api as om
from openmdao.api import Problem

class HthFEReorderComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('nonmatching_opt')
        self.options.declare('initial_order')
        self.options.declare('input_h_th_fe_name', default='thickness_FE_init')
        self.options.declare('output_h_th_fe_name', default='thickness_FE')

    def init_parameters(self):
        self.nonmatching_opt = self.options['nonmatching_opt']
        self.initial_order = self.options['initial_order']
        self.input_h_th_fe_name = self.options['input_h_th_fe_name']
        self.output_h_th_fe_name = self.options['output_h_th_fe_name']

        self.input_shape = self.nonmatching_opt.vec_scalar_fe_dof
        self.output_shape = self.nonmatching_opt.vec_scalar_fe_dof

        self.num_splines = self.nonmatching_opt.num_splines
        self.h_th_sizes = self.nonmatching_opt.h_th_sizes

        self.deriv_mat = self.get_derivative(coo=True)

    def setup(self):
        self.add_input(self.input_h_th_fe_name, shape=self.input_shape,
                       val=self.nonmatching_opt.init_h_th_fe)
        self.add_output(self.output_h_th_fe_name, shape=self.output_shape)
        self.declare_partials(self.output_h_th_fe_name,
                              self.input_h_th_fe_name,
                              val=self.deriv_mat.data,
                              rows=self.deriv_mat.row,
                              cols=self.deriv_mat.col)

    def compute(self, inputs, outputs):
        outputs[self.output_h_th_fe_name] = \
            self.deriv_mat*inputs[self.input_h_th_fe_name]

    def get_derivative(self, coo=True):
        diag_vecs = []
        for s_ind in range(self.num_splines):
            diag_vecs += [np.eye((self.h_th_sizes[s_ind]))]

        block_mat_list = [[] for s_ind in range(self.num_splines)]
        for s_ind in range(self.num_splines):
            for init_ind in self.initial_order:
                if s_ind == init_ind:
                    block_mat_list[s_ind] += [np.eye(self.h_th_sizes[s_ind])]
                else:
                    block_mat_list[s_ind] += [np.zeros((self.h_th_sizes[s_ind],
                                             self.h_th_sizes[init_ind]))]

        deriv_mat = np.block(block_mat_list)
        if coo:
            return coo_matrix(deriv_mat)
        else:
            return deriv_mat


if __name__ == "__main__":
    from GOLDFISH.tests.test_tbeam import nonmatching_opt
    # from GOLDFISH.tests.test_slr import nonmatching_opt

    prob = Problem()
    comp = HthFEReorderComp(nonmatching_opt=nonmatching_opt,
                            initial_order=[1,0])
    comp.init_parameters()
    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print('check_partials:')
    prob.check_partials(compact_print=True)