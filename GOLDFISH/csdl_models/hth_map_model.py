from GOLDFISH.nonmatching_opt_ffd import *
from GOLDFISH.operations.volume_exop import *

import csdl_alpha as csdl

from scipy.linalg import block_diag
from scipy.sparse import coo_matrix

class HthMapModel(csdl.CustomExplicitOperation):

    def __init__(self, nonmatching_opt, order=0,
                 input_h_th_name_design='h_th_design', 
                 output_h_th_name_full='h_th'):
        super().__init__()
        csdl.check_parameter(nonmatching_opt, 'nonmatching_opt')
        csdl.check_parameter(order, 'order')
        csdl.check_parameter(input_h_th_name_design, 'input_h_th_name_design')
        csdl.check_parameter(output_h_th_name_full, 'output_h_th_name_full')

        self.nonmatching_opt = nonmatching_opt
        self.order = order
        self.input_h_th_name_design = input_h_th_name_design
        self.output_h_th_name_full = output_h_th_name_full
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

    def evaluate(self, inputs: csdl.VariableGroup):
        self.declare_input(self.input_h_th_name_design, inputs.h_th_design)

        h_th_full = self.create_output(self.output_h_th_name_full,
                    shape=(self.output_shape,))
        h_th_full.add_name(self.output_h_th_name_full)
        # output = csdl.VariableGroup()
        # output.h_th_full = h_th_full

        self.declare_derivative_parameters(self.output_h_th_name_full,
                                           self.input_h_th_name_design,
                                           dependent=True)

        return h_th_full

    def compute(self, input_vals, output_vals):
        output_vals[self.output_h_th_name_full] = \
            self.deriv_mat*input_vals[self.input_h_th_name_design]

    def compute_derivatives(self, input_vals, output_vals, derivatives):
        derivatives[self.output_h_th_name_full, 
                    self.input_h_th_name_design] = self.deriv_mat

    def get_derivative(self, coo=True):
        diag_vecs = []
        for s_ind in range(self.num_splines):
            diag_vecs += [np.ones((self.h_th_sizes[s_ind],1))]
        deriv_mat = block_diag(*diag_vecs)
        if coo:
            return coo_matrix(deriv_mat)
        else:
            return deriv_mat

if __name__ == "__main__":
    from GOLDFISH.tests.test_tbeam import nonmatching_opt
    # from GOLDFISH.tests.test_slr import nonmatching_opt
    # from GOLDFISH.tests.test_dRdt import nonmatching_opt

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    h_th_design_name = 'h_th_design'
    init_val = np.array([np.average(h_th_sub_array) for h_th_sub_array
                         in nonmatching_opt.init_h_th_list])

    inputs = csdl.VariableGroup()
    inputs.h_th_design = csdl.Variable(value=init_val, 
                                        name=h_th_design_name)

    m = HthMapModel(nonmatching_opt=nonmatching_opt)
    h_th_full = m.evaluate(inputs)
    # h_th_full = outputs.h_th_full

    print(h_th_full.value)

    from csdl_alpha.src.operations.derivative.utils import verify_derivatives_inline
    verify_derivatives_inline([h_th_full], [inputs.h_th_design], 
                              step_size=1e-6, raise_on_error=False)
    recorder.stop()