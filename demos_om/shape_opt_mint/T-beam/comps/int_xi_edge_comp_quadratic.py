from GOLDFISH.nonmatching_opt import *
import openmdao.api as om
from openmdao.api import Problem

class IntXiEdgeComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('nonmatching_opt')
        self.options.declare('input_xi_name', default='int_xi')
        self.options.declare('output_name', default='int_xi_edge')

    def init_parameters(self):
        self.nonmatching_opt = self.options['nonmatching_opt']
        self.input_xi_name = self.options['input_xi_name']
        self.output_name = self.options['output_name']

        self.int_edge_cons_dofs, self.int_edge_cons_vals = \
            self.nonmatching_opt.cpiga2xi.get_diff_intersections_edge_cons_info()

        self.input_shape = self.nonmatching_opt.xi_size
        self.output_shape = 1
        self.init_xi = get_petsc_vec_array(self.nonmatching_opt.xi_nest)

    def setup(self):
        self.add_input(self.input_xi_name, shape=self.input_shape,
                       val=self.init_xi+np.random.random(self.input_shape))
        self.add_output(self.output_name)
        self.declare_partials(self.output_name,
                              self.input_xi_name)

    def compute(self, inputs, outputs):
        input_array = inputs[self.input_xi_name]
        output_val = 0
        for i, xi_dof in enumerate(self.int_edge_cons_dofs):
            output_val += (input_array[xi_dof] - self.int_edge_cons_vals[i])**2
        outputs[self.output_name] = output_val

    def compute_partials(self, inputs, partials):
        input_array = inputs[self.input_xi_name]

        deriv_array = np.zeros(self.input_shape)
        for i, xi_dof in enumerate(self.int_edge_cons_dofs):
            deriv_array[xi_dof] = 2*(input_array[xi_dof] - self.int_edge_cons_vals[i])

        partials[self.output_name, self.input_xi_name] = deriv_array

if __name__ == "__main__":
    from GOLDFISH.tests.test_tbeam_mint import preprocessor, nonmatching_opt

    #################################
    preprocessor.check_intersections_type()
    preprocessor.get_diff_intersections()
    nonmatching_opt.set_diff_intersections(preprocessor)
    #################################

    prob = Problem()
    comp = IntXiEdgeComp(nonmatching_opt=nonmatching_opt)
    comp.init_parameters()
    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print('check_partials:')
    prob.check_partials(compact_print=True)