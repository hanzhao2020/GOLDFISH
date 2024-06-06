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

        # self.int_edge_cons_dofs, self.int_edge_cons_vals = \
        #     self.nonmatching_opt.cpiga2xi.get_diff_intersections_edge_cons_info()

        self.int_edge_cons_dofs = self.nonmatching_opt.cpiga2xi.int_edge_cons_dofs
        self.int_edge_cons_vals = self.nonmatching_opt.cpiga2xi.int_edge_cons_vals

        self.input_shape = self.nonmatching_opt.xi_size
        self.output_shape = self.int_edge_cons_dofs.size
        self.init_xi = get_petsc_vec_array(self.nonmatching_opt.xi_nest)

        self.deriv = self.get_derivative()

    def setup(self):
        self.add_input(self.input_xi_name, shape=self.input_shape,
                       val=self.init_xi)
        self.add_output(self.output_name,
                        shape=self.output_shape)
        self.declare_partials(self.output_name,
                              self.input_xi_name,
                              val=self.deriv.data,
                              rows=self.deriv.row,
                              cols=self.deriv.col)

    def compute(self, inputs, outputs):
        input_array = inputs[self.input_xi_name]
        output_array = np.zeros(self.output_shape)
        for i, xi_dof in enumerate(self.int_edge_cons_dofs):
            output_array[i] = input_array[xi_dof] - self.int_edge_cons_vals[i]
        outputs[self.output_name] = output_array

    def get_derivative(self, coo=True):
        deriv_mat = np.zeros((self.output_shape, self.input_shape))
        for i, xi_dof in enumerate(self.int_edge_cons_dofs):
            deriv_mat[i, xi_dof] = 1.
        if coo:
            deriv_mat = coo_matrix(deriv_mat)
        return deriv_mat


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



    # cp0 = np.array([0,1,-1,0.5])
    # bs0 = BSplines.BSpline([2,], [[0,0,0,0.5,1,1,1]])
    # # xi_list = np.linspace(0,1,4)
    # xi_list = np.array([0.1,0.2,0.6,0.9])

    # A_mat = np.zeros((4,4))
    # for i in range(4):
    #     nodes_evals = bs0.getNodesAndEvals([xi_list[i]])
    #     nodes = [item[0] for item in nodes_evals]
    #     evals = [item[1] for item in nodes_evals]
    #     for j, node in enumerate(nodes):
    #         A_mat[i,node] = evals[j]

    # bs_eval = np.dot(A_mat,cp0)
    # cp1 = np.linalg.solve(A_mat, bs_eval)