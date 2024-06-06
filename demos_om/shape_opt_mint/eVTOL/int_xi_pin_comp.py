from GOLDFISH.nonmatching_opt import *
import openmdao.api as om
from openmdao.api import Problem

class IntXiPinComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('nonmatching_opt')
        self.options.declare('input_xi_name', default='int_xi')
        self.options.declare('output_name', default='int_xi_pin')

    def init_parameters(self):
        self.nonmatching_opt = self.options['nonmatching_opt']
        self.input_xi_name = self.options['input_xi_name']
        self.output_name = self.options['output_name']

        self.cpiga2xi = self.nonmatching_opt.cpiga2xi

        # Pin xi coordinates to fix leading edge parametric coordinate
        #  on ribs w.r.t. to top surf
        self.xi_pin_dofs = []

        rib_inds = [5,6,7,8,9,10]
        surf_ind = 1

        for i, diff_int_ind in enumerate(self.cpiga2xi.diff_int_inds):
            s_ind0, s_ind1 = self.nonmatching_opt.mapping_list[diff_int_ind]
            # print(f's0: {s_ind0} --- s1:{s_ind1}')
            if s_ind0 == surf_ind and s_ind1 in rib_inds:
                # fix ribs leading edge
                pin_dof = int((self.cpiga2xi.xi_flat_inds[i]
                               +self.cpiga2xi.xi_flat_inds[i+1])/2-1)
                # # Fix ribs trailing edge
                # pin_dof = int(self.cpiga2xi.xi_flat_inds[i]+1)
                # print("---------- pin dof:", pin_dof)
                self.xi_pin_dofs += [pin_dof]
        print("xi_pin_dofs:", self.xi_pin_dofs)

        self.xi_pin_vals = np.ones(len(self.xi_pin_dofs))*0.18
        # self.xi_pin_vals = np.ones(len(self.xi_pin_dofs))*0.8


        # pin_dof = []
        # for i, j in enumerate(nonmatching_opt.cpiga2xi.diff_int_inds):
        #     s0, s1 = nonmatching_opt.mapping_list[j]
        #     if s0 == 1 and s1 in [5,6,7,8,9,10]:
        #             # pin_dof += [int(nonmatching_opt.cpiga2xi.xi_flat_inds[i]+1)]
        #             pin_dof += [int((nonmatching_opt.cpiga2xi.xi_flat_inds[i]+nonmatching_opt.cpiga2xi.xi_flat_inds[i+1])/2-1)]
            
        self.input_shape = self.nonmatching_opt.xi_size
        self.output_shape = len(self.xi_pin_dofs)
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
        for i, xi_dof in enumerate(self.xi_pin_dofs):
            output_array[i] = input_array[xi_dof] - self.xi_pin_vals[i]
        outputs[self.output_name] = output_array

    def get_derivative(self, coo=True):
        deriv_mat = np.zeros((self.output_shape, self.input_shape))
        for i, xi_dof in enumerate(self.xi_pin_dofs):
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
    comp = IntXiPinComp(nonmatching_opt=nonmatching_opt)
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