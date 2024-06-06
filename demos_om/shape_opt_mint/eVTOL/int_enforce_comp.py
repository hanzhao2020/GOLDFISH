from GOLDFISH.nonmatching_opt import *
import openmdao.api as om
from openmdao.api import Problem

class IntEnforceComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('nonmatching_opt')
        self.options.declare('cpdesign2analysis')
        self.options.declare('input_xi_name', default='int_xi')
        self.options.declare('input_cp_name', default='cp_design')
        self.options.declare('output_name', default='int_cp_diff')

    def init_parameters(self):
        self.nonmatching_opt = self.options['nonmatching_opt']
        self.cpsurfd2a = self.options['cpdesign2analysis']
        self.input_xi_name = self.options['input_xi_name']
        self.input_cp_name = self.options['input_cp_name']
        self.output_name = self.options['output_name']

        self.cpiga2xi = self.nonmatching_opt.cpiga2xi

        self.input_xi_shape = self.nonmatching_opt.xi_size
        self.input_cp_shape = len(self.cpsurfd2a.init_cp_design[2])


        self.output_shape = len(self.input_cp_shape)

        self.init_xi = get_petsc_vec_array(self.nonmatching_opt.xi_nest)
        self.init_cp = self.cpsurfd2a.init_cp_design[2]

        self.deriv = self.get_derivative()


    def setup(self):
        self.add_input(self.input_xi_name, shape=self.input_xi_shape,
                       val=self.init_xi)
        self.add_input(self.input_cp_name, shape=self.input_cp_shape,
                       val=self.init_cp)
        self.add_output(self.output_name,
                        shape=self.output_shape)

        self.declare_partials(self.output_name, self.input_xi_name)
        self.declare_partials(self.output_name, self.input_cp_name)

        # self.declare_partials(self.output_name,
        #                       self.input_xi_name,
        #                       val=self.deriv.data,
        #                       rows=self.deriv.row,
        #                       cols=self.deriv.col)

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



    def implicit_edge_residual(self, xi_flat, cp_edges, cp_mids=None):

        res_vec = np.zeros(cp_edge.size)
        edge_int_ind = 0
        for i, diff_int_ind in enumerate(self.diff_int_inds):
            int_type = self.preprocessor.intersections_type[diff_int_ind]
            if int_type[0] == 'surf-edge' or int_type[0] == 'edge-surf':
                int_surf_inds = self.local_int_surf_inds(i)
                edge_indicator = self.preprocessor.diff_int_edge_cons[i]
                side = int(edge_indicator[edge_indicator.index('-')-1])
                para_dir = int(edge_indicator[edge_indicator.index('-')+1])
                edge_val = int(edge_indicator[edge_indicator.index('.')+1])

                xi_flat_sub = xi_flat[self.xi_flat_inds[i]:self.xi_flat_inds[i+1]]
                xi_size = self.xi_sizes[i]
                local_cons_dof = self.int_edge_cons_local_dofs_list[edge_int_ind]

                if para_dir == 0:
                    edge_surf_xi_eval_dofs = np.sort(np.concatenate([local_cons_dof, local_cons_dof+1]))
                else:
                    edge_surf_xi_eval_dofs = np.sort(np.concatenate([local_cons_dof, local_cons_dof-1]))

                if side == 0:
                    mid_surf_xi_eval_dofs = edge_surf_xi_eval_dofs+int(xi_sizes/2)
                else:
                    mid_surf_xi_eval_dofs = edge_surf_xi_eval_dofs-int(xi_sizes/2)


                edge_surf_xi_eval = xi_flat_sub[edge_surf_xi_eval_dofs].reshape(-1,2)
                mid_surf_xi_eval = xi_flat_sub[mid_surf_xi_eval_dofs].reshape(-1,2)

                edge_surf_ind = int_surf_inds[side]
                mid_surf_ind = int_surf_inds[int(1-side)]

                edge_surf_basis = self.int_surf_basis[edge_surf_ind]
                mid_surf_basis = self.int_surf_basis[mid_surf_ind]

                edge_surf_cp = cp_edges[self.cp_flat_inds[edge_surf_ind]:
                                        self.cp_flat_inds[edge_surf_ind+1]]

                if cp_mids is None:
                    mid_surf_cp = self.cp_flat_global\
                              [self.cp_flat_inds[mid_surf_ind]:
                               self.cp_flat_inds[mid_surf_ind+1],
                               self.int_surf_avg_normal_dir[mid_surf_ind]]
                else:
                    mid_surf_cp = cp_mids[self.cp_flat_inds[mid_surf_ind]:
                                          self.cp_flat_inds[mid_surf_ind+1]]

                res_vec[self.cp_flat_inds[mid_surf_ind]:
                        self.cp_flat_inds[mid_surf_ind+1]] += \
                        self.implicit_edge_residual_sub(
                            edge_surf_xi_eval, edge_surf_cp, edge_surf_basis,
                            mid_surf_xi_eval, mid_surf_cp, mid_surf_basis)

                edge_int_ind += 1


if __name__ == "__main__":
    from GOLDFISH.tests.test_tbeam_mint import preprocessor, nonmatching_opt

    #################################
    preprocessor.check_intersections_type()
    preprocessor.get_diff_intersections()
    nonmatching_opt.set_diff_intersections(preprocessor)
    #################################

    prob = Problem()
    comp = IntEnforceComp(nonmatching_opt=nonmatching_opt)
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