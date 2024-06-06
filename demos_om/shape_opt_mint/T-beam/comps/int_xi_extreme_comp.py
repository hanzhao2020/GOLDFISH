from GOLDFISH.nonmatching_opt import *
import openmdao.api as om
from openmdao.api import Problem

class IntXiExtremeComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('nonmatching_opt')
        self.options.declare('num_eval_pts', default=None)
        self.options.declare('input_xi_name', default='int_xi')
        self.options.declare('output_name', default='int_xi_cons')

    def init_parameters(self):
        self.nonmatching_opt = self.options['nonmatching_opt']
        self.input_xi_name = self.options['input_xi_name']
        self.output_name = self.options['output_name']

        self.cpiga2xi = self.nonmatching_opt.cpiga2xi
        self.preprocessor = self.nonmatching_opt.preprocessor

        self.num_eval_pts = self.options['num_eval_pts']
        if self.num_eval_pts is not None:
            if not isinstance(self.num_eval_pts, list):
                self.num_eval_pts = [self.num_eval_pts]*len(self.cpiga2xi.diff_int_inds)

            self.int_xi_extreme_cons_dofs = []
            for i, diff_int_ind in enumerate(self.cpiga2xi.diff_int_inds):
                int_type = self.preprocessor.intersections_type[diff_int_ind]
                # print("int_type:", int_type)
                # if 'surf' not in int_type:
                if int_type[0] == 'surf-surf':
                    for extre_cons_side in range(self.cpiga2xi.num_sides):
                        for para_dir in range(self.cpiga2xi.para_dim):
                            if extre_cons_side == 0:
                                start_ind = self.cpiga2xi.xi_flat_inds[i]+para_dir
                                end_ind = int((self.cpiga2xi.xi_flat_inds[i]
                                          +self.cpiga2xi.xi_flat_inds[i+1])/2)
                            elif extre_cons_side == 1:
                                start_ind = int((self.cpiga2xi.xi_flat_inds[i]
                                      +self.cpiga2xi.xi_flat_inds[i+1])/2)+para_dir
                                end_ind = self.cpiga2xi.xi_flat_inds[i+1]

                            xi_dofs = np.arange(start_ind, end_ind, 
                                                self.cpiga2xi.para_dim, dtype='int32')
                            dofs_inds = np.linspace(0, int(self.cpiga2xi.xi_sizes[i]/4-1), 
                                                    self.num_eval_pts[i], dtype='int32')
                            xi_dofs_temp = xi_dofs[dofs_inds]
                            self.int_xi_extreme_cons_dofs += [xi_dofs_temp]

                elif int_type[0] == 'surf-edge':
                    extre_cons_side = 0
                    for para_dir in range(self.cpiga2xi.para_dim):
                        start_ind = self.cpiga2xi.xi_flat_inds[i]+para_dir
                        end_ind = int((self.cpiga2xi.xi_flat_inds[i]
                                       +self.cpiga2xi.xi_flat_inds[i+1])/2)

                        xi_dofs = np.arange(start_ind, end_ind, 
                                            self.cpiga2xi.para_dim, dtype='int32')
                        dofs_inds = np.linspace(0, int(self.cpiga2xi.xi_sizes[i]/4-1), 
                                                self.num_eval_pts[i], dtype='int32')
                        # print('i:', i)
                        # print('dof_inds:', dofs_inds)
                        xi_dofs_temp = xi_dofs[dofs_inds]
                        self.int_xi_extreme_cons_dofs += [xi_dofs_temp]

                    extre_cons_side = 1
                    edge_indicator = self.preprocessor.diff_int_edge_cons[i]
                    edge_para_dir = int(edge_indicator[edge_indicator.index('-')+1])
                    para_dir = int(1-edge_para_dir)

                    start_ind = int((self.cpiga2xi.xi_flat_inds[i]
                                +self.cpiga2xi.xi_flat_inds[i+1])/2)+para_dir
                    end_ind = self.cpiga2xi.xi_flat_inds[i+1]

                    xi_dofs = np.arange(start_ind, end_ind, 
                                        self.cpiga2xi.para_dim, dtype='int32')
                    dofs_inds = np.linspace(0, int(self.cpiga2xi.xi_sizes[i]/4-1), 
                                            self.num_eval_pts[i], dtype='int32')
                    xi_dofs_temp = xi_dofs[dofs_inds]
                    self.int_xi_extreme_cons_dofs += [xi_dofs_temp]


                elif int_type[0] == 'edge-surf':
                    extre_cons_side = 1
                    for para_dir in range(self.cpiga2xi.para_dim):
                        start_ind = int((self.cpiga2xi.xi_flat_inds[i]
                              +self.cpiga2xi.xi_flat_inds[i+1])/2)+para_dir
                        end_ind = self.cpiga2xi.xi_flat_inds[i+1]

                        xi_dofs = np.arange(start_ind, end_ind, 
                                            self.cpiga2xi.para_dim, dtype='int32')
                        dofs_inds = np.linspace(0, int(self.cpiga2xi.xi_sizes[i]/4-1), 
                                                self.num_eval_pts[i], dtype='int32')
                        xi_dofs_temp = xi_dofs[dofs_inds]
                        self.int_xi_extreme_cons_dofs += [xi_dofs_temp]

                    extre_cons_side = 0
                    edge_indicator = self.preprocessor.diff_int_edge_cons[i]
                    edge_para_dir = int(edge_indicator[edge_indicator.index('-')+1])
                    para_dir = int(1-edge_para_dir)

                    start_ind = self.cpiga2xi.xi_flat_inds[i]+para_dir
                    end_ind = int((self.cpiga2xi.xi_flat_inds[i]
                                  +self.cpiga2xi.xi_flat_inds[i+1])/2)

                    xi_dofs = np.arange(start_ind, end_ind, 
                                        self.cpiga2xi.para_dim, dtype='int32')
                    dofs_inds = np.linspace(0, int(self.cpiga2xi.xi_sizes[i]/4-1), 
                                            self.num_eval_pts[i], dtype='int32')
                    xi_dofs_temp = xi_dofs[dofs_inds]
                    self.int_xi_extreme_cons_dofs += [xi_dofs_temp]



            self.int_xi_extreme_cons_dofs = np.concatenate(self.int_xi_extreme_cons_dofs)

        else:
            self.int_xi_extreme_cons_dofs = np.array(self.cpiga2xi.int_xi_free_dofs)

        self.input_shape = self.nonmatching_opt.xi_size
        self.output_shape = self.int_xi_extreme_cons_dofs.size
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
        for i, xi_dof in enumerate(self.int_xi_extreme_cons_dofs):
            output_array[i] = input_array[xi_dof]
        outputs[self.output_name] = output_array

    def get_derivative(self, coo=True):
        deriv_mat = np.zeros((self.output_shape, self.input_shape))
        for i, xi_dof in enumerate(self.int_xi_extreme_cons_dofs):
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
    comp = IntXiExtremeComp(nonmatching_opt=nonmatching_opt)
    comp.init_parameters()
    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print('check_partials:')
    prob.check_partials(compact_print=True)