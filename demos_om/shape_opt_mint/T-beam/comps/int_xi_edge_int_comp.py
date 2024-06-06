import sys
sys.path.append("../opers/")
sys.path.append("../")

from GOLDFISH.nonmatching_opt import *

from int_xi_edge_int_exop import *

import openmdao.api as om
from openmdao.api import Problem

class IntXiEdgeIntComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('nonmatching_opt')
        self.options.declare('input_xi_name', default='int_xi')
        self.options.declare('output_name', default='int_xi_edge_int')

    def init_parameters(self):
        self.nonmatching_opt = self.options['nonmatching_opt']
        self.input_xi_name = self.options['input_xi_name']
        self.output_name = self.options['output_name']

        self.int_xi_edge_int_exop = IntXiEdgeIntExop(self.nonmatching_opt)

        self.input_shape = self.nonmatching_opt.xi_size
        self.init_xi = get_petsc_vec_array(self.nonmatching_opt.xi_nest)
        # self.init_xi = np.random.random(self.input_shape)

    def setup(self):
        self.add_input(self.input_xi_name, shape=self.input_shape,
                       val=self.init_xi)
        self.add_output(self.output_name)
        self.declare_partials(self.output_name,
                              self.input_xi_name)

    def update_inputs(self, inputs):
        self.nonmatching_opt.update_xi(inputs[self.input_xi_name])

    def compute(self, inputs, outputs):
        self.update_inputs(inputs)
        outputs[self.output_name] = self.int_xi_edge_int_exop.int_edge_int()

    def compute_partials(self, inputs, partials):
        self.update_inputs(inputs)
        dintdxi_vec = self.int_xi_edge_int_exop.dint_xi_edge_int_dxi(array=True)
        partials[self.output_name, self.input_xi_name] = dintdxi_vec


if __name__ == "__main__":
    from GOLDFISH.tests.test_tbeam_mint import preprocessor, nonmatching_opt

    #################################
    opt_field = [0]
    opt_surf_inds = [[1]]

    nonmatching_opt.set_geom_preprocessor(preprocessor)


    nonmatching_opt.set_shopt_surf_inds(opt_field=opt_field, shopt_surf_inds=opt_surf_inds)
    nonmatching_opt.get_init_CPIGA()
    nonmatching_opt.set_shopt_align_CP(align_surf_inds=[[1]],
                                       align_dir=[[1]])
                                       
    # nonmatching_opt.create_mortar_meshes(preprocessor.mortar_nels)

    # if mpirank == 0:
    #     print("Setting up mortar meshes...")
    # nonmatching_opt.mortar_meshes_setup(preprocessor.mapping_list, 
    #                     preprocessor.intersections_para_coords, 
    #                     penalty_coefficient, 2)
    # pressure = -Constant(1.)
    # f = as_vector([Constant(0.), Constant(0.), pressure])
    # source_terms = []
    # residuals = []
    # for s_ind in range(nonmatching_opt.num_splines):
    #     z = nonmatching_opt.splines[s_ind].rationalize(
    #         nonmatching_opt.spline_test_funcs[s_ind])
    #     source_terms += [inner(f, z)*nonmatching_opt.splines[s_ind].dx]
    #     residuals += [SVK_residual(nonmatching_opt.splines[s_ind], 
    #                   nonmatching_opt.spline_funcs[s_ind], 
    #                   nonmatching_opt.spline_test_funcs[s_ind], 
    #                   E, nu, h_th, source_terms[s_ind])]
    # nonmatching_opt.set_residuals(residuals)

    preprocessor.check_intersections_type()
    preprocessor.get_diff_intersections()
    nonmatching_opt.create_diff_intersections()
    #################################

    prob = Problem()
    comp = IntXiEdgeIntComp(nonmatching_opt=nonmatching_opt)
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