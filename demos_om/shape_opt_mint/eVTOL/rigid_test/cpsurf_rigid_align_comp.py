from GOLDFISH.nonmatching_opt_ffd import *
import openmdao.api as om
from openmdao.api import Problem

# For rib front end line evaluation
# splines[1].cpFuncs[0](0, 0.15) -> 4.669310
# splines[1].cpFuncs[1](0, 0.15) -> 0.528595
# splines[1].cpFuncs[0](1, 0.15) -> 3.860719
# splines[1].cpFuncs[1](1, 0.15) -> 5.318053

# solve line x=a0*y+a1 -> a0 = -0.16882725, a1 = 4.75855124

# For rib rear end line evaluation
# splines[1].cpFuncs[0](0, 0.8) -> 5.439442
# splines[1].cpFuncs[1](0, 0.8) -> 0.528595
# splines[1].cpFuncs[0](1, 0.8) -> 4.410813
# splines[1].cpFuncs[1](1, 0.8) -> 5.318053

# solve line x=b0*y+b1 -> b0 = -0.2147694, b1 = 5.55296803

# a0 = -0.16882725
# a1 = 4.75855124
# b0 = -0.2147694
# b1 = 5.55296803

# Front xi1 = 0.15
x1fr = 0.528595
x1ft = 5.318053
x0fr = 4.669310
x0ft = 3.860719

# Rear xi1 = 0.85
x0rr = 5.497967
x0rt = 4.452617

A0 = np.array([[x1fr, 1], [x1ft, 1]])
x0 = np.array([x0fr, x0ft])
x1 = np.array([x0rr, x0rt])

a0, a1 = np.linalg.solve(A0, x0)
b0, b1 = np.linalg.solve(A0, x1)


class CPSurfRigidAlignComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('cpdesign2analysis')
        self.options.declare('diff_vec', default=None)
        self.options.declare('input_cp_design_name_pre', default='CP_design')
        self.options.declare('output_cp_coarse_name_pre', default='CP_coarse')

    def init_parameters(self):
        self.cpdesign2analysis = self.options['cpdesign2analysis']
        self.diff_vec = self.options['diff_vec']
        self.input_cp_design_name_pre = self.options['input_cp_design_name_pre']
        self.output_cp_coarse_name_pre = self.options['output_cp_coarse_name_pre']

        self.opt_field = self.cpdesign2analysis.opt_field
        self.derivs_temp = self.cpdesign2analysis.cp_coarse_align_deriv_list
        self.init_cp_design_temp = self.cpdesign2analysis.init_cp_design

        self.input_shapes_temp = [mat.shape[1] for mat in self.derivs_temp]
        self.output_shapes = [mat.shape[0] for mat in self.derivs_temp]

        if self.diff_vec is None:
            self.diff_vec = [np.zeros(output_shape) for output_shape 
                             in self.output_shapes]

        self.input_cp_design_name_list = []
        self.output_cp_coarse_name_list = []
        for i, field in enumerate(self.opt_field):
            self.input_cp_design_name_list += \
                [self.input_cp_design_name_pre+str(field)]
            self.output_cp_coarse_name_list += \
                [self.output_cp_coarse_name_pre+str(field)]

        # Customized operations for rigid ribs
        self.input_shapes = self.input_shapes_temp.copy()
        self.init_cp_design = [cp.copy() for cp in self.init_cp_design_temp]
        self.derivs = [mat.copy() for mat in self.derivs_temp]

        # num_x_spar_input = 2
        # num_x_spar_output = 8
        # x_rib_size = 4
        # field0 = 0
        # field1 = 1
        # self.diff_vec[field0][num_x_spar_output:] = 0
        # # self.init_cp_design[field0] = self.init_cp_design[field0][0:num_x_spar_input]
        # self.input_shapes[field0] = num_x_spar_input
        # self.derivs[field0] = coo_matrix(self.derivs_temp[field0].todense()[:,0:num_x_spar_input])

        # self.derivs_rib_dxdy = np.zeros((self.output_shapes[field0], 
        #                                  self.input_shapes[field1]))
        # self.derivs_rib_dxdy_diff = np.zeros(self.output_shapes[field0])

        # sub_dxdy = np.array([a0, b0, a0, b0])
        # sub_diff_vec = np.array([a1, b1, a1, b1])
        # for i in range(self.input_shapes[field1]):
        #     self.derivs_rib_dxdy[num_x_spar_output+i*x_rib_size:
        #                          num_x_spar_output+(i+1)*x_rib_size,i] = sub_dxdy
        #     self.derivs_rib_dxdy_diff[num_x_spar_output+i*x_rib_size:
        #                               num_x_spar_output+(i+1)*x_rib_size] = sub_diff_vec

        # self.derivs_rib_dxdy = coo_matrix(self.derivs_rib_dxdy)

    def setup(self):
        for i, field in enumerate(self.opt_field):
            self.add_input(self.input_cp_design_name_list[i],
                           shape=self.input_shapes[i],
                           val=self.init_cp_design[i])
            self.add_output(self.output_cp_coarse_name_list[i],
                            shape=self.output_shapes[i])
            self.declare_partials(self.output_cp_coarse_name_list[i],
                                  self.input_cp_design_name_list[i],
                                  val=self.derivs[i].data,
                                  rows=self.derivs[i].row,
                                  cols=self.derivs[i].col)

        # # Customized operation
        # self.declare_partials(self.output_cp_coarse_name_list[0],
        #                       self.input_cp_design_name_list[1],
        #                       val=self.derivs_rib_dxdy.data,
        #                       rows=self.derivs_rib_dxdy.row,
        #                       cols=self.derivs_rib_dxdy.col)

    def compute(self, inputs, outputs):
        for i, field in enumerate(self.opt_field):
            if field == 0:
                outputs[self.output_cp_coarse_name_list[i]] = \
                    self.derivs[i]*inputs[self.input_cp_design_name_list[i]] \
                    + self.diff_vec[i] #+ \
                    # self.derivs_rib_dxdy*inputs[self.input_cp_design_name_list[i+1]] \
                    # + self.derivs_rib_dxdy_diff
            else:
                outputs[self.output_cp_coarse_name_list[i]] = \
                    self.derivs[i]*inputs[self.input_cp_design_name_list[i]] \
                    + self.diff_vec[i]


if __name__ == "__main__":
    import time
    from datetime import datetime

    import sys
    sys.path.append("/Users/hanzhao/OneDrive/github/GOLDFISH/demos_om/shape_opt_mint/eVTOL/")

    import numpy as np
    import matplotlib.pyplot as plt
    import openmdao.api as om
    from igakit.cad import *
    from igakit.io import VTK
    from GOLDFISH.nonmatching_opt_om import *

    from create_geom_evtol_1spar import preprocessor


    opt_field = [0,2]
    shopt_surf_inds = [[3], [0,1,3]]
    cpsurfd2a = CPSurfDesign2Analysis(preprocessor, opt_field=opt_field, 
                shopt_surf_inds=shopt_surf_inds)
    design_degree = [2,1]
    p_list = [[design_degree]*len(shopt_surf_inds[0]), 
              [design_degree]*len(shopt_surf_inds[1])]
    design_knots = [[0]*(design_degree[0]+1)+[1]*(design_degree[0]+1),
                    [0]*(design_degree[1]+1)+[1]*(design_degree[1]+1)]
    knots_list = [[design_knots]*len(shopt_surf_inds[0]), 
                  [design_knots]*len(shopt_surf_inds[1])]

    cpsurfd2a.set_init_knots(p_list, knots_list)
    cpsurfd2a.get_init_cp_coarse()
    t0 = cpsurfd2a.set_cp_align(2, [None, 1, None])

    prob = Problem()
    comp = CPSurfRigidAlignComp(cpdesign2analysis=cpsurfd2a)
    comp.init_parameters()
    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print('check_partials:')
    prob.check_partials(compact_print=True)