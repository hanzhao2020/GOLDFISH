from GOLDFISH.nonmatching_opt_ffd import *
import openmdao.api as om
from openmdao.api import Problem

class CPSurfOrderElevationComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('cpdesign2analysis')
        self.options.declare('input_cp_coarse_name_pre', default='CP_coarse')
        self.options.declare('output_cp_order_ele_name_pre', default='CP_order_ele')

    def init_parameters(self):
        self.cpdesign2analysis = self.options['cpdesign2analysis']
        self.input_cp_coarse_name_pre = self.options['input_cp_coarse_name_pre']
        self.output_cp_order_ele_name_pre = self.options['output_cp_order_ele_name_pre']

        self.opt_field = self.cpdesign2analysis.opt_field
        self.derivs = self.cpdesign2analysis.order_ele_operator_list
        self.init_cp_coarse = self.cpdesign2analysis.init_cp_coarse

        self.input_shapes = [mat.shape[1] for mat in self.derivs]
        self.output_shapes = [mat.shape[0] for mat in self.derivs]

        self.input_cp_coarse_name_list = []
        self.output_cp_order_ele_name_list = []
        for i, field in enumerate(self.opt_field):
            self.input_cp_coarse_name_list += \
                [self.input_cp_coarse_name_pre+str(field)]
            self.output_cp_order_ele_name_list += \
                [self.output_cp_order_ele_name_pre+str(field)]

    def setup(self):
        for i, field in enumerate(self.opt_field):
            self.add_input(self.input_cp_coarse_name_list[i],
                           shape=self.input_shapes[i],
                           val=self.init_cp_coarse[i])
            self.add_output(self.output_cp_order_ele_name_list[i],
                            shape=self.output_shapes[i])
            self.declare_partials(self.output_cp_order_ele_name_list[i],
                                  self.input_cp_coarse_name_list[i],
                                  val=self.derivs[i].data,
                                  rows=self.derivs[i].row,
                                  cols=self.derivs[i].col)

    def compute(self, inputs, outputs):
        for i, field in enumerate(self.opt_field):
            outputs[self.output_cp_order_ele_name_list[i]] = \
                self.derivs[i]*inputs[self.input_cp_coarse_name_list[i]]


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
    comp = CPSurfOrderElevationComp(cpdesign2analysis=cpsurfd2a)
    comp.init_parameters()
    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print('check_partials:')
    prob.check_partials(compact_print=True)