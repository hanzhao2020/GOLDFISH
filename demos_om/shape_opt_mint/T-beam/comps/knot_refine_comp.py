import openmdao.api as om
from openmdao.api import Problem
from scipy.sparse import coo_matrix
from scipy.linalg import block_diag

from PENGoLINS.occ_preprocessing import *
from PENGoLINS.nonmatching_shell import *
from IGAOPT.opt_utils import *
from IGAOPT.bsp_utils import *

class KnotRefineComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('design_knots_list')
        self.options.declare('ref_knots_list', default=None)
        self.options.declare('ref_operator', default=None)
        self.options.declare('opt_field', default=[0,1,2])
        self.options.declare('input_CP_name_pre', default='CP_design')
        self.options.declare('output_CP_name_pre', default='CP_analysis')

    def init_parameters(self):
        self.design_knots_list = self.options['design_knots_list']
        self.ref_knots_list = self.options['ref_knots_list']
        self.ref_operator = self.options['ref_operator']
        self.opt_field = self.options['opt_field']
        self.input_CP_name_pre = self.options['input_CP_name_pre']
        self.output_CP_name_pre = self.options['output_CP_name_pre']

        self.num_surfs = len(self.design_knots_list)
        self.input_shape = 0
        for s_ind in range(self.num_surfs):
            deg0 = spline_degree(self.design_knots_list[s_ind][0])
            deg1 = spline_degree(self.design_knots_list[s_ind][1])
            CP_shape0 = len(self.design_knots_list[s_ind][0])-deg0-1
            CP_shape1 = len(self.design_knots_list[s_ind][1])-deg1-1
            self.input_shape += CP_shape0*CP_shape1

        if self.ref_knots_list is None:
            self.ref_operator = coo_matrix(np.eye(self.input_shape))
        if self.ref_operator is None:
            self.ref_operator_list = []
            for i in range(self.num_surfs):
                ref_operator = surface_knot_refine_operator(
                               self.design_knots_list[i], 
                               self.ref_knots_list[i], 
                               coo=False)
                self.ref_operator_list += [ref_operator]
            self.ref_operator = coo_matrix(block_diag(
                                           *self.ref_operator_list))

        self.input_CP_name_list = []
        self.output_CP_name_list = []
        for i, field in enumerate(self.opt_field):
            self.input_CP_name_list += [self.input_CP_name_pre+str(field)]
            self.output_CP_name_list += [self.output_CP_name_pre+str(field)]

        self.CP_init_val = [np.ones(self.input_shape)]*len(self.opt_field)

    def set_initial_val(self, CP_init_val):
        self.CP_init_val = CP_init_val

    def setup(self):
        for i, field in enumerate(self.opt_field):
            self.add_input(self.input_CP_name_list[i], 
                           shape=self.input_shape,
                           val=self.CP_init_val[i])
            self.add_output(self.output_CP_name_list[i], 
                            shape=self.ref_operator.shape[0])
            self.declare_partials(self.output_CP_name_list[i],
                                  self.input_CP_name_list[i],
                                  val=self.ref_operator.data,
                                  rows=self.ref_operator.row,
                                  cols=self.ref_operator.col)

    def compute(self, inputs, outputs):
        for i, field in enumerate(self.opt_field):
            outputs[self.output_CP_name_list[i]] = \
                self.ref_operator*inputs[self.input_CP_name_list[i]]


if __name__ == '__main__':

    filename_igs = "./geometry/initial_geometry.igs"
    igs_shapes = read_igs_file(filename_igs, as_compound=False)
    occ_surf_list = [topoface2surface(face, BSpline=True) 
                     for face in igs_shapes]
    occ_surf_data_list = [BSplineSurfaceData(surf) for surf in occ_surf_list]

    num_surfs = len(occ_surf_list)
    num_sides = 2
    opt_field = [0,1,2]
    p = occ_surf_data_list[0].degree[0]
    design_knots0 = [0]*(p+1) + [0.5] + [1]*(p+1)
    design_knots1 = [0]*(p+1) + [0.5] + [1]*(p+1)
    design_knots = [np.array(design_knots0), np.array(design_knots1)]
    design_knots_list = [design_knots]*3

    ref_knots_list = []
    ref_knots_list_occ_surf = []

    for s_ind in range(num_surfs):
        ref_knots_list += [[],]
        ref_knots_list_occ_surf += [[],]
        for side in range(num_sides):
            ref_knots_list[s_ind] += [[],]
            ref_knots_list_occ_surf[s_ind] += [[],]
            ref_knots_temp = []
            ref_knots_temp_occ_surf = []
            for knot in occ_surf_data_list[s_ind].knots[side][p+1:-(p+1)]:
                if knot not in design_knots_list[s_ind][side]:
                    ref_knots_temp += [knot]
            if len(ref_knots_temp) > 0:
                ref_knots_list[s_ind][side] = np.array(ref_knots_temp)
            else:
                ref_knots_list[s_ind][side] = None
            for knot in design_knots_list[s_ind][side][p+1:-(p+1)]:
                if knot not in occ_surf_data_list[s_ind].knots[side]:
                    ref_knots_temp_occ_surf += [knot]
            if len(ref_knots_temp_occ_surf) > 0:
                ref_knots_list_occ_surf[s_ind][side] = \
                    np.array(ref_knots_temp_occ_surf)
            else:
                ref_knots_list_occ_surf[s_ind][side] = None

    occ_surf_list_opt = []
    for s_ind in range(num_surfs):
        for side in range(num_sides):
            if ref_knots_list_occ_surf[s_ind][side] is not None:
                occ_surf_ref_knots = array2TColStdArray1OfReal(
                    ref_knots_list_occ_surf[s_ind][side])
                occ_surf_ref_mult = array2TColStdArray1OfInteger(
                    np.ones(len(ref_knots_list_occ_surf[s_ind][side])))
                if side == 0:
                    occ_surf_list[s_ind].InsertUKnots(
                        occ_surf_ref_knots, occ_surf_ref_mult)
                if side == 1:
                    occ_surf_list[s_ind].InsertVKnots(
                        occ_surf_ref_knots, occ_surf_ref_mult)
        occ_surf_list_opt += [occ_surf_list[s_ind]]
    occ_surf_data_list_opt = [BSplineSurfaceData(surf) 
                              for surf in occ_surf_list_opt]

    init_CP_analysis_list = [[] for i in range(len(opt_field))]
    for i, field in enumerate(opt_field):
        for s_ind in range(num_surfs):
            init_CP_analysis_list[i] += [occ_surf_data_list_opt[s_ind]\
                                         .control[:,:,field].T.reshape(-1)]
    init_CP_analysis_list = [np.concatenate(init_CP_analysis_list[i]) 
                             for i in range(len(opt_field))]
        
    prob = Problem()
    comp = KnotRefineComp(design_knots_list=design_knots_list, 
                          ref_knots_list=ref_knots_list)
    comp.init_parameters()

    init_CP_design_list = []
    for i, field in enumerate(opt_field):
        init_CP_design = solve_nonsquare(comp.ref_operator.todense(),
                         init_CP_analysis_list[i].reshape(-1,1))
        init_CP_design_list += [np.asarray(init_CP_design).reshape(-1)]
    comp.set_initial_val(init_CP_design_list)

    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print('check_partials:')
    prob.check_partials(compact_print=True)