import openmdao.api as om
from openmdao.api import Problem
from scipy.sparse import coo_matrix
from PENGoLINS.occ_preprocessing import *
from IGAOPT.opt_utils import *
from IGAOPT.bsp_utils import *

class CPPinComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('design_knots_list')
        self.options.declare('opt_field', default=[0,1,2])
        self.options.declare('surf_inds', default=[0,2])
        self.options.declare('pin_dir', default=[[0], [0]])
        self.options.declare('pin_side', default=[[[0], None],
                                                  [[1], None]])
        self.options.declare('pin_one_end', default=False)
        self.options.declare('input_CP_name_pre', default='CP_design')
        self.options.declare('output_pin_name_pre', default='CP_design_pin')

    def init_parameters(self):
        self.design_knots_list = self.options['design_knots_list']
        self.opt_field = self.options['opt_field']
        self.surf_inds = self.options['surf_inds']
        self.pin_dir = self.options['pin_dir']
        self.pin_side = self.options['pin_side']
        self.pin_one_end = self.options['pin_one_end']
        self.input_CP_name_pre = self.options['input_CP_name_pre']
        self.output_pin_name_pre = self.options['output_pin_name_pre']

        self.num_surfs = len(self.design_knots_list)
        self.CP_shape_list = []
        self.CP_length_list = []
        self.input_shape = 0
        for s_ind in range(self.num_surfs):
            deg0 = spline_degree(self.design_knots_list[s_ind][0])
            deg1 = spline_degree(self.design_knots_list[s_ind][1])
            CP_shape0 = len(self.design_knots_list[s_ind][0])-deg0-1
            CP_shape1 = len(self.design_knots_list[s_ind][1])-deg1-1
            self.input_shape += CP_shape0*CP_shape1
            self.CP_shape_list += [[CP_shape0, CP_shape1]]
            self.CP_length_list += [CP_shape0*CP_shape1]

        pin_dof_list = []
        self.dof_array_list = []
        dofs = np.arange(self.input_shape)
        for s_ind in range(self.num_surfs):
            num_col, num_row = self.CP_shape_list[s_ind]
            self.dof_array_list += [dofs[
                                    int(np.sum(self.CP_length_list[0:s_ind])):
                                    int(np.sum(self.CP_length_list[0:s_ind+1])
                                    )].reshape(num_row, num_col)]

        for i, s_ind in enumerate(self.surf_inds):
            if self.pin_dir[i] is not None:
                for direction in self.pin_dir[i]:
                    if self.pin_side[i][direction] is not None:
                        for side in self.pin_side[i][direction]:
                            # print(s_ind, "--", direction, "--", side)
                            if direction == 0:
                                if self.pin_one_end:
                                    pin_dof_list += [[self.dof_array_list
                                                     [s_ind][:,-1*side][0],]]
                                else:
                                    pin_dof_list += [[self.dof_array_list
                                                     [s_ind][:,-1*side],]]
                            else:
                                if self.pin_one_end:
                                    pin_dof_list += [[self.dof_array_list
                                                     [s_ind][-1*side,:][0],]]
                                else:
                                    pin_dof_list += [[self.dof_array_list
                                                     [s_ind][-1*side,:],]]
        self.pin_dof = np.concatenate(pin_dof_list).reshape(-1)
        self.output_shape = self.pin_dof.shape[0]
        self.pin_val = [np.zeros(self.output_shape)]*len(self.opt_field)
        self.partial_mat = self.get_derivative()

        self.input_CP_name_list = []
        self.output_pin_name_list = []
        for i, field in enumerate(self.opt_field):
            self.input_CP_name_list += [self.input_CP_name_pre+str(field),]
            self.output_pin_name_list += [self.output_pin_name_pre+str(field)]

    def get_derivative(self):
        derivative = np.zeros((self.output_shape, self.input_shape))
        for i in range(self.output_shape):
            derivative[i, self.pin_dof[i]] = 1
        derivative_coo = coo_matrix(derivative)
        return derivative_coo

    def set_pin_val(self, pin_val):
        self.pin_val = pin_val

    def setup(self):
        for i, field in enumerate(self.opt_field):
            self.add_input(self.input_CP_name_list[i], 
                           shape=self.input_shape,
                           val=np.arange(self.input_shape)*(i+1))
            self.add_output(self.output_pin_name_list[i], 
                            shape=self.output_shape)
            self.declare_partials(self.output_pin_name_list[i],
                                  self.input_CP_name_list[i],
                                  val=self.partial_mat.data,
                                  rows=self.partial_mat.row,
                                  cols=self.partial_mat.col)

    def compute(self, inputs, outputs):
        for i, field in enumerate(self.opt_field):
            outputs[self.output_pin_name_list[i]] = \
                inputs[self.input_CP_name_list[i]]\
                [self.pin_dof] - self.pin_val[i]

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

    # ref_knots_list = []
    # ref_knots_list_occ_surf = []

    # for s_ind in range(num_surfs):
    #     ref_knots_list += [[],]
    #     ref_knots_list_occ_surf += [[],]
    #     for side in range(num_sides):
    #         ref_knots_list[s_ind] += [[],]
    #         ref_knots_list_occ_surf[s_ind] += [[],]
    #         ref_knots_temp = []
    #         ref_knots_temp_occ_surf = []
    #         for knot in occ_surf_data_list[s_ind].knots[side][p+1:-(p+1)]:
    #             if knot not in design_knots_list[s_ind][side]:
    #                 ref_knots_temp += [knot]
    #         if len(ref_knots_temp) > 0:
    #             ref_knots_list[s_ind][side] = np.array(ref_knots_temp)
    #         else:
    #             ref_knots_list[s_ind][side] = None
    #         for knot in design_knots_list[s_ind][side][p+1:-(p+1)]:
    #             if knot not in occ_surf_data_list[s_ind].knots[side]:
    #                 ref_knots_temp_occ_surf += [knot]
    #         if len(ref_knots_temp_occ_surf) > 0:
    #             ref_knots_list_occ_surf[s_ind][side] = \
    #                 np.array(ref_knots_temp_occ_surf)
    #         else:
    #             ref_knots_list_occ_surf[s_ind][side] = None

    # occ_surf_list_opt = []
    # for s_ind in range(num_surfs):
    #     for side in range(num_sides):
    #         if ref_knots_list_occ_surf[s_ind][side] is not None:
    #             occ_surf_ref_knots = array2TColStdArray1OfReal(
    #                 ref_knots_list_occ_surf[s_ind][side])
    #             occ_surf_ref_mult = array2TColStdArray1OfInteger(
    #                 np.ones(len(ref_knots_list_occ_surf[s_ind][side])))
    #             if side == 0:
    #                 occ_surf_list[s_ind].InsertUKnots(
    #                     occ_surf_ref_knots, occ_surf_ref_mult)
    #             if side == 1:
    #                 occ_surf_list[s_ind].InsertVKnots(
    #                     occ_surf_ref_knots, occ_surf_ref_mult)
    #     occ_surf_list_opt += [occ_surf_list[s_ind]]
    # occ_surf_data_list_opt = [BSplineSurfaceData(surf) 
    #                           for surf in occ_surf_list_opt]

    # init_CP_analysis_list = [[] for i in range(len(opt_field))]
    # for i, field in enumerate(opt_field):
    #     for s_ind in range(num_surfs):
    #         init_CP_analysis_list[i] += [occ_surf_data_list_opt[s_ind]\
    #                                      .control[:,:,field].T.reshape(-1)]
    # init_CP_analysis_list = [np.concatenate(init_CP_analysis_list[i]) 
    #                          for i in range(len(opt_field))]
        
    prob = Problem()
    comp = CPPinComp(design_knots_list=design_knots_list, pin_one_end=True)
    comp.init_parameters()

    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print('check_partials:')
    prob.check_partials(compact_print=True)