import openmdao.api as om
from openmdao.api import Problem
from scipy.sparse import coo_matrix
from PENGoLINS.occ_preprocessing import *
from IGAOPT.opt_utils import *
from IGAOPT.bsp_utils import *

class CPEnforceComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('design_knots_list')
        self.options.declare('enforce_dir', default=[1,1,1])
        self.options.declare('opt_field', default=[0,1,2])
        self.options.declare('input_CP_name_pre', default='CP_design')
        self.options.declare('output_CP_enforce_name_pre', 
                             default='CP_design_enforce')

    def init_parameters(self):
        self.design_knots_list = self.options['design_knots_list']
        self.enforce_dir = self.options['enforce_dir']
        self.opt_field = self.options['opt_field']
        self.input_CP_name_pre = self.options['input_CP_name_pre']
        self.output_CP_enforce_name_pre = \
            self.options['output_CP_enforce_name_pre']

        self.CP_shape_list = []
        self.CP_length_list = []
        self.input_shape = 0
        self.num_surfs = len(self.design_knots_list)
        for s_ind in range(self.num_surfs):
            deg0 = spline_degree(self.design_knots_list[s_ind][0])
            deg1 = spline_degree(self.design_knots_list[s_ind][1])
            CP_shape0 = len(self.design_knots_list[s_ind][0])-deg0-1
            CP_shape1 = len(self.design_knots_list[s_ind][1])-deg1-1
            self.input_shape += CP_shape0*CP_shape1
            self.CP_shape_list += [[CP_shape0, CP_shape1]]
            self.CP_length_list += [CP_shape0*CP_shape1]
        
        self.output_shape = 0
        self.output_shape_list = []

        for s_ind in range(self.num_surfs):
            num_col, num_row = self.CP_shape_list[s_ind]
            if self.enforce_dir[s_ind] == 0:
                self.output_shape += (num_col-1)*num_row
                self.output_shape_list += [(num_col-1)*num_row]
            elif self.enforce_dir[s_ind] == 1:
                self.output_shape += num_col*(num_row-1)
                self.output_shape_list += [num_col*(num_row-1)]
            else:
                raise ValueError("Undefined direction: {}"
                                 .format(self.enforce_dir[s_ind]))

        self.input_CP_name_list = []
        self.output_enforce_name_list = []
        for i, field in enumerate(self.opt_field):
            self.input_CP_name_list += [self.input_CP_name_pre+str(field)]
            self.output_enforce_name_list += \
                [self.output_CP_enforce_name_pre+str(field)]

        self.partial_mat = self.get_derivative()


    def setup(self):
        for i, field in enumerate(self.opt_field):
            self.add_input(self.input_CP_name_list[i], 
                           shape=self.input_shape,
                           val=np.arange(self.input_shape)*(i+1))
            self.add_output(self.output_enforce_name_list[i], 
                            shape=self.output_shape)
            self.declare_partials(self.output_enforce_name_list[i],
                                  self.input_CP_name_list[i],
                                  val=self.partial_mat.data,
                                  rows=self.partial_mat.row,
                                  cols=self.partial_mat.col)

    def compute(self, inputs, outputs):
        for i in range(len(self.opt_field)):
            outputs[self.output_enforce_name_list[i]] = \
                self.compute_enforce_all(inputs[self.input_CP_name_list[i]])

    def compute_enforce(self, CP, s_ind):
        num_col, num_row = self.CP_shape_list[s_ind]
        # num_row = self.num_row
        # num_col = self.num_col
        CP_diff_array = np.zeros(self.output_shape_list[s_ind])
        if self.enforce_dir[s_ind] == 0:
            for i in range(num_row):
                CP_diff_array[i*(num_col-1):(i+1)*(num_col-1)] = \
                    CP[i*num_col+1:(i+1)*num_col] - CP[i*num_col]
        elif self.enforce_dir[s_ind] == 1:
            for i in range(num_row-1):
                CP_diff_array[i*num_col:(i+1)*num_col] = \
                    CP[(i+1)*num_col:(i+2)*num_col] - CP[0:num_col]
        else:
            raise ValueError("Undefined direction: {}"
                             .format(self.enforce_dir[s_ind]))
        return CP_diff_array

    def compute_enforce_all(self, CP):
        CP_diff_list = []
        for s_ind in range(self.num_surfs):
            CP_diff_list += [self.compute_enforce(
                CP[int(np.sum(self.CP_length_list[0:s_ind])):
                int(np.sum(self.CP_length_list[0:s_ind+1]))], s_ind)]
        return np.concatenate(CP_diff_list)

    def get_sub_derivative(self, s_ind, coo=True):
        num_col, num_row = self.CP_shape_list[s_ind]
        derivative = np.zeros((self.output_shape_list[s_ind], 
                               self.CP_length_list[s_ind]))
        if self.enforce_dir[s_ind] == 0:
            for i in range(num_row):
                derivative[i*(num_col-1):(i+1)*(num_col-1), i*num_col] = -1.
                for j in range(1, num_col):
                    derivative[i*(num_col-1)+j-1, i*num_col+j] = 1.
        elif self.enforce_dir[s_ind] == 1:
            for i in range(num_row-1):
                for j in range(num_col):
                    derivative[i*num_col+j, j] = -1.
                    derivative[i*num_col+j, (i+1)*num_col+j] = 1.
        else:
            raise ValueError("Undefined direction: {}"
                             .format(self.enforce_dir[s_ind]))
        if coo:
            derivative = coo_matrix(derivative)
        return derivative

    def get_derivative(self, coo=True):
        sub_derivative_list = []
        for s_ind in range(self.num_surfs):
            sub_derivative_list += [self.get_sub_derivative(s_ind, False)]
        if coo:
            derivative = coo_matrix(block_diag(*sub_derivative_list))
        else:
            derivative = block_diag(*sub_derivative_list)
        return derivative


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
            
    prob = Problem()
    comp = CPEnforceComp(design_knots_list=design_knots_list)
    comp.init_parameters()

    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print('check_partials:')
    prob.check_partials(compact_print=True)