import openmdao.api as om
from openmdao.api import Problem
from scipy.sparse import coo_matrix
from PENGoLINS.occ_preprocessing import *
from GOLDFISH.utils.opt_utils import *
from GOLDFISH.utils.bsp_utils import *

class CPSurfAlignComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('nonmatching_opt')
        self.options.declare('align_surf_ind', default=[1])
        self.options.declare('align_dir', default=[1])
        self.options.declare('input_cp_iga_name_pre', default='CP_IGA')
        self.options.declare('output_cp_align_name_pre', default='CPS_align')

    def init_parameters(self):
        self.nonmatching_opt = self.options['nonmatching_opt']
        self.align_surf_ind = self.options['align_surf_ind']
        self.align_dir = self.options['align_dir']
        self.input_cp_iga_name_pre = self.options['input_cp_iga_name_pre']
        self.output_cp_align_name_pre = self.options['output_cp_align_name_pre']

        self.num_surfs = self.nonmatching_opt.num_splines
        self.opt_field = self.nonmatching_opt.opt_field
        self.init_cp_iga = self.nonmatching_opt.get_init_CPIGA()

        self.input_shape = self.nonmatching_opt.vec_scalar_iga_dof
        self.cp_info = self.nonmatching_opt.cpiga2xi
        self.cp_shapes = self.cp_info.cp_shapes
        self.cp_sizes = self.cp_info.cp_sizes

        self.output_shape = 0
        self.output_shape_list = []

        for ind, s_ind in enumerate(self.align_surf_ind):
            num_col, num_row = self.cp_shapes[s_ind]
            if self.align_dir[ind] == 0:
                self.output_shape += (num_col-1)*num_row
                self.output_shape_list += [(num_col-1)*num_row]
            elif self.align_dir[ind] == 1:
                self.output_shape += num_col*(num_row-1)
                self.output_shape_list += [num_col*(num_row-1)]
            else:
                raise ValueError("Undefined direction: {}"
                                 .format(self.align_dir[ind]))


        sub_vecs = self.nonmatching_opt.vec_scalar_iga_nest\
                   .copy().getNestSubVecs()
        self.align_dofs = []
        self.align_shape = 0
        for i, s_ind in enumerate(self.align_surf_ind):
            ind_off = np.sum(self.cp_sizes[0:s_ind])
            sub_vec_range = sub_vecs[s_ind].getOwnershipRange()
            sub_vec_size = sub_vecs[s_ind].size
            self.align_dofs += list(range(ind_off+sub_vec_range[0],
                                        ind_off+sub_vec_range[1]))
            self.align_shape += sub_vec_size



        self.partial_mat = self.get_derivative()

        self.input_cp_iga_name_list = []
        self.output_cp_align_name_list = []
        for i, field in enumerate(self.opt_field):
            self.input_cp_iga_name_list += [self.input_cp_iga_name_pre+str(field),]
            self.output_cp_align_name_list += [self.output_cp_align_name_pre+str(field)]

    def setup(self):
        for i, field in enumerate(self.opt_field):
            self.add_input(self.input_cp_iga_name_list[i], 
                           shape=self.input_shape,
                           val=self.init_cp_iga[:,field])
            self.add_output(self.output_cp_align_name_list[i], 
                            shape=self.output_shape)
            self.declare_partials(self.output_cp_align_name_list[i],
                                  self.input_cp_iga_name_list[i],
                                  val=self.partial_mat.data,
                                  rows=self.partial_mat.row,
                                  cols=self.partial_mat.col)

    def compute(self, inputs, outputs):
        for i in range(len(self.opt_field)):
            outputs[self.output_cp_align_name_list[i]] = \
                self.compute_enforce_all(inputs[self.input_cp_iga_name_list[i]])


    def compute_enforce(self, CP, ind):
        num_col, num_row = self.cp_shapes[self.align_surf_ind[ind]]
        # num_row = self.num_row
        # num_col = self.num_col
        CP_diff_array = np.zeros(self.output_shape_list[ind])
        if self.align_dir[ind] == 0:
            for i in range(num_row):
                CP_diff_array[i*(num_col-1):(i+1)*(num_col-1)] = \
                    CP[i*num_col+1:(i+1)*num_col] - CP[i*num_col]
        elif self.align_dir[ind] == 1:
            for i in range(num_row-1):
                CP_diff_array[i*num_col:(i+1)*num_col] = \
                    CP[(i+1)*num_col:(i+2)*num_col] - CP[0:num_col]
        else:
            raise ValueError("Undefined direction: {}"
                             .format(self.align_dir[ind]))
        return CP_diff_array

    def compute_enforce_all(self, CP):
        CP_diff_list = []
        for ind, s_ind in enumerate(self.align_surf_ind):
            CP_diff_list += [self.compute_enforce(
                CP[int(np.sum(self.cp_sizes[0:s_ind])):
                int(np.sum(self.cp_sizes[0:s_ind+1]))], ind)]
        return np.concatenate(CP_diff_list)

    def get_sub_derivative(self, ind, coo=True):
        num_col, num_row = self.cp_shapes[self.align_surf_ind[ind]]
        derivative = np.zeros((self.output_shape_list[ind], 
                               self.cp_sizes[self.align_surf_ind[ind]]))
        if self.align_dir[ind] == 0:
            for i in range(num_row):
                derivative[i*(num_col-1):(i+1)*(num_col-1), i*num_col] = -1.
                for j in range(1, num_col):
                    derivative[i*(num_col-1)+j-1, i*num_col+j] = 1.
        elif self.align_dir[ind] == 1:
            for i in range(num_row-1):
                for j in range(num_col):
                    derivative[i*num_col+j, j] = -1.
                    derivative[i*num_col+j, (i+1)*num_col+j] = 1.
        else:
            raise ValueError("Undefined direction: {}"
                             .format(self.align_dir[ind]))
        if coo:
            derivative = coo_matrix(derivative)
        return derivative

    def get_derivative(self, coo=True):
        sub_derivative_list = []
        for ind, s_ind in enumerate(self.align_surf_ind):
            sub_derivative_list += [self.get_sub_derivative(ind, False)]

        # for ind, s_ind in enumerate(self.align_surf_ind):
        #         sub_derivative_list += [self.get_sub_derivative(ind, False)]
        # if coo:
        #     derivative = coo_matrix(block_diag(*sub_derivative_list))
        # else:
        #     derivative = block_diag(*sub_derivative_list)

        if len(sub_derivative_list) == 1:
            derivative = sub_derivative_list[0]
        else:
            derivative = block_diag(*sub_derivative_list)

        partial_mat2 = np.zeros((self.align_shape, self.input_shape))
        for i in range(self.align_shape):
            partial_mat2[i, self.align_dofs[i]] = 1.

        res_mat = np.dot(derivative, partial_mat2)

        if coo:
            return coo_matrix(res_mat)
        else:
            return res_mat



if __name__ == '__main__':
    from PENGoLINS.occ_preprocessing import *
    from GOLDFISH.nonmatching_opt_om import *
    from time import perf_counter

    class SplineBC(object):
        """
        Setting Dirichlet boundary condition to tIGAr spline generator.
        """
        def __init__(self, directions=[0,1], sides=[[0,1],[0,1]], 
                     fields=[[[0,1,2],[0,1,2]],[[0,1,2],[0,1,2]]],
                     n_layers=[[1,1],[1,1]]):
            self.fields = fields
            self.directions = directions
            self.sides = sides
            self.n_layers = n_layers

        def set_bc(self, spline_generator):
            for direction in self.directions:
                for side in self.sides[direction]:
                    for field in self.fields[direction][side]:
                        scalar_spline = spline_generator.getScalarSpline(field)
                        side_dofs = scalar_spline.getSideDofs(direction,
                                    side, nLayers=self.n_layers[direction][side])
                        spline_generator.addZeroDofs(field, side_dofs)

    def OCCBSpline2tIGArSpline(surface, num_field=3, quad_deg_const=3, 
                               spline_bc=None, index=0):
        """
        Generate ExtractedBSpline from OCC B-spline surface.
        """
        quad_deg = surface.UDegree()*quad_deg_const
        # DIR = SAVE_PATH+"spline_data/extraction_"+str(index)+"_init"
        # spline = ExtractedSpline(DIR, quad_deg)
        spline_mesh = NURBSControlMesh4OCC(surface, useRect=False)
        spline_generator = EqualOrderSpline(worldcomm, num_field, spline_mesh)
        if spline_bc is not None:
            spline_bc.set_bc(spline_generator)
        # spline_generator.writeExtraction(DIR)
        spline = ExtractedSpline(spline_generator, quad_deg)
        return spline

    test_ind = 3
    optimizer = 'SNOPT'
    opt_field = [0]
    ffd_block_num_el = [6,2,1]
    # save_path = './'
    save_path = '/home/han/Documents/test_results/'
    # folder_name = "results/"
    folder_name = "results"+str(test_ind)+"/"

    filename_igs = "../geometry/init_Tbeam_geom_moved.igs"
    igs_shapes = read_igs_file(filename_igs, as_compound=False)
    occ_surf_list = [topoface2surface(face, BSpline=True) 
                     for face in igs_shapes]
    occ_surf_data_list = [BSplineSurfaceData(surf) for surf in occ_surf_list]
    num_surfs = len(occ_surf_list)
    p = occ_surf_data_list[0].degree[0]

    # Define material and geometric parameters
    E = Constant(1.0e12)
    nu = Constant(0.)
    h_th = Constant(0.1)
    penalty_coefficient = 1.0e3
    pressure = Constant(1.)

    fields0 = [None, [[0,1,2]],]
    spline_bc0 = SplineBC(directions=[1], sides=[None, [0]],
                          fields=fields0, n_layers=[None, [1]])
    spline_bcs = [spline_bc0,]*2

    # Geometry preprocessing and surface-surface intersections computation
    preprocessor = OCCPreprocessing(occ_surf_list, reparametrize=False, 
                                    refine=False)
    print("Computing intersections...")
    int_data_filename = "int_data.npz"
    if os.path.isfile(int_data_filename):
        preprocessor.load_intersections_data(int_data_filename)
    else:
        preprocessor.compute_intersections(mortar_refine=2)
        preprocessor.save_intersections_data(int_data_filename)

    if mpirank == 0:
        print("Total DoFs:", preprocessor.total_DoFs)
        print("Number of intersections:", preprocessor.num_intersections_all)

    # cpiga2xi = CPIGA2Xi(preprocessor)


    if mpirank == 0:
        print("Creating splines...")
    # Create tIGAr extracted spline instances
    splines = []
    for i in range(num_surfs):
            spline = OCCBSpline2tIGArSpline(preprocessor.BSpline_surfs[i], 
                                            spline_bc=spline_bcs[i], index=i)
            splines += [spline,]

    # Create non-matching problem
    nonmatching_opt_ffd = NonMatchingOpt(splines, E, h_th, nu, comm=worldcomm)
    nonmatching_opt_ffd.set_shape_opt(opt_field=opt_field, opt_surf_inds=[[1]])
    nonmatching_opt_ffd.create_mortar_meshes(preprocessor.mortar_nels)

    if mpirank == 0:
        print("Setting up mortar meshes...")
    nonmatching_opt_ffd.mortar_meshes_setup(preprocessor.mapping_list, 
                        preprocessor.intersections_para_coords, 
                        penalty_coefficient, 2)
    pressure = -Constant(1.)
    f = as_vector([Constant(0.), Constant(0.), pressure])
    source_terms = []
    residuals = []
    for s_ind in range(nonmatching_opt_ffd.num_splines):
        z = nonmatching_opt_ffd.splines[s_ind].rationalize(
            nonmatching_opt_ffd.spline_test_funcs[s_ind])
        source_terms += [inner(f, z)*nonmatching_opt_ffd.splines[s_ind].dx]
        residuals += [SVK_residual(nonmatching_opt_ffd.splines[s_ind], 
                      nonmatching_opt_ffd.spline_funcs[s_ind], 
                      nonmatching_opt_ffd.spline_test_funcs[s_ind], 
                      E, nu, h_th, source_terms[s_ind])]
    nonmatching_opt_ffd.set_residuals(residuals)

    preprocessor.check_intersections_type()
    preprocessor.get_diff_intersections()
    nonmatching_opt_ffd.set_diff_intersections(preprocessor)  

    nonmatching_opt_ffd.set_init_CPIGA(nonmatching_opt_ffd.cpiga2xi.cp_flat_global)


    prob = Problem()
    comp = CPSurfAlignComp(nonmatching_opt=nonmatching_opt_ffd,
                           align_surf_ind=[1],
                           align_dir=[1])
    comp.init_parameters()

    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print('check_partials:')
    prob.check_partials(compact_print=True)