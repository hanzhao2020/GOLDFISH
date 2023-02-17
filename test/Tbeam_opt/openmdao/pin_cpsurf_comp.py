import openmdao.api as om
from openmdao.api import Problem
from scipy.sparse import coo_matrix
from PENGoLINS.occ_preprocessing import *
from GOLDFISH.utils.opt_utils import *
from GOLDFISH.utils.bsp_utils import *

class CPSurfPinComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('nonmatching_opt')
        self.options.declare('pin_surf_inds', default=[0])
        self.options.declare('input_cp_iga_name_pre', default='CP_IGA')
        self.options.declare('output_cp_pin_name_pre', default='CPS_pin')

    def init_paramters(self):
        self.nonmatching_opt = self.options['nonmatching_opt']
        self.pin_surf_inds = self.options['pin_surf_inds']
        self.input_cp_iga_name_pre = self.options['input_cp_iga_name_pre']
        self.output_cp_pin_name_pre = self.options['output_cp_pin_name_pre']

        self.opt_field = self.nonmatching_opt.opt_field
        self.input_shape = self.nonmatching_opt.vec_scalar_iga_dof
        self.init_cp_iga = self.nonmatching_opt.get_init_CPIGA()

        sub_vecs = self.nonmatching_opt.vec_scalar_iga_nest\
                   .copy().getNestSubVecs()
        self.cp_sizes = [vec.size for vec in sub_vecs]
        self.pin_dofs = []
        self.output_shape = 0
        for i, s_ind in enumerate(self.pin_surf_inds):
            ind_off = np.sum(self.cp_sizes[0:s_ind])
            sub_vec_range = sub_vecs[s_ind].getOwnershipRange()
            sub_vec_size = sub_vecs[s_ind].size
            self.pin_dofs += list(range(int(ind_off+sub_vec_range[0]),
                                        int(ind_off+sub_vec_range[1])))
            self.output_shape += sub_vec_size

        self.pin_val = []
        for i, field in enumerate(self.opt_field):
            self.pin_val += [self.init_cp_iga[self.pin_dofs, field]]

        self.partial_mat = self.get_derivative()

        self.input_CP_name_list = []
        self.output_pin_name_list = []
        for i, field in enumerate(self.opt_field):
            self.input_CP_name_list += [self.input_cp_iga_name_pre+str(field),]
            self.output_pin_name_list += [self.output_cp_pin_name_pre+str(field)]


    def setup(self):
        for i, field in enumerate(self.opt_field):
            self.add_input(self.input_CP_name_list[i], 
                           shape=self.input_shape,
                           val=self.init_cp_iga[:,field])
            self.add_output(self.output_pin_name_list[i], 
                            shape=self.output_shape)
            self.declare_partials(self.output_pin_name_list[i],
                                  self.input_CP_name_list[i],
                                  val=self.partial_mat.data,
                                  rows=self.partial_mat.row,
                                  cols=self.partial_mat.col)

    def get_derivative(self, coo=True):
        partial_mat = np.zeros((self.output_shape, self.input_shape))
        for i in range(self.output_shape):
            partial_mat[i, self.pin_dofs[i]] = 1.
        if coo:
            partial_mat = coo_matrix(partial_mat)
        return partial_mat

    def compute(self, inputs, outputs):
        for i, field in enumerate(self.opt_field):
            outputs[self.output_pin_name_list[i]] = \
                self.partial_mat*inputs[self.input_CP_name_list[i]] \
                - self.pin_val[i]

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

    filename_igs = "./geometry/init_Tbeam_geom_moved.igs"
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
    nonmatching_opt_ffd = NonMatchingOptFFD(splines, E, h_th, nu, 
                                            opt_field=opt_field, comm=worldcomm)
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
    nonmatching_opt_ffd.set_xi_diff_info(preprocessor)  

    nonmatching_opt_ffd.set_init_CPIGA(nonmatching_opt_ffd.cpiga2xi.cp_flat_global)


    prob = Problem()
    comp = CPSurfPinComp(nonmatching_opt=nonmatching_opt_ffd,
                         pin_surf_inds=[1])
    comp.init_paramters()

    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print('check_partials:')
    prob.check_partials(compact_print=True)