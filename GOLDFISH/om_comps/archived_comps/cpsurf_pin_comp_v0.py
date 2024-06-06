import openmdao.api as om
from openmdao.api import Problem
from scipy.sparse import coo_matrix
from PENGoLINS.occ_preprocessing import *
from GOLDFISH.utils.opt_utils import *
from GOLDFISH.utils.bsp_utils import *

class CPSurfPinComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('nonmatching_opt')
        self.options.declare('input_cp_iga_name_pre', default='CP_IGA')
        self.options.declare('output_cp_pin_name_pre', default='CPS_pin')

    def init_parameters(self):
        self.nonmatching_opt = self.options['nonmatching_opt']
        # self.pin_surf_inds = self.options['pin_surf_inds']
        self.input_cp_iga_name_pre = self.options['input_cp_iga_name_pre']
        self.output_cp_pin_name_pre = self.options['output_cp_pin_name_pre']

        self.opt_field = self.nonmatching_opt.opt_field
        self.init_cp_iga = self.nonmatching_opt.init_cp_iga_design

        self.input_shapes = []
        self.output_shapes = []
        for field_ind, field in enumerate(self.opt_field):
            self.input_shapes += [len(self.nonmatching_opt.cpdes_iga_dofs[field_ind])]
            self.output_shapes += [len(self.nonmatching_opt.pin_cp_dofs[field_ind])]

        self.partial_mat = self.nonmatching_opt.shopt_dcppindcpsurf
        self.pin_vals = self.nonmatching_opt.pin_vals

        self.input_CP_name_list = []
        self.output_pin_name_list = []
        for i, field in enumerate(self.opt_field):
            self.input_CP_name_list += [self.input_cp_iga_name_pre+str(field),]
            self.output_pin_name_list += [self.output_cp_pin_name_pre+str(field)]


    def setup(self):
        for i, field in enumerate(self.opt_field):
            self.add_input(self.input_CP_name_list[i], 
                           shape=self.input_shapes[i],
                           val=self.init_cp_iga[field])
            self.add_output(self.output_pin_name_list[i], 
                            shape=self.output_shapes[i])
            self.declare_partials(self.output_pin_name_list[i],
                                  self.input_CP_name_list[i],
                                  val=self.partial_mat[i].data,
                                  rows=self.partial_mat[i].row,
                                  cols=self.partial_mat[i].col)

    def compute(self, inputs, outputs):
        for i, field in enumerate(self.opt_field):
            outputs[self.output_pin_name_list[i]] = \
                self.partial_mat[i]*inputs[self.input_CP_name_list[i]] \
                - self.pin_vals[i]

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
    nonmatching_opt_ffd = NonMatchingOpt(splines, E, h_th, nu, comm=worldcomm)
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

    # nonmatching_opt_ffd.set_init_CPIGA(nonmatching_opt_ffd.cpiga2xi.cp_flat_global.T)

    opt_field = [0,1]
    opt_surf_inds0 = list(range(nonmatching_opt_ffd.num_splines))
    opt_surf_inds1 = [1]
    opt_surf_inds = [opt_surf_inds0, opt_surf_inds1]

    nonmatching_opt_ffd.set_shape_opt(opt_field=opt_field, opt_surf_inds=opt_surf_inds)
    nonmatching_opt_ffd.get_init_CPIGA()
    nonmatching_opt_ffd.set_shopt_align_CP(align_surf_inds=[[0,1], [1]],
                                       align_dir=[[0,1], [0]])
    nonmatching_opt_ffd.set_shopt_pin_CP(pin_surf_inds=[[0,1], [1]],
                                     pin_dir=[[1,0], [1]],
                                     pin_side=[[0,0], [0]])


    prob = Problem()
    comp = CPSurfPinComp(nonmatching_opt=nonmatching_opt_ffd)
    comp.init_parameters()

    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print('check_partials:')
    prob.check_partials(compact_print=True)