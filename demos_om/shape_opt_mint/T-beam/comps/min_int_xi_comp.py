from GOLDFISH.nonmatching_opt_ffd import *
from GOLDFISH.operations.compliance_exop import *

import openmdao.api as om
from openmdao.api import Problem

import sys
sys.path.append("../opers/")
from min_int_xi_exop import *

class MinIntXiComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('nonmatching_opt')
        self.options.declare('rho', default=1e3)
        self.options.declare('method', default='induced power')
        self.options.declare('input_xi_name', default='int_xi')
        self.options.declare('output_name', default='min_xi')

    def init_parameters(self):
        self.nonmatching_opt = self.options['nonmatching_opt']
        self.rho = self.options['rho']
        self.method = self.options['method']
        self.input_xi_name = self.options['input_xi_name']
        self.output_name = self.options['output_name']

        self.min_int_xi_exop = MinIntXiExop(self.nonmatching_opt,
                                            self.rho, self.method)
        self.init_xi = get_petsc_vec_array(self.nonmatching_opt.xi_nest)
        self.input_shape = self.min_int_xi_exop.xi_size

    def setup(self):
        self.add_input(self.input_xi_name, shape=self.input_shape,
                       val=self.init_xi)
        self.add_output(self.output_name)
        self.declare_partials(self.output_name, self.input_xi_name)

    def update_inputs(self, inputs):
        self.nonmatching_opt.update_xi(inputs[self.input_xi_name])

    def compute(self, inputs, outputs):
        self.update_inputs(inputs)
        xi_array = get_petsc_vec_array(self.nonmatching_opt.xi_nest)
        # xi_array = inputs[self.input_xi_name]
        outputs[self.output_name] = self.min_int_xi_exop.min_xi(xi_array)

    def compute_partials(self, inputs, partials):
        self.update_inputs(inputs)
        xi_array = get_petsc_vec_array(self.nonmatching_opt.xi_nest)
        # xi_array = inputs[self.input_xi_name]
        partials[self.output_name, self.input_xi_name] = \
            self.min_int_xi_exop.derivative(xi_array)

if __name__ == "__main__":
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
    ####################################################
    nonmatching_opt_ffd.set_xi_diff_info(preprocessor)  
    ####################################################

    prob = Problem()
    comp = MinIntXiComp(nonmatching_opt=nonmatching_opt_ffd)
    comp.init_parameters()
    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print('check_partials:')
    prob.check_partials(compact_print=True)