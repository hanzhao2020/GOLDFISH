from GOLDFISH.nonmatching_opt_ffd import *
from GOLDFISH.operations.max_vmstress_exop import *

import csdl
from csdl import Model, CustomExplicitOperation
from csdl_om import Simulator

class VMStressModel(Model):
    def initialize(self):
        self.parameters.declare('nonmatching_opt')
        self.parameters.declare('surf', default='top')
        self.parameters.declare('linearize_stress', default=False)
        self.parameters.declare('input_u_name', default='displacements')
        self.parameters.declare('input_cp_iga_name_pre', default='CP_IGA')
        self.parameters.declare('input_h_th_name', default='thickness')
        self.parameters.declare('output_vM_name', default='vM_stress')

    def init_parameters(self):
        self.nonmatching_opt = self.parameters['nonmatching_opt']
        self.surf = self.parameters['surf']
        self.linearize_stress = self.parameters['linearize_stress']
        self.input_cp_iga_name_pre = self.parameters['input_cp_iga_name_pre']
        self.input_h_th_name = self.parameters['input_h_th_name']
        self.input_u_name = self.parameters['input_u_name']
        self.output_vM_name = self.parameters['output_vM_name']

        self.op = MaxvMStressOperation(
                  nonmatching_opt=self.nonmatching_opt,
                  surf=self.surf,
                  linearize_stress=self.linearize_stress,
                  input_cp_iga_name_pre=self.input_cp_iga_name_pre,
                  input_h_th_name=self.input_h_th_name,
                  input_u_name=self.input_u_name,
                  output_vM_name=self.output_vM_name)
        self.op.init_parameters()

    def define(self):
        input_list = []
        disp = self.declare_variable(self.op.input_u_name,
               shape=(self.op.input_u_shape,),
               val=self.op.init_disp_array)
        input_list += [disp]
        if self.op.opt_shape:
            cp_iga_list = [None for i in range(len(self.op.opt_field))]
            for i, field in enumerate(self.op.opt_field):
                cp_iga_list[i] = self.declare_variable(
                                 self.op.input_cp_iga_name_list[i],
                                 shape=(self.op.input_cpiga_shape),
                                 val=self.op.init_cp_iga[:,field])
            input_list += cp_iga_list
        if self.op.opt_thickness:
            h_th = self.declare_variable(self.op.input_h_th_name,
                   shape=(self.op.input_h_th_shape),
                   val=self.op.init_h_th)
            input_list += [h_th]

        vm_stress = csdl.custom(*input_list, op=self.op)
        self.register_output(self.op.output_vM_name, vm_stress)


class MaxvMStressOperation(CustomExplicitOperation):
    def initialize(self):
        self.parameters.declare('nonmatching_opt')
        self.parameters.declare('surf', default='top')
        self.parameters.declare('linearize_stress', default=False)
        self.parameters.declare('input_u_name', default='displacements')
        self.parameters.declare('input_cp_iga_name_pre', default='CP_IGA')
        self.parameters.declare('input_h_th_name', default='thickness')
        self.parameters.declare('output_vM_name', default='vM_stress')

    def init_parameters(self):
        self.nonmatching_opt = self.parameters['nonmatching_opt']
        self.surf = self.parameters['surf']
        self.linearize_stress = self.parameters['linearize_stress']
        self.input_cp_iga_name_pre = self.parameters['input_cp_iga_name_pre']
        self.input_h_th_name = self.parameters['input_h_th_name']
        self.input_u_name = self.parameters['input_u_name']
        self.output_vM_name = self.parameters['output_vM_name']

        self.vMstress_exop = MaxvMStressExOperation(self.nonmatching_opt,
                             1., 1., 1., self.surf,
                             'pnorm', self.linearize_stress)

        self.vMstress_symexp = self.vMstress_exop.vMstress
        self.vMstress_forms = [Form(vMstress) for vMstress 
                               in self.vMstress_symexp]

        self.input_u_shape = self.nonmatching_opt.vec_iga_dof
        # self.init_disp_array = get_petsc_vec_array(
        #                        self.nonmatching_opt.u_iga_nest)
        # self.init_disp_array = u_iga_array
        self.init_disp_array = np.ones(self.nonmatching_opt.vec_iga_dof)

        self.dvMdu_symexp = []
        self.dvMdu_forms = []
        for s_ind in range(self.nonmatching_opt.num_splines):
            self.dvMdu_symexp += [derivative(self.vMstress_symexp[s_ind],
                                  self.nonmatching_opt.spline_funcs[s_ind]),]
            self.dvMdu_forms += [Form(self.dvMdu_symexp[s_ind])]

        self.opt_field = self.nonmatching_opt.opt_field
        self.opt_shape = self.nonmatching_opt.opt_shape
        self.opt_thickness = self.nonmatching_opt.opt_thickness
        self.var_thickness = self.nonmatching_opt.var_thickness
        if self.opt_shape:
            self.input_cpiga_shape = self.nonmatching_opt.vec_scalar_iga_dof
            self.init_cp_iga = self.nonmatching_opt.get_init_CPIGA()
            self.input_cp_iga_name_list = []
            for i, field in enumerate(self.opt_field):
                self.input_cp_iga_name_list += \
                    [self.input_cp_iga_name_pre+str(field)]

            self.dvMdcp_symexp = [[] for i in range(len(self.opt_field))]
            self.dvMdcp_forms = [[] for i in range(len(self.opt_field))]
            for i, field in enumerate(self.opt_field):
                self.dvMdcp_symexp[i] += [
                    derivative(self.vMstress_symexp[s_ind],
                    self.nonmatching_opt.splines[s_ind].cpFuncs[field]),]
                self.dvMdcp_forms[i] += [Form(self.dvMdcp_symexp[i][s_ind])]

        if self.opt_thickness:
            if self.var_thickness:
                self.input_h_th_shape = self.nonmatching_opt.vec_scalar_iga_dof
                self.init_h_th = np.ones(self.nonmatching_opt.vec_scalar_iga_dof)*0.1
            else:
                self.input_h_th_shape = self.nonmatching_opt.h_th_dof
                self.init_h_th = self.nonmatching_opt.init_h_th

            self.dvMdh_th_symexp = []
            self.dvMdh_th_forms = []
            for s_ind in range(self.nonmatching_opt.num_splines):
                self.dvMdh_th_symexp += [derivative(self.vMstress_symexp[s_ind],
                                    self.nonmatching_opt.h_th[s_ind]),]
                self.dvMdh_th_forms += [Form(self.dvMdh_th_symexp[s_ind])]

    def define(self):
        self.add_output(self.output_vM_name)
        self.add_input(self.input_u_name, shape=self.input_u_shape,)
                       # val=self.init_disp_array)
        self.declare_derivatives(self.output_vM_name,
                                 self.input_u_name)
        if self.opt_shape:
            for i, field in enumerate(self.opt_field):
                self.add_input(self.input_cp_iga_name_list[i],
                               shape=self.input_cpiga_shape,)
                               # val=self.init_cp_iga[:,field])
                self.declare_derivatives(self.output_vM_name,
                                         self.input_cp_iga_name_list[i])
        if self.opt_thickness:
            self.add_input(self.input_h_th_name, shape=self.input_h_th_shape,)
                           # val=self.init_h_th)
            self.declare_derivatives(self.output_vM_name, 
                                     self.input_h_th_name)

    def update_inputs(self, inputs):
        if self.opt_shape:
            for i, field in enumerate(self.opt_field):
                self.nonmatching_opt.update_CPIGA(
                    inputs[self.input_cp_iga_name_list[i]], field)
        if self.opt_thickness:
            if self.var_thickness:
                self.nonmatching_opt.update_h_th_IGA(
                                     inputs[self.input_h_th_name])
            else:
                self.nonmatching_opt.update_h_th(inputs[self.input_h_th_name])
        self.nonmatching_opt.update_uIGA(inputs[self.input_u_name])

    def compute(self, inputs, outputs):
        self.update_inputs(inputs)
        outputs[self.output_vM_name] = self.max_vm_exop.\
                                           vM_stress_global()

    def compute_derivatives(self, inputs, derivatives):
        self.update_inputs(inputs)
        dmax_vmdu_IGA = self.max_vm_exop.dmax_vMduIGA_global(
                        array=True, apply_bcs=True)
        derivatives[self.output_vM_name, self.input_u_name] = dmax_vmdu_IGA

        if self.opt_shape:
            for i, field in enumerate(self.opt_field):
                dmax_vmdcp_IGA = self.max_vm_exop.dmax_vMdCPIGA_global(
                                 field, array=True)
                derivatives[self.output_vM_name, 
                         self.input_cp_iga_name_list[i]] = dmax_vmdcp_IGA
        if self.opt_thickness:
            dmax_vmdh_th = self.max_vm_exop.dmax_vMdh_th_global(array=True)
            derivatives[self.output_vM_name, self.input_h_th_name] = \
                dmax_vmdh_th


if __name__ == "__main__":
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

    filename_igs = "./geometry/init_Tbeam_geom.igs"
    igs_shapes = read_igs_file(filename_igs, as_compound=False)
    occ_surf_list = [topoface2surface(face, BSpline=True) 
                     for face in igs_shapes]
    occ_surf_data_list = [BSplineSurfaceData(surf) for surf in occ_surf_list]
    num_surfs = len(occ_surf_list)
    p = occ_surf_data_list[0].degree[0]

    # Define material and geometric parameters
    E = Constant(1.0e12)
    nu = Constant(0.)
    # h_th = Constant(0.1)
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

    # Display B-spline surfaces and intersections using 
    # PythonOCC build-in 3D viewer.
    # display, start_display, add_menu, add_function_to_menu = init_display()
    # preprocessor.display_surfaces(display, save_fig=False)
    # preprocessor.display_intersections(display, save_fig=False)

    if mpirank == 0:
        print("Creating splines...")
    # Create tIGAr extracted spline instances
    splines = []
    for i in range(num_surfs):
            spline = OCCBSpline2tIGArSpline(preprocessor.BSpline_surfs[i], 
                                            spline_bc=spline_bcs[i], index=i)
            splines += [spline,]

    h_th = []
    expression = Expression("0.1*sin(pi*x[0])*sin(pi*x[1])+0.05", degree=2)
    for i in range(num_surfs):
        h_th += [Function(splines[i].V_control)]
        h_th[i].interpolate(Constant(0.1))
        # temp_vec = project(expression, splines[i].V_linear)
        # h_th[i].assign(temp_vec)

    # Create non-matching problem
    nonmatching_opt = NonMatchingOptFFD(splines, E, h_th, nu, opt_shape=True, 
                                        opt_thickness=True, var_thickness=True, comm=worldcomm)
    nonmatching_opt.create_mortar_meshes(preprocessor.mortar_nels)

    if mpirank == 0:
        print("Setting up mortar meshes...")
    nonmatching_opt.mortar_meshes_setup(preprocessor.mapping_list, 
                    preprocessor.intersections_para_coords, 
                    penalty_coefficient)
    pressure = -Constant(10.)
    f = as_vector([Constant(0.), Constant(0.), pressure])
    source_terms = []
    residuals = []
    for s_ind in range(nonmatching_opt.num_splines):
        z = nonmatching_opt.splines[s_ind].rationalize(
            nonmatching_opt.spline_test_funcs[s_ind])
        source_terms += [inner(f, z)*nonmatching_opt.splines[s_ind].dx]
        residuals += [SVK_residual(nonmatching_opt.splines[s_ind], 
                      nonmatching_opt.spline_funcs[s_ind], 
                      nonmatching_opt.spline_test_funcs[s_ind], 
                      E, nu, h_th[s_ind], source_terms[s_ind])]
    nonmatching_opt.set_residuals(residuals)

    if mpirank == 0:
        print("Solving linear non-matching problem...")
    # _, u_iga = nonmatching_opt.solve_nonlinear_nonmatching_problem(solver="direct", iga_dofs=True)
    # u_iga_array = get_petsc_vec_array(u_iga)

    nonmatching_opt.solve_linear_nonmatching_problem(solver="direct")
    u_iga_array = get_petsc_vec_array(nonmatching_opt.u)

    m = MaxvMStressModel(nonmatching_opt=nonmatching_opt,
                          rho=1e2, alpha=0.01, m=1e4, 
                          # method='induced power', 
                          method='pnorm', 
                          linearize_stress=True)
    m.init_parameters()
    sim = Simulator(m)
    sim.run()
    print('check_partials:')
    sim.check_partials(compact_print=True)

    vM1 = splines[1].projectScalarOntoLinears(m.op.vMstress[1])
    vM1v = v2p(vM1.vector())
    max_val = vM1v.max()[1]