"""
The compressed T-beam geometry can be downloaded from:
    https://drive.google.com/file/d/1FJLAzM33YIFseXUI6t8U4uu4P0stCNXi/view?usp=sharing
"""
import numpy as np
import matplotlib.pyplot as plt
import openmdao.api as om
from igakit.cad import *
from igakit.io import VTK
from GOLDFISH.nonmatching_opt_om import *

class ShapeOptGroup(om.Group):

    def initialize(self):
        self.options.declare('nonmatching_opt_ffd')
        self.options.declare('cpffd_design_name_pre', default='CP_design_FFD')
        self.options.declare('cpffd_full_name_pre', default='CP_FFD')
        self.options.declare('cpsurf_fe_name_pre', default='CPS_FE')
        self.options.declare('cpsurf_iga_name_pre', default='CPS_IGA')
        self.options.declare('disp_name', default='displacements')
        self.options.declare('int_energy_name', default='int_E')
        self.options.declare('cpffd_pin_name_pre', default='CP_FFD_pin')
        self.options.declare('cpffd_regu_name_pre', default='CP_FFD_regu')

    def init_parameters(self):
        self.nonmatching_opt_ffd = self.options['nonmatching_opt_ffd']
        self.cpffd_design_name_pre = self.options['cpffd_design_name_pre']
        self.cpffd_full_name_pre = self.options['cpffd_full_name_pre']
        self.cpsurf_fe_name_pre = self.options['cpsurf_fe_name_pre']
        self.cpsurf_iga_name_pre = self.options['cpsurf_iga_name_pre']
        self.disp_name = self.options['disp_name']
        self.int_energy_name = self.options['int_energy_name']
        self.cpffd_pin_name_pre = self.options['cpffd_pin_name_pre']
        self.cpffd_regu_name_pre = self.options['cpffd_regu_name_pre']

        self.opt_field = self.nonmatching_opt_ffd.opt_field
        self.init_cpffd = self.nonmatching_opt_ffd.shopt_init_cpffd_design
        self.input_cpffd_shapes = [cpffd.size for cpffd in self.init_cpffd]
        self.design_var_lower = -1.e-3
        self.design_var_upper = 2

        self.cpffd_design_name_list = []
        self.cpffd_full_name_list = []
        self.cpsurf_fe_name_list = []
        self.cpsurf_iga_name_list = []
        self.cpffd_pin_name_list = []
        self.cpffd_regu_name_list = []
        for i, field in enumerate(self.opt_field):
            self.cpffd_design_name_list += [self.cpffd_design_name_pre+str(field)]
            self.cpffd_full_name_list += [self.cpffd_full_name_pre+str(field)]
            self.cpsurf_fe_name_list += [self.cpsurf_fe_name_pre+str(field)]
            self.cpsurf_iga_name_list += [self.cpsurf_iga_name_pre+str(field)]
            self.cpffd_pin_name_list += [self.cpffd_pin_name_pre+str(field)]
            self.cpffd_regu_name_list += [self.cpffd_regu_name_pre+str(field)]

        # Create components' names
        self.inputs_comp_name = 'inputs_comp'
        self.cpffd_design2full_comp_name = 'CPFFDDesign2Full_comp'
        self.cpffd2fe_comp_name = 'CPFFD2FE_comp'
        self.cpfe2iga_comp_name = 'CPFE2IGA_comp'
        self.disp_states_comp_name = 'disp_states_comp'
        self.int_energy_comp_name = 'internal_energy_comp'
        self.cpffd_pin_comp_name = 'CPFFD_pin_comp'
        self.cpffd_regu_comp_name = 'CPFFD_regu_comp'

    def setup(self):
        # Add inputs comp
        inputs_comp = om.IndepVarComp()
        for i, field in enumerate(self.opt_field):
            inputs_comp.add_output(self.cpffd_design_name_list[i],
                        shape=self.input_cpffd_shapes[i],
                        val=self.init_cpffd[i])
        self.add_subsystem(self.inputs_comp_name, inputs_comp)

        # Add CP FFD design to full comp
        self.cpffd_design2full_comp = CPFFDesign2FullComp(
                        nonmatching_opt_ffd=self.nonmatching_opt_ffd,
                        input_cpffd_design_name_pre=self.cpffd_design_name_pre,
                        output_cpffd_full_name_pre=self.cpffd_full_name_pre)
        self.cpffd_design2full_comp.init_parameters()
        self.add_subsystem(self.cpffd_design2full_comp_name, self.cpffd_design2full_comp)

        # Add FFD comp
        self.ffd2surf_comp = CPFFD2SurfComp(
                        nonmatching_opt_ffd=self.nonmatching_opt_ffd,
                        input_cpffd_name_pre=self.cpffd_full_name_pre,
                        output_cpsurf_name_pre=self.cpsurf_fe_name_pre)
        self.ffd2surf_comp.init_parameters()
        self.add_subsystem(self.cpffd2fe_comp_name, self.ffd2surf_comp)

        # Add CPFE2IGA comp
        self.cpfe2iga_comp = CPFE2IGAComp(
                        nonmatching_opt=self.nonmatching_opt_ffd,
                        input_cp_fe_name_pre=self.cpsurf_fe_name_pre,
                        output_cp_iga_name_pre=self.cpsurf_iga_name_pre)
        self.cpfe2iga_comp.init_parameters()
        self.add_subsystem(self.cpfe2iga_comp_name, self.cpfe2iga_comp)

        # Add disp_states_comp
        self.disp_states_comp = DispStatesComp(
                           nonmatching_opt=self.nonmatching_opt_ffd,
                           input_cp_iga_name_pre=self.cpsurf_iga_name_pre,
                           output_u_name=self.disp_name)
        self.disp_states_comp.init_parameters(save_files=save_files,
                                             nonlinear_solver_rtol=1e-3)
        self.add_subsystem(self.disp_states_comp_name, self.disp_states_comp)

        # Add internal energy comp (objective function)
        self.int_energy_comp = IntEnergyComp(
                          nonmatching_opt=self.nonmatching_opt_ffd,
                          input_cp_iga_name_pre=self.cpsurf_iga_name_pre,
                          input_u_name=self.disp_name,
                          output_wint_name=self.int_energy_name)
        self.int_energy_comp.init_parameters()
        self.add_subsystem(self.int_energy_comp_name, self.int_energy_comp)

        # Add CP FFD pin comp (linear constraint)
        self.cpffd_pin_comp = CPFFDPinComp(
                         nonmatching_opt_ffd=self.nonmatching_opt_ffd,
                         input_cpffd_design_name_pre=self.cpffd_design_name_pre,
                         output_cppin_name_pre=self.cpffd_pin_name_pre)
        self.cpffd_pin_comp.init_parameters()
        self.add_subsystem(self.cpffd_pin_comp_name, self.cpffd_pin_comp)
        self.cpffd_pin_cons_val = []
        for i, field in enumerate(self.opt_field):
            self.cpffd_pin_cons_val += [self.nonmatching_opt_ffd.shopt_cpffd_flat
                                [:,field][self.nonmatching_opt_ffd.shopt_cpffd_pin_dof[i]]]

        # Add CP FFD regu comp (linear constraint)
        self.cpffd_regu_comp = CPFFDReguComp(
                           nonmatching_opt_ffd=self.nonmatching_opt_ffd,
                           input_cpffd_design_name_pre=self.cpffd_design_name_pre,
                           output_cpregu_name_pre=self.cpffd_regu_name_pre)
        self.cpffd_regu_comp.init_parameters()
        self.add_subsystem(self.cpffd_regu_comp_name, self.cpffd_regu_comp)
        self.cpffd_regu_lower = [np.ones(self.cpffd_regu_comp.\
                                 output_shapes[i])*1.e-1
                                 for i in range(len(self.opt_field))]

        # Connect names between components
        for i, field in enumerate(self.opt_field):
            # For optimization components
            self.connect(self.inputs_comp_name+'.'
                         +self.cpffd_design_name_list[i],
                         self.cpffd_design2full_comp_name+'.'
                         +self.cpffd_design_name_list[i])

            self.connect(self.cpffd_design2full_comp_name+'.'
                         +self.cpffd_full_name_list[i],
                         self.cpffd2fe_comp_name+'.'
                         +self.cpffd_full_name_list[i])

            self.connect(self.cpffd2fe_comp_name+'.'
                         +self.cpsurf_fe_name_list[i],
                         self.cpfe2iga_comp_name+'.'
                         +self.cpsurf_fe_name_list[i])

            self.connect(self.cpfe2iga_comp_name+'.'
                         +self.cpsurf_iga_name_list[i],
                         self.disp_states_comp_name+'.'
                         +self.cpsurf_iga_name_list[i])

            self.connect(self.cpfe2iga_comp_name+'.'
                         +self.cpsurf_iga_name_list[i],
                         self.int_energy_comp_name+'.'
                         +self.cpsurf_iga_name_list[i])

            # For constraints
            self.connect(self.inputs_comp_name+'.'
                         +self.cpffd_design_name_list[i],
                         self.cpffd_pin_comp_name+'.'
                         +self.cpffd_design_name_list[i])
            self.connect(self.inputs_comp_name+'.'
                         +self.cpffd_design_name_list[i],
                         self.cpffd_regu_comp_name+'.'
                         +self.cpffd_design_name_list[i])

        self.connect(self.disp_states_comp_name+'.'+self.disp_name,
                     self.int_energy_comp_name+'.'+self.disp_name)

        # Add design variable, constraints and objective
        for i, field in enumerate(self.opt_field):
            self.add_design_var(self.inputs_comp_name+'.'
                                +self.cpffd_design_name_list[i],
                                lower=self.design_var_lower,
                                upper=self.design_var_upper)
            self.add_constraint(self.cpffd_pin_comp_name+'.'
                                +self.cpffd_pin_name_list[i],
                                equals=self.cpffd_pin_cons_val[i])
            self.add_constraint(self.cpffd_regu_comp_name+'.'
                                +self.cpffd_regu_name_list[i],
                                lower=self.cpffd_regu_lower[i])
        self.add_objective(self.int_energy_comp_name+'.'
                           +self.int_energy_name,
                           scaler=1e6)


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

save_files = False
optimizer = 'SLSQP'
# optimizer = 'SNOPT'
opt_field = [0,1]
ffd_block_num_el = [2,2,1]
p_ffd = 3
save_path = './'
folder_name = "results/"

filename_igs = "./geometry/init_tube_geom.igs"
igs_shapes = read_igs_file(filename_igs, as_compound=False)
occ_surf_list = [topoface2surface(face, BSpline=True) 
                 for face in igs_shapes]
occ_surf_data_list = [BSplineSurfaceData(surf) for surf in occ_surf_list]
num_surfs = len(occ_surf_list)
p = occ_surf_data_list[0].degree[0]

# Define material and geometric parameters
E = Constant(1.0e12)
nu = Constant(0.)
h_th = Constant(0.01)
penalty_coefficient = 1.0e3
pressure = Constant(1.)

fields0 = [[[0,2]],]
fields1 = [[None,[1,2]],]
spline_bc0 = SplineBC(directions=[0], sides=[[0],],
                     fields=fields0, n_layers=[[1],])
spline_bc1 = SplineBC(directions=[0], sides=[[1],],
                     fields=fields1, n_layers=[[None, 1],])
spline_bcs = [spline_bc0, spline_bc1]*2

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

if mpirank == 0:
    print("Creating splines...")
# Create tIGAr extracted spline instances
splines = []
for i in range(num_surfs):
        spline = OCCBSpline2tIGArSpline(preprocessor.BSpline_surfs[i], 
                                        spline_bc=spline_bcs[i], index=i)
        splines += [spline,]

# Create non-matching problem
nonmatching_opt_ffd = NonMatchingOptFFD(splines, E, h_th, nu, comm=worldcomm)
nonmatching_opt_ffd.create_mortar_meshes(preprocessor.mortar_nels)

nonmatching_opt_ffd.set_shopt_surf_inds_FFD(opt_field, list(range(num_surfs)))

if mpirank == 0:
    print("Setting up mortar meshes...")
nonmatching_opt_ffd.mortar_meshes_setup(preprocessor.mapping_list, 
                    preprocessor.intersections_para_coords, 
                    penalty_coefficient)
pressure = -Constant(1.)
source_terms = []
residuals = []
for s_ind in range(nonmatching_opt_ffd.num_splines):
    X = nonmatching_opt_ffd.splines[s_ind].F
    x = X + nonmatching_opt_ffd.splines[s_ind].rationalize(
            nonmatching_opt_ffd.spline_funcs[s_ind])
    z = nonmatching_opt_ffd.splines[s_ind].rationalize(
        nonmatching_opt_ffd.spline_test_funcs[s_ind])
    A0,A1,A2,deriv_A2,A,B = surfaceGeometry(
                            nonmatching_opt_ffd.splines[s_ind], X)
    a0,a1,a2,deriv_a2,a,b = surfaceGeometry(
                            nonmatching_opt_ffd.splines[s_ind], x)
    pressure_dir = sqrt(det(a)/det(A))*a2        
    normal_pressure = pressure*pressure_dir
    source_terms += [inner(normal_pressure, z)\
                     *nonmatching_opt_ffd.splines[s_ind].dx]
    residuals += [SVK_residual(nonmatching_opt_ffd.splines[s_ind], 
                  nonmatching_opt_ffd.spline_funcs[s_ind], 
                  nonmatching_opt_ffd.spline_test_funcs[s_ind], 
                  E, nu, h_th, source_terms[s_ind])]
nonmatching_opt_ffd.set_residuals(residuals)

# Create FFD block in igakit format
cp_ffd_lims = nonmatching_opt_ffd.cpsurf_des_lims
FFD_block = create_3D_block(ffd_block_num_el, p_ffd, cp_ffd_lims)

# Set FFD to non-matching optimization instance
nonmatching_opt_ffd.set_shopt_FFD(FFD_block.knots, FFD_block.control)
# Set constraint info
nonmatching_opt_ffd.set_shopt_align_CPFFD(align_dir=[[2],[2]])
nonmatching_opt_ffd.set_shopt_pin_CPFFD(pin_dir0=[0,0], pin_side0=[[0],[0]],
                                        pin_dir1=[2,2], pin_side1=[[0],[0]])
nonmatching_opt_ffd.set_shopt_pin_CPFFD(pin_dir0=[1,1], pin_side0=[[0],[0]],
                                        pin_dir1=[2,2], pin_side1=[[0],[0]])
nonmatching_opt_ffd.set_shopt_regu_CPFFD()
# Set up optimization
if save_files:
    nonmatching_opt_ffd.create_files(save_path=save_path, 
                                     folder_name=folder_name, 
                                     refine_mesh=True, ref_nel=64)
model = ShapeOptGroup(nonmatching_opt_ffd=nonmatching_opt_ffd)
model.init_parameters()
prob = om.Problem(model=model)

if optimizer.upper() == 'SNOPT':
    prob.driver = om.pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'
    prob.driver.opt_settings['Minor feasibility tolerance'] = 1e-6
    prob.driver.opt_settings['Major feasibility tolerance'] = 1e-6
    prob.driver.opt_settings['Major optimality tolerance'] = 1e-3
    prob.driver.opt_settings['Major iterations limit'] = 50000
    prob.driver.opt_settings['Summary file'] = './SNOPT_summary.out'
    prob.driver.opt_settings['Print file'] = './SNOPT_print.out'
    prob.driver.options['debug_print'] = ['objs']
    prob.driver.options['print_results'] = True
elif optimizer.upper() == 'SLSQP':
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1e-12
    prob.driver.options['disp'] = True
    prob.driver.options['debug_print'] = ['objs']
    prob.driver.options['maxiter'] = 50000
else:
    raise ValueError("Undefined optimizer: {}".format(optimizer))

# # Create a recorder variable
# recorder_name = './opt_data/recorder.sql'
# FFD_data_name = './opt_data/FFD_data.npz'

# prob.driver.recording_options['includes'] = ['*']
# prob.driver.recording_options['record_objectives'] = True
# prob.driver.recording_options['record_derivatives'] = True
# prob.driver.recording_options['record_constraints'] = True
# prob.driver.recording_options['record_desvars'] = True
# prob.driver.recording_options['record_inputs'] = True
# prob.driver.recording_options['record_outputs'] = True
# prob.driver.recording_options['record_residuals'] = True

# recorder = om.SqliteRecorder(recorder_name)
# prob.driver.add_recorder(recorder)

prob.setup()
# prob.set_solver_print(0)
prob.run_driver()

# major_iter_inds = model.disp_states_comp.func_eval_major_ind
# np.savez(FFD_data_name, opt_field=opt_field,
#                         major_iter_ind=major_iter_inds,
#                         ffd_control=np.array(FFD_block.control,dtype=object),
#                         ffd_knots=np.array(FFD_block.knots,dtype=object),
#                         QoI=np.array([max_F0, max_F1]))

# Plot optimized cross sectional view 
num_eval_pts = 101
xi0_pts = np.linspace(0,1,num_eval_pts)
xi1_pts = 0.1
s0_x_pts = np.zeros(num_eval_pts)
s0_y_pts = np.zeros(num_eval_pts)
s1_x_pts = np.zeros(num_eval_pts)
s1_y_pts = np.zeros(num_eval_pts)
for i in range(num_eval_pts):
    s0_x_pts[i] = nonmatching_opt_ffd.splines[2].cpFuncs[0]([xi0_pts[i], xi1_pts])
    s0_y_pts[i] = nonmatching_opt_ffd.splines[2].cpFuncs[1]([xi0_pts[i], xi1_pts])
    s1_x_pts[i] = nonmatching_opt_ffd.splines[3].cpFuncs[0]([xi0_pts[i], xi1_pts])
    s1_y_pts[i] = nonmatching_opt_ffd.splines[3].cpFuncs[1]([xi0_pts[i], xi1_pts])

y_ana_pts = np.zeros(num_eval_pts)
for i in range(num_eval_pts):
    y_ana_pts[i] = np.sqrt(1-xi0_pts[i]**2)


plt.figure()
plt.plot(s0_x_pts, s0_y_pts, linewidth=3)
plt.plot(s1_x_pts, s1_y_pts, linewidth=3)
plt.plot(xi0_pts, y_ana_pts, '--', color='gray', linewidth=2)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.show()