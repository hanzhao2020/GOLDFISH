"""
Initial geometry can be downloaded from the following link:
https://drive.google.com/file/d/1fRaho_xzmChlgLdrMM9CQ7WTqr9_DItt/view?usp=share_link
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
        self.options.declare('cpffd_name_pre', default='CP_FFD')
        self.options.declare('cpsurf_fe_name_pre', default='CPS_FE')
        self.options.declare('cpsurf_iga_name_pre', default='CPS_IGA')
        self.options.declare('disp_name', default='displacements')
        # self.options.declare('int_energy_name', default='int_E')
        self.options.declare('cpffd_align_name_pre', default='CP_FFD_align')
        self.options.declare('cpffd_pin_name_pre', default='CP_FFD_pin')
        self.options.declare('cpffd_regu_name_pre', default='CP_FFD_regu')
        self.options.declare('max_vM_name', default='max_vM_stress')
        self.options.declare('volume_name', default='volume')

    def init_parameters(self, design_var_lims):
        self.nonmatching_opt_ffd = self.options['nonmatching_opt_ffd']
        self.cpffd_name_pre = self.options['cpffd_name_pre']
        self.cpsurf_fe_name_pre = self.options['cpsurf_fe_name_pre']
        self.cpsurf_iga_name_pre = self.options['cpsurf_iga_name_pre']
        self.disp_name = self.options['disp_name']
        # self.int_energy_name = self.options['int_energy_name']
        self.cpffd_align_name_pre = self.options['cpffd_align_name_pre']
        self.cpffd_pin_name_pre = self.options['cpffd_pin_name_pre']
        self.cpffd_regu_name_pre = self.options['cpffd_regu_name_pre']
        self.max_vM_name = self.options['max_vM_name']
        self.volume_name = self.options['volume_name']
        self.design_var_lims = design_var_lims

        self.opt_field = self.nonmatching_opt_ffd.opt_field

        self.cpffd_name_list = []
        self.cpsurf_fe_name_list = []
        self.cpsurf_iga_name_list = []
        self.cpffd_pin_name_list = []
        self.cpffd_regu_name_list = []
        self.cpffd_align_name_list = []
        for i, field in enumerate(self.opt_field):
            self.cpffd_name_list += [self.cpffd_name_pre+str(field)]
            self.cpsurf_fe_name_list += [self.cpsurf_fe_name_pre+str(field)]
            self.cpsurf_iga_name_list += [self.cpsurf_iga_name_pre+str(field)]
            self.cpffd_align_name_list += [self.cpffd_align_name_pre+str(field)]
            self.cpffd_pin_name_list += [self.cpffd_pin_name_pre+str(field)]
            self.cpffd_regu_name_list += [self.cpffd_regu_name_pre+str(field)]

        self.inputs_comp_name = 'inputs_comp'
        self.ffd2surf_comp_name = 'FFD2Surf_comp'
        self.cpfe2iga_comp_name = 'CPFE2IGA_comp'
        self.disp_states_comp_name = 'disp_states_comp'
        self.cpffd_pin_comp_name = 'CPFFD_pin_comp'
        self.cpffd_regu_comp_name = 'CPFFD_regu_comp'
        self.max_vM_comp_name = 'max_vM_comp'
        self.volume_comp_name = 'volume_comp'


    def setup(self):
        # Add inputs comp
        inputs_comp = om.IndepVarComp()
        for i, field in enumerate(self.opt_field):
            inputs_comp.add_output(self.cpffd_name_list[i],
                        shape=self.nonmatching_opt_ffd.cpffd_size,
                        val=self.nonmatching_opt_ffd.cpffd_flat[:,field])
        self.add_subsystem(self.inputs_comp_name, inputs_comp)

        # Add FFD comp
        self.ffd2surf_comp = FFD2SurfComp(
                        nonmatching_opt_ffd=self.nonmatching_opt_ffd,
                        input_cpffd_name_pre=self.cpffd_name_pre,
                        output_cpsurf_name_pre=self.cpsurf_fe_name_pre)
        self.ffd2surf_comp.init_parameters()
        self.add_subsystem(self.ffd2surf_comp_name, self.ffd2surf_comp)

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
        self.disp_states_comp.init_parameters(save_files=True,
                                             nonlinear_solver_rtol=1e-3)
        self.add_subsystem(self.disp_states_comp_name, self.disp_states_comp)

        # Add CP FFD pin comp (linear constraint)
        self.cpffd_pin_comp = CPFFDPinComp(
                         nonmatching_opt_ffd=self.nonmatching_opt_ffd,
                         input_cpffd_name_pre=self.cpffd_name_pre,
                         output_cppin_name_pre=self.cpffd_pin_name_pre)
        self.cpffd_pin_comp.init_parameters()
        self.add_subsystem(self.cpffd_pin_comp_name, self.cpffd_pin_comp)
        self.cpffd_pin_cons_val = []
        for i, field in enumerate(self.opt_field):
            self.cpffd_pin_cons_val += [self.nonmatching_opt_ffd.cpffd_flat
                                [:,field][self.nonmatching_opt_ffd.pin_dof]]

        # Add CP FFD regu comp (linear constraint)
        self.cpffd_regu_comp = CPFFDReguComp(
                           nonmatching_opt_ffd=self.nonmatching_opt_ffd,
                           input_cpffd_name_pre=self.cpffd_name_pre,
                           output_cpregu_name_pre=self.cpffd_regu_name_pre)
        self.cpffd_regu_comp.init_parameters()
        self.add_subsystem(self.cpffd_regu_comp_name, self.cpffd_regu_comp)
        self.cpffd_regu_lower = [np.ones(self.cpffd_regu_comp.\
                                 output_shapes[i])*2.e-2
                                 for i in range(len(self.opt_field))]

        # Add max von Mises stress comp (constraint)
        rho = 1.5e2
        upper_vM = 1e5
        self.max_vM_comp = MaxvMStressComp(
                               nonmatching_opt=self.nonmatching_opt_ffd,
                               rho=rho, alpha=None, m=upper_vM, method='pnorm', 
                               linearize_stress=False, 
                               input_u_name=self.disp_name,
                               input_cp_iga_name_pre=self.cpsurf_iga_name_pre,
                               output_max_vM_name=self.max_vM_name)
        self.max_vM_comp.init_parameters()
        self.add_subsystem(self.max_vM_comp_name, self.max_vM_comp)
        #########################################################

        # Add volume comp (objective)
        self.volume_comp = VolumeComp(
                           nonmatching_opt=self.nonmatching_opt_ffd,
                           input_cp_iga_name_pre=self.cpsurf_iga_name_pre,
                           output_vol_name=self.volume_name)
        self.volume_comp.init_parameters()
        self.add_subsystem(self.volume_comp_name, self.volume_comp)
        self.vol_val = 0
        for s_ind in range(self.nonmatching_opt_ffd.num_splines):
            self.vol_val += assemble(self.nonmatching_opt_ffd.h_th[s_ind]
                            *self.nonmatching_opt_ffd.splines[s_ind].dx)

        # Connect names between components
        for i, field in enumerate(self.opt_field):
            # For optimization components
            self.connect(self.inputs_comp_name+'.'
                         +self.cpffd_name_list[i],
                         self.ffd2surf_comp_name+'.'
                         +self.cpffd_name_list[i])
            self.connect(self.ffd2surf_comp_name+'.'
                         +self.cpsurf_fe_name_list[i],
                         self.cpfe2iga_comp_name+'.'
                         +self.cpsurf_fe_name_list[i])
            self.connect(self.cpfe2iga_comp_name+'.'
                         +self.cpsurf_iga_name_list[i],
                         self.disp_states_comp_name+'.'
                         +self.cpsurf_iga_name_list[i])
            self.connect(self.cpfe2iga_comp_name+'.'
                         +self.cpsurf_iga_name_list[i],
                         self.volume_comp_name+'.'
                         +self.cpsurf_iga_name_list[i])

            # For constraints
            self.connect(self.inputs_comp_name+'.'
                         +self.cpffd_name_list[i],
                         self.cpffd_pin_comp_name+'.'
                         +self.cpffd_name_list[i])
            self.connect(self.inputs_comp_name+'.'
                         +self.cpffd_name_list[i],
                         self.cpffd_regu_comp_name+'.'
                         +self.cpffd_name_list[i])

            self.connect(self.cpfe2iga_comp_name+'.'
                         +self.cpsurf_iga_name_list[i],
                         self.max_vM_comp_name+'.'
                         +self.cpsurf_iga_name_list[i])

        self.connect(self.disp_states_comp_name+'.'+self.disp_name,
                     self.max_vM_comp_name+'.'+self.disp_name)

        # Add design variable, constraints and objective
        for i, field in enumerate(self.opt_field):
            self.add_design_var(self.inputs_comp_name+'.'
                                +self.cpffd_name_list[i],
                                lower=self.design_var_lims[i][0],
                                upper=self.design_var_lims[i][1])
            self.add_constraint(self.cpffd_pin_comp_name+'.'
                                +self.cpffd_pin_name_list[i],
                                equals=self.cpffd_pin_cons_val[i])
            self.add_constraint(self.cpffd_regu_comp_name+'.'
                                +self.cpffd_regu_name_list[i],
                                lower=self.cpffd_regu_lower[i])
        self.add_constraint(self.max_vM_comp_name+'.'
                            +self.max_vM_name,
                            upper=upper_vM,
                            scaler=1e-7)

        self.add_objective(self.volume_comp_name+'.'
                           +self.volume_name,
                           scaler=1.e1)

def clampedBC(spline_generator, side=0, direction=0):
    """
    Apply clamped boundary condition to spline.
    """
    for field in [0,1,2]:
        if field in [1]:
            n_layers = 1
        else:
            n_layers = 2
        scalar_spline = spline_generator.getScalarSpline(field)
        side_dofs = scalar_spline.getSideDofs(direction, side, 
                                              nLayers=n_layers)
        spline_generator.addZeroDofs(field, side_dofs)

def OCCBSpline2tIGArSpline(surface, num_field=3, quad_deg_const=4, 
                           setBCs=None, side=0, direction=0, index=0):
    """
    Generate ExtractedBSpline from OCC B-spline surface.
    """
    quad_deg = surface.UDegree()*quad_deg_const
    # DIR = SAVE_PATH+"spline_data/extraction_"+str(index)+"_init"
    # spline = ExtractedSpline(DIR, quad_deg)
    spline_mesh = NURBSControlMesh4OCC(surface, useRect=False)
    spline_generator = EqualOrderSpline(worldcomm, num_field, spline_mesh)
    if setBCs is not None:
        setBCs(spline_generator, side, direction)
    # spline_generator.writeExtraction(DIR)
    spline = ExtractedSpline(spline_generator, quad_deg)
    return spline

test_ind = 2
# optimizer = 'SLSQP'
optimizer = 'SNOPT'
opt_field = [2]
ffd_block_num_el = [3,6,1]
# save_path = './'
save_path = '/home/han/Documents/test_results/'
# folder_name = "results/"
folder_name = "results"+str(test_ind)+"/"

# Define parameters
# Scale down the geometry using ``geom_scale``to make the length 
# of the wing in the span-wise direction is around 11 m 
# (original length 4.54e5).
geom_scale = 2.54e-5  # Convert current length unit to m
E = Constant(68e9)  # Young's modulus, Pa
nu = Constant(0.35)  # Poisson's ratio
h_th = Constant(3.0e-3)  # Thickness of surfaces, m

p = 3  # spline order
penalty_coefficient = 1.0e3

print("Importing geometry...")
filename_igs = "./geometry/pegasus_wing.iges"
igs_shapes = read_igs_file(filename_igs, as_compound=False)
pegasus_surfaces = [topoface2surface(face, BSpline=True) 
                  for face in igs_shapes]

# Upper skins: 0, 4, 8, 12, ..., 64
# Lower skins: 1, 5, 9, 13, ..., 65
# Front spars: 2, 6, 19, 14, ..., 66
# Rear spars: 3, 7, 11, 15, ..., 67
# Ribs: 72, 73, 74, ..., 85 (18 ribs)
num_secs = 2
wing_indices = list(range(0, len(pegasus_surfaces))) # All surfaces
# wing_indices = list(range(0,8)) + list(range(72,74)) # First two sections
# wing_indices = list(range(0,12)) + list(range(72,75)) # First three sections
# wing_indices = list(range(0,num_secs*4)) + list(range(72,72+num_secs))
wing_surfaces = [pegasus_surfaces[i] for i in wing_indices]
num_surfs = len(wing_surfaces)
if mpirank == 0:
    print("Number of surfaces:", num_surfs)

# # Save original surfaces
# from PENGoLINS.igakit_utils import *
# from igakit.io import VTK
# ik_surfs = []
# for i in range(num_surfs):
#     ik_surf = BSpline_surface2ikNURBS(wing_surfaces[i])
#     ik_surfs += [ik_surf]
#     VTK().write("./geometry/wing_init_surf_"+str(i)+".vtk", ik_surf)
# exit()

num_pts_eval = [6]*num_surfs
ref_level_list = [1]*num_surfs

u_insert_list = [4]*num_surfs
v_insert_list = [2]*num_surfs

u_num_insert = []
v_num_insert = []
for i in range(len(u_insert_list)):
    u_num_insert += [ref_level_list[i]*u_insert_list[i]]
    v_num_insert += [ref_level_list[i]*v_insert_list[i]]

# Geometry preprocessing and surface-surface intersections computation
preprocessor = OCCPreprocessing(wing_surfaces, reparametrize=True, 
                                refine=True)
preprocessor.reparametrize_BSpline_surfaces(num_pts_eval, num_pts_eval,
                                            geom_scale=geom_scale,
                                            remove_dense_knots=True,
                                            rtol=1e-4)
preprocessor.refine_BSpline_surfaces(p, p, u_num_insert, v_num_insert, 
                                     correct_element_shape=True)

if mpirank == 0:
    print("Computing intersections...")
int_data_filename = "pegasus_wing_int_data"+str(test_ind)+".npz"
if os.path.isfile(int_data_filename):
    preprocessor.load_intersections_data(int_data_filename)
else:
    preprocessor.compute_intersections(rtol=1e-6, mortar_refine=2, 
                                       edge_rel_ratio=1e-3)
    preprocessor.save_intersections_data(int_data_filename)

if mpirank == 0:
    print("Total DoFs:", preprocessor.total_DoFs)
    print("Number of intersections:", preprocessor.num_intersections_all)

# # Display B-spline surfaces and intersections using 
# # PythonOCC build-in 3D viewer.
# display, start_display, add_menu, add_function_to_menu = init_display()
# preprocessor.display_surfaces(display, save_fig=False)
# preprocessor.display_intersections(display, save_fig=False)

if mpirank == 0:
    print("Creating splines...")
# Create tIGAr extracted spline instances
splines = []
for i in range(num_surfs):
    if i in [0, 1, 2, 3]:
        # Apply clamped BC to surfaces near root
        spline = OCCBSpline2tIGArSpline(
                 preprocessor.BSpline_surfs_refine[i], 
                 setBCs=clampedBC, side=0, direction=1, index=i)
        splines += [spline,]
    else:
        spline = OCCBSpline2tIGArSpline(
                 preprocessor.BSpline_surfs_refine[i], index=i)
        splines += [spline,]

h_th = Constant(3.0e-3)

# Create non-matching problem
nonmatching_opt_ffd = NonMatchingOptFFD(splines, E, h_th, nu, opt_shape=True,
                                 opt_field=opt_field, 
                                 opt_thickness=False, comm=worldcomm)
nonmatching_opt_ffd.create_mortar_meshes(preprocessor.mortar_nels)

if mpirank == 0:
    print("Setting up mortar meshes...")

nonmatching_opt_ffd.mortar_meshes_setup(preprocessor.mapping_list, 
                                    preprocessor.intersections_para_coords, 
                                    penalty_coefficient)

# Define magnitude of load
pressure = Constant(-10) # The load should be in the unit of N/m^2
f0 = as_vector([Constant(0.), Constant(0.), Constant(0.)])

loads_inds = []
for i in range(72):
    loads_inds += [i*4]
    loads_inds += [i*4+1]

source_terms = []
residuals = []
for s_ind in range(nonmatching_opt_ffd.num_splines):
    if s_ind in loads_inds:
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
    else:
        z = nonmatching_opt_ffd.splines[s_ind].rationalize(
            nonmatching_opt_ffd.spline_test_funcs[s_ind])
        source_terms += [inner(f0, z)*nonmatching_opt_ffd.splines[s_ind].dx]
        residuals += [SVK_residual(nonmatching_opt_ffd.splines[s_ind], 
                      nonmatching_opt_ffd.spline_funcs[s_ind], 
                      nonmatching_opt_ffd.spline_test_funcs[s_ind], 
                      E, nu, h_th, source_terms[s_ind])]
nonmatching_opt_ffd.set_residuals(residuals)

# Create FFD block in igakit format
cp_ffd_lims = nonmatching_opt_ffd.cpsurf_lims
for field in [2]:
    cp_range = cp_ffd_lims[field][1] - cp_ffd_lims[field][0]
    cp_ffd_lims[field][1] = cp_ffd_lims[field][1] + 0.1*cp_range
    cp_ffd_lims[field][0] = cp_ffd_lims[field][0] - 0.1*cp_range
FFD_block = create_3D_block(ffd_block_num_el, p, cp_ffd_lims)

# Set FFD to non-matching optimization instance
nonmatching_opt_ffd.set_FFD(FFD_block.knots, FFD_block.control)
# Set constraint info
nonmatching_opt_ffd.set_pin_CPFFD(pin_dir0=0, pin_side0=[0,1],
                                  pin_dir1=2, pin_side1=[0,1])
nonmatching_opt_ffd.set_regu_CPFFD(regu_dir=[None], regu_side=[None])


# Set up optimization
nonmatching_opt_ffd.create_files(save_path=save_path, 
                                 folder_name=folder_name)
model = ShapeOptGroup(nonmatching_opt_ffd=nonmatching_opt_ffd)
model.init_parameters(design_var_lims=[[cp_ffd_lims[2][0]/2, cp_ffd_lims[2][1]*2]])
prob = om.Problem(model=model)

if optimizer.upper() == 'SNOPT':
    prob.driver = om.pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'
    prob.driver.opt_settings['Minor feasibility tolerance'] = 1e-6
    prob.driver.opt_settings['Major feasibility tolerance'] = 1e-6
    prob.driver.opt_settings['Major optimality tolerance'] = 1e-4
    prob.driver.opt_settings['Major iterations limit'] = 50000
    prob.driver.opt_settings['Summary file'] = './SNOPT_report/SNOPT_summary'+str(test_ind)+'.out'
    prob.driver.opt_settings['Print file'] = './SNOPT_report/SNOPT_print'+str(test_ind)+'.out'
    prob.driver.options['debug_print'] = ['objs']
    prob.driver.options['print_results'] = True
elif optimizer.upper() == 'SLSQP':
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1e-8
    prob.driver.options['disp'] = True
    prob.driver.options['debug_print'] = ['objs']
    prob.driver.options['maxiter'] = 50000
else:
    raise ValueError("Undefined optimizer: {}".format(optimizer))

prob.setup()
prob.run_driver()

# if mpirank == 0:
#     print("Maximum F2: {:8.6f}".
#           format(np.max(nonmatching_opt_ffd.splines[0].cpFuncs[2]
#                  .vector().get_local())))
#     print("Miminum F2: {:8.6f}".
#           format(np.min(nonmatching_opt_ffd.splines[1].cpFuncs[2]
#                  .vector().get_local())))

#### Save final shape of FFD block
VTK().write("./geometry/FFD_block_initial"+str(test_ind)+".vtk", FFD_block)
init_CP_FFD = FFD_block.control[:,:,:,0:3].transpose(2,1,0,3).reshape(-1,3)
final_CP_FFD = init_CP_FFD.copy()
final_FFD_CP1 = prob[model.inputs_comp_name+'.'+model.cpffd_name_list[0]]
final_CP_FFD[:,opt_field[0]] = final_FFD_CP1
final_CP_FFD = final_CP_FFD.reshape(FFD_block.control[:,:,:,0:3]\
               .transpose(2,1,0,3).shape)
final_CP_FFD = final_CP_FFD.transpose(2,1,0,3)
final_FFD_block = NURBS(FFD_block.knots, final_CP_FFD)
VTK().write('./geometry/FFD_block_final'+str(test_ind)+'.vtk', final_FFD_block)