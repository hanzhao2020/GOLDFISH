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

class ThicknessOptGroup(om.Group):

    def initialize(self):
        self.options.declare('nonmatching_opt')
        self.options.declare('h_th_name_design', default='thickness')
        self.options.declare('h_th_name_full', default='thickness_full')
        self.options.declare('disp_name', default='displacements')
        self.options.declare('max_vM_name', default='max_vM_stress')
        self.options.declare('volume_name', default='volume')

    def init_parameters(self):
        self.nonmatching_opt = self.options['nonmatching_opt']
        self.h_th_name_design = self.options['h_th_name_design']
        self.h_th_name_full = self.options['h_th_name_full']
        self.disp_name = self.options['disp_name']
        self.volume_name = self.options['volume_name']
        self.max_vM_name = self.options['max_vM_name']

        self.design_var_lower = 1.e-3
        self.design_var_upper = 1.e-1

        self.num_splines = self.nonmatching_opt.num_splines
        self.init_h_th = [np.average(h_th_sub_array) for h_th_sub_array
                          in self.nonmatching_opt.init_h_th_list]

        self.inputs_comp_name = 'inputs_comp'
        self.h_th_map_comp_name = 'h_th_map_comp'
        self.disp_states_comp_name = 'disp_states_comp'
        self.volume_comp_name = 'volume_comp'
        self.max_vM_comp_name = 'max_vM_comp'

    def setup(self):
        # Add inputs comp
        inputs_comp = om.IndepVarComp()
        inputs_comp.add_output(self.h_th_name_design,
                    shape=self.num_splines, val=self.init_h_th)
        self.add_subsystem(self.inputs_comp_name, inputs_comp)

        # Add h_th map comp
        self.h_th_map_comp = HthMapComp(
                        nonmatching_opt=self.nonmatching_opt,
                        input_h_th_name_design=self.h_th_name_design,
                        output_h_th_name_full=self.h_th_name_full)
        self.h_th_map_comp.init_parameters()
        self.add_subsystem(self.h_th_map_comp_name, self.h_th_map_comp)

        # Add disp_states_comp
        self.disp_states_comp = DispStatesComp(
                           nonmatching_opt=self.nonmatching_opt,
                           input_h_th_name=self.h_th_name_full,
                           output_u_name=self.disp_name)
        self.disp_states_comp.init_parameters(save_files=True)
        self.add_subsystem(self.disp_states_comp_name, self.disp_states_comp)

        # Add volume comp (objective function)
        self.volume_comp = VolumeComp(
                           nonmatching_opt=self.nonmatching_opt,
                           input_h_th_name=self.h_th_name_full,
                           output_vol_name=self.volume_name)
        self.volume_comp.init_parameters()
        self.add_subsystem(self.volume_comp_name, self.volume_comp)
        self.vol_val = 0
        for s_ind in range(self.nonmatching_opt.num_splines):
            self.vol_val += assemble(self.nonmatching_opt.h_th[s_ind]
                            *self.nonmatching_opt.splines[s_ind].dx)

        # Add max von Mises stress comp (constraint)
        # xi2 = 0.
        rho = 2e2
        upper_vM = 8e5
        self.max_vM_comp = MaxvMStressComp(
                               nonmatching_opt=nonmatching_opt,
                               rho=rho, alpha=None, m=upper_vM, method='pnorm', 
                               linearize_stress=False, 
                               input_u_name=self.disp_name,
                               input_h_th_name=self.h_th_name_full,
                               output_max_vM_name=self.max_vM_name)
        self.max_vM_comp.init_parameters()
        self.add_subsystem(self.max_vM_comp_name, self.max_vM_comp)

        # Connect names between components
        self.connect(self.inputs_comp_name+'.'
                     +self.h_th_name_design,
                     self.h_th_map_comp_name+'.'
                     +self.h_th_name_design)

        self.connect(self.h_th_map_comp_name+'.'
                     +self.h_th_name_full,
                     self.disp_states_comp_name+'.'
                     +self.h_th_name_full)

        self.connect(self.h_th_map_comp_name+'.'
                     +self.h_th_name_full,
                     self.volume_comp_name+'.'
                     +self.h_th_name_full)

        self.connect(self.h_th_map_comp_name+'.'
                     +self.h_th_name_full,
                     self.max_vM_comp_name+'.'
                     +self.h_th_name_full)

        self.connect(self.disp_states_comp_name+'.'+self.disp_name,
                     self.max_vM_comp_name+'.'+self.disp_name)

        # Add design variable, constraints and objective
        self.add_design_var(self.inputs_comp_name+'.'
                            +self.h_th_name_design,
                            lower=self.design_var_lower,
                            upper=self.design_var_upper,
                            scaler=1e3)

        self.add_constraint(self.max_vM_comp_name+'.'
                            +self.max_vM_name,
                            upper=upper_vM,
                            scaler=1e-6)
        # Use scaler 1e10 for SNOPT optimizer, 1e8 for SLSQP
        self.add_objective(self.volume_comp_name+'.'
                           +self.volume_name,
                           scaler=1.e2)


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
# Ribs: 68, 69, 70, ..., 85 (18 ribs)
num_secs = 8
wing_indices = list(range(0,len(pegasus_surfaces))) # All surfaces
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

u_insert_list = [6]*num_surfs
v_insert_list = [4]*num_surfs

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

# write_geom_file(preprocessor.BSpline_surfs_refine, "pegasus_wing_geom.igs")
# exit()

if mpirank == 0:
    print("Computing intersections...")
int_data_filename = "pegasus_wing_int_data.npz"
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
# # preprocessor.display_surfaces(display, transparency=0.5, show_bdry=False, save_fig=False)
# preprocessor.display_intersections(display, color='RED', save_fig=False)
# print(aaa)

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

h_th = []
for i in range(num_surfs):
    h_th += [Function(splines[i].V_linear)]
    h_th[i].interpolate(Constant(3.0e-3))

# Create non-matching problem
nonmatching_opt = NonMatchingOpt(splines, E, h_th, nu, opt_shape=False, 
                                 opt_thickness=True, comm=worldcomm)
nonmatching_opt.create_mortar_meshes(preprocessor.mortar_nels)

if mpirank == 0:
    print("Setting up mortar meshes...")

nonmatching_opt.mortar_meshes_setup(preprocessor.mapping_list, 
                                    preprocessor.intersections_para_coords, 
                                    penalty_coefficient)

# Define magnitude of load
load = Constant(10) # The load should be in the unit of N/m^2
f1 = as_vector([Constant(0.0), Constant(0.0), load])

# Distributed downward load
loads = [f1]*num_surfs
source_terms = []
residuals = []
for i in range(num_surfs):
    source_terms += [inner(loads[i], nonmatching_opt.splines[i].rationalize(
        nonmatching_opt.spline_test_funcs[i]))*nonmatching_opt.splines[i].dx]
    residuals += [SVK_residual(nonmatching_opt.splines[i], nonmatching_opt.spline_funcs[i], 
        nonmatching_opt.spline_test_funcs[i], E, nu, h_th[i], source_terms[i])]
nonmatching_opt.set_residuals(residuals)

# nonmatching_opt.solve_nonlinear_nonmatching_problem(solver='direct')

# Set up optimization
nonmatching_opt.create_files(save_path=save_path, folder_name=folder_name, 
                             thickness=nonmatching_opt.opt_thickness)
model = ThicknessOptGroup(nonmatching_opt=nonmatching_opt)
model.init_parameters()
prob = om.Problem(model=model)

if optimizer.upper() == 'SNOPT':
    prob.driver = om.pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'
    prob.driver.opt_settings['Minor feasibility tolerance'] = 1e-6
    prob.driver.opt_settings['Major feasibility tolerance'] = 1e-6
    prob.driver.opt_settings['Major optimality tolerance'] = 1e-4
    prob.driver.opt_settings['Major iterations limit'] = 50000
    prob.driver.opt_settings['Summary file'] = \
        './SNOPT_report/SNOPT_summary'+str(test_ind)+'.out'
    prob.driver.opt_settings['Print file'] = \
        './SNOPT_report/SNOPT_print'+str(test_ind)+'.out'
    prob.driver.options['debug_print'] = ['objs', 'desvars']
    prob.driver.options['print_results'] = True
elif optimizer.upper() == 'SLSQP':
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1e-6
    prob.driver.options['disp'] = True
    prob.driver.options['debug_print'] = ['objs', 'desvars']
    prob.driver.options['maxiter'] = 50000
else:
    raise ValueError("Undefined optimizer: {}".format(optimizer))

prob.setup()
prob.run_driver()

if mpirank == 0:
    for i in range(num_surfs):
        print("Thickness for patch {:2d}: {:8.4f}".format(i, 
              nonmatching_opt.h_th[i].vector().get_local()[0]))

thickness_vec = np.zeros(num_surfs)
for i in range(num_surfs):
    thickness_vec[i] = nonmatching_opt.h_th[i].vector().get_local()[0]
np.savez("./pegasus_wing_thickness.npz",
         name1=thickness_vec)

if mpirank == 0:
    print("Solving nonlinear non-matching problem ...")

# SAVE_PATH = "/home/han/Documents/test_results/"
# FOLDER_NAME = 'results/'

# nonmatching_opt.solve_nonlinear_nonmatching_problem()

# # print out vertical displacement at the tip of trailing edge
# right_srf_ind = 67
# xi = array([1, 1])
# z_disp_hom = eval_func(nonmatching_opt.splines[right_srf_ind].mesh, 
#                        nonmatching_opt.spline_funcs[right_srf_ind][2], xi)
# w = eval_func(nonmatching_opt.splines[right_srf_ind].mesh, 
#               nonmatching_opt.splines[right_srf_ind].cpFuncs[3], xi)
# QoI = z_disp_hom/w

# if mpirank == 0:
#     print("Trailing edge tip vertical displacement: {:10.8f}.\n".format(QoI))

# Compute von Mises stress
if mpirank == 0:
    print("Computing von Mises stresses...")

von_Mises_funcs = []
vM_max_list = []
for i in range(nonmatching_opt.num_splines):
    von_Mises_proj = nonmatching_opt.splines[i].projectScalarOntoLinears(
                     model.max_vM_comp.max_vm_exop.vMstress[i], lumpMass=False)
    von_Mises_funcs += [von_Mises_proj,]
    vM_max_list += [v2p(von_Mises_funcs[i].vector()).max()[1]]

print("vM_list:", vM_max_list)
print("Max von Mises stress:", np.max(vM_max_list))


# von_Mises_tops = []
# von_Mises_tops_proj = []
# max_vM_list = []
# for i in range(nonmatching_opt.num_splines):
#     print("Computing von Mises stress for surface: ", i)
#     spline_stress = ShellStressSVK(nonmatching_opt.splines[i], 
#                                    nonmatching_opt.spline_funcs[i],
#                                    E, nu, h_th[i], linearize=False,) 
#                                    # G_det_min=5e-2)
#     # von Mises stresses on top surfaces
#     von_Mises_top = spline_stress.vonMisesStress(h_th[i]/2)
#     von_Mises_tops += [von_Mises_top]
#     von_Mises_top_proj = nonmatching_opt.splines[i].projectScalarOntoLinears(
#                          von_Mises_top, lumpMass=False)
#     von_Mises_tops_proj += [von_Mises_top_proj]
#     max_vM_list += [np.max(von_Mises_top_proj.vector().get_local())]

# print("vM_list:", max_vM_list)
# print("Max von Mises stress:", np.max(max_vM_list))

if mpirank == 0:
    print("Saving results...")

for i in range(nonmatching_opt.num_splines):
    save_results(splines[i], nonmatching_opt.spline_funcs[i], i, 
                 save_path=save_path, folder=folder_name, 
                 save_cpfuncs=True, comm=worldcomm)

for i in range(nonmatching_opt.num_splines):
    von_Mises_funcs[i].rename("von_Mises_top_"+str(i), 
                             "von_Mises_top_"+str(i))
    File(save_path+folder_name+"von_Mises_top_"+str(i)+".pvd") \
        << von_Mises_funcs[i]