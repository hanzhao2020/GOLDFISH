## Caddee
from caddee.utils.aircraft_models.pav.pav_geom_mesh import PavGeomMesh
import caddee.api as cd

## Solvers
from VAST.core.vast_solver import VASTFluidSover
from VAST.core.fluid_problem import FluidProblem
from VAST.core.generate_mappings_m3l import VASTNodalForces
from VAST.core.vlm_llt.viscous_correction import ViscousCorrectionModel

# Other lsdo lab stuff
import csdl
from python_csdl_backend import Simulator
from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem
import m3l
from m3l.utils.utils import index_functions
import lsdo_geo as lg
import array_mapper as am
from m3l.core.function_spaces import IDWFunctionSpace

## Other stuff
import numpy as np
from mpi4py import MPI
import pickle
import pathlib
import sys

# KL shell related components
from GOLDFISH.nonmatching_opt_ffd import *
from PENGoLINS.igakit_utils import *
from PENGoLINS.occ_preprocessing import*
import klshell_module as klshell
from klshell_pde import *

sys.setrecursionlimit(100000)

SAVE_PATH = '/home/han/Documents/test_results/'


debug_geom_flag = False
force_reprojection = False
visualize_flag = False
fewer_ribs = True
# Dashboard and xdmf recorder cannot be turned on at the same time
dashboard = False
xdmf_record = False

ft2m = 0.3048
in2m = 0.0254

# wing_cl0 = 0.3366
# pitch_angle_list = [-0.02403544, 6, 12.48100761]
# h_0 = 0.02*in2m

wing_cl0 = 0.3662
pitch_angle_list = [-0.38129494, 6, 12.11391141]
h_0 = 0.05*in2m
pitch_angle = np.deg2rad(pitch_angle_list[2])


caddee = cd.CADDEE()
caddee.system_model = system_model = cd.SystemModel()

# region Geometry and meshes
pav_geom_mesh = PavGeomMesh()
pav_geom_mesh.setup_geometry(
    include_wing_flag=True,
    include_htail_flag=False,
)
pav_geom_mesh.setup_internal_wingbox_geometry(debug_geom_flag=debug_geom_flag,
                                              force_reprojection=force_reprojection,
                                              fewer_ribs=fewer_ribs)
pav_geom_mesh.sys_rep.spatial_representation.assemble()
pav_geom_mesh.oml_mesh(include_wing_flag=True,
                       debug_geom_flag=debug_geom_flag, force_reprojection=force_reprojection)
pav_geom_mesh.vlm_meshes(include_wing_flag=True, num_wing_spanwise_vlm=21, num_wing_chordwise_vlm=5,
                         visualize_flag=visualize_flag, force_reprojection=force_reprojection)
pav_geom_mesh.setup_index_functions()



caddee.system_representation = sys_rep = pav_geom_mesh.sys_rep
caddee.system_parameterization = sys_param = pav_geom_mesh.sys_param
sys_param.setup()
spatial_rep = sys_rep.spatial_representation
# endregion

# region save_surfaces

knots_u = []
knots_v = []
order_u = []
order_v = []
control_points = []
for name in pav_geom_mesh.geom_data['primitive_names']['structural_left_wing_names']:
    primitive = spatial_rep.get_primitives([name])[name].geometry_primitive
    control_points.append(primitive.control_points)
    knots_u.append(primitive.knots_u)
    knots_v.append(primitive.knots_v)
    order_u.append(primitive.order_u)
    order_v.append(primitive.order_v)

# print(knots_u)

# PENGoLINS related geometry preprocessing
p = 3
geom_scale = 1

ikNURBS_surfs = []
for s_ind in range(len(knots_u)):
    ikNURBS_surfs += [NURBS((knots_u[s_ind], knots_v[s_ind]), 
                           control_points[s_ind]),]

occ_surfs = [ikNURBS2BSpline_surface(ik_surf) for ik_surf in ikNURBS_surfs]
write_geom_file(occ_surfs, 'pav_wing_geom_m.igs')
print(aaa)

occ_surfs_init = []
for s_ind in range(len(ikNURBS_surfs)):
    occ_surfs_init += [ikNURBS2BSpline_surface(ikNURBS_surfs[s_ind])]

num_surfs = len(occ_surfs_init)

# Create PENGoLINS preprocessor instance
num_pts_eval = [6]*num_surfs
ref_level_list = [1]*num_surfs

u_insert_list = [8]*num_surfs
v_insert_list = [8]*num_surfs

u_num_insert = []
v_num_insert = []
for i in range(len(u_insert_list)):
    u_num_insert += [ref_level_list[i]*u_insert_list[i]]
    v_num_insert += [ref_level_list[i]*v_insert_list[i]]

# Geometry preprocessing and surface-surface intersections computation
preprocessor = OCCPreprocessing(occ_surfs_init, reparametrize=False, 
                                refine=False)
# preprocessor.reparametrize_BSpline_surfaces(num_pts_eval, num_pts_eval,
#                                             geom_scale=geom_scale,
#                                             remove_dense_knots=True,
#                                             rtol=1e-4)
# preprocessor.refine_BSpline_surfaces(p, p, u_num_insert, v_num_insert, 
#                                      correct_element_shape=True)

ik_surfs = [BSpline_surface2ikNURBS(occ_bs_surf, p=3, 
            u_num_insert=0, v_num_insert=0, 
            refine=False, geom_scale=1.0e3) 
            for occ_bs_surf in preprocessor.BSpline_surfs]

# Front spars: [0, 2, 4, 6, 8]
# Rear spars: [1, 3, 5, 7, 9]
# ribs: [10, 11, 12, 13, 14, 15]
# Upper skin: [16, 18, 20, 22, 24]
# Lower skin: [17, 19, 21, 23, 25]

if mpirank == 0:
    print("Computing intersections...")
int_data_filename = "pav_wing_int_data.npz"
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

# for i in range(num_surfs):
#     VTK().write('./geometry/surf_refine'+str(i)+'.vtk',
#                 BSpline_surface2ikNURBS(preprocessor.BSpline_surfs_refine[i]))

# for i in range(preprocessor.num_intersections_all):
#     mesh_phy = generate_mortar_mesh(preprocessor.intersections_phy_coords[i], 
#                                     num_el=128)
#     File('./geometry/int_curve'+str(i)+'.pvd') << mesh_phy


if mpirank == 0:
    print("Creating splines...")
# Create tIGAr extracted spline instances
splines = []
lin_splines = []
# lin_spline_inds = [i for i in range(num_surfs)
#                    if i not in [10, 11, 12, 13, 14, 15] ]

lin_spline_inds = [i for i in range(num_surfs)]
for i in range(num_surfs):
    if i in [0, 1, 16, 17]:
        # Apply clamped BC to surfaces near root
        spline = OCCBSpline2tIGArSpline(
                 preprocessor.BSpline_surfs_refine[i], 
                 setBCs=clampedBC, side=0, direction=1, index=i)
        splines += [spline,]
    else:
        spline = OCCBSpline2tIGArSpline(
                 preprocessor.BSpline_surfs_refine[i], index=i)
        splines += [spline,]
    # Create linear splines for ourter surfaces
    if i in lin_spline_inds:
        lin_splines += [OCCBSpline2LoadSpline(
                        preprocessor.BSpline_surfs_refine[i], 
                        splines[i], n_fields=1, index=i)]

# Unstiffened Aluminum 2024 (T4)
# reference: https://asm.matweb.com/search/SpecificMaterial.asp?bassnum=ma2024t4
E = 73.1E9 # unit: Pa
nu = 0.33
h = h_0 # unit: m
rho = 2780 # unit: kg/m^3
f_d = -rho*h*9.81 # self-weight unit: N
tensile_yield_strength = 324E6 # unit: Pa
safety_factor = 1.5

penalty_coefficient = 1e3

h_th = []
for i in range(num_surfs):
    h_th += [Function(splines[i].V_linear)]
    h_th[i].interpolate(Constant(h))

# Create non-matching problem
nonmatching_opt = NonMatchingOpt(splines, E, h_th, nu, opt_shape=False, 
                                 opt_thickness=True, comm=worldcomm)
nonmatching_opt.create_mortar_meshes(preprocessor.mortar_nels)

if mpirank == 0:
    print("Setting up mortar meshes...")

nonmatching_opt.mortar_meshes_setup(preprocessor.mapping_list, 
                                    preprocessor.intersections_para_coords, 
                                    penalty_coefficient)


lin_spline_loads = PENGoLINS_dist_loads(lin_splines)

# Question? How this pickle data generated
with open('./pav_wing/pav_wing_v2_paneled_mesh_data_2303.pickle', 'rb') as f:
    nodes_parametric = pickle.load(f)

for i in range(len(nodes_parametric)):
    nodes_parametric[i] = (nodes_parametric[i][0].replace(' ', '_').replace(',',''), 
                           np.array([nodes_parametric[i][1]]))

wing_thickness = pav_geom_mesh.functions['wing_thickness']

test_ind = 15
save_path = SAVE_PATH
folder_name = 'results'+str(test_ind)+'/'

nonmatching_opt.create_files(save_path=save_path, folder_name=folder_name, 
                             thickness=nonmatching_opt.opt_thickness)

# nodes_parametric_new = []
# for node_para in nodes_parametric:
#     if node_para[0] in wing_thickness.coefficients.keys():
#         nodes_parametric_new += [node_para,]


nodes_parametric_new = []
for s_ind, key in enumerate(wing_thickness.coefficients.keys()):
    for nodal_val in nonmatching_opt.splines[s_ind].mesh.coordinates():
        nodes_parametric_new += [(key, nodal_val)]

thickness_nodes = wing_thickness.evaluate(nodes_parametric_new)

klshell_pde = KLShellPDE(nonmatching_opt, lin_spline_loads, lin_spline_inds)

# create the shell dictionaries:
shells = {}
shells['wing_shell'] = {'E': E, 'nu': nu, 'rho': rho,}
shell_name = list(shells.keys())[0]

# tempm = KLShellModule(nonmatching_opt, klshell_pde, shells)


# # Create force functions and set residuals
# spline_forces = []
# source_terms = []
# residuals = []
# for s_ind in range(nonmatching_opt.num_splines):
#     a0,a1,a2,_,a,_ = surfaceGeometry(
#                      nonmatching_opt.splines[s_ind], 
#                      nonmatching_opt.splines[s_ind].F\
#                      +nonmatching_opt.spline_funcs[s_ind])
#     A0,A1,A2,_,A,_ = surfaceGeometry(
#                      nonmatching_opt.splines[s_ind], 
#                      nonmatching_opt.splines[s_ind].F)
#     if s_ind in klshell_pde.lin_spline_inds:
#         lin_spline = klshell_pde.lin_splines.surfaces[
#                                klshell_pde.lin_spline_inds.
#                                index(s_ind)]
#         spline_forces += [Function(lin_spline.V)]
#         source_terms += [lin_spline.rationalize(
#             spline_forces[s_ind])*sqrt(det(a)/det(A))
#             *inner(a2, nonmatching_opt.spline_test_funcs[s_ind])*
#             nonmatching_opt.splines[s_ind].dx]
#     else:
#         spline_forces += [Constant((0.))]
#         source_terms += [
#             spline_forces[s_ind]*sqrt(det(a)/det(A))
#             *inner(a2, nonmatching_opt.spline_test_funcs[s_ind])*
#             nonmatching_opt.splines[s_ind].dx]            
#     residuals += [SVK_residual(
#         nonmatching_opt.splines[s_ind], 
#         nonmatching_opt.spline_funcs[s_ind], 
#         nonmatching_opt.spline_test_funcs[s_ind], 
#         nonmatching_opt.E[s_ind], 
#         nonmatching_opt.nu[s_ind], 
#         nonmatching_opt.h_th[s_ind], 
#         source_terms[s_ind])]

# nonmatching_opt.set_residuals(residuals)

# nonmatching_opt.set_aero_linear_splines(
#     klshell_pde.lin_splines.surfaces,
#     spline_forces)

# raise RuntimeError()

# # Wing shell Mesh
# z_offset = 0.0
# wing_shell_mesh = am.MappedArray(input=fenics_mesh.geometry.x).reshape((-1,3))
# shell_mesh = klshell.LinearKLShellSurface(
#                     meshes=dict(
#                     wing_shell_mesh=wing_shell_mesh,
#                     ))


# region Mission
design_scenario_name = 'structural_sizing'
design_scenario = cd.DesignScenario(name=design_scenario_name)
# endregion

# region Cruise condition
cruise_name = "cruise_3"
cruise_model = m3l.Model()
cruise_condition = cd.CruiseCondition(name=cruise_name)
cruise_condition.atmosphere_model = cd.SimpleAtmosphereModel()
cruise_condition.set_module_input(name='altitude', val=600 * ft2m)
cruise_condition.set_module_input(name='mach_number', val=0.145972)  # 112 mph = 0.145972 Mach
cruise_condition.set_module_input(name='range', val=80467.2)  # 50 miles = 80467.2 m
cruise_condition.set_module_input(name='pitch_angle', val=pitch_angle)
cruise_condition.set_module_input(name='flight_path_angle', val=0)
cruise_condition.set_module_input(name='roll_angle', val=0)
cruise_condition.set_module_input(name='yaw_angle', val=0)
cruise_condition.set_module_input(name='wind_angle', val=0)
cruise_condition.set_module_input(name='observer_location', val=np.array([0, 0, 600 * ft2m]))

cruise_ac_states = cruise_condition.evaluate_ac_states()
cruise_model.register_output(cruise_ac_states)

# region VLM Solver
vlm_model = VASTFluidSover(
    surface_names=[pav_geom_mesh.mesh_data['vlm']['mesh_name']['wing'],],
    surface_shapes=[(1,) + pav_geom_mesh.mesh_data['vlm']['chamber_surface']['wing']
                    .evaluate().shape[1:],],
    fluid_problem=FluidProblem(solver_option='VLM', problem_type='fixed_wake'),
    mesh_unit='m', cl0=[wing_cl0, ])
wing_vlm_panel_forces, vlm_forces, vlm_moments = vlm_model.evaluate(ac_states=cruise_ac_states)
cruise_model.register_output(vlm_forces)
cruise_model.register_output(vlm_moments)

vlm_force_mapping_model = VASTNodalForces(
    surface_names=[pav_geom_mesh.mesh_data['vlm']['mesh_name']['wing'],],
    surface_shapes=[(1,) + pav_geom_mesh.mesh_data['vlm']['chamber_surface']['wing']
                    .evaluate().shape[1:],],
    initial_meshes=[pav_geom_mesh.mesh_data['vlm']['chamber_surface']['wing'],])

wing_oml_mesh = pav_geom_mesh.mesh_data['oml']['oml_geo_nodes']['wing']
oml_forces = vlm_force_mapping_model.evaluate(vlm_forces=wing_vlm_panel_forces,
                                              nodal_force_meshes=[wing_oml_mesh, ])
wing_forces = oml_forces[0]

# endregion

# region Strucutral Loads
wing_force = pav_geom_mesh.functions['wing_force']
oml_para_nodes = pav_geom_mesh.mesh_data['oml']['oml_para_nodes']['wing']

wing_force.inverse_evaluate(oml_para_nodes, wing_forces)
cruise_model.register_output(wing_force.coefficients)

left_wing_oml_para_coords = pav_geom_mesh.mesh_data['oml']['oml_para_nodes']['left_wing']
left_oml_geo_nodes = spatial_rep.evaluate_parametric(left_wing_oml_para_coords)

left_wing_forces = wing_force.evaluate(left_wing_oml_para_coords)
wing_component = pav_geom_mesh.geom_data['components']['wing']

shell_force_map_model = klshell.KLShellForces(
                        component=wing_component,
                        nonmatching_opt=nonmatching_opt,
                        klshell_pde=klshell_pde,
                        shells=shells)
cruise_structural_wing_mesh_forces = shell_force_map_model.evaluate(
                        nodal_forces=left_wing_forces,
                        nodal_forces_mesh=left_oml_geo_nodes)
# endregion

# region Structures

# klshell_module = KLShellModule(nonmatching_opt=nonmatching_opt,
#                                klshell_pde=klshell_pde,
#                                shells=shells)
# klshell_module.init_parameters()

shell_displacements_model = klshell.KLShell(
                            component=wing_component,
                            nonmatching_opt=nonmatching_opt,
                            klshell_pde=klshell_pde,
                            shells=shells)

cruise_structural_wing_mesh_displacements, \
    cruise_structural_wing_mesh_stresses, wing_mass = \
    shell_displacements_model.evaluate(
        forces=cruise_structural_wing_mesh_forces)
cruise_model.register_output(cruise_structural_wing_mesh_stresses)
cruise_model.register_output(cruise_structural_wing_mesh_displacements)
cruise_model.register_output(wing_mass)

# # # TEMP: remove stress
# cruise_structural_wing_mesh_displacements, wing_mass = \
#     shell_displacements_model.evaluate(
#         forces=cruise_structural_wing_mesh_forces)
# cruise_model.register_output(cruise_structural_wing_mesh_displacements)
# cruise_model.register_output(wing_mass)

grid_num = 10
transfer_para_mesh = []
structural_left_wing_names = pav_geom_mesh.geom_data['primitive_names']\
                             ['structural_left_wing_names']
for name in structural_left_wing_names:
    for u in np.linspace(0,1,grid_num):
        for v in np.linspace(0,1,grid_num):
            transfer_para_mesh.append((name, np.array([u,v]).reshape((1,2))))

transfer_geo_nodes_ma = spatial_rep.evaluate_parametric(transfer_para_mesh)

shell_nodal_displacements_model = klshell.KLShellNodalDisplacements(
                                  component=wing_component,
                                  nonmatching_opt=nonmatching_opt,
                                  klshell_pde=klshell_pde,
                                  shells=shells)

nodal_displacements, tip_displacement = shell_nodal_displacements_model.\
    evaluate(cruise_structural_wing_mesh_displacements, transfer_geo_nodes_ma)
wing_displacement = pav_geom_mesh.functions['wing_displacement']

wing_displacement.inverse_evaluate(transfer_para_mesh, nodal_displacements)
cruise_model.register_output(wing_displacement.coefficients)

# wing_stress = pav_geom_mesh.functions['wing_stress']
# wing_stress.inverse_evaluate(nodes_parametric_new, 
#         cruise_structural_wing_mesh_stresses, regularization_coeff=1e-3)
# cruise_model.register_output(wing_stress.coefficients)

cruise_model.register_output(tip_displacement)
cruise_model.register_output(nodal_displacements)

# Add cruise m3l model to cruise condition
cruise_condition.add_m3l_model('cruise_model', cruise_model)
# Add design condition to design scenario
design_scenario.add_design_condition(cruise_condition)

system_model.add_design_scenario(design_scenario=design_scenario)

caddee_csdl_model = caddee.assemble_csdl()

system_model_name = 'system_model.'+design_scenario_name+'.'\
                    +cruise_name+'.'+cruise_name+'.'

caddee_csdl_model.add_constraint(system_model_name+
    'Wing_klshell_displacement_map.wing_shell_tip_displacement',
    upper=0.05,scaler=1E1)

caddee_csdl_model.add_constraint(system_model_name+
    'Wing_klshell_model.klshell.max_vM_model.max_vM_stress',
    upper=324E6/1.5/20,scaler=1E-8)

caddee_csdl_model.add_objective(system_model_name+
    'Wing_klshell_model.klshell.volume_model.volume', scaler=1e1)

h_min = h

caddee_csdl_model.create_input(shell_name+'_hth_design',
    shape=(nonmatching_opt.num_splines),
    val=np.ones(nonmatching_opt.num_splines)*h_min)

caddee_csdl_model.add_design_variable(shell_name+'_hth_design',
    lower=0.005*in2m,
    upper=0.1*in2m,
    scaler=1e2)
caddee_csdl_model.connect(shell_name+'_hth_design', 
        system_model_name+\
        'Wing_klshell_model.klshell.h_th_map_model.'+shell_name+'_hth_design')


# h_min = h

# i = 0
# shape = (9, 1)
# valid_structural_left_wing_names = structural_left_wing_names

# for name in valid_structural_left_wing_names:
#     primitive = spatial_rep.get_primitives([name])[name].geometry_primitive
#     name = name.replace(' ', '_').replace(',','')
#     surface_id = i
#     h_init = caddee_csdl_model.create_input('wing_thickness_dv_'+name, val=h_min)
#     caddee_csdl_model.add_design_variable('wing_thickness_dv_'+name, # 0.02 in
#                                           lower=0.005 * in2m,
#                                           upper=0.1 * in2m,
#                                           scaler=1000,
#                                           )
#     caddee_csdl_model.register_output('wing_thickness_surface_'+name, csdl.expand(h_init, shape))
#     caddee_csdl_model.connect('wing_thickness_surface_'+name,
#                                 system_model_name+'wing_thickness_function_evaluation.'+\
#                                 name+'_wing_thickness_coefficients')
#     i += 1

if dashboard:
    import lsdo_dash.api as ld
    index_functions_map = {}
    
    index_functions_map['wing_thickness'] = wing_thickness  
    index_functions_map['wing_force'] = wing_force
    index_functions_map['wing_displacement'] = wing_displacement
    index_functions_map['wing_stress'] = wing_stress

    rep = csdl.GraphRepresentation(caddee_csdl_model)

    # profiler.disable()
    # profiler.dump_stats('output')

    caddee_viz = ld.caddee_plotters.CaddeeViz(
        caddee = caddee,
        system_m3l_model = system_model,
        design_configuration_map={},
    )

if __name__ == '__main__':
    if dashboard:
        from dash_pav import TC2DB
        dashbuilder = TC2DB()
        sim = Simulator(rep, analytics=True, dashboard = dashbuilder)
    else:
        sim = Simulator(caddee_csdl_model, analytics=True)

    print("Inspection opt 0: Memory usage: {:8.2f} MB.\n"\
          .format(memory_usage_psutil()))

    sim.run()

    print("Inspection opt 1: Memory usage: {:8.2f} MB.\n"\
          .format(memory_usage_psutil()))


    prob = CSDLProblem(problem_name='pav', simulator=sim)

    optimizer = SLSQP(prob, maxiter=50, ftol=1E-7)

    # # from modopt.snopt_library import SNOPT
    # optimizer = SNOPT(prob,
    #                   Major_iterations = 100,
    #                   Major_optimality = 1e-5,
    #                   append2file=False)

    optimizer.solve()
    optimizer.print_results()