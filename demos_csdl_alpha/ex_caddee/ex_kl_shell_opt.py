import csdl_alpha as csdl
import CADDEE_alpha as cd
from CADDEE_alpha import functions as fs
import numpy as np
import os
import sys
from VortexAD.core.vlm.vlm_solver import vlm_solver
import aframe as af

# lsdo_airfoil must be cloned and installed from https://github.com/LSDOlab/lsdo_airfoil
from lsdo_airfoil.core.three_d_airfoil_aero_model import ThreeDAirfoilMLModelMaker

# IGA shell related packages
from PENGoLINS.igakit_utils import *
from PENGoLINS.nonmatching_coupling import *
import igakit.cad as ikcad
from igakit.io import VTK

from kl_shell_group import KLShellModel
from GOLDFISH.nonmatching_opt_csdl import *

# Settings
couple = False
optimize = True
inline = False
shell = False

# # The shell solver requires FEniCSx and femo, which require manual installation.
# # See https://github.com/LSDOlab/femo_alpha
# if shell:
#     from femo_alpha.rm_shell.rm_shell_model import RMShellModel
#     from femo_alpha.fea.utils_dolfinx import readFEAMesh, reconstructFEAMesh

def return_ik_knots(cd_knot, return_order=False):
    xi0_0_count = 0
    num_int_konts0 = 0
    for i in cd_knot:
        if i == 0:
            xi0_0_count += 1
        elif i == 1:
            break
        else:
            num_int_konts0 += 1
    p0 = xi0_0_count-1
    knot0 = cd_knot[0:int(xi0_0_count*2+num_int_konts0)]
    knot1 = cd_knot[int(xi0_0_count*2+num_int_konts0):]
    xi1_0_count = 0
    for i in knot1:
        if i == 0:
            xi1_0_count += 1
        else:
            break
    p1 = xi1_0_count-1
    if return_order:
        return [knot0, knot1], [p0, p1]
    else:
        return [knot0, knot1]

# Quantities
skin_thickness = 0.007
spar_thickness = 0.001
rib_thickness = 0.001
mesh_fname = 'c172_tri.msh'
mass = 1000
stress_bound = 1e8
num_ribs = 10
load_factor = 3

# Start recording
rec = csdl.Recorder(inline=True, debug=True)
rec.start()

# Initialize CADDEE and import geometry
caddee = cd.CADDEE()
c172_geom = cd.import_geometry('c172.stp')

# def define_base_config(caddee : cd.CADDEE):
aircraft = cd.aircraft.components.Aircraft(geometry=c172_geom)
base_config = cd.Configuration(system=aircraft)
caddee.base_configuration = base_config

airframe = aircraft.comps["airframe"] = cd.Component()

# Setup wing component
wing_geometry = aircraft.create_subgeometry(
    search_names=['MainWing'],
    # The wing coming out of openVSP has some extra surfaces that we don't need
    ignore_names=['0, 8', '0, 9', '0, 14', '0, 15', '1, 16', '1, 17', '1, 22', '1, 23']
)
wing = cd.aircraft.components.Wing(AR=1, S_ref=1, geometry=wing_geometry)
airframe.comps["wing"] = wing


# print("1 ------------------")

# Generate internal geometry
wing.construct_ribs_and_spars(
    c172_geom,
    num_ribs=num_ribs,
    LE_TE_interpolation="ellipse",
    full_length_ribs=True,
    spanwise_multiplicity=10,
    offset=np.array([0.,0.,.15]),
    finite_te=True,
)

# extract relevant geometries
right_wing = wing.create_subgeometry(search_names=[''], ignore_names=[', 1, ', '_r_', '-'])
right_wing_oml = wing.create_subgeometry(search_names=['MainWing, 0'])
left_wing_oml = wing.create_subgeometry(search_names=['MainWing, 1'])
right_wing_spars = wing.create_subgeometry(search_names=['spar'], ignore_names=['_r_', '-'])
right_wing_ribs = wing.create_subgeometry(search_names=['rib'], ignore_names=['_r_', '-'])
wing_oml = wing.create_subgeometry(search_names=['MainWing'])
wing.quantities.right_wing_oml = right_wing_oml
wing.quantities.oml = wing_oml

# print("2 ------------------")
####################################################
# refit_space = fs.BSplineSpace(2, (3, 3), (16, 8))
# right_wing_refit = right_wing.refit(refit_space, grid_resolution=(16, 8))

te_surfs = wing.create_subgeometry(search_names=['0, 10', '0, 13'])
general_refit_space = fs.BSplineSpace(2, (2, 2), (16, 10))
te_refit_space = fs.BSplineSpace(2, (2, 2), (16,3))
refit_spaces = {}
for ind in right_wing.functions:
    if ind in te_surfs.functions:
        refit_spaces[ind] = te_refit_space
    else:
        refit_spaces[ind] = general_refit_space
right_wing_refit = right_wing.refit(refit_spaces, 
                    grid_resolution=(16, 10))


# For control points
cp_list = []
for key, value in right_wing_refit.functions.items():
	# print(key, value.coefficients.value)
    cp_list += [value.coefficients.value]

# For knots
knot_list = []
for key, value in right_wing_refit.functions.items():
	# print(key, value.space.knots)
    knot_list += [value.space.knots]

num_surfs = len(knot_list)

bc_list = [[2, 0, 0], [3, 0, 0]]
kl_shell_model = KLShellModel(knot_list, cp_list, bc_list)

# # For knots
# knot_list = []
# ik_knot_list = []
# for key, value in right_wing_refit.functions.items():
# 	# print(key, value.space.knots)
#     knot_list += [value.space.knots]
#     ik_knot_list += [return_ik_knots(knot_list[-1]),]

# ik_surfs = []
# for i in range(len(ik_knot_list)):
#     ik_surfs += [ikcad.NURBS(ik_knot_list[i],
#                  cp_list[i])]
#     VTK().write('./geometry/ik_surf'+str(i)+'.vtk', 
#                 ik_surfs[-1])

# ###################################################
# # IGA KL shell analysis
# num_surfs = len(knot_list)
# h_th = []
# h_val_list = [1e-3]*num_surfs
# for i in range(num_surfs):
#     h_th += [Function(kl_shell_model.splines[i].V_linear)]
#     h_th[i].interpolate(Constant(h_val_list[i]))

# E = 69e9
# nu = 0.33
# penalty_coefficient = Constant(1e3)

# # Create non-matching problem
# nonmatching_opt = NonMatchingOpt(kl_shell_model.splines, E, h_th, nu, comm=worldcomm)
# nonmatching_opt.create_mortar_meshes(kl_shell_model.preprocessor.mortar_nels)

# if mpirank == 0:
#     print("Setting up mortar meshes...")

# nonmatching_opt.mortar_meshes_setup(kl_shell_model.preprocessor.mapping_list, 
#                             kl_shell_model.preprocessor.intersections_para_coords, 
#                             penalty_coefficient)

# # # Distributed load
# # Define magnitude of load
# load = Constant(500) # The load should be in the unit of N/m^2
# f1 = as_vector([Constant(0.0), Constant(0.0), load])
# f0 = as_vector([Constant(0.0), Constant(0.0), Constant(0.0)])

# # Distributed downward load
# loads = [f0]*num_surfs
# loads[3] = f1
# source_terms = []
# residuals = []
# for i in range(num_surfs):
#     z = nonmatching_opt.splines[i].rationalize(
#         nonmatching_opt.spline_test_funcs[i])
#     source_terms += [inner(loads[i], z)\
#                      *nonmatching_opt.splines[i].dx]
#     residuals += [SVK_residual(nonmatching_opt.splines[i], 
#                   nonmatching_opt.spline_funcs[i], 
#                   nonmatching_opt.spline_test_funcs[i], E, nu, h_th[i], 
#                   source_terms[i])]
# nonmatching_opt.set_residuals(residuals)


# save_path = "./"
# folder_name = "results/"
# nonmatching_opt.create_files(save_path=save_path, folder_name=folder_name)

# nonmatching_opt.solve_nonlinear_nonmatching_problem()
# nonmatching_opt.save_files()
# raise RuntimeError
# ###############################################


# print("3 ------------------")

# material
E = csdl.Variable(value=69E9, name='E')
G = csdl.Variable(value=26E9, name='G')
density = csdl.Variable(value=2700, name='density')
nu = csdl.Variable(value=0.33, name='nu')
aluminum = cd.materials.IsotropicMaterial(name='aluminum', E=E, G=G, 
                                            density=density, nu=nu)

# Define thickness functions
# The ribs and spars have a constant thickness, while the skin has a variable thickness that we will optimize
thickness_fs = fs.ConstantSpace(2)
skin_fs = fs.BSplineSpace(2, (2,1), (5,2))
r_skin_fss = right_wing_oml.create_parallel_space(skin_fs)
skin_t_coeffs, skin_fn = r_skin_fss.initialize_function(1, value=skin_thickness)
spar_fn = fs.Function(thickness_fs, spar_thickness)
rib_fn = fs.Function(thickness_fs, rib_thickness)

# correlate the left and right wing skin thickness functions - want symmetry
oml_lr_map = {rind:lind for rind, lind in zip(right_wing_oml.functions, left_wing_oml.functions)}
wing.quantities.oml_lr_map = oml_lr_map

# build function set out of the thickness functions
functions = skin_fn.functions.copy()
for ind in wing.geometry.functions:
    name = wing.geometry.function_names[ind]
    if "spar" in name:
        functions[ind] = spar_fn
    elif "rib" in name:
        functions[ind] = rib_fn

for rind, lind in oml_lr_map.items():
    # the v coord is flipped left to right
    functions[lind] = fs.Function(skin_fs, functions[rind].coefficients[:,::-1,:])


# print("4 ------------------")

thickness_function_set = fs.FunctionSet(functions)
wing.quantities.material_properties.set_material(aluminum, thickness_function_set)

# set skin thickness as a design variable
skin_t_coeffs.set_as_design_variable(upper=0.05, lower=0.0001, scaler=5e2)
skin_t_coeffs.add_name('skin_thickness')

# print("5 ------------------")

# Spaces for states
# pressure
pressure_function_space = fs.IDWFunctionSpace(num_parametric_dimensions=2, order=4, grid_size=(240, 40), conserve=False)
indexed_pressue_function_space = wing_oml.create_parallel_space(pressure_function_space)
wing.quantities.pressure_space = indexed_pressue_function_space

# displacement
displacement_space = fs.BSplineSpace(2, (1,1), (3,3))
wing.quantities.displacement_space = wing_geometry.create_parallel_space(
                                                displacement_space)
wing.quantities.oml_displacement_space = wing_oml.create_parallel_space(
                                                displacement_space)

# print("6 ------------------")

# meshing
mesh_container = base_config.mesh_container

# vlm mesh
vlm_mesh = cd.mesh.VLMMesh()
wing_chord_surface = cd.mesh.make_vlm_surface(
    wing, 40, 1, LE_interp="ellipse", TE_interp="ellipse", 
    spacing_spanwise="cosine", ignore_camber=True, plot=False,
)
wing_chord_surface.project_airfoil_points(oml_geometry=wing_oml)
vlm_mesh.discretizations["wing_chord_surface"] = wing_chord_surface
mesh_container["vlm_mesh"] = vlm_mesh

# print("7 ------------------")

# # beam mesh
# beam_mesh = cd.mesh.BeamMesh()
# beam_disc = cd.mesh.make_1d_box_beam(wing, num_ribs*2-1, 0.5, project_spars=True, spar_search_names=['1', '2'], make_half_beam=True)
# beam_mesh.discretizations["wing"] = beam_disc
# mesh_container["beam_mesh"] = beam_mesh

# # shell meshing
# if shell:
# import shell mesh
    # file_path = os.path.dirname(os.path.abspath(__file__))
    # wing_shell_mesh = cd.mesh.import_shell_mesh(
    #     file_path+"/"+mesh_fname, 
    #     right_wing,
    #     rescale=[1e-3, 1e-3, 1e-3],
    #     grid_search_n=5,
    #     priority_inds=[i for i in right_wing_oml.functions],
    #     priority_eps=3e-6,
    # )
    # process_elements(wing_shell_mesh, right_wing_oml, right_wing_ribs, right_wing_spars)

    # nodes = wing_shell_mesh.nodal_coordinates
    # connectivity = wing_shell_mesh.connectivity
    # filename = mesh_fname+"_reconstructed.xdmf"
    # if os.path.isfile(filename) and False:
    #     wing_shell_mesh_fenics = readFEAMesh(filename)
    # else:
    #     # Reconstruct the mesh using the projected nodes and connectivity
    #     wing_shell_mesh_fenics = reconstructFEAMesh(filename, 
    #                                                 nodes.value, connectivity)
    # # store the xdmf mesh object for shell analysis
    # wing_shell_mesh.fea_mesh = wing_shell_mesh_fenics

    # wing_shell_mesh_cd = cd.mesh.ShellMesh()
    # wing_shell_mesh_cd.discretizations['wing'] = wing_shell_mesh
    # mesh_container['shell_mesh'] = wing_shell_mesh_cd

# def define_conditions(caddee: cd.CADDEE):
conditions = caddee.conditions
base_config = caddee.base_configuration

# Cruise
pitch_angle = csdl.Variable(shape=(1, ), value=np.deg2rad(2.69268269))
cruise = cd.aircraft.conditions.CruiseCondition(
    altitude=1,
    range=70 * cd.Units.length.kilometer_to_m,
    mach_number=0.18,
    pitch_angle=pitch_angle, # np.linspace(np.deg2rad(-4), np.deg2rad(10), 15),
)
cruise.quantities.pitch_angle = pitch_angle
cruise.configuration = base_config.copy()
conditions["cruise"] = cruise

# print("8 ------------------")


# raise RuntimeError

def run_vlm(mesh_containers, conditions):

    # print("run vlm 1.0 ------------------")

    # implicit displacement input
    wing = conditions[0].configuration.system.comps["airframe"].comps["wing"]
    displacement_space:fs.FunctionSetSpace = wing.quantities.oml_displacement_space
    implicit_disp_coeffs = []
    implicit_disp_fns = []
    for i in range(len(mesh_containers)):
        coeffs, function = displacement_space.initialize_function(3, implicit=True)
        implicit_disp_coeffs.append(coeffs)
        implicit_disp_fns.append(function)

    # set up VLM analysis
    nodal_coords = []
    nodal_vels = []

    # print("run vlm 2.0 ------------------")

    for mesh_container, condition, disp_fn in zip(mesh_containers, conditions, implicit_disp_fns):
        transfer_mesh_para = disp_fn.generate_parametric_grid((5, 5))
        transfer_mesh_phys = wing.geometry.evaluate(transfer_mesh_para)
        transfer_mesh_disp = disp_fn.evaluate(transfer_mesh_para)
        
        wing_lattice = mesh_container["vlm_mesh"].discretizations["wing_chord_surface"]
        wing_lattic_coords = wing_lattice.nodal_coordinates

        map = fs.NodalMap()
        weights = map.evaluate(csdl.reshape(wing_lattic_coords, (np.prod(wing_lattic_coords.shape[0:-1]), 3)), transfer_mesh_phys)
        wing_camber_mesh_displacement = (weights @ transfer_mesh_disp).reshape(wing_lattic_coords.shape)
        
        nodal_coords.append(wing_lattic_coords + wing_camber_mesh_displacement)
        nodal_vels.append(wing_lattice.nodal_velocities) # the velocities should be the same for every node in this case

    if len(nodal_coords) == 1:
        nodal_coordinates = nodal_coords[0]
        nodal_velocities = nodal_vels[0]
    else:
        nodal_coordinates = csdl.vstack(nodal_coords)
        nodal_velocities = csdl.vstack(nodal_vels)

    # print("run vlm 3.0 ------------------")

    # Add an airfoil model
    nasa_langley_airfoil_maker = ThreeDAirfoilMLModelMaker(
        airfoil_name="ls417",
        aoa_range=np.linspace(-12, 16, 50), 
        reynolds_range=[1e5, 2e5, 5e5, 1e6, 2e6, 4e6, 7e6, 10e6], 
        mach_range=[0., 0.2, 0.3, 0.4, 0.5, 0.6],
        num_interp=120,
    )

    # print("run vlm 3.1 ------------------")

    Cl_model = nasa_langley_airfoil_maker.get_airfoil_model(quantities=["Cl"])
    # print("run vlm 3.2 ------------------")
    Cd_model = nasa_langley_airfoil_maker.get_airfoil_model(quantities=["Cd"])
    # print("run vlm 3.3 ------------------")
    # Causes seg fault on Mac
    Cp_model = nasa_langley_airfoil_maker.get_airfoil_model(quantities=["Cp"])
    # print("run vlm 3.4 ------------------")
    alpha_stall_model = nasa_langley_airfoil_maker.get_airfoil_model(quantities=["alpha_Cl_min_max"])

    # print("run vlm 4.0 ------------------")

    vlm_outputs = vlm_solver(
        mesh_list=[nodal_coordinates],
        mesh_velocity_list=[nodal_velocities],
        atmos_states=conditions[0].quantities.atmos_states,
        airfoil_alpha_stall_models=[alpha_stall_model],
        airfoil_Cd_models=[Cd_model],
        airfoil_Cl_models=[Cl_model],
        airfoil_Cp_models=[Cp_model], 
    )

    # print("run vlm 5.0 ------------------")

    return vlm_outputs, implicit_disp_coeffs


def fit_pressure_fn(mesh_container, condition, spanwise_Cp):
    wing = condition.configuration.system.comps["airframe"].comps["wing"]
    vlm_mesh = mesh_container["vlm_mesh"]
    wing_lattice = vlm_mesh.discretizations["wing_chord_surface"]
    rho = condition.quantities.atmos_states.density
    v_inf = condition.parameters.speed
    airfoil_upper_nodes = wing_lattice._airfoil_upper_para
    airfoil_lower_nodes = wing_lattice._airfoil_lower_para

    spanwise_p = spanwise_Cp * 0.5 * rho * v_inf**2
    spanwise_p = csdl.blockmat([[spanwise_p[:, 0:120].T()], [spanwise_p[:, 120:].T()]])

    pressure_indexed_space : fs.FunctionSetSpace = wing.quantities.pressure_space
    pressure_function = pressure_indexed_space.fit_function_set(
        values=spanwise_p.reshape((-1, 1)), parametric_coordinates=airfoil_upper_nodes+airfoil_lower_nodes,
        regularization_parameter=1e-4,
    )

    return pressure_function


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, *args):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# def define_analysis(caddee: cd.CADDEE):
conditions = caddee.conditions
wing = caddee.base_configuration.system.comps["airframe"].comps["wing"]

# finalize meshes
cruise:cd.aircraft.conditions.CruiseCondition = conditions["cruise"]
cruise.finalize_meshes()
mesh_container = cruise.configuration.mesh_container


# print("9 ------------------")

# run VLM 
vlm_outputs, implicit_disp_coeffs = run_vlm([mesh_container], [cruise])
forces = vlm_outputs.surface_force[0][0]
Cp = vlm_outputs.surface_spanwise_Cp[0][0]

# print("10 ------------------")

# trim the aircraft
pitch_angle = cruise.quantities.pitch_angle
z_force = -forces[2]*csdl.cos(pitch_angle) + forces[0]*csdl.sin(pitch_angle)
residual = z_force - mass*9.81*load_factor
trim_solver = csdl.nonlinear_solvers.BracketedSearch()
trim_solver.add_state(pitch_angle, residual, (np.deg2rad(0), np.deg2rad(10)))
with HiddenPrints():
    # The vlm solver prints stuff every time it's called and it annoys me
    trim_solver.run()

# fit pressure function to trimmed VLM results
# we can actually do this before the trim if we wanted, it would be updated automatically
pressure_fn = fit_pressure_fn(mesh_container, cruise, Cp)


# Transfer pressure to IGA outer surfaces
spline2_para_coords = kl_shell_model.splines[2].mesh.coordinates()
oml_nodes_parametric11 = [(11, coord) for coord in spline2_para_coords]
spline3_para_coords = kl_shell_model.splines[3].mesh.coordinates()
oml_nodes_parametric12 = [(12, coord) for coord in spline3_para_coords]

pressure_magnitudes11 = pressure_fn.evaluate(oml_nodes_parametric11)
pressure_normals11 = wing.geometry.evaluate_normals(oml_nodes_parametric11)
oml_pressures2 = pressure_normals11*csdl.expand(pressure_magnitudes11, pressure_normals11.shape, 'i->ij')

pressure_magnitudes12 = pressure_fn.evaluate(oml_nodes_parametric12)
pressure_normals12 = wing.geometry.evaluate_normals(oml_nodes_parametric12)
oml_pressures3 = pressure_normals12*csdl.expand(pressure_magnitudes12, pressure_normals11.shape, 'i->ij')

# wing_pressure = csdl.VariableGroup()
# wing_pressure.spline2_pressure = oml_pressures2
# wing_pressure.spline3_pressure = oml_pressures3
pressure_list = [oml_pressures2, oml_pressures3]
pressure_inds = [2,3]

# Transfer thickness
material = wing.quantities.material_properties.material
names_list = []
for key in right_wing.function_names:
    names_list += [key]
coords_list = []
for i in range(num_surfs):
    para_coords = kl_shell_model.splines[i].mesh.coordinates()
    coords_list += [(names_list[i], coord) for coord in para_coords]
h_th = wing.quantities.material_properties.evaluate_thickness(coords_list)

# Evaluate in IGA shell
kl_shell_model.evaluate(pressure_list, h_th, E, nu, density, pressure_inds=pressure_inds)


raise RuntimeError











# Run structural analysis
if shell:
    displacement, shell_outputs = run_shell(mesh_container, cruise, pressure_fn, rec=True)
    max_stress:csdl.Variable = shell_outputs.aggregated_stress
    wing_mass:csdl.Variable = shell_outputs.mass
else:
    displacement, max_stress, wing_mass = run_beam(mesh_container, cruise, pressure_fn)

mirror_function(displacement, wing.quantities.oml_lr_map)

max_stress.set_as_constraint(upper=stress_bound, scaler=1e-8)
max_stress.add_name('max_stress')
wing_mass.set_as_objective(scaler=1e-2)
wing_mass.add_name('wing_mass')
wing.quantities.oml_displacement = displacement
wing.quantities.pressure_function = pressure_fn

# Solver for aerostructural coupling
# we iterate between the VLM and structural analysis until the displacement converges
if couple:
    coeffs = displacement.stack_coefficients()
    disp_res = implicit_disp_coeffs[0] - coeffs
    solver = csdl.nonlinear_solvers.Jacobi(max_iter=10, tolerance=1e-6)
    solver.add_state(implicit_disp_coeffs[0], disp_res)
    solver.run()








def run_shell(mesh_container, condition:cd.aircraft.conditions.CruiseCondition, pressure_function, rec=False):
    wing = condition.configuration.system.comps["airframe"].comps["wing"]
    
    # Shell
    pav_shell_mesh = mesh_container["shell_mesh"]
    wing_shell_mesh = pav_shell_mesh.discretizations['wing']
    nodes = wing_shell_mesh.nodal_coordinates
    nodes_parametric = wing_shell_mesh.nodes_parametric
    connectivity = wing_shell_mesh.connectivity
    wing_shell_mesh_fenics = wing_shell_mesh.fea_mesh
    element_centers_parametric = wing_shell_mesh.element_centers_parametric
    oml_node_inds = wing_shell_mesh.oml_node_inds
    oml_nodes_parametric = wing_shell_mesh.oml_nodes_parametric
    node_disp = wing.geometry.evaluate(nodes_parametric) - nodes.reshape((-1,3))

    # transfer aero peressures
    pressure_magnitudes = pressure_function.evaluate(oml_nodes_parametric)
    pressure_normals = wing.geometry.evaluate_normals(oml_nodes_parametric)
    oml_pressures = pressure_normals*csdl.expand(pressure_magnitudes, pressure_normals.shape, 'i->ij')

    shell_pressures = csdl.Variable(value=np.zeros(nodes.shape[1:]))
    shell_pressures = shell_pressures.set(csdl.slice[oml_node_inds], oml_pressures)
    f_input = shell_pressures

    material = wing.quantities.material_properties.material
    element_thicknesses = wing.quantities.material_properties.evaluate_thickness(element_centers_parametric)

    E0, nu0, G0 = material.from_compliance()
    density0 = material.density

    # create node-wise material properties
    nel = connectivity.shape[0]
    E = E0*np.ones(nel)
    E.add_name('E')
    nu = nu0*np.ones(nel)
    nu.add_name('nu')
    density = density0*np.ones(nel)
    density.add_name('density')

    # define boundary conditions
    def clamped_boundary(x):
        eps = 1e-3
        return np.less_equal(x[1], eps)

    # run solver
    shell_model = RMShellModel(mesh=wing_shell_mesh_fenics,
                               shell_bc_func=clamped_boundary,
                               element_wise_material=True,
                               record=rec, # record=true doesn't work with 2 shell instances
                               rho=4)
    shell_outputs = shell_model.evaluate(f_input, 
                                         element_thicknesses, E, nu, density,
                                         node_disp,
                                         debug_mode=False)
    disp_extracted = shell_outputs.disp_extracted

    # fit displacement function
    oml_displacement_space:fs.FunctionSetSpace = wing.quantities.oml_displacement_space
    oml_displacement_function = oml_displacement_space.fit_function_set(disp_extracted[oml_node_inds], oml_nodes_parametric)

    return oml_displacement_function, shell_outputs

# def run_beam(mesh_container, condition, pressure_fn):
#     wing = condition.configuration.system.comps["airframe"].comps["wing"]
#     beam_mesh = mesh_container["beam_mesh"]
#     wing_box = beam_mesh.discretizations["wing"]
#     aluminum = wing.quantities.material_properties.material

#     box_beam = mesh_container["beam_mesh"].discretizations["wing"]
#     beam_nodes = box_beam.nodal_coordinates

#     box_cs = af.CSBox(
#         ttop=wing_box.top_skin_thickness,
#         tbot=wing_box.bottom_skin_thickness,
#         tweb=wing_box.shear_web_thickness,
#         height=wing_box.beam_height,
#         width=wing_box.beam_width,
#     )
#     beam = af.Beam(
#         name="wing_beam", 
#         mesh=wing_box.nodal_coordinates, 
#         cs=box_cs,
#         material=aluminum,
#     )

#     # transfer aero forces
#     right_wing_oml_inds = list(wing.quantities.right_wing_oml.functions)
#     force_magnitudes, force_para_coords = pressure_fn.integrate(wing.geometry, grid_n=30, indices=right_wing_oml_inds)
#     force_coords = wing.geometry.evaluate(force_para_coords)
#     force_normals = wing.geometry.evaluate_normals(force_para_coords)
#     force_vectors = force_normals*csdl.expand(force_magnitudes.flatten(), force_normals.shape, 'i->ij')

#     mapper = fs.NodalMap(weight_eps=5)
#     force_map = mapper.evaluate(force_coords, beam_nodes.reshape((-1, 3)))
#     beam_forces = force_map.T() @ force_vectors

#     beam_forces_plus_moments = csdl.Variable(shape=(beam_forces.shape[0], 6), value=0)
#     beam_forces_plus_moments = beam_forces_plus_moments.set(
#         csdl.slice[:, 0:3], beam_forces
#     )

#     beam.add_boundary_condition(node=0, dof=[1, 1, 1, 1, 1, 1])
#     beam.add_load(beam_forces_plus_moments)

#     frame = af.Frame()
#     frame.add_beam(beam)

#     struct_solution = frame.evaluate()

#     beam_displacement = struct_solution.get_displacement(beam)
#     beam_stress = struct_solution.get_stress(beam)

#     # transfer displacements to oml
#     mapper = fs.NodalMap(weight_eps=5, weight_to_be_normalized=True)
#     oml_mesh_parametric = wing.quantities.right_wing_oml.generate_parametric_grid((10, 10))
#     oml_mesh_phys = wing.geometry.evaluate(oml_mesh_parametric)
#     disp_map = mapper.evaluate(oml_mesh_phys, beam_nodes.reshape((-1, 3)))
#     disp = disp_map @ beam_displacement
#     oml_displacement_space:fs.FunctionSetSpace = wing.quantities.oml_displacement_space
#     oml_displacement_function = oml_displacement_space.fit_function_set(disp, oml_mesh_parametric)

#     # get wing mass
#     mass_fn = af.FrameMass()
#     mass_fn.add_beam(beam)
#     wing_mass_prop = mass_fn.evaluate()
#     wing_mass = wing_mass_prop.mass

#     return oml_displacement_function, beam_stress, wing_mass

def process_elements(wing_shell_mesh, right_wing_oml, right_wing_ribs, right_wing_spars):
    """
    Process the elements of the shell mesh to determine the type of element (rib, spar, skin)
    """

    nodes = wing_shell_mesh.nodal_coordinates
    connectivity = wing_shell_mesh.connectivity

    # figure out type of surface each element is (rib/spar/skin)
    grid_n = 20
    oml_errors = np.linalg.norm(right_wing_oml.evaluate(right_wing_oml.project(nodes.value, grid_search_density_parameter=grid_n, plot=False), non_csdl=True) - nodes.value, axis=1)
    rib_errors = np.linalg.norm(right_wing_ribs.evaluate(right_wing_ribs.project(nodes.value, grid_search_density_parameter=grid_n, plot=False), non_csdl=True) - nodes.value, axis=1)
    spar_errors = np.linalg.norm(right_wing_spars.evaluate(right_wing_spars.project(nodes.value, grid_search_density_parameter=grid_n, plot=False), non_csdl=True) - nodes.value, axis=1)

    element_centers = np.array([np.mean(nodes.value[connectivity[i].astype(int)], axis=0) for i in range(connectivity.shape[0])])

    rib_correction = 1e-4
    element_centers_parametric = []
    oml_inds = []
    rib_inds = []
    spar_inds = []
    for i in range(connectivity.shape[0]):
        inds = connectivity[i].astype(int)
        
        # rib projection is messed up so we use an alternitive approach - if all the points are in an x-z plane, it's a rib
        if np.all(np.isclose(nodes.value[inds, 1], nodes.value[inds[0], 1], atol=rib_correction)):
            rib_inds.append(i)
            continue

        errors = [np.sum(oml_errors[inds]), np.sum(rib_errors[inds]), np.sum(spar_errors[inds])]
        ind = np.argmin(errors)
        if ind == 0:
            oml_inds.append(i)
        elif ind == 1:
            rib_inds.append(i)
        elif ind == 2:
            spar_inds.append(i)
        else:
            raise ValueError('Error in determining element type')

    oml_centers = right_wing_oml.project(element_centers[oml_inds], grid_search_density_parameter=5, plot=False, force_reprojection=True)
    rib_centers = right_wing_ribs.project(element_centers[rib_inds], grid_search_density_parameter=5, plot=False, force_reprojection=True)
    spar_centers = right_wing_spars.project(element_centers[spar_inds], grid_search_density_parameter=5, plot=False, force_reprojection=True)
    oml_inds_copy = oml_inds.copy()

    for i in range(connectivity.shape[0]):
        if oml_inds and oml_inds[0] == i:
            element_centers_parametric.append(oml_centers.pop(0))
            oml_inds.pop(0)
        elif rib_inds and rib_inds[0] == i:
            element_centers_parametric.append(rib_centers.pop(0))
            rib_inds.pop(0)
        elif spar_inds and spar_inds[0] == i:
            element_centers_parametric.append(spar_centers.pop(0))
            spar_inds.pop(0)
        else:
            raise ValueError('Error in sorting element centers')
        
    wing_shell_mesh.element_centers_parametric = element_centers_parametric
    
    oml_node_inds = []
    for c_ind in oml_inds_copy:
        n_inds = connectivity[c_ind].astype(int)
        for n_ind in n_inds:
            if not n_ind in oml_node_inds:
                oml_node_inds.append(n_ind)

    oml_nodes_parametric = right_wing_oml.project(nodes.value[oml_node_inds], grid_search_density_parameter=5)
    wing_shell_mesh.oml_node_inds = oml_node_inds
    wing_shell_mesh.oml_nodes_parametric = oml_nodes_parametric
    wing_shell_mesh.oml_el_inds = oml_inds_copy

def mirror_function(displacement, oml_lr_map):
    for rind, lind in oml_lr_map.items():
        displacement.functions[lind].coefficients = displacement.functions[rind].coefficients[:,::-1,:]



def load_dv_values(fname, group):
    inputs = csdl.inline_import(fname, group)
    recorder = csdl.get_current_recorder()
    dvs = recorder.design_variables
    for var in dvs:
        var_name = var.name
        var.set_value(inputs[var_name].value)
        scale = 1/np.linalg.norm(inputs[var_name].value)
        dvs[var] = (scale, dvs[var][1], dvs[var][2])

define_base_config(caddee)
rec.inline = inline
define_conditions(caddee)
define_analysis(caddee)
csdl.save_optimization_variables()

fname = 'structural_opt_beam_test'
if optimize:
    from modopt import CSDLAlphaProblem
    from modopt import PySLSQP

    
    # If you have a GPU, you can set gpu=True - but it may not be faster
    # I think this is because the ml airfoil model can't be run on the GPU when Jax is using it
    sim = csdl.experimental.JaxSimulator(rec, gpu=False, save_on_update=True, filename=fname)
    
    # If you don't have jax installed, you can use the PySimulator instead (it's slower)
    # To install jax, see https://jax.readthedocs.io/en/latest/installation.html
    # sim = csdl.experimental.PySimulator(rec)

    # It's a good idea to check the totals of the simulator before running the optimizer
    # sim.check_totals()
    # exit()

    prob = CSDLAlphaProblem(problem_name=fname, simulator=sim)
    optimizer = PySLSQP(prob, solver_options={'maxiter':200, 'acc':1e-6})

    # Solve your optimization problem
    optimizer.solve()
    optimizer.print_results()
    csdl.inline_export(fname+'_final')


# Plotting
# load dv values and perform an inline execution to get the final results
load_dv_values(fname+'_final.hdf5', 'inline')
rec.execute()
wing = caddee.base_configuration.system.comps["airframe"].comps["wing"]
mesh = wing.quantities.oml.plot_but_good(color=wing.quantities.material_properties.thickness)
wing.quantities.oml.plot_but_good(color=wing.quantities.oml_displacement)
wing.quantities.oml.plot_but_good(color=wing.quantities.pressure_function)
