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

##### Obtain Control points and knot vectors for IGA ####
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
############################################################

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

thickness_function_set = fs.FunctionSet(functions)
wing.quantities.material_properties.set_material(aluminum, thickness_function_set)

# set skin thickness as a design variable
skin_t_coeffs.set_as_design_variable(upper=0.05, lower=0.0001, scaler=5e2)
skin_t_coeffs.add_name('skin_thickness')


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


def run_vlm(mesh_containers, conditions):


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


    # Add an airfoil model
    nasa_langley_airfoil_maker = ThreeDAirfoilMLModelMaker(
        airfoil_name="ls417",
        aoa_range=np.linspace(-12, 16, 50), 
        reynolds_range=[1e5, 2e5, 5e5, 1e6, 2e6, 4e6, 7e6, 10e6], 
        mach_range=[0., 0.2, 0.3, 0.4, 0.5, 0.6],
        num_interp=120,
    )


    Cl_model = nasa_langley_airfoil_maker.get_airfoil_model(quantities=["Cl"])
    Cd_model = nasa_langley_airfoil_maker.get_airfoil_model(quantities=["Cd"])
    # Causes seg fault on Mac
    Cp_model = nasa_langley_airfoil_maker.get_airfoil_model(quantities=["Cp"])
    alpha_stall_model = nasa_langley_airfoil_maker.get_airfoil_model(quantities=["alpha_Cl_min_max"])


    vlm_outputs = vlm_solver(
        mesh_list=[nodal_coordinates],
        mesh_velocity_list=[nodal_velocities],
        atmos_states=conditions[0].quantities.atmos_states,
        airfoil_alpha_stall_models=[alpha_stall_model],
        airfoil_Cd_models=[Cd_model],
        airfoil_Cl_models=[Cl_model],
        airfoil_Cp_models=[Cp_model], 
    )


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

# run VLM 
vlm_outputs, implicit_disp_coeffs = run_vlm([mesh_container], [cruise])
forces = vlm_outputs.surface_force[0][0]
Cp = vlm_outputs.surface_spanwise_Cp[0][0]


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

#####################################################################
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