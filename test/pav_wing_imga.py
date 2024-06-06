"""
Structural analysis of the PAV wing
with the Reissner--Mindlin shell model

-----------------------------------------------------------
Test the integration of m3l and shell model
-----------------------------------------------------------
"""

## Caddee
from caddee.core.caddee_core.system_representation.component.component import LiftingSurface
from caddee.core.caddee_core.system_representation.system_primitive.system_primitive import SystemPrimitive
import caddee.core.primitives.bsplines.bspline_functions as bsf
from caddee import GEOMETRY_FILES_FOLDER
import caddee.api as cd

# Other lsdo lab stuff
import array_mapper as am
import m3l
from m3l.utils.utils import index_functions
import lsdo_geo as lg
from python_csdl_backend import Simulator

## Other stuff
import numpy as np
import pathlib
import sys


sys.setrecursionlimit(100000)

in2m = 0.0254
ft2m = 0.3048
lbs2kg = 0.453592
psf2pa = 50

do_plots = False
force_reprojection = False # set this to false after first run, set to true if you get weird shape mismatch errors

# CADDEE geometry initialization
caddee = cd.CADDEE()
caddee.system_model = system_model = cd.SystemModel()
caddee.system_representation = sys_rep = cd.SystemRepresentation()
caddee.system_parameterization = sys_param = cd.SystemParameterization(system_representation=sys_rep)
spatial_rep = sys_rep.spatial_representation


## Generate geometry

# import initial geomrty
file_name = '/pav_wing/pav_new_SI.stp'
cfile = str(pathlib.Path(__file__).parent.resolve())
spatial_rep.import_file(file_name=cfile+file_name)
spatial_rep.refit_geometry(file_name=cfile+file_name)


# region Geometry processing/id

# fix naming
primitives_new = {}
indicies_new = {}
for key, item in spatial_rep.primitives.items():
    item.name = item.name.replace(' ','_').replace(',','')
    primitives_new[key.replace(' ','_').replace(',','')] = item

for key, item in spatial_rep.primitive_indices.items():
    indicies_new[key.replace(' ','_').replace(',','')] = item

spatial_rep.primitives = primitives_new
spatial_rep.primitive_indices = indicies_new

wing_primitive_names = list(spatial_rep.get_primitives(search_names=['Wing']).keys())

# Manual surface identification
if False:
    for key in wing_primitive_names:
        surfaces = wing_primitive_names
        surfaces.remove(key)
        print(key)
        spatial_rep.plot(primitives=surfaces)



# make wing components
left_wing_names = []
left_wing_top_names = []
left_wing_bottom_names = []
left_wing_te_top_names = []
left_wing_te_bottom_names = []
for i in range(22+172,37+172):
    surf_name = 'Wing_1_' + str(i)
    left_wing_names.append(surf_name)
    if i%4 == 2:
        left_wing_te_bottom_names.append(surf_name)
    elif i%4 == 3:
        left_wing_bottom_names.append(surf_name)
    elif i%4 == 0:
        left_wing_top_names.append(surf_name)
    else:
        left_wing_te_top_names.append(surf_name)



right_wing_names = []
right_wing_top_names = []
right_wing_bottom_names = []
right_wing_te_top_names = []
right_wing_te_bottom_names = []
for i in range(174,189):
    surf_name = 'Wing_0_' + str(i)
    right_wing_names.append(surf_name)
    if i%4 == 2:
        right_wing_te_bottom_names.append(surf_name)
    elif i%4 == 3:
        right_wing_bottom_names.append(surf_name)
    elif i%4 == 0:
        right_wing_top_names.append(surf_name)
    else:
        right_wing_te_top_names.append(surf_name)

# Important points from openVSP
root_te = np.array([15.170, 0., 1.961]) * ft2m
root_le = np.array([8.800, 0, 1.989]) * ft2m
l_tip_te = np.array([11.300, -14.000, 1.978]) * ft2m
l_tip_le = np.array([8.796, -14.000, 1.989]) * ft2m
r_tip_te = np.array([11.300, 14.000, 1.978]) * ft2m
r_tip_le = np.array([8.796, 14.000, 1.989]) * ft2m



# endregion

# region Components

wing_left = LiftingSurface(name='wing_left', spatial_representation=spatial_rep, primitive_names=left_wing_names)
wing_left_top = LiftingSurface(name='wing_left_top', spatial_representation=spatial_rep, primitive_names=left_wing_top_names)
wing_left_bottom = LiftingSurface(name='wing_left_bottom', spatial_representation=spatial_rep, primitive_names=left_wing_bottom_names)

wing_oml = LiftingSurface(name='wing_oml', spatial_representation=spatial_rep, primitive_names=left_wing_names+right_wing_names)
wing_top = LiftingSurface(name='wing_top', spatial_representation=spatial_rep, primitive_names=left_wing_top_names+right_wing_top_names)
wing_bottom = LiftingSurface(name='wing_bottom', spatial_representation=spatial_rep, primitive_names=left_wing_bottom_names+right_wing_bottom_names)
wing_te = LiftingSurface(name='wing_te', spatial_representation=spatial_rep, primitive_names=left_wing_te_top_names+left_wing_te_bottom_names+right_wing_te_top_names+right_wing_te_bottom_names)

# endregion

# region Internal structure generation

structural_left_wing_names = []

# projections for internal structure
num_pts = 10
spar_rib_spacing_ratio = 3
num_rib_pts = 20

tip_te = l_tip_te
tip_le = l_tip_le

root_25 = (3*root_le+root_te)/4
root_75 = (root_le+3*root_te)/4
tip_25 = (3*tip_le+tip_te)/4
tip_75 = (tip_le+3*tip_te)/4


avg_spar_spacing = (np.linalg.norm(root_25-root_75)+np.linalg.norm(tip_25-tip_75))/2
half_span = root_le[1] - tip_le[1]
num_ribs = int(spar_rib_spacing_ratio*half_span/avg_spar_spacing)+1

f_spar_projection_points = np.linspace(root_25, tip_25, num_ribs)
r_spar_projection_points = np.linspace(root_75, tip_75, num_ribs)

rib_projection_points = np.linspace(f_spar_projection_points, r_spar_projection_points, num_rib_pts)

f_spar_top = wing_left_top.project(f_spar_projection_points, plot=do_plots, force_reprojection=force_reprojection)
f_spar_bottom = wing_left_bottom.project(f_spar_projection_points, plot=do_plots, force_reprojection=force_reprojection)

r_spar_top = wing_left_top.project(r_spar_projection_points, plot=do_plots, force_reprojection=force_reprojection)
r_spar_bottom = wing_left_bottom.project(r_spar_projection_points, plot=do_plots, force_reprojection=force_reprojection)

ribs_top = wing_left_top.project(rib_projection_points, direction=[0.,0.,1.], plot=do_plots, grid_search_n=100, force_reprojection=force_reprojection)
ribs_bottom = wing_left_bottom.project(rib_projection_points, direction=[0.,0.,1.], plot=do_plots, grid_search_n=100, force_reprojection=force_reprojection)


# make multi-patch spars - for coherence
n_cp = (2,2)
order = (2,)

for i in range(num_ribs-1):
    f_spar_points = np.zeros((2,2,3))
    f_spar_points[0,:,:] = ribs_top.value[0,(i,i+1),:]
    f_spar_points[1,:,:] = ribs_bottom.value[0,(i,i+1),:]
    f_spar_bspline = bsf.fit_bspline(f_spar_points, num_control_points=n_cp, order=order)
    f_spar = SystemPrimitive('f_spar_' + str(i), f_spar_bspline)
    spatial_rep.primitives[f_spar.name] = f_spar
    structural_left_wing_names.append(f_spar.name)

    r_spar_points = np.zeros((2,2,3))
    r_spar_points[0,:,:] = ribs_top.value[-1,(i,i+1),:]
    r_spar_points[1,:,:] = ribs_bottom.value[-1,(i,i+1),:]
    r_spar_bspline = bsf.fit_bspline(r_spar_points, num_control_points=n_cp, order=order)
    r_spar = SystemPrimitive('r_spar_' + str(i), r_spar_bspline)
    spatial_rep.primitives[r_spar.name] = r_spar
    structural_left_wing_names.append(r_spar.name)

# make ribs
n_cp_rib = (num_rib_pts,2)
order_rib = (2,)

for i in range(num_ribs):
    rib_points = np.zeros((num_rib_pts, 2, 3))
    rib_points[:,0,:] = ribs_top.value[:,i,:]
    rib_points[:,1,:] = ribs_bottom.value[:,i,:]
    rib_bspline = bsf.fit_bspline(rib_points, num_control_points=n_cp_rib, order = order_rib)
    rib = SystemPrimitive('rib_' + str(i), rib_bspline)
    spatial_rep.primitives[rib.name] = rib
    structural_left_wing_names.append(rib.name)

# make surface panels
n_cp = (num_rib_pts,2)
order = (2,)



surface_dict = {}
for i in range(num_ribs-1):
    t_panel_points = ribs_top.value[:,(i,i+1),:]
    t_panel_bspline = bsf.fit_bspline(t_panel_points, num_control_points=n_cp, order=order)
    t_panel = SystemPrimitive('t_panel_' + str(i), t_panel_bspline)
    surface_dict[t_panel.name] = t_panel
    structural_left_wing_names.append(t_panel.name)

    b_panel_points = ribs_bottom.value[:,(i,i+1),:]
    b_panel_bspline = bsf.fit_bspline(b_panel_points, num_control_points=n_cp, order=order)
    b_panel = SystemPrimitive('b_panel_' + str(i), b_panel_bspline)
    surface_dict[b_panel.name] = b_panel
    structural_left_wing_names.append(b_panel.name)



surface_dict.update(spatial_rep.primitives)
spatial_rep.primitives = surface_dict

spatial_rep.assemble()
# spatial_rep.plot(plot_types=['wireframe'])
# endregion

# region control points and knots for imga


knots_u = []
knots_v = []
order_u = []
order_v = []
control_points = []
for name in structural_left_wing_names:
    primitive = spatial_rep.get_primitives([name])[name].geometry_primitive
    control_points.append(primitive.control_points)
    knots_u.append(primitive.knots_u)
    knots_v.append(primitive.knots_v)
    order_u.append(primitive.order_u)
    order_v.append(primitive.order_v)
# Han - use these lists for whatever you need
print(aaa)
exit()

# endregion


# region geometry function

coefficients = {}
geo_space = lg.BSplineSpace(name='geo_base_space',
                            order=(4,4),
                            control_points_shape=(25,25))
wing_oml_geo = index_functions(left_wing_names+right_wing_names, 'wing_oml_geo', geo_space, 3)
spatial_rep = sys_rep.spatial_representation
for name in left_wing_names+right_wing_names:
    primitive = spatial_rep.get_primitives([name])[name].geometry_primitive
    coefficients[name] = m3l.Variable(name = name + '_geo_coefficients', shape = primitive.control_points.shape, value = primitive.control_points)

wing_oml_geo.coefficients = coefficients

# endregion

# region thickness functions

# this generates a single thickness field over the whole wing with a constant value t.
# in principle, this can be modified to whatever you want

t1 = 0.01 # thickness of material 1
t2 = 0.02 # thickness of material 2
order = 2
shape = 3
space_t = lg.BSplineSpace(name='thickness_base_space',
                        order=(order, order),
                        control_points_shape=(shape, shape))
wing_thickness_mat1 = index_functions(left_wing_names+right_wing_names, 'wing_thickness', space_t, 1, value=t1*np.ones((shape,shape)))

wing_thickness_mat2 = index_functions(left_wing_names+right_wing_names, 'wing_thickness', space_t, 1, value=t2*np.ones((shape,shape)))

# endregion


if False:
    spatial_rep.plot(plot_types=['wireframe'])




# design scenario
design_scenario_name = 'structural_sizing'
design_scenario = cd.DesignScenario(name=design_scenario_name)

# region Cruise condition
cruise_name = "cruise_3"
cruise_model = m3l.Model()
cruise_condition = cd.CruiseCondition(name=cruise_name)
cruise_condition.atmosphere_model = cd.SimpleAtmosphereModel()
cruise_condition.set_module_input(name='altitude', val=600*ft2m)
cruise_condition.set_module_input(name='mach_number', val=0.145972)  # 112 mph = 0.145972 Mach = 50m/s
cruise_condition.set_module_input(name='range', val=80467.2)  # 50 miles = 80467.2 m
cruise_condition.set_module_input(name='pitch_angle', val=np.deg2rad(6))
cruise_condition.set_module_input(name='flight_path_angle', val=0)
cruise_condition.set_module_input(name='roll_angle', val=0)
cruise_condition.set_module_input(name='yaw_angle', val=0)
cruise_condition.set_module_input(name='wind_angle', val=0)
cruise_condition.set_module_input(name='observer_location', val=np.array([0, 0, 600*ft2m]))

cruise_ac_states = cruise_condition.evaluate_ac_states()
cruise_model.register_output(cruise_ac_states)
# endregion


# run model
cruise_condition.add_m3l_model('cruise_model', cruise_model)
design_scenario.add_design_condition(cruise_condition)
system_model.add_design_scenario(design_scenario=design_scenario)
caddee_csdl_model = caddee.assemble_csdl()
system_model_name = 'system_model.'+design_scenario_name+'.'+cruise_name+'.'+cruise_name+'.'

sim = Simulator(caddee_csdl_model, analytics=True)
sim.run()