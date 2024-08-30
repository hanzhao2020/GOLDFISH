import numpy as np
import csdl_alpha as csdl
from matplotlib import pyplot as plt

import time
from datetime import datetime

from igakit.cad import *
from igakit.io import VTK
from GOLDFISH.nonmatching_opt_csdl import *
# from GOLDFISH.nonmatching_opt_om import *

def clampedBC(spline_generator, side=0, direction=0):
    """
    Apply clamped boundary condition to spline.
    """
    for field in [0,1,2]:
        if field in [0]:
            n_layers = 1
        else:
            n_layers = 2
        scalar_spline = spline_generator.getScalarSpline(field)
        side_dofs = scalar_spline.getSideDofs(direction, side, nLayers=n_layers)
        spline_generator.addZeroDofs(field, side_dofs)

def OCCBSpline2tIGArSpline(surface, num_field=3, quad_deg_const=4, 
                        setBCs=None, side=0, direction=0, index=0):
    """
    Generate ExtractedBSpline from OCC B-spline surface.
    """
    quad_deg = surface.UDegree()*quad_deg_const
    DIR = SAVE_PATH+"spline_data/extraction_"+str(index)+"_init"
    # spline = ExtractedSpline(DIR, quad_deg)
    spline_mesh = NURBSControlMesh4OCC(surface, useRect=False)
    spline_generator = EqualOrderSpline(worldcomm, num_field, spline_mesh)
    if setBCs is not None:
        setBCs(spline_generator, side, direction)
    # spline_generator.writeExtraction(DIR)
    spline = ExtractedSpline(spline_generator, quad_deg)
    return spline

run_verify_forward_eval = False
run_check_derivatives = False
run_optimization = True

save_files = False
optimizer = 'SLSQP'
# optimizer = 'SNOPT'

save_path = './'
folder_name = "results/"


geom_scale = 1.  # Convert current length unit to m
E = Constant(68e9)  # Young's modulus, Pa
nu = Constant(0.35)  # Poisson's ratio
h_th_val = Constant(1.0e-2)  # Thickness of surfaces, m

# p = 3  # spline order
penalty_coefficient = 1.0e3

print("Importing geometry...")
filename_igs = "./geometry/plate_geometry.igs"
igs_shapes = read_igs_file(filename_igs, as_compound=False)
plate_surfaces = [topoface2surface(face, BSpline=True) 
                  for face in igs_shapes]

num_surfs = len(plate_surfaces)
if mpirank == 0:
    print("Number of surfaces:", num_surfs)

# Geometry preprocessing and surface-surface intersections computation
preprocessor = OCCPreprocessing(plate_surfaces, reparametrize=False, 
                                refine=False)

if mpirank == 0:
    print("Computing intersections...")
int_data_filename = "plate_int_data.npz"
if os.path.isfile(int_data_filename):
    preprocessor.load_intersections_data(int_data_filename)
else:
    preprocessor.compute_intersections(rtol=1e-6, mortar_refine=2, 
                                       edge_rel_ratio=1e-3)
    preprocessor.save_intersections_data(int_data_filename)

if mpirank == 0:
    print("Total DoFs:", preprocessor.total_DoFs)
    print("Number of intersections:", preprocessor.num_intersections_all)

if mpirank == 0:
    print("Creating splines...")
# Create tIGAr extracted spline instances
splines = []
for i in range(num_surfs):
    if i == 0:
        # Apply clamped BC to surfaces near root
        spline = OCCBSpline2tIGArSpline(
                 preprocessor.BSpline_surfs[i], 
                 setBCs=clampedBC, side=0, direction=0, index=i)
        splines += [spline,]
    else:
        spline = OCCBSpline2tIGArSpline(
                 preprocessor.BSpline_surfs[i], index=i)
        splines += [spline,]

h_th = []
h_val_list = [1e-2]*num_surfs
for i in range(num_surfs):
    h_th += [Function(splines[i].V_linear)]
    h_th[i].interpolate(Constant(h_val_list[i]))

# Create non-matching problem
nonmatching_opt = NonMatchingOpt(splines, E, h_th, nu, comm=worldcomm)
nonmatching_opt.create_mortar_meshes(preprocessor.mortar_nels)

if mpirank == 0:
    print("Setting up mortar meshes...")

nonmatching_opt.mortar_meshes_setup(preprocessor.mapping_list, 
                            preprocessor.intersections_para_coords, 
                            penalty_coefficient)

nonmatching_opt.set_thickness_opt(var_thickness=False)

# Define magnitude of load
load = Constant(-100) # The load should be in the unit of N/m^3
f1 = as_vector([Constant(0.0), Constant(0.0), load])
f0 = as_vector([Constant(0.0), Constant(0.0), Constant(0.0)])

xi5 = nonmatching_opt.splines[5].parametricCoordinates()
bdry1 = conditional(gt(xi5[0],1.-1e-3), Constant(1.), Constant(0.))

f_list = [f0]*(num_surfs-1) + [f1*bdry1]

# Distributed downward load
# loads = [f1]*num_surfs
source_terms = []
residuals = []
for i in range(num_surfs):
    source_terms += [inner(f_list[i], nonmatching_opt.splines[i].rationalize(
        nonmatching_opt.spline_test_funcs[i]))*nonmatching_opt.splines[i].ds]
    residuals += [SVK_residual(nonmatching_opt.splines[i], 
                               nonmatching_opt.spline_funcs[i], 
                               nonmatching_opt.spline_test_funcs[i], 
                               E, nu, h_th[i], source_terms[i])]
nonmatching_opt.set_residuals(residuals)

# if mpirank == 0:
#     print("Solving linear non-matching problem ...")
# nonmatching_opt.solve_linear_nonmatching_problem()

vol_val = 0
for s_ind in range(nonmatching_opt.num_splines):
    vol_val += assemble(nonmatching_opt.h_th[s_ind]
                    *nonmatching_opt.splines[s_ind].dx)

# Set up optimization
if save_files:
    nonmatching_opt.create_files(save_path=save_path, folder_name=folder_name, 
                                 thickness=nonmatching_opt.opt_thickness)

class ThicknessOptModel():
    def __init__(self, nonmatching_opt):
        self.nonmatching_opt = nonmatching_opt
        self.h_th_design_name = 'h_th_design'
        self.h_th_full_name = 'h_th'
        self.disp_name = 'u'
        self.wint_name = 'w_int'
        self.vol_name = 'vol'

    def evaluate(self, inputs:csdl.VariableGroup, debug_mode=False):

        variable_dict = inputs
        h_th_map_model = HthMapModel(nonmatching_opt=self.nonmatching_opt)
        h_th_full = h_th_map_model.evaluate(variable_dict)
        setattr(variable_dict, self.h_th_full_name, h_th_full)

        disp_model = DispStatesModel(nonmatching_opt=nonmatching_opt)
        u = disp_model.evaluate(variable_dict)
        setattr(variable_dict, self.disp_name, u)

        int_energy_model = IntEnergyModel(nonmatching_opt=nonmatching_opt)
        w_int = int_energy_model.evaluate(variable_dict)
        setattr(variable_dict, self.wint_name, w_int)

        volume_model = VolumeModel(nonmatching_opt=nonmatching_opt)
        vol = volume_model.evaluate(variable_dict)
        setattr(variable_dict, self.vol_name, vol)

        return variable_dict

init_h_th_val = np.array([np.average(h_th_sub_array) for h_th_sub_array
                         in nonmatching_opt.init_h_th_list])


recorder = csdl.Recorder(inline=True)
recorder.start()

thopt_model = ThicknessOptModel(nonmatching_opt=nonmatching_opt)

h_th_design = csdl.Variable(value=init_h_th_val, 
                            name=thopt_model.h_th_design_name)
inputs = csdl.VariableGroup()
inputs.h_th_design = h_th_design


outputs = thopt_model.evaluate(inputs)
u = outputs.u
h_th = outputs.h_th
w_int = outputs.w_int
vol = outputs.vol

if run_verify_forward_eval:
    outputs = thopt_model.evaluate(inputs)

    print("Forward evaluation ...")
    print(" "*4, w_int.names, w_int.value)
    print(" "*4, vol.names, vol.value)

if run_check_derivatives:
    sim = csdl.experimental.PySimulator(recorder)
    sim.check_totals([u,vol,w_int,h_th], [h_th_design], 
                     step_size=1e-6, raise_on_error=False)

start_time = time.perf_counter()
start_current_time = datetime.now().strftime("%H:%M:%S")
if mpirank == 0:
    print("Start current time:", start_current_time)

if run_optimization:
    from modopt import CSDLAlphaProblem
    from modopt import SLSQP
    h_th_design.set_as_design_variable(lower=4e-3, upper=5e-2)

    vol.set_as_constraint(lower=vol_val, upper=vol_val)
    w_int.set_as_objective(scaler=1e3)
    sim = csdl.experimental.PySimulator(recorder)

    prob = CSDLAlphaProblem(problem_name='plate_thopt', simulator=sim)

    optimizer = SLSQP(prob, solver_options={'ftol':1e-6, 'maxiter':10000, 'disp':True})

    # Solve your optimization problem
    optimizer.solve()
    optimizer.print_results()
    print("Optimization results:")
    print(" "*4, w_int.names, w_int.value)
    print(" "*4, vol.names, vol.value)

recorder.stop()

end_time = time.perf_counter()
run_time = end_time - start_time
end_current_time = datetime.now().strftime("%H:%M:%S")

if mpirank == 0:
    print("End current time:", end_current_time)
    print("Simulation run time: {:.2f} s".format(run_time))


h_th_profile = []
num_pts = 101
xi0_array = np.linspace(0,1,num_pts)
xi1 = 0.5
for i in range(num_surfs):
    for xi_ind in range(num_pts):
        h_th_profile += [nonmatching_opt.h_th[i]((xi0_array[xi_ind], xi1))]

x0_array = np.linspace(0.,1.,num_pts*num_surfs)
np.savez("h_th_profile.npz", h=h_th_profile, x=x0_array)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(x0_array, h_th_profile, '--')
plt.show()