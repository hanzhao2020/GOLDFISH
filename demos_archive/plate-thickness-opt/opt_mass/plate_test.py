"""
The required eVTOL geometry can be downloaded from:
    https://drive.google.com/file/d/1xpY8ACQlodmwkUZsiEQvTZPcUu-uezgi/view?usp=sharing
and extracted using the command "tar -xvzf eVTOL_wing_structure.tgz".
"""
from PENGoLINS.occ_preprocessing import *
from PENGoLINS.nonmatching_coupling import *

# complier = 'ffc'
complier = 'tsfc'

if complier == 'tsfc':
    # Use TSFC representation, due to complicated forms:
    parameters["form_compiler"]["representation"] = "tsfc"
    sys.setrecursionlimit(10000)

import time
from datetime import datetime
start_time = time.perf_counter()
start_current_time = datetime.now().strftime("%H:%M:%S")
if mpirank == 0:
    print("Start current time:", start_current_time)

SAVE_PATH = "/home/han/Documents/test_results/"
FOLDER_NAME = "results3/"

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
    spline_generator.writeExtraction(DIR)
    spline = ExtractedSpline(spline_generator, quad_deg)
    return spline

save_disp = True
save_stress = True
# Define parameters
# Scale down the geometry using ``geom_scale``to make the length 
# of the wing in the span-wise direction is around 11 m 
# (original length 4.54e5).
geom_scale = 1.  # Convert current length unit to m
E = Constant(68e9)  # Young's modulus, Pa
nu = Constant(0.35)  # Poisson's ratio
h_th_val = Constant(1.0e-2)  # Thickness of surfaces, m

p = 3  # spline order
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

# write_geom_file(preprocessor.BSpline_surfs, "pegasus_wing_geom.igs")
# exit()

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
h_val_list = [0.017412, 0.013612, 0.010129, 0.006712, 0.003324]
for i in range(num_surfs):
    h_th += [Function(splines[i].V_linear)]
    h_th[i].interpolate(Constant(h_val_list[i]))

# Create non-matching problem
problem = NonMatchingCoupling(splines, E, h_th, nu, comm=worldcomm)
problem.create_mortar_meshes(preprocessor.mortar_nels)

if mpirank == 0:
    print("Setting up mortar meshes...")

problem.mortar_meshes_setup(preprocessor.mapping_list, 
                            preprocessor.intersections_para_coords, 
                            penalty_coefficient)

# Define magnitude of load
load = Constant(-100) # The load should be in the unit of N/m^3
f1 = as_vector([Constant(0.0), Constant(0.0), load])

# Distributed downward load
loads = [f1]*num_surfs
source_terms = []
residuals = []
for i in range(num_surfs):
    source_terms += [inner(loads[i], problem.splines[i].rationalize(
        problem.spline_test_funcs[i]))*problem.splines[i].dx]
    residuals += [SVK_residual(problem.splines[i], problem.spline_funcs[i], 
        problem.spline_test_funcs[i], E, nu, h_th[i], source_terms[i])]
problem.set_residuals(residuals)

if mpirank == 0:
    print("Solving linear non-matching problem...")

# problem.solve_linear_nonmatching_problem()
problem.solve_nonlinear_nonmatching_problem()

# print out vertical displacement at the tip of trailing edge
right_srf_ind = 4
xi = array([1, 1])
z_disp_hom = eval_func(problem.splines[right_srf_ind].mesh, 
                       problem.spline_funcs[right_srf_ind][2], xi)
w = eval_func(problem.splines[right_srf_ind].mesh, 
              problem.splines[right_srf_ind].cpFuncs[3], xi)
QoI = z_disp_hom/w

if mpirank == 0:
    print("Free edge vertical displacement: {:10.8f}.\n".format(QoI))


end_time = time.perf_counter()
run_time = end_time - start_time
end_current_time = datetime.now().strftime("%H:%M:%S")

if mpirank == 0:
    print("End current time:", end_current_time)
    print("Simulation run time: {:.2f} s".format(run_time))

# Compute von Mises stress
if mpirank == 0:
    print("Computing von Mises stresses...")

von_Mises_tops = []
for i in range(problem.num_splines):
    spline_stress = ShellStressSVK(problem.splines[i], 
                                   problem.spline_funcs[i],
                                   E, nu, h_th, linearize=False,) 
                                   # G_det_min=5e-2)
    # problem.spline_funcs[i].interpolate(Constant((0.,0.,0.)))
    # von Mises stresses on top surfaces
    von_Mises_top = spline_stress.vonMisesStress(-h_th[i]/2)
    # _, von_Mises_top = spline_stress.secondPiolaKirchhoffStress(h_th/2)
    # von_Mises_top = spline_stress.membraneStrain()
    # von_Mises_top = spline_stress.curvatureChange()
    # von_Mises_top_proj = problem.splines[i].projectScalarOntoLinears(
    #                      von_Mises_top, lumpMass=True)
    VvM = FunctionSpace(problem.splines[i].mesh, 'CG', 2)
    von_Mises_top_proj = spline_project(problem.splines[i], VvM,
                         von_Mises_top)
    # von_Mises_top_proj = project_scalar(problem.splines[i], von_Mises_top[0,0], rationalize=False)
    von_Mises_tops += [von_Mises_top_proj]

if mpirank == 0:
    print("Saving results...")

if save_disp:
    for i in range(problem.num_splines):
        save_results(splines[i], problem.spline_funcs[i], i, 
                     save_path=SAVE_PATH, folder=FOLDER_NAME, 
                     save_cpfuncs=True, comm=worldcomm)
if save_stress:
    for i in range(problem.num_splines):
        von_Mises_tops[i].rename("von_Mises_top_"+str(i), 
                                 "von_Mises_top_"+str(i))
        File(SAVE_PATH+FOLDER_NAME+"von_Mises_top_"+str(i)+".pvd") \
            << von_Mises_tops[i]