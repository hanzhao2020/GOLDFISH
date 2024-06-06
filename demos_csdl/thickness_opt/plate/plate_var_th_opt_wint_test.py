import numpy as np
import matplotlib.pyplot as plt
from igakit.cad import *
from igakit.io import VTK
from GOLDFISH.nonmatching_opt_csdl import *

import csdl
from csdl import Model
from csdl_om import Simulator as om_simulator
from python_csdl_backend import Simulator as py_simulator
from modopt.csdl_library import CSDLProblem
# from modopt.snopt_library import SNOPT
from modopt.scipy_library import SLSQP


class VarThOptGroup(Model):

    def initialize(self):
        self.parameters.declare('nonmatching_opt_ffd')
        self.parameters.declare('h_th_ffd_name', default='thickness')
        self.parameters.declare('h_th_fe_name', default='thickness_FE')
        self.parameters.declare('h_th_iga_name', default='thickness_IGA')
        self.parameters.declare('disp_name', default='displacements')
        self.parameters.declare('int_energy_name', default='w_int')
        self.parameters.declare('h_th_ffd_align_name', default='thickness_FFD_align')
        self.parameters.declare('h_th_ffd_regu_name', default='thickness_FFD_regu')
        self.parameters.declare('volume_name', default='volume')

    def init_parameters(self):
        self.nonmatching_opt_ffd = self.parameters['nonmatching_opt_ffd']
        self.h_th_ffd_name = self.parameters['h_th_ffd_name']
        self.h_th_fe_name = self.parameters['h_th_fe_name']
        self.h_th_iga_name = self.parameters['h_th_iga_name']
        self.disp_name = self.parameters['disp_name']
        self.volume_name = self.parameters['volume_name']
        self.h_th_ffd_align_name = self.parameters['h_th_ffd_align_name']
        self.int_energy_name = self.parameters['int_energy_name']
        self.h_th_ffd_regu_name = self.parameters['h_th_ffd_regu_name']

        self.design_var_lower = 5e-4
        self.design_var_upper = 5e-1

        self.num_splines = self.nonmatching_opt_ffd.num_splines
        self.init_h_th_ffd = self.nonmatching_opt_ffd.get_init_h_th_FFD().copy()

        self.h_th_ffd2fe_model_name = 'h_th_ffd2fe_model'
        self.h_th_fe2iga_model_name = 'h_th_fe2iga_model'
        self.disp_states_model_name = 'disp_states_model'
        self.volume_model_name = 'volume_model'
        self.int_energy_model_name = 'int_energy_model'
        self.h_th_ffd_align_model_name = 'h_th_ffd_align_model'
        self.h_th_ffd_regu_model_name = 'h_th_ffd_regu_model'

    def define(self):
        self.create_input(self.h_th_ffd_name, 
                          shape=(self.nonmatching_opt_ffd.thopt_cpffd_size),
                          val=self.init_h_th_ffd)
        
        # Add h_th FFD2FE model
        self.h_th_ffd2fe_model = HthFFD2FEModel(
                        nonmatching_opt_ffd=self.nonmatching_opt_ffd,
                        input_h_th_ffd_name=self.h_th_ffd_name,
                        output_h_th_fe_name=self.h_th_fe_name)
        self.h_th_ffd2fe_model.init_parameters()
        self.add(self.h_th_ffd2fe_model, 
                 name=self.h_th_ffd2fe_model_name, promotes=[])

        # Add h_th FE2IGA model
        self.h_th_fe2iga_model = HthFE2IGAModel(
                        nonmatching_opt=self.nonmatching_opt_ffd,
                        input_h_th_fe_name=self.h_th_fe_name,
                        output_h_th_iga_name=self.h_th_iga_name)
        self.h_th_fe2iga_model.init_parameters()
        self.add(self.h_th_fe2iga_model, 
                 name=self.h_th_fe2iga_model_name, promotes=[])

        # Add disp_states_model
        self.disp_states_model = DispStatesModel(
                           nonmatching_opt=self.nonmatching_opt_ffd,
                           input_h_th_name=self.h_th_iga_name,
                           output_u_name=self.disp_name)
        self.disp_states_model.init_parameters(save_files=True)
        self.add(self.disp_states_model, 
                 name=self.disp_states_model_name, promotes=[])

        # Add internal energy comp (objective function)
        self.int_energy_model = IntEnergyModel(
                          nonmatching_opt=self.nonmatching_opt_ffd,
                          input_h_th_name=self.h_th_iga_name,
                          input_u_name=self.disp_name,
                          output_wint_name=self.int_energy_name)
        self.int_energy_model.init_parameters()
        self.add(self.int_energy_model, 
                 name=self.int_energy_model_name, promotes=[])

        # Add volume constraint
        self.volume_model = VolumeModel(
                           nonmatching_opt=self.nonmatching_opt_ffd,
                           input_h_th_name=self.h_th_iga_name,
                           output_vol_name=self.volume_name)
        self.volume_model.init_parameters()
        self.add(self.volume_model, name=self.volume_model_name, promotes=[])
        self.vol_val = 0
        for s_ind in range(self.nonmatching_opt_ffd.num_splines):
            self.vol_val += assemble(self.nonmatching_opt_ffd.h_th[s_ind]
                            *self.nonmatching_opt_ffd.splines[s_ind].dx)

        # Add thickness FFD align comp (linear constraint)
        self.h_th_ffd_align_model = HthFFDAlignModel(
            nonmatching_opt_ffd=self.nonmatching_opt_ffd,
            input_h_th_name=self.h_th_ffd_name,
            output_h_th_align_name=self.h_th_ffd_align_name)
        self.h_th_ffd_align_model.init_parameters()
        self.add(self.h_th_ffd_align_model, 
                 name=self.h_th_ffd_align_model_name, promotes=[])
        self.cpffd_align_cons_val = \
            np.zeros(self.h_th_ffd_align_model.op.output_shape)

        # Add thickness FFD regu comp (linear constraint)
        self.h_th_ffd_regu_model = HthFFDReguModel(
            nonmatching_opt_ffd=self.nonmatching_opt_ffd,
            input_h_th_name=self.h_th_ffd_name,
            output_h_th_regu_name=self.h_th_ffd_regu_name)
        self.h_th_ffd_regu_model.init_parameters()
        self.add(self.h_th_ffd_regu_model, name=self.h_th_ffd_regu_model_name, promotes=[])
        self.h_th_ffd_regu_cons_val = \
             np.ones(self.h_th_ffd_regu_model.op.output_shape)*1e-5

        # Connect names between components
        # For optimization components
        self.connect(self.h_th_ffd_name,
                     self.h_th_ffd2fe_model_name+'.'
                     +self.h_th_ffd_name)
        self.connect(self.h_th_ffd2fe_model_name+'.'
                     +self.h_th_fe_name,
                     self.h_th_fe2iga_model_name+'.'
                     +self.h_th_fe_name)
        self.connect(self.h_th_fe2iga_model_name+'.'
                     +self.h_th_iga_name,
                     self.disp_states_model_name+'.'
                     +self.h_th_iga_name)
        self.connect(self.h_th_fe2iga_model_name+'.'
                     +self.h_th_iga_name,
                     self.int_energy_model_name+'.'
                     +self.h_th_iga_name)
        # For constraints
        self.connect(self.h_th_fe2iga_model_name+'.'
                     +self.h_th_iga_name,
                     self.volume_model_name+'.'
                     +self.h_th_iga_name)
        
        self.connect(self.h_th_ffd_name,
                     self.h_th_ffd_align_model_name+'.'
                     +self.h_th_ffd_name)
        self.connect(self.h_th_ffd_name,
                     self.h_th_ffd_regu_model_name+'.'
                     +self.h_th_ffd_name)

        self.connect(self.disp_states_model_name+'.'+self.disp_name,
                     self.int_energy_model_name+'.'+self.disp_name)

        # Add design variable, constraints and objective
        self.add_design_variable(self.h_th_ffd_name,
                            lower=self.design_var_lower,
                            upper=self.design_var_upper)
        self.add_constraint(self.h_th_ffd_align_model_name+'.'
                            +self.h_th_ffd_align_name,
                            equals=self.cpffd_align_cons_val)
        self.add_constraint(self.h_th_ffd_regu_model_name+'.'
                            +self.h_th_ffd_regu_name,
                            lower=self.h_th_ffd_regu_cons_val)
        self.add_constraint(self.volume_model_name+'.'
                            +self.volume_name,
                            equals=self.vol_val)
        self.add_objective(self.int_energy_model_name+'.'
                           +self.int_energy_name,
                           scaler=1e1)

def clampedBC(spline_generator, side=0, direction=0):
    """
    Apply clamped boundary condition to spline.
    """
    for field in [0,1,2]:
        # if field in [0]:
        #     n_layers = 1
        # else:
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


test_ind = 22
ftol=1e-7
# optimizer = 'SLSQP'
optimizer = 'SNOPT'

# save_path = './'
save_path = '/home/han/Documents/test_results/'
# save_path = '/Users/hanzhao/Documents/test_results/'
folder_name = "results"+str(test_ind)+"/"

p_ffd = 3
ffd_block_num_el = [3,1,1]

geom_scale = 1.  # Convert current length unit to m
E = Constant(68e9)  # Young's modulus, Pa
nu = Constant(0.35)  # Poisson's ratio
# h_th_val = Constant(1.0e-2)  # Thickness of surfaces, m

# p = test_ind  # spline order
penalty_coefficient = 1.0e3

print("Importing geometry...")
# filename_igs = "./geometry/plate_geometry_quadratic.igs"
filename_igs = "./geometry/plate_geometry_cubic.igs"
# filename_igs = "./geometry/plate_geometry_quartic.igs"
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

preprocessor.compute_intersections(rtol=1e-6, mortar_refine=2,) 

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
    h_th += [Function(splines[i].V_control)]
    h_th[i].interpolate(Constant(h_val_list[i]))

# Create non-matching problem
nonmatching_opt = NonMatchingOptFFD(splines, E, h_th, nu, opt_shape=False, 
                                 opt_thickness=True, var_thickness=True, 
                                 comm=worldcomm)

# # Save initial discretized geometry
# nonmatching_opt.create_files(save_path=save_path, 
#                                  folder_name='results_temp/', 
#                                  refine_mesh=False)
# nonmatching_opt.save_files()
# # Save intersection curves
# for i in range(preprocessor.num_intersections_all):
#     mesh_phy = generate_mortar_mesh(preprocessor.intersections_phy_coords[i], num_el=128)
#     File('./geometry/int_curve'+str(i)+'.pvd') << mesh_phy
# exit()

nonmatching_opt.create_mortar_meshes(preprocessor.mortar_nels)

if mpirank == 0:
    print("Setting up mortar meshes...")

nonmatching_opt.mortar_meshes_setup(preprocessor.mapping_list, 
                            preprocessor.intersections_para_coords, 
                            penalty_coefficient)

# # Define magnitude of load
# load = Constant(-100) # The load should be in the unit of N/m^3
# f1 = as_vector([Constant(0.0), Constant(0.0), load])

# # Distributed downward load
# loads = [f1]*num_surfs
# source_terms = []
# residuals = []
# for i in range(num_surfs):
#     source_terms += [inner(loads[i], nonmatching_opt.splines[i].rationalize(
#         nonmatching_opt.spline_test_funcs[i]))*nonmatching_opt.splines[i].dx]
#     residuals += [SVK_residual(nonmatching_opt.splines[i], 
#                                nonmatching_opt.spline_funcs[i], 
#                                nonmatching_opt.spline_test_funcs[i], 
#                                E, nu, h_th[i], source_terms[i])]
# nonmatching_opt.set_residuals(residuals)

# Define magnitude of load
load = Constant(-1) # The load should be in the unit of N/m^3
f1 = as_vector([Constant(0.0), Constant(0.0), load])
f0 = as_vector([Constant(0.0), Constant(0.0), Constant(0.0)])

xi_end = nonmatching_opt.splines[-1].parametricCoordinates()
bdry1 = conditional(gt(xi_end[0],1.-1e-3), Constant(1.), Constant(0.))

f_list = [f0]*(num_surfs-1) + [f1*bdry1]

# Line load
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

# nonmatching_opt.solve_nonlinear_nonmatching_problem()
# print("*"*50)

# Create FFD block in igakit format
cp_ffd_lims = nonmatching_opt.cpsurf_lims
for field in [2]:
    cp_ffd_lims[field][1] = 0.05
    cp_ffd_lims[field][0] = -0.05
FFD_block = create_3D_block(ffd_block_num_el, p_ffd, cp_ffd_lims)

# VTK().write("./geometry/ffd_block_init.vtk", FFD_block)
# vtk_writer = VTKWriter()
# vtk_writer.write_cp("./geometry/ffd_cp_init.vtk", FFD_block)

# Set FFD to non-matching optimization instance
nonmatching_opt.set_thopt_FFD(FFD_block.knots, FFD_block.control)
nonmatching_opt.set_thopt_align_CPFFD(thopt_align_dir=1)
nonmatching_opt.set_thopt_regu_CPFFD([None], [None], None)

# # Set constraint info
# nonmatching_opt_ffd.set_pin_CPFFD(pin_dir0=2, pin_side0=[1],
#                                   pin_dir1=1, pin_side1=[1])
# nonmatching_opt_ffd.set_regu_CPFFD(regu_dir=[None], regu_side=[None])

# Set up optimization
nonmatching_opt.create_files(save_path=save_path, folder_name=folder_name, 
                             thickness=nonmatching_opt.opt_thickness)

model = VarThOptGroup(nonmatching_opt_ffd=nonmatching_opt)
model.init_parameters()
sim = py_simulator(model, analytics=False)
sim.run()

prob = CSDLProblem(problem_name='plate-var-th-opt', simulator=sim)

optimizer = SLSQP(prob, maxiter=1000, ftol=ftol)
# optimizer = SNOPT(prob, Major_iterations = 1000, 
#                   Major_optimality = 1e-9, append2file=False)

optimizer.solve()
optimizer.print_results()

if mpirank == 0:
    for i in range(num_surfs):
        print("Thickness for patch {:2d}: {:10.6f}".format(i, 
              nonmatching_opt.h_th[i].vector().get_local()[0]))

h_th_profile = []
num_pts = 101
xi0_array = np.linspace(0,1,num_pts)
xi1 = 0.5
for i in range(num_surfs):
    for xi_ind in range(num_pts):
        h_th_profile += [nonmatching_opt.h_th[i]((xi0_array[xi_ind], xi1))]

x0_array = np.linspace(0.,1./num_surfs, num_pts)
x_array = np.concatenate([(x0_array + 1./num_surfs*i) for i in range(num_surfs)])

h_th_norm = h_th_profile/np.max(h_th_profile)

th = array([1.        , 0.99533525, 0.99065238, 0.98594663, 0.9812117 ,
       0.9764666 , 0.97169185, 0.96689298, 0.96207135, 0.9572236 ,
       0.95235104, 0.94745466, 0.94253387, 0.93758512, 0.93261512,
       0.92761303, 0.92258259, 0.91753498, 0.91244501, 0.90733233,
       0.90219221, 0.89702172, 0.89182206, 0.88659221, 0.88132931,
       0.87603696, 0.87071223, 0.86535394, 0.85996202, 0.85453616,
       0.84907611, 0.84357974, 0.8380473 , 0.83247997, 0.82687318,
       0.82123094, 0.81554595, 0.80982255, 0.80405715, 0.79825372,
       0.79240659, 0.78651524, 0.78057901, 0.77459696, 0.76857067,
       0.7624971 , 0.75637052, 0.75019487, 0.74396437, 0.73769083,
       0.73135868, 0.72497079, 0.71852691, 0.71202989, 0.70546131,
       0.69884278, 0.6921521 , 0.68539935, 0.67857958, 0.67168948,
       0.66473061, 0.65769385, 0.6505834 , 0.6433963 , 0.63612632,
       0.62877217, 0.62133124, 0.61380058, 0.60617544, 0.59845286,
       0.59063134, 0.58270281, 0.57466879, 0.56651583, 0.55824537,
       0.54985143, 0.54132523, 0.53266407, 0.52386036, 0.51490575,
       0.50579169, 0.49651135, 0.4870552 , 0.47740267, 0.46756671,
       0.45751289, 0.44723197, 0.4367098 , 0.42592717, 0.41486508,
       0.40349899, 0.39180443, 0.37974976, 0.36730008, 0.35441388,
       0.34104125, 0.32712289, 0.31258686, 0.29734216, 0.28127334,
       0.26423203, 0.246017  , 0.22634664, 0.20480777, 0.18074422,
       0.15300365, 0.11920342, 0.07328464])


th_norm = th/np.max(th)
x_coord = np.linspace(0.,1.,len(th))

import matplotlib.pyplot as plt
plt.figure()
# plt.plot(x0_array, h_th_profile, '--')
plt.plot(x_coord, th_norm, '--', color='gray', label='Bernoulli beam')
plt.plot(x_array, h_th_norm, '-', label='Kirchhoff–Love shell, y=0.5')
# plt.plot(np.linspace(0,1,len(cp)), cp, '-*', label='Shell opt cp')
plt.legend()
plt.xlabel("x")
plt.ylabel("Normalized thickness")
plt.savefig("temp"+str(test_ind)+".png")
plt.show()