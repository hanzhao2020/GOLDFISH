import numpy as np
import matplotlib.pyplot as plt
import openmdao.api as om
from igakit.cad import *
from igakit.io import VTK
from GOLDFISH.nonmatching_opt_om import *

class VarThOptGroup(om.Group):

    def initialize(self):
        self.options.declare('nonmatching_opt_ffd')
        self.options.declare('h_th_ffd_name', default='thickness')
        self.options.declare('h_th_fe_name', default='thickness_FE')
        self.options.declare('h_th_iga_name', default='thickness_IGA')
        self.options.declare('disp_name', default='displacements')
        self.options.declare('int_energy_name', default='w_int')
        self.options.declare('h_th_ffd_align_name', default='thickness_FFD_align')
        self.options.declare('volume_name', default='volume')

    def init_parameters(self):
        self.nonmatching_opt_ffd = self.options['nonmatching_opt_ffd']
        self.h_th_ffd_name = self.options['h_th_ffd_name']
        self.h_th_fe_name = self.options['h_th_fe_name']
        self.h_th_iga_name = self.options['h_th_iga_name']
        self.disp_name = self.options['disp_name']
        self.volume_name = self.options['volume_name']
        self.h_th_ffd_align_name = self.options['h_th_ffd_align_name']
        self.int_energy_name = self.options['int_energy_name']

        self.design_var_lower = 1.0e-3
        self.design_var_upper = 1.0e-1

        self.num_splines = self.nonmatching_opt_ffd.num_splines
        self.nonmatching_opt_ffd.get_init_h_th_multiFFD()

        self.inputs_comp_name = 'inputs_comp'
        self.h_th_ffd2fe_comp_name = 'h_th_ffd2fe_comp'
        self.h_th_fe2iga_comp_name = 'h_th_fe2iga_comp'
        self.h_th_ffd_align_comp_name = 'h_th_ffd_align_comp'
        self.disp_states_comp_name = 'disp_states_comp'
        self.volume_comp_name = 'volume_comp'
        self.int_energy_comp_name = 'int_energy_comp'

    def setup(self):
        # Add inputs comp
        inputs_comp = om.IndepVarComp()
        inputs_comp.add_output(self.h_th_ffd_name,
                    shape=self.nonmatching_opt_ffd.thopt_cpffd_design_size, 
                    val=self.nonmatching_opt_ffd.init_h_th_multiffd)
        self.add_subsystem(self.inputs_comp_name, inputs_comp)

        # Add h_th FFD2FE comp
        self.h_th_ffd2fe_comp = HthFFD2FEComp(
                        nonmatching_opt_ffd=self.nonmatching_opt_ffd,
                        input_h_th_ffd_name=self.h_th_ffd_name,
                        output_h_th_fe_name=self.h_th_fe_name)
        self.h_th_ffd2fe_comp.init_parameters()
        self.add_subsystem(self.h_th_ffd2fe_comp_name, self.h_th_ffd2fe_comp)

        # Add h_th FE2IGA comp
        self.h_th_fe2iga_comp = HthFE2IGAComp(
                        nonmatching_opt=self.nonmatching_opt_ffd,
                        input_h_th_fe_name=self.h_th_fe_name,
                        output_h_th_iga_name=self.h_th_iga_name)
        self.h_th_fe2iga_comp.init_parameters()
        self.add_subsystem(self.h_th_fe2iga_comp_name, self.h_th_fe2iga_comp)

        # Add disp_states_comp
        self.disp_states_comp = DispStatesComp(
                           nonmatching_opt=self.nonmatching_opt_ffd,
                           input_h_th_name=self.h_th_iga_name,
                           output_u_name=self.disp_name)
        self.disp_states_comp.init_parameters(save_files=True)
        self.add_subsystem(self.disp_states_comp_name, self.disp_states_comp)

        # Add internal energy comp (objective function)
        self.int_energy_comp = IntEnergyComp(
                          nonmatching_opt=self.nonmatching_opt_ffd,
                          input_h_th_name=self.h_th_iga_name,
                          input_u_name=self.disp_name,
                          output_wint_name=self.int_energy_name)
        self.int_energy_comp.init_parameters()
        self.add_subsystem(self.int_energy_comp_name, self.int_energy_comp)

        # Add volume comp (objective function)
        self.volume_comp = VolumeComp(
                           nonmatching_opt=self.nonmatching_opt_ffd,
                           input_h_th_name=self.h_th_iga_name,
                           output_vol_name=self.volume_name)
        self.volume_comp.init_parameters()
        self.add_subsystem(self.volume_comp_name, self.volume_comp)
        self.vol_val = 0
        for s_ind in range(self.nonmatching_opt_ffd.num_splines):
            self.vol_val += assemble(self.nonmatching_opt_ffd.h_th[s_ind]
                            *self.nonmatching_opt_ffd.splines[s_ind].dx)

        # # Add max von Mises stress comp (constraint)
        # xi2 = 0.
        # rho = 1.5e2
        # upper_vM = 1e6
        # self.max_vM_comp = MaxvMStressComp(
        #                        nonmatching_opt=self.nonmatching_opt_ffd,
        #                        rho=rho, alpha=None, m=upper_vM, method='pnorm', 
        #                        linearize_stress=False, 
        #                        input_u_name=self.disp_name,
        #                        input_h_th_name=self.h_th_iga_name,
        #                        output_int_energy_name=self.int_energy_name)
        # self.max_vM_comp.init_parameters()
        # self.add_subsystem(self.int_energy_comp_name, self.max_vM_comp)

        # # Add thickness FFD align comp (linear constraint)
        # self.h_th_ffd_align_comp = HthFFDAlignComp(
        #     nonmatching_opt_ffd=self.nonmatching_opt_ffd,
        #     input_h_th_name=self.h_th_ffd_name,
        #     output_h_th_align_name=self.h_th_ffd_align_name)
        # self.h_th_ffd_align_comp.init_parameters()
        # self.add_subsystem(self.h_th_ffd_align_comp_name, self.h_th_ffd_align_comp)
        # self.cpffd_align_cons_val = \
        #     np.zeros(self.h_th_ffd_align_comp.output_shape)


        # Connect names between components
        self.connect(self.inputs_comp_name+'.'
                     +self.h_th_ffd_name,
                     self.h_th_ffd2fe_comp_name+'.'
                     +self.h_th_ffd_name)

        self.connect(self.h_th_ffd2fe_comp_name+'.'
                     +self.h_th_fe_name,
                     self.h_th_fe2iga_comp_name+'.'
                     +self.h_th_fe_name)

        self.connect(self.h_th_fe2iga_comp_name+'.'
                     +self.h_th_iga_name,
                     self.disp_states_comp_name+'.'
                     +self.h_th_iga_name)

        self.connect(self.h_th_fe2iga_comp_name+'.'
                     +self.h_th_iga_name,
                     self.volume_comp_name+'.'
                     +self.h_th_iga_name)

        self.connect(self.h_th_fe2iga_comp_name+'.'
                     +self.h_th_iga_name,
                     self.int_energy_comp_name+'.'
                     +self.h_th_iga_name)

        self.connect(self.disp_states_comp_name+'.'+self.disp_name,
                     self.int_energy_comp_name+'.'+self.disp_name)

        # self.connect(self.inputs_comp_name+'.'+self.h_th_ffd_name,
        #              self.h_th_ffd_align_comp_name+'.'+self.h_th_ffd_name)

        # Add design variable, constraints and objective
        self.add_design_var(self.inputs_comp_name+'.'
                            +self.h_th_ffd_name,
                            lower=self.design_var_lower,
                            upper=self.design_var_upper,
                            scaler=1e1)

        # self.add_constraint(self.h_th_ffd_align_comp_name+'.'
        #                     +self.h_th_ffd_align_name,
        #                     equals=self.cpffd_align_cons_val)

        self.add_constraint(self.volume_comp_name+'.'
                            +self.volume_name,
                            equals=self.vol_val)

        self.add_objective(self.int_energy_comp_name+'.'
                           +self.int_energy_name,
                           scaler=1e1)

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


test_ind = 47

# optimizer = 'SLSQP'
optimizer = 'SNOPT'

# save_path = './'
save_path = '/home/han/Documents/test_results/'
# save_path = '/Users/hanzhao/Documents/test_results/'
folder_name = "results"+str(test_ind)+"/"

# Define parameters
# Scale down the geometry using ``geom_scale``to make the length 
# of the wing in the span-wise direction is around 11 m 
# (original length 4.54e5).
geom_scale = 2.54e-5  # Convert current length unit to m
E = Constant(68e9)  # Young's modulus, Pa
nu = Constant(0.35)  # Poisson's ratio
# h_th = Constant(3.0e-3)  # Thickness of surfaces, m

p = 3  # spline order
penalty_coefficient = 1.0e3

print("Importing geometry...")
filename_igs = "./geometry/pegasus_wing.iges"
igs_shapes = read_igs_file(filename_igs, as_compound=False)
pegasus_surfaces = [topoface2surface(face, BSpline=True) 
                  for face in igs_shapes]

# Upper skins: 0, 4, 8, 12, ..., 68
# Lower skins: 1, 5, 9, 13, ..., 69
# Front spars: 2, 6, 19, 14, ..., 70
# Rear spars: 3, 7, 11, 15, ..., 71
# Ribs: 72, 73, 74, ..., 89 (18 ribs)
num_secs = 18
# wing_indices = list(range(0,len(pegasus_surfaces))) # All surfaces
# wing_indices = list(range(0,8)) + list(range(72,74)) # First two sections
# wing_indices = list(range(0,12)) + list(range(72,75)) # First three sections
wing_indices = list(range(0,num_secs*4)) + list(range(72,72+num_secs))
wing_surfaces = [pegasus_surfaces[i] for i in wing_indices]
num_surfs = len(wing_surfaces)
if mpirank == 0:
    print("Number of surfaces:", num_surfs)

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
# # preprocessor.display_surfaces(display, save_fig=False)
# preprocessor.display_intersections(display, color='RED', save_fig=False)

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
    h_th += [Function(splines[i].V_control)]
    h_th[i].interpolate(Constant(5.0e-3))

# Create non-matching problem
# problem = NonMatchingCoupling(splines, E, h_th, nu, comm=worldcomm)
nonmatching_opt = NonMatchingOptFFD(splines, E, h_th, nu, opt_shape=False, 
                                 opt_thickness=True, var_thickness=True, 
                                 comm=worldcomm)

print(aaa)
# nonmatching_opt.create_mortar_meshes(preprocessor.mortar_nels)

# if mpirank == 0:
#     print("Setting up mortar meshes...")

# nonmatching_opt.mortar_meshes_setup(preprocessor.mapping_list, 
#                                     preprocessor.intersections_para_coords, 
#                                     penalty_coefficient)

# # Define magnitude of load
# load = Constant(1) # The load should be in the unit of N/m^2
# f1 = as_vector([Constant(0.0), Constant(0.0), load])

# # Distributed downward load
# loads = [f1]*num_surfs
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

###############################################################
print("Creating FFD blocks ...")
outer_skin_end_ind = num_secs*4
step = 4
ffd_p = 2

ffd0_surf_inds = np.arange(0, outer_skin_end_ind, step)
ffd1_surf_inds = np.arange(1, outer_skin_end_ind, step)
ffd2_surf_inds = np.arange(2, outer_skin_end_ind, step)
ffd3_surf_inds = np.arange(3, outer_skin_end_ind, step)

nonmatching_opt.set_thopt_multiFFD_surf_inds([ffd0_surf_inds, ffd1_surf_inds,
                                              ffd2_surf_inds, ffd3_surf_inds])
num_ffd_blocks = nonmatching_opt.num_thopt_ffd
cp_ffd_lims_multiffd = nonmatching_opt.thopt_cpsurf_lims_multiffd

ffd_block_list = []
ffd_block_num_el0 = [2,6,1]
for field in [2]:
    cp_range = cp_ffd_lims_multiffd[0][field][1] - cp_ffd_lims_multiffd[0][field][0]
    cp_ffd_lims_multiffd[0][field][0] = cp_ffd_lims_multiffd[0][field][0] - 0.1*cp_range
    cp_ffd_lims_multiffd[0][field][1] = cp_ffd_lims_multiffd[0][field][1] + 0.1*cp_range
    ffd_block_list += [create_3D_block(ffd_block_num_el0, ffd_p, cp_ffd_lims_multiffd[0])]
ffd_block_num_el1 = [2,6,1]
for field in [2]:
    cp_range = cp_ffd_lims_multiffd[1][field][1] - cp_ffd_lims_multiffd[1][field][0]
    cp_ffd_lims_multiffd[1][field][0] = cp_ffd_lims_multiffd[1][field][0] - 0.1*cp_range
    cp_ffd_lims_multiffd[1][field][1] = cp_ffd_lims_multiffd[1][field][1] + 0.1*cp_range
    ffd_block_list += [create_3D_block(ffd_block_num_el1, ffd_p, cp_ffd_lims_multiffd[1])]

ffd_block_num_el2 = [1,6,2]
for field in [0]:
    cp_range = cp_ffd_lims_multiffd[2][field][1] - cp_ffd_lims_multiffd[2][field][0]
    cp_ffd_lims_multiffd[2][field][0] = cp_ffd_lims_multiffd[2][field][0] - 0.1*cp_range
    cp_ffd_lims_multiffd[2][field][1] = cp_ffd_lims_multiffd[2][field][1] + 0.1*cp_range
    ffd_block_list += [create_3D_block(ffd_block_num_el2, ffd_p, cp_ffd_lims_multiffd[2])]
ffd_block_num_el3 = [1,6,2]
for field in [0]:
    cp_range = cp_ffd_lims_multiffd[3][field][1] - cp_ffd_lims_multiffd[3][field][0]
    cp_ffd_lims_multiffd[3][field][0] = cp_ffd_lims_multiffd[3][field][0] - 0.1*cp_range
    cp_ffd_lims_multiffd[3][field][1] = cp_ffd_lims_multiffd[3][field][1] + 0.1*cp_range
    ffd_block_list += [create_3D_block(ffd_block_num_el3, ffd_p, cp_ffd_lims_multiffd[3])]

vtk_writer = VTKWriter()
for i in range(len(ffd_block_list)):
    # VTK().write("./geometry/ffd_block_init"+str(i)+".vtk", ffd_block_list[i])
    vtk_writer.write('./geometry/thopt_ffd_block'+str(i)+'.vtk', ffd_block_list[i], ref_level=3)
    vtk_writer.write_cp('./geometry/thopt_ffd_block_cp'+str(i)+'.vtk', ffd_block_list[i])

print(aaa)
ffd_knots_list = [ffd_block.knots for ffd_block in ffd_block_list]
ffd_control_list = [ffd_block.control for ffd_block in ffd_block_list]
print("Setting multiple FFD blocks ...")
nonmatching_opt.set_thopt_multiFFD(ffd_knots_list, ffd_control_list)
# print("Setting FFD blocks alignments ...")
# nonmatching_opt.set_align_CP_multiFFD([[0], [0], [2], [2]])
###############################################################

# # Single FFD block
# # # Create FFD block in igakit format
# # cp_ffd_lims = nonmatching_opt.cpsurf_lims
# # for field in [2]:
# #     cp_ffd_lims[field][1] = 0.02
# #     cp_ffd_lims[field][0] = -0.02
# # FFD_block = create_3D_block(ffd_block_num_el, p, cp_ffd_lims)

# # Set FFD to non-matching optimization instance
# nonmatching_opt.set_FFD(FFD_block.knots, FFD_block.control)
# nonmatching_opt.set_align_CPFFD(align_dir=1)
# # # Set constraint info
# # nonmatching_opt_ffd.set_pin_CPFFD(pin_dir0=2, pin_side0=[1],
# #                                   pin_dir1=1, pin_side1=[1])
# # nonmatching_opt_ffd.set_regu_CPFFD(regu_dir=[None], regu_side=[None])

print("Creating files ...")
# Set up optimization
nonmatching_opt.create_files(save_path=save_path, folder_name=folder_name, 
                             thickness=nonmatching_opt.opt_thickness)
print("Setting variable thickness optimization group ...")
model = VarThOptGroup(nonmatching_opt_ffd=nonmatching_opt)
print("Initializing gropu parameters ...")
model.init_parameters()
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
    prob.driver.options['debug_print'] = ['objs']#, 'desvars']
    prob.driver.options['print_results'] = True
elif optimizer.upper() == 'SLSQP':
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1e-5
    prob.driver.options['disp'] = True
    prob.driver.options['debug_print'] = ['objs']#, 'desvars']
    prob.driver.options['maxiter'] = 50000
else:
    raise ValueError("Undefined optimizer: {}".format(optimizer))

# prob.setup()
# prob.run_driver()

# if mpirank == 0:
#     for i in range(num_surfs):
#         print("Thickness for patch {:2d}: {:10.6f}".format(i, 
#               nonmatching_opt.h_th[i].vector().get_local()[0]))

save_disp = False

if mpirank == 0:
    print("Saving results...")

if save_disp:
    for i in range(nonmatching_opt.num_splines):
        save_results(splines[i], nonmatching_opt.spline_funcs[i], i, 
                     save_path=save_path, folder=folder_name, 
                     save_cpfuncs=True, comm=worldcomm)

# #### Save final shape of FFD block
# # VTK().write("./geometry/FFD_block_initial.vtk", FFD_block)
# init_CP_FFD = ffd_block_list[0].control[:,:,:,0:3].transpose(2,1,0,3).reshape(-1,3)
# final_CP_FFD = init_CP_FFD.copy()
# final_FFD_CP1 = prob[model.inputs_comp_name+'.'+model.cpffd_name_list[0]]
# final_CP_FFD[:,1] = final_FFD_CP1
# final_CP_FFD = final_CP_FFD.reshape(ffd_block_list[0].control[:,:,:,0:3]\
#                .transpose(2,1,0,3).shape)
# final_CP_FFD = final_CP_FFD.transpose(2,1,0,3)
# final_FFD_block = NURBS(ffd_block_list[0].knots, final_CP_FFD)
# VTK().write('./geometry/FFD_block_final.vtk', final_FFD_block)

# h_th_profile0 = []
# h_th_profile1 = []
# h_th_profile2 = []
# h_th_profile3 = []
# num_pts = 101
# xi1_array = np.linspace(0,1,num_pts)
# xi0 = 0.5
# for i in range(len(ffd2_surf_inds)):
#     for xi_ind in range(num_pts):
#         h_th_profile0 += [nonmatching_opt.h_th[ffd2_surf_inds[i]]((xi0, xi1_array[xi_ind]))]

# x0_array = np.linspace(0.,nonmatching_opt.cpsurf_lims_multiffd[0][1][1],num_pts*len(ffd2_surf_inds))
# # np.savez("h_th_profile"+str(test_ind)+".npz", h=h_th_profile, x=x0_array)

# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(x0_array, h_th_profile0, '--')
# plt.show()


# Create a recorder variable
opt_data_dir = save_path+folder_name+'opt_data/'
if not os.path.isdir(opt_data_dir):
    os.mkdir(opt_data_dir)

recorder_name = opt_data_dir+'recorder'+str(test_ind)+'.sql'
shopt_data_name = opt_data_dir+'shopt_ffd_data'+str(test_ind)+'.npz'

prob.driver.recording_options['includes'] = ['*']
prob.driver.recording_options['record_objectives'] = True
prob.driver.recording_options['record_derivatives'] = False
prob.driver.recording_options['record_constraints'] = True
prob.driver.recording_options['record_desvars'] = True
prob.driver.recording_options['record_inputs'] = True
prob.driver.recording_options['record_outputs'] = True
prob.driver.recording_options['record_residuals'] = True

recorder = om.SqliteRecorder(recorder_name)
prob.driver.add_recorder(recorder)

prob.setup()
prob.run_driver()

major_iter_inds = model.disp_states_comp.func_eval_major_ind
np.savez(shopt_data_name,major_iter_ind=major_iter_inds,ffd_control=None,
                          ffd_knots=None,
                          QoI=0.)

# Save initial thickness optimization FFD blocks
thopt_ffd_dir = opt_data_dir+'thopt_ffd_files/'
thopt_ffd_block_name_pre = 'thopt_ffd_block'
thopt_ffd_cp_name_pre = 'thopt_ffd_cp'
if not os.path.isdir(thopt_ffd_dir):
    os.mkdir(thopt_ffd_dir)
for i in range(num_thopt_ffd):
    VTKWriter().write(thopt_ffd_dir+thopt_ffd_block_name_pre+str(i)+".vtk", 
                      thopt_ffd_block_list[i])
    VTKWriter().write_cp(thopt_ffd_dir+thopt_ffd_cp_name_pre+str(i)+".vtk", 
                         thopt_ffd_block_list[i])