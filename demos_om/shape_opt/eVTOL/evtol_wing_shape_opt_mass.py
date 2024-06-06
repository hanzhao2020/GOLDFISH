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
        # self.design_var_lower = self.design_var_lim[0]
        # self.design_var_upper = self.design_var_lim[1]

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
        # self.int_energy_comp_name = 'internal_energy_comp'
        # self.cpffd_align_comp_name = 'CPFFD_align_comp'
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

        # # Add internal energy comp (objective function)
        # self.int_energy_comp = IntEnergyComp(
        #                   nonmatching_opt=self.nonmatching_opt_ffd,
        #                   input_cp_iga_name_pre=self.cpsurf_iga_name_pre,
        #                   input_u_name=self.disp_name,
        #                   output_wint_name=self.int_energy_name)
        # self.int_energy_comp.init_parameters()
        # self.add_subsystem(self.int_energy_comp_name, self.int_energy_comp)

        # # Add CP FFD align comp (linear constraint)
        # self.cpffd_align_comp = CPFFDAlignComp(
        #     nonmatching_opt_ffd=self.nonmatching_opt_ffd,
        #     input_cpffd_name_pre=self.cpffd_name_pre,
        #     output_cpalign_name_pre=self.cpffd_align_name_pre)
        # self.cpffd_align_comp.init_parameters()
        # self.add_subsystem(self.cpffd_align_comp_name, self.cpffd_align_comp)
        # self.cpffd_align_cons_val = \
        #     np.zeros(self.cpffd_align_comp.output_shape)

        # # Add CP FFD pin comp (linear constraint)
        # self.cpffd_pin_comp = CPFFDPinComp(
        #                  nonmatching_opt_ffd=self.nonmatching_opt_ffd,
        #                  input_cpffd_name_pre=self.cpffd_name_pre,
        #                  output_cppin_name_pre=self.cpffd_pin_name_pre)
        # self.cpffd_pin_comp.init_parameters()
        # self.add_subsystem(self.cpffd_pin_comp_name, self.cpffd_pin_comp)
        # self.cpffd_pin_cons_val = []
        # for i, field in enumerate(self.opt_field):
        #     self.cpffd_pin_cons_val += [self.nonmatching_opt_ffd.cpffd_flat
        #                         [:,field][self.nonmatching_opt_ffd.pin_dof]]

        # Add CP FFD regu comp (linear constraint)
        self.cpffd_regu_comp = CPFFDReguComp(
                           nonmatching_opt_ffd=self.nonmatching_opt_ffd,
                           input_cpffd_name_pre=self.cpffd_name_pre,
                           output_cpregu_name_pre=self.cpffd_regu_name_pre)
        self.cpffd_regu_comp.init_parameters()
        self.add_subsystem(self.cpffd_regu_comp_name, self.cpffd_regu_comp)
        self.cpffd_regu_lower = [np.ones(self.cpffd_regu_comp.\
                                 output_shapes[i])*1.e-1
                                 for i in range(len(self.opt_field))]

        # Add max von Mises stress comp (constraint)
        xi2 = 0.
        rho = 1.0e2
        upper_vM = 1e7
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
            # self.connect(self.cpfe2iga_comp_name+'.'
            #              +self.cpsurf_iga_name_list[i],
            #              self.int_energy_comp_name+'.'
            #              +self.cpsurf_iga_name_list[i])
            self.connect(self.cpfe2iga_comp_name+'.'
                         +self.cpsurf_iga_name_list[i],
                         self.volume_comp_name+'.'
                         +self.cpsurf_iga_name_list[i])



            # For constraints
            # self.connect(self.inputs_comp_name+'.'
            #              +self.cpffd_name_list[i],
            #              self.cpffd_align_comp_name +'.'
            #              +self.cpffd_name_list[i])
            # self.connect(self.inputs_comp_name+'.'
            #              +self.cpffd_name_list[i],
            #              self.cpffd_pin_comp_name+'.'
            #              +self.cpffd_name_list[i])
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

        # self.connect(self.disp_states_comp_name+'.'+self.disp_name,
        #              self.int_energy_comp_name+'.'+self.disp_name)

        # Add design variable, constraints and objective
        for i, field in enumerate(self.opt_field):
            self.add_design_var(self.inputs_comp_name+'.'
                                +self.cpffd_name_list[i],
                                lower=self.design_var_lims[i][0],
                                upper=self.design_var_lims[i][1])
            # self.add_constraint(self.cpffd_align_comp_name+'.'
            #                     +self.cpffd_align_name_list[i],
            #                     equals=self.cpffd_align_cons_val[i])
            # self.add_constraint(self.cpffd_pin_comp_name+'.'
            #                     +self.cpffd_pin_name_list[i],
            #                     equals=self.cpffd_pin_cons_val[i])
            self.add_constraint(self.cpffd_regu_comp_name+'.'
                                +self.cpffd_regu_name_list[i],
                                lower=self.cpffd_regu_lower[i])
        # self.add_constraint(self.volume_comp_name+'.'
        #                     +self.volume_name,
        #                     # equals=self.vol_val)
        #                     upper=1.05*self.vol_val,
        #                     lower=0.98*self.vol_val)

        self.add_constraint(self.max_vM_comp_name+'.'
                            +self.max_vM_name,
                            upper=upper_vM,
                            scaler=1e-7)
        # # Use scaler 1e10 for SNOPT optimizer, 1e8 for SLSQP
        # self.add_objective(self.int_energy_comp_name+'.'
        #                    +self.int_energy_name,
        #                    scaler=1e5)

        self.add_objective(self.volume_comp_name+'.'
                           +self.volume_name,
                           scaler=1.e2)


def clampedBC(spline_generator, side=0, direction=0):
    """
    Apply clamped boundary condition to spline.
    """
    for field in [0,1,2]:
        scalar_spline = spline_generator.getScalarSpline(field)
        side_dofs = scalar_spline.getSideDofs(direction, side, nLayers=2)
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

test_ind = 1
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
# of the wing in the span-wise direction is around 5 m.
geom_scale = 2.54e-5  # Convert current length unit to m
E = Constant(68e9)  # Young's modulus, Pa
nu = Constant(0.35)  # Poisson's ratio
h_th = Constant(3.0e-3)  # Thickness of surfaces, m

p = 3  # spline order
penalty_coefficient = 1.0e3

print("Importing geometry...")
filename_igs = "./geometry/eVTOL_wing_structure.igs"
igs_shapes = read_igs_file(filename_igs, as_compound=False)
evtol_surfaces = [topoface2surface(face, BSpline=True) 
                  for face in igs_shapes]

# Outer skin indices: list(range(12, 18))
# Spars indices: [78, 92, 79]
# Ribs indices: list(range(80, 92))
# wing_indices = list(range(12, 18)) + [78, 92, 79]  + list(range(80, 92))
wing_indices = list(range(12, 18)) + [78, 79]  + [81, 84, 87, 90]
wing_surfaces = [evtol_surfaces[i] for i in wing_indices]
num_surfs = len(wing_surfaces)
if mpirank == 0:
    print("Number of surfaces:", num_surfs)

num_pts_eval = [16]*num_surfs
ref_level_list = [1]*num_surfs

# Meshes that are close to the results in the paper
u_insert_list = [16, 15, 14, 13, 1, 1] \
              + [16, 18] + [4]*4  
v_insert_list = [8, 7, 6, 5, 12, 11] \
              + [1]*2 + [1]*4
# # Meshes with equal numbers of knot insertion, this scheme
# # has slightly smaller QoI due to skinny elements at wingtip
# u_insert_list = [8]*num_surfs
# v_insert_list = [8]*num_surfs

# For the two small NURBS patches at the wingtip, we control the
# refinement level less than 3 to prevent over refinement.
for i in [4,5]:
    if ref_level_list[i] > 4:
        ref_level_list[i] = 2
    elif ref_level_list[i] <=4 and ref_level_list[i] >= 1:
        ref_level_list[i] = 1

u_num_insert = []
v_num_insert = []
for i in range(num_surfs):
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

# write_geom_file(preprocessor.BSpline_surfs_repara, "eVTOL_wing_geom.igs")
# exit()

if mpirank == 0:
    print("Computing intersections...")
int_data_filename = "eVTOL_wing_int_data.npz"
if os.path.isfile(int_data_filename):
    preprocessor.load_intersections_data(int_data_filename)
else:
    preprocessor.compute_intersections(mortar_refine=2)
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
    if i in [0, 1]:
        # Apply clamped BC to surfaces near root
        spline = OCCBSpline2tIGArSpline(
                 preprocessor.BSpline_surfs_refine[i], 
                 setBCs=clampedBC, side=0, direction=0, index=i)
        splines += [spline,]
    else:
        spline = OCCBSpline2tIGArSpline(
                 preprocessor.BSpline_surfs_refine[i], index=i)
        splines += [spline,]

# Create non-matching problem
# problem = NonMatchingCoupling(splines, E, h_th, nu, comm=worldcomm)
nonmatching_opt_ffd = NonMatchingOptFFD(splines, E, h_th, nu, 
                                        opt_field=opt_field, comm=worldcomm)
nonmatching_opt_ffd.create_mortar_meshes(preprocessor.mortar_nels)

if mpirank == 0:
    print("Setting up mortar meshes...")

nonmatching_opt_ffd.mortar_meshes_setup(preprocessor.mapping_list, 
                                        preprocessor.intersections_para_coords, 
                                        penalty_coefficient)

# Define magnitude of load
load = Constant(1e2) # The load should be in the unit of N/m^2
f1 = as_vector([Constant(0.0), Constant(0.0), load])

# Distributed downward load
loads = [f1]*num_surfs
source_terms = []
residuals = []
for i in range(num_surfs):
    z = nonmatching_opt_ffd.splines[i].rationalize(
        nonmatching_opt_ffd.spline_test_funcs[i])
    source_terms += [inner(loads[i], z)\
                     *nonmatching_opt_ffd.splines[i].dx]
    residuals += [SVK_residual(nonmatching_opt_ffd.splines[i], 
                  nonmatching_opt_ffd.spline_funcs[i], 
                  nonmatching_opt_ffd.spline_test_funcs[i], E, nu, h_th, 
                  source_terms[i])]
nonmatching_opt_ffd.set_residuals(residuals)

# Create FFD block in igakit format
cp_ffd_lims = nonmatching_opt_ffd.cpsurf_lims
for field in [2]:
    cp_range = cp_ffd_lims[field][1] - cp_ffd_lims[field][0]
    cp_ffd_lims[field][1] = cp_ffd_lims[field][1] + 0.05*cp_range
    cp_ffd_lims[field][0] = cp_ffd_lims[field][0] - 0.05*cp_range
FFD_block = create_3D_block(ffd_block_num_el, p, cp_ffd_lims)

# Set FFD to non-matching optimization instance
nonmatching_opt_ffd.set_FFD(FFD_block.knots, FFD_block.control)
# Set constraint info
# nonmatching_opt_ffd.set_pin_CPFFD(pin_dir0=1, pin_side0=[0,1],
#                                   pin_dir1=2, pin_side1=[0])
# nonmatching_opt_ffd.set_pin_CPFFD(pin_dir0=1, pin_side0=[0,1],
#                                   pin_dir1=2, pin_side1=[0])
# nonmatching_opt_ffd.set_align_CPFFD(align_dir=[0,2])
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
    prob.driver.opt_settings['Summary file'] = './SNOPT_summary.out'
    prob.driver.opt_settings['Print file'] = './SNOPT_print.out'
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
VTK().write("./geometry/FFD_block_initial.vtk", FFD_block)
init_CP_FFD = FFD_block.control[:,:,:,0:3].transpose(2,1,0,3).reshape(-1,3)
final_CP_FFD = init_CP_FFD.copy()
final_FFD_CP1 = prob[model.inputs_comp_name+'.'+model.cpffd_name_list[0]]
final_CP_FFD[:,1] = final_FFD_CP1
final_CP_FFD = final_CP_FFD.reshape(FFD_block.control[:,:,:,0:3]\
               .transpose(2,1,0,3).shape)
final_CP_FFD = final_CP_FFD.transpose(2,1,0,3)
final_FFD_block = NURBS(FFD_block.knots, final_CP_FFD)
VTK().write('./geometry/FFD_block_final.vtk', final_FFD_block)