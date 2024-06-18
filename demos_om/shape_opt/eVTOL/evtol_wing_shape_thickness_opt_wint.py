"""
The compressed eVTOL geometry can be downloaded from:
    https://drive.google.com/file/d/1IVNmFAEEMyM0p4QuEITgGMuG43UF2q5U/view?usp=sharing
"""
import numpy as np
import matplotlib.pyplot as plt
import openmdao.api as om
from igakit.cad import *
from igakit.io import VTK
from GOLDFISH.nonmatching_opt_om import *
from int_energy_regu_comp import *

class ShapeOptGroup(om.Group):

    def initialize(self):
        self.options.declare('nonmatching_opt_ffd')
        # Shape optimization related arguments
        self.options.declare('cpffd_design_name_pre', default='CP_design_FFD')
        self.options.declare('cpffd_full_name_pre', default='CP_FFD')
        self.options.declare('cpsurf_fe_name_pre', default='CPS_FE')
        self.options.declare('cpsurf_iga_name_pre', default='CPS_IGA')
        # self.options.declare('cpffd_pin_name_pre', default='CP_FFD_pin')
        self.options.declare('cpffd_regu_name_pre', default='CP_FFD_regu')
        # Thickness optimization related arguments
        self.options.declare('h_th_ffd_name', default='thickness')
        self.options.declare('h_th_fe_name', default='thickness_FE')
        self.options.declare('h_th_iga_name', default='thickness_IGA')
        self.options.declare('h_th_ffd_align_name', default='th_FFD_align')
        # Displacement state argument
        self.options.declare('disp_name', default='displacements')
        # Volume constraint
        self.options.declare('volume_name', default='volume')
        # Objective
        self.options.declare('int_energy_name', default='int_E')
        self.options.declare('regu_para', default=0.)

    def init_parameters(self, shape_var_lims, 
                       thickness_var_lims=[1.0e-3, 5.0e-2]):
        self.nonmatching_opt_ffd = self.options['nonmatching_opt_ffd']
        # Shape optimization attributes
        self.cpffd_design_name_pre = self.options['cpffd_design_name_pre']
        self.cpffd_full_name_pre = self.options['cpffd_full_name_pre']
        self.cpsurf_fe_name_pre = self.options['cpsurf_fe_name_pre']
        self.cpsurf_iga_name_pre = self.options['cpsurf_iga_name_pre']
        # self.cpffd_pin_name_pre = self.options['cpffd_pin_name_pre']
        self.cpffd_regu_name_pre = self.options['cpffd_regu_name_pre']
        # Thickness optimization attributes
        self.h_th_ffd_name = self.options['h_th_ffd_name']
        self.h_th_fe_name = self.options['h_th_fe_name']
        self.h_th_iga_name = self.options['h_th_iga_name']
        self.h_th_ffd_align_name = self.options['h_th_ffd_align_name']
        # Displacement attributes
        self.disp_name = self.options['disp_name']
        # Volume constraint
        self.volume_name = self.options['volume_name']
        # Objective
        self.int_energy_name = self.options['int_energy_name']
        # Shape variable limits
        self.shape_var_lims = shape_var_lims
        # Thickness variable limits
        self.thickness_var_lims = thickness_var_lims
        # Shape optimization field
        self.opt_field = self.nonmatching_opt_ffd.opt_field
        self.init_cpffd = self.nonmatching_opt_ffd.shopt_init_cpffd_design
        self.input_cpffd_shapes = [cpffd.size for cpffd in self.init_cpffd]

        self.regu_para = self.options['regu_para']

        self.cpffd_design_name_list = []
        self.cpffd_full_name_list = []
        self.cpsurf_fe_name_list = []
        self.cpsurf_iga_name_list = []
        # self.cpffd_pin_name_list = []
        self.cpffd_regu_name_list = []
        for i, field in enumerate(self.opt_field):
            self.cpffd_design_name_list += [self.cpffd_design_name_pre+str(field)]
            self.cpffd_full_name_list += [self.cpffd_full_name_pre+str(field)]
            self.cpsurf_fe_name_list += [self.cpsurf_fe_name_pre+str(field)]
            self.cpsurf_iga_name_list += [self.cpsurf_iga_name_pre+str(field)]
            # self.cpffd_pin_name_list += [self.cpffd_pin_name_pre+str(field)]
            self.cpffd_regu_name_list += [self.cpffd_regu_name_pre+str(field)]

        self.num_splines = self.nonmatching_opt_ffd.num_splines
        self.nonmatching_opt_ffd.get_init_h_th_multiFFD()

        self.inputs_comp_name = 'inputs_comp'
        # Shape optimization comp names
        self.cpffd_design2full_comp_name = 'CPFFDDesign2Full_comp'
        self.cpffd2fe_comp_name = 'CPFFDFE_comp'
        self.cpfe2iga_comp_name = 'CPFE2IGA_comp'
        self.cpffd_align_comp_name = 'CPFFD_align_comp'
        # self.cpffd_pin_comp_name = 'CPFFD_pin_comp'
        self.cpffd_regu_comp_name = 'CPFFD_regu_comp'
        # Thickness optimization comp names
        self.h_th_ffd2fe_comp_name = 'h_th_ffd2fe_comp'
        self.h_th_fe2iga_comp_name = 'h_th_fe2iga_comp'
        self.h_th_ffd_align_comp_name = 'h_th_ffd_align_comp'
        # Displacement state comp name
        self.disp_states_comp_name = 'disp_states_comp'
        # Volume constraint comp name    
        self.volume_comp_name = 'volume_comp'
        # Internal energy comp name
        self.int_energy_comp_name = 'internal_energy_comp'

    def setup(self):
        # Add inputs comp
        inputs_comp = om.IndepVarComp()
        for i, field in enumerate(self.opt_field):
            inputs_comp.add_output(self.cpffd_design_name_list[i],
                        shape=self.input_cpffd_shapes[i],
                        val=self.init_cpffd[i])
        inputs_comp.add_output(self.h_th_ffd_name,
                    shape=self.nonmatching_opt_ffd.thopt_cpffd_design_size, 
                    val=self.nonmatching_opt_ffd.init_h_th_multiffd)
        self.add_subsystem(self.inputs_comp_name, inputs_comp)

        ###################################
        ## Shape optimization components ##
        ###################################

        # Add CP FFD design to full comp 
        self.cpffd_design2full_comp = CPFFDesign2FullComp(
                        nonmatching_opt_ffd=self.nonmatching_opt_ffd,
                        input_cpffd_design_name_pre=self.cpffd_design_name_pre,
                        output_cpffd_full_name_pre=self.cpffd_full_name_pre)
        self.cpffd_design2full_comp.init_parameters()
        self.add_subsystem(self.cpffd_design2full_comp_name, self.cpffd_design2full_comp)

        # Add FFD comp
        self.ffd2surf_comp = CPFFD2SurfComp(
                        nonmatching_opt_ffd=self.nonmatching_opt_ffd,
                        input_cpffd_name_pre=self.cpffd_full_name_pre,
                        output_cpsurf_name_pre=self.cpsurf_fe_name_pre)
        self.ffd2surf_comp.init_parameters()
        self.add_subsystem(self.cpffd2fe_comp_name, self.ffd2surf_comp)

        # CPFE2IGA comp
        self.cpfe2iga_comp = CPFE2IGAComp(
                        nonmatching_opt=self.nonmatching_opt_ffd,
                        input_cp_fe_name_pre=self.cpsurf_fe_name_pre,
                        output_cp_iga_name_pre=self.cpsurf_iga_name_pre)
        self.cpfe2iga_comp.init_parameters()
        self.add_subsystem(self.cpfe2iga_comp_name, self.cpfe2iga_comp)

        # Add CP FFD regu comp (linear constraint)
        self.cpffd_regu_comp = CPFFDReguComp(
                           nonmatching_opt_ffd=self.nonmatching_opt_ffd,
                           input_cpffd_design_name_pre=self.cpffd_design_name_pre,
                           output_cpregu_name_pre=self.cpffd_regu_name_pre)
        self.cpffd_regu_comp.init_parameters()
        self.add_subsystem(self.cpffd_regu_comp_name, self.cpffd_regu_comp)
        self.cpffd_regu_lower = [np.ones(self.cpffd_regu_comp.\
                                 output_shapes[i])*1.e-1
                                 for i in range(len(self.opt_field))]

        #######################################
        ## Thickness optimization components ##
        #######################################

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


        ########################################
        # Add disp_states_comp
        self.disp_states_comp = DispStatesComp(
                           nonmatching_opt=self.nonmatching_opt_ffd,
                           input_cp_iga_name_pre=self.cpsurf_iga_name_pre,
                           input_h_th_name=self.h_th_iga_name,
                           output_u_name=self.disp_name)
        self.disp_states_comp.init_parameters(save_files=save_files,
                                             nonlinear_solver_rtol=1e-3)
        self.add_subsystem(self.disp_states_comp_name, self.disp_states_comp)


        # Add volume comp (constraint)
        self.volume_comp = VolumeComp(
                           nonmatching_opt=self.nonmatching_opt_ffd,
                           input_cp_iga_name_pre=self.cpsurf_iga_name_pre,
                           input_h_th_name=self.h_th_iga_name,
                           output_vol_name=self.volume_name)
        self.volume_comp.init_parameters()
        self.add_subsystem(self.volume_comp_name, self.volume_comp)
        self.vol_val = 0
        for s_ind in range(self.nonmatching_opt_ffd.num_splines):
            self.vol_val += assemble(self.nonmatching_opt_ffd.h_th[s_ind]
                            *self.nonmatching_opt_ffd.splines[s_ind].dx)

        # Add internal energy comp (objective function)
        # self.regu_para = regu_para
        self.int_energy_comp = IntEnergyReguComp(
                          nonmatching_opt=self.nonmatching_opt_ffd,
                          input_cp_iga_name_pre=self.cpsurf_iga_name_pre,
                          input_h_th_name=self.h_th_iga_name,
                          input_u_name=self.disp_name,
                          output_wint_name=self.int_energy_name,
                          regu_para=self.regu_para)
        self.int_energy_comp.init_parameters()
        self.add_subsystem(self.int_energy_comp_name, self.int_energy_comp)


        ########################
        ## Connect components ##
        ########################

        # Connect names between components for shape optimization
        for i, field in enumerate(self.opt_field):
            # For optimization components
            self.connect(self.inputs_comp_name+'.'
                         +self.cpffd_design_name_list[i],
                         self.cpffd_design2full_comp_name+'.'
                         +self.cpffd_design_name_list[i])

            self.connect(self.cpffd_design2full_comp_name+'.'
                         +self.cpffd_full_name_list[i],
                         self.cpffd2fe_comp_name+'.'
                         +self.cpffd_full_name_list[i])

            self.connect(self.cpffd2fe_comp_name+'.'
                         +self.cpsurf_fe_name_list[i],
                         self.cpfe2iga_comp_name+'.'
                         +self.cpsurf_fe_name_list[i])

            self.connect(self.cpfe2iga_comp_name+'.'
                         +self.cpsurf_iga_name_list[i],
                         self.disp_states_comp_name+'.'
                         +self.cpsurf_iga_name_list[i])

            # For constraints
            self.connect(self.inputs_comp_name+'.'
                         +self.cpffd_design_name_list[i],
                         self.cpffd_regu_comp_name+'.'
                         +self.cpffd_design_name_list[i])

            self.connect(self.cpfe2iga_comp_name+'.'
                         +self.cpsurf_iga_name_list[i],
                         self.volume_comp_name+'.'
                         +self.cpsurf_iga_name_list[i])
            self.connect(self.cpfe2iga_comp_name+'.'
                         +self.cpsurf_iga_name_list[i],
                         self.int_energy_comp_name+'.'
                         +self.cpsurf_iga_name_list[i])

        # Connect names between components for thickness optimization
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


        # Add CPFFD design variable and constraints
        for i, field in enumerate(self.opt_field):
            self.add_design_var(self.inputs_comp_name+'.'
                                +self.cpffd_design_name_list[i],
                                lower=self.shape_var_lims[i][0],
                                upper=self.shape_var_lims[i][1])
            self.add_constraint(self.cpffd_regu_comp_name+'.'
                                +self.cpffd_regu_name_list[i],
                                lower=self.cpffd_regu_lower[i])
        # Add thickness design variable
        self.add_design_var(self.inputs_comp_name+'.'
                            +self.h_th_ffd_name,
                            lower=self.thickness_var_lims[0],
                            upper=self.thickness_var_lims[1],
                            scaler=1e1)
        # Add volume constraint
        self.add_constraint(self.volume_comp_name+'.'
                            +self.volume_name,
                            equals=self.vol_val)
        # Add internal energy as objective function
        self.add_objective(self.int_energy_comp_name+'.'
                           +self.int_energy_name,
                           scaler=1e0)

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

save_files = False
regu_para = 1e-3
optimizer = 'SNOPT'
opt_field = [2]
save_path = './'
folder_name = "results/"

# Define parameters
# Scale down the geometry using ``geom_scale``to make the length 
# of the wing in the span-wise direction is around 5 m.
geom_scale = 2.54e-5  # Convert current length unit to m
E = Constant(68e9)  # Young's modulus, Pa
nu = Constant(0.35)  # Poisson's ratio
# h_th = Constant(3.0e-3)  # Thickness of surfaces, m

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
wing_indices = list(range(12, 18)) + [78, 92, 79]  + list(range(80, 92))
wing_surfaces = [evtol_surfaces[i] for i in wing_indices]
num_surfs = len(wing_surfaces)
if mpirank == 0:
    print("Number of surfaces:", num_surfs)

num_pts_eval = [16]*num_surfs
ref_level_list = [1]*num_surfs

# Meshes that are close to the results in the paper
u_insert_list = [16, 15, 14, 13, 1, 1] \
              + [16, 18, 17] + [4]*12
v_insert_list = [8, 7, 6, 5, 12, 11] \
              + [1]*3 + [1]*12

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

h_th = []
for i in range(num_surfs):
    h_th += [Function(splines[i].V_control)]
    h_th[i].interpolate(Constant(3.0e-3))

# Create non-matching problem
nonmatching_opt_ffd = NonMatchingOptFFD(splines, E, h_th, nu, comm=worldcomm)
nonmatching_opt_ffd.create_mortar_meshes(preprocessor.mortar_nels)

nonmatching_opt_ffd.set_shopt_surf_inds_FFD(opt_field, list(range(num_surfs)))

if mpirank == 0:
    print("Setting up mortar meshes...")

nonmatching_opt_ffd.mortar_meshes_setup(preprocessor.mapping_list, 
                                        preprocessor.intersections_para_coords, 
                                        penalty_coefficient)
################################
## Set shape optimization FFD ##
################################
# Create FFD block in igakit format
shopt_ffd_num_el = [4,8,2]
shopt_cpffd_lims = nonmatching_opt_ffd.cpsurf_des_lims
for field in [2]:
    cp_range = shopt_cpffd_lims[field][1] - shopt_cpffd_lims[field][0]
    shopt_cpffd_lims[field][1] = shopt_cpffd_lims[field][1] + 0.05*cp_range
    shopt_cpffd_lims[field][0] = shopt_cpffd_lims[field][0] - 0.05*cp_range
shopt_ffd_block = create_3D_block(shopt_ffd_num_el, p, shopt_cpffd_lims)
shape_var_lims = [shopt_cpffd_lims[2][0]-cp_range*2, 
                  shopt_cpffd_lims[2][1]+cp_range*2]

nonmatching_opt_ffd.set_shopt_FFD(shopt_ffd_block.knots, 
                                  shopt_ffd_block.control)
# Set constraint info
nonmatching_opt_ffd.set_shopt_align_CPFFD()
nonmatching_opt_ffd.set_shopt_regu_CPFFD()

####################################
## Set thickness optimization FFD ##
####################################
# Variable outer skin setting
lower_skin_inds = [0,2]
upper_skin_inds = [1,3]
thopt_multi_ffd_inds = [lower_skin_inds, upper_skin_inds]
nonmatching_opt_ffd.set_thopt_multiFFD_surf_inds(thopt_multi_ffd_inds)
num_thopt_ffd = nonmatching_opt_ffd.num_thopt_ffd
thopt_ffd_lims_multiffd = nonmatching_opt_ffd.thopt_cpsurf_lims_multiffd

thopt_ffd_num_el = [[3,6,1]]*2
thopt_ffd_p = [2]*num_thopt_ffd
thopt_field = [2]*2

thopt_ffd_block_list = []
for ffd_ind in range(num_thopt_ffd):
    field = thopt_field[ffd_ind]
    cp_range = thopt_ffd_lims_multiffd[ffd_ind][field][1]\
              -thopt_ffd_lims_multiffd[ffd_ind][field][0]
    thopt_ffd_lims_multiffd[ffd_ind][field][1] = \
        thopt_ffd_lims_multiffd[ffd_ind][field][1] + 0.1*cp_range
    thopt_ffd_lims_multiffd[ffd_ind][field][0] = \
        thopt_ffd_lims_multiffd[ffd_ind][field][0] - 0.1*cp_range
    thopt_ffd_block_list += [create_3D_block(thopt_ffd_num_el[ffd_ind],
                                       thopt_ffd_p[ffd_ind],
                                       thopt_ffd_lims_multiffd[ffd_ind])]
thopt_ffd_knots_list = [ffd_block.knots for ffd_block 
                        in thopt_ffd_block_list]
thopt_ffd_control_list = [ffd_block.control for ffd_block 
                          in thopt_ffd_block_list]
print("Setting multiple thickness FFD blocks ...")
nonmatching_opt_ffd.set_thickness_opt(var_thickness=True)
nonmatching_opt_ffd.set_thopt_multiFFD(thopt_ffd_knots_list, 
                                       thopt_ffd_control_list)


# # Distributed load
# Define magnitude of load
load = Constant(120) # The load should be in the unit of N/m^2
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
                  nonmatching_opt_ffd.spline_test_funcs[i], E, nu, h_th[i], 
                  source_terms[i])]
nonmatching_opt_ffd.set_residuals(residuals)

#########################
## Set up optimization ##
#########################
if save_files:
    nonmatching_opt_ffd.create_files(save_path=save_path, folder_name=folder_name, 
                                     thickness=True, refine_mesh=True, ref_nel=48)

t0 = IntEnergyReguExOperation(nonmatching_opt_ffd, Constant(0.))
init_wint = t0.Wint()
print("initial wint:", init_wint)

model = ShapeOptGroup(nonmatching_opt_ffd=nonmatching_opt_ffd, regu_para=regu_para)
model.init_parameters(shape_var_lims=[shape_var_lims])
prob = om.Problem(model=model)

if optimizer.upper() == 'SNOPT':
    prob.driver = om.pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'
    prob.driver.opt_settings['Minor feasibility tolerance'] = 1e-6
    prob.driver.opt_settings['Major feasibility tolerance'] = 1e-6
    prob.driver.opt_settings['Major optimality tolerance'] = 1e-3
    prob.driver.opt_settings['Major iterations limit'] = 50000
    prob.driver.opt_settings['Summary file'] = \
        './SNOPT_report/SNOPT_summary.out'
    prob.driver.opt_settings['Print file'] = \
        './SNOPT_report/SNOPT_print.out'
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

# Create a recorder variable
opt_data_dir = save_path+folder_name+'opt_data/'
if not os.path.isdir(opt_data_dir):
    os.makedirs(opt_data_dir)

recorder_name = opt_data_dir+'recorder.sql'
shopt_data_name = opt_data_dir+'shopt_ffd_data.npz'

prob.driver.recording_options['includes'] = ['*']
prob.driver.recording_options['record_objectives'] = True
prob.driver.recording_options['record_derivatives'] = True
prob.driver.recording_options['record_constraints'] = True
prob.driver.recording_options['record_desvars'] = True
prob.driver.recording_options['record_inputs'] = True
prob.driver.recording_options['record_outputs'] = True
prob.driver.recording_options['record_residuals'] = True

# recorder = om.SqliteRecorder(recorder_name)
# prob.driver.add_recorder(recorder)

prob.setup()
prob.run_driver()

# major_iter_inds = model.disp_states_comp.func_eval_major_ind
# np.savez(shopt_data_name, opt_field=opt_field,
#                           major_iter_ind=major_iter_inds,
#                           ffd_control=np.array(shopt_ffd_block.control,dtype=object),
#                           ffd_knots=np.array(shopt_ffd_block.knots,dtype=object),
#                           QoI=0.)

# # Save initial thickness optimization FFD blocks
# thopt_ffd_dir = opt_data_dir+'thopt_ffd_files/'
# thopt_ffd_block_name_pre = 'thopt_ffd_block'
# thopt_ffd_cp_name_pre = 'thopt_ffd_cp'
# if not os.path.isdir(thopt_ffd_dir):
#     os.mkdir(thopt_ffd_dir)
# for i in range(num_thopt_ffd):
#     VTKWriter().write(thopt_ffd_dir+thopt_ffd_block_name_pre+str(i)+".vtk", 
#                       thopt_ffd_block_list[i])
#     VTKWriter().write_cp(thopt_ffd_dir+thopt_ffd_cp_name_pre+str(i)+".vtk", 
#                          thopt_ffd_block_list[i])

t1 = IntEnergyReguExOperation(nonmatching_opt_ffd, Constant(0.))
final_wint = t1.Wint()
print("final wint:", final_wint)