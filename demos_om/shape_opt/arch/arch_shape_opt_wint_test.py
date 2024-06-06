import numpy as np
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
        self.options.declare('int_energy_name', default='int_E')
        self.options.declare('cpffd_align_name_pre', default='CP_FFD_align')
        self.options.declare('cpffd_pin_name_pre', default='CP_FFD_pin')
        self.options.declare('cpffd_regu_name_pre', default='CP_FFD_regu')

    def init_parameters(self):
        self.nonmatching_opt_ffd = self.options['nonmatching_opt_ffd']
        self.cpffd_name_pre = self.options['cpffd_name_pre']
        self.cpsurf_fe_name_pre = self.options['cpsurf_fe_name_pre']
        self.cpsurf_iga_name_pre = self.options['cpsurf_iga_name_pre']
        self.disp_name = self.options['disp_name']
        self.int_energy_name = self.options['int_energy_name']
        self.cpffd_align_name_pre = self.options['cpffd_align_name_pre']
        self.cpffd_pin_name_pre = self.options['cpffd_pin_name_pre']
        self.cpffd_regu_name_pre = self.options['cpffd_regu_name_pre']

        self.opt_field = self.nonmatching_opt_ffd.opt_field
        self.design_var_lower = -1.e-3
        self.design_var_upper = 12.

        self.cpffd_name_list = []
        self.cpsurf_fe_name_list = []
        self.cpsurf_iga_name_list = []
        self.cpffd_align_name_list = []
        self.cpffd_pin_name_list = []
        self.cpffd_regu_name_list = []
        for i, field in enumerate(self.opt_field):
            self.cpffd_name_list += [self.cpffd_name_pre+str(field)]
            self.cpsurf_fe_name_list += [self.cpsurf_fe_name_pre+str(field)]
            self.cpsurf_iga_name_list += [self.cpsurf_iga_name_pre+str(field)]
            self.cpffd_align_name_list += [self.cpffd_align_name_pre+str(field)]
            self.cpffd_pin_name_list += [self.cpffd_pin_name_pre+str(field)]
            self.cpffd_regu_name_list += [self.cpffd_regu_name_pre+str(field)]

        # Create components' names
        self.inputs_comp_name = 'inputs_comp'
        self.ffd2surf_comp_name = 'FFD2Surf_comp'
        self.cpfe2iga_comp_name = 'CPFE2IGA_comp'
        self.disp_states_comp_name = 'disp_states_comp'
        self.int_energy_comp_name = 'internal_energy_comp'
        self.cpffd_align_comp_name = 'CPFFD_align_comp'
        self.cpffd_pin_comp_name = 'CPFFD_pin_comp'
        self.cpffd_regu_comp_name = 'CPFFD_regu_comp'


    def setup(self):
        # Add inputs comp
        inputs_comp = om.IndepVarComp()
        for i, field in enumerate(self.opt_field):
            inputs_comp.add_output(self.cpffd_name_list[i],
                shape=self.nonmatching_opt_ffd.shopt_cpffd_size,
                val=self.nonmatching_opt_ffd.shopt_cpffd_flat[:,field])
        self.add_subsystem(self.inputs_comp_name, inputs_comp)

        # Add FFD comp
        self.ffd2surf_comp = CPFFD2SurfComp(
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
        self.disp_states_comp.init_parameters(save_files=False, 
                              nonlinear_solver_rtol=1e-3)
        self.add_subsystem(self.disp_states_comp_name, self.disp_states_comp)

        # Add internal energy comp (objective function)
        self.int_energy_comp = IntEnergyComp(
            nonmatching_opt=self.nonmatching_opt_ffd,
            input_cp_iga_name_pre=self.cpsurf_iga_name_pre,
            input_u_name=self.disp_name,
            output_wint_name=self.int_energy_name)
        self.int_energy_comp.init_parameters()
        self.add_subsystem(self.int_energy_comp_name, self.int_energy_comp)

        # Add CP FFD align comp (linear constraint)
        self.cpffd_align_comp = CPFFDAlignComp(
            nonmatching_opt_ffd=self.nonmatching_opt_ffd,
            input_cpffd_name_pre=self.cpffd_name_pre,
            output_cpalign_name_pre=self.cpffd_align_name_pre)
        self.cpffd_align_comp.init_parameters()
        self.add_subsystem(self.cpffd_align_comp_name, self.cpffd_align_comp)
        self.cpffd_align_cons_val = \
            np.zeros(self.cpffd_align_comp.output_shape)

        # Add CP FFD pin comp (linear constraint)
        self.cpffd_pin_comp = CPFFDPinComp(
            nonmatching_opt_ffd=self.nonmatching_opt_ffd,
            input_cpffd_name_pre=self.cpffd_name_pre,
            output_cppin_name_pre=self.cpffd_pin_name_pre)
        self.cpffd_pin_comp.init_parameters()
        self.add_subsystem(self.cpffd_pin_comp_name, self.cpffd_pin_comp)
        self.cpffd_pin_cons_val = np.zeros(self.cpffd_pin_comp.output_shape)

        self.cpffd_regu_comp = CPFFDReguComp(
                           nonmatching_opt_ffd=self.nonmatching_opt_ffd,
                           input_cpffd_name_pre=self.cpffd_name_pre,
                           output_cpregu_name_pre=self.cpffd_regu_name_pre)
        self.cpffd_regu_comp.init_parameters()
        self.add_subsystem(self.cpffd_regu_comp_name, self.cpffd_regu_comp)
        self.cpffd_regu_lower = [np.ones(self.cpffd_regu_comp.\
                                 output_shapes[i])*1.e-1
                                 for i in range(len(self.opt_field))]

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
            self.connect(self.cpfe2iga_comp_name+'.'
                         +self.cpsurf_iga_name_list[i],
                         self.int_energy_comp_name+'.'
                         +self.cpsurf_iga_name_list[i])
            # For constraints
            self.connect(self.inputs_comp_name+'.'
                         +self.cpffd_name_list[i],
                         self.cpffd_align_comp_name+'.'
                         +self.cpffd_name_list[i])
            self.connect(self.inputs_comp_name+'.'
                         +self.cpffd_name_list[i],
                         self.cpffd_pin_comp_name+'.'
                         +self.cpffd_name_list[i])
            self.connect(self.inputs_comp_name+'.'
                         +self.cpffd_name_list[i],
                         self.cpffd_regu_comp_name+'.'
                         +self.cpffd_name_list[i])

        self.connect(self.disp_states_comp_name+'.'+self.disp_name,
                     self.int_energy_comp_name+'.'+self.disp_name)

        # Add design variable, constraints and objective
        for i, field in enumerate(self.opt_field):
            self.add_design_var(self.inputs_comp_name+'.'
                                +self.cpffd_name_list[i],
                                lower=self.design_var_lower,
                                upper=self.design_var_upper)
            self.add_constraint(self.cpffd_align_comp_name+'.'
                                +self.cpffd_align_name_list[i],
                                equals=self.cpffd_align_cons_val)
            self.add_constraint(self.cpffd_pin_comp_name+'.'
                                +self.cpffd_pin_name_list[i],
                                equals=self.cpffd_pin_cons_val)
            self.add_constraint(self.cpffd_regu_comp_name+'.'
                                +self.cpffd_regu_name_list[i],
                                lower=self.cpffd_regu_lower[i])
        self.add_objective(self.int_energy_comp_name+'.'
                           +self.int_energy_name,
                           scaler=1e8)


class SplineBC(object):
    """
    Setting Dirichlet boundary condition to tIGAr spline generator.
    """
    def __init__(self, directions=[0,1], sides=[[0,1],[0,1]], 
                 fields=[[[0,1,2],[0,1,2]],[[0,1,2],[0,1,2]]],
                 n_layers=[[1,1],[1,1]]):
        self.fields = fields
        self.directions = directions
        self.sides = sides
        self.n_layers = n_layers

    def set_bc(self, spline_generator):
        for direction in self.directions:
            for side in self.sides[direction]:
                for field in self.fields[direction][side]:
                    scalar_spline = spline_generator.getScalarSpline(field)
                    side_dofs = scalar_spline.getSideDofs(direction,
                                side, nLayers=self.n_layers[direction][side])
                    spline_generator.addZeroDofs(field, side_dofs)


def OCCBSpline2tIGArSpline(surface, num_field=3, quad_deg_const=3, 
                           spline_bc=None, index=0):
    """
    Generate ExtractedBSpline from OCC B-spline surface.
    """
    quad_deg = surface.UDegree()*quad_deg_const
    # DIR = SAVE_PATH+"spline_data/extraction_"+str(index)+"_init"
    # spline = ExtractedSpline(DIR, quad_deg)
    spline_mesh = NURBSControlMesh4OCC(surface, useRect=False)
    spline_generator = EqualOrderSpline(worldcomm, num_field, spline_mesh)
    if spline_bc is not None:
        spline_bc.set_bc(spline_generator)
    # spline_generator.writeExtraction(DIR)
    spline = ExtractedSpline(spline_generator, quad_deg)
    return spline

optimizer = 'SLSQP'
# optimizer = 'SNOPT'
test_ind = 1

opt_field = [2]
ffd_block_num_el = [4,1,1]
p_ffd = 2
# save_path = './'
# save_path = '/home/han/Documents/test_results/'
save_path = '/Users/hanzhao/Documents/test_results/'
folder_name = "results"+str(test_ind)+"/"

filename_igs = "./geometry/init_arch_geom_ref1.igs"
igs_shapes = read_igs_file(filename_igs, as_compound=False)
occ_surf_list = [topoface2surface(face, BSpline=True) 
                 for face in igs_shapes]
occ_surf_data_list = [BSplineSurfaceData(surf) for surf in occ_surf_list]
num_surfs = len(occ_surf_list)
p = occ_surf_data_list[0].degree[0]

# Define material and geometric parameters
E = Constant(1.0e12)
nu = Constant(0.)
h_th = Constant(0.01)
penalty_coefficient = 1.0e3
pressure = Constant(1.)

fields0 = [[[0,1,2]],]
fields1 = [[None,[0,1,2]],]
spline_bc0 = SplineBC(directions=[0], sides=[[0],],
                     fields=fields0, n_layers=[[1],])
spline_bc1 = SplineBC(directions=[0], sides=[[1],],
                     fields=fields1, n_layers=[[None, 1],])
spline_bcs = [spline_bc0, None, None, spline_bc1]*1

# Geometry preprocessing and surface-surface intersections computation
preprocessor = OCCPreprocessing(occ_surf_list, reparametrize=False, 
                                refine=False)
print("Computing intersections...")
int_data_filename = "int_data.npz"
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
        spline = OCCBSpline2tIGArSpline(preprocessor.BSpline_surfs[i], 
                                        spline_bc=spline_bcs[i], index=i)
        splines += [spline,]

# Create non-matching problem
nonmatching_opt_ffd = NonMatchingOptFFD(splines, E, h_th, nu, 
                                        opt_field=opt_field, comm=worldcomm)

# # Save initial discretized geometry
# nonmatching_opt_ffd.create_files(save_path=save_path, 
#                                  folder_name='results_temp/', 
#                                  refine_mesh=False)
# nonmatching_opt_ffd.save_files()
# exit()

nonmatching_opt_ffd.create_mortar_meshes(preprocessor.mortar_nels)

if mpirank == 0:
    print("Setting up mortar meshes...")
nonmatching_opt_ffd.mortar_meshes_setup(preprocessor.mapping_list, 
                    preprocessor.intersections_para_coords, 
                    penalty_coefficient)

# Define magnitude of load
load = Constant(1.) # The load should be in the unit of N/m^2
force_list = []
source_terms = []
residuals = []
for i in range(num_surfs):
    X = nonmatching_opt_ffd.splines[i].F
    A0,A1,A2,_,A,B = surfaceGeometry(nonmatching_opt_ffd.splines[i],X)
    v_vec = as_vector([Constant(0.), Constant(0.), Constant(1.)])
    cos_beta = inner(v_vec, A2)
    force = as_vector([Constant(0.), Constant(0.), -load*cos_beta])
    # force = as_vector([Constant(0.), Constant(0.), -Constant(1.)])
    force_list += [force]
    source_terms += [inner(force_list[i], nonmatching_opt_ffd.splines[i]\
                    .rationalize(nonmatching_opt_ffd.spline_test_funcs[i]))\
                    *nonmatching_opt_ffd.splines[i].dx]
    residuals += [SVK_residual(nonmatching_opt_ffd.splines[i], 
                  nonmatching_opt_ffd.spline_funcs[i], 
                  nonmatching_opt_ffd.spline_test_funcs[i], 
                  E, nu, h_th, source_terms[i])]
nonmatching_opt_ffd.set_residuals(residuals)

# Create FFD block in igakit format
cp_ffd_lims = nonmatching_opt_ffd.cpsurf_lims
for field in [2]:
    cp_range = cp_ffd_lims[field][1] - cp_ffd_lims[field][0]
    cp_ffd_lims[field][1] = cp_ffd_lims[field][1] + 0.2*cp_range
FFD_block = create_3D_block(ffd_block_num_el, p_ffd, cp_ffd_lims)

# Set FFD to non-matching optimization instance
nonmatching_opt_ffd.set_shopt_FFD(FFD_block.knots, FFD_block.control)
# Set constraint info
nonmatching_opt_ffd.set_shopt_align_CPFFD(shopt_align_dir=1)
nonmatching_opt_ffd.set_shopt_pin_CPFFD(pin_dir0=2, pin_side0=[0],
                                        pin_dir1=1, pin_side1=[0])
nonmatching_opt_ffd.set_shopt_regu_CPFFD(shopt_regu_dir=[None], 
                                         shopt_regu_side=[None],)
                                         # shopt_regu_align=[1])

# Set up optimization
nonmatching_opt_ffd.create_files(save_path=save_path, 
                                 folder_name=folder_name, 
                                 refine_mesh=True)
model = ShapeOptGroup(nonmatching_opt_ffd=nonmatching_opt_ffd)
model.init_parameters()
prob = om.Problem(model=model)

if optimizer.upper() == 'SNOPT':
    prob.driver = om.pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'
    prob.driver.opt_settings['Minor feasibility tolerance'] = 1e-6
    prob.driver.opt_settings['Major feasibility tolerance'] = 1e-6
    prob.driver.opt_settings['Major optimality tolerance'] = 1e-2
    prob.driver.opt_settings['Major iterations limit'] = 50000
    prob.driver.opt_settings['Summary file'] = './SNOPT_report/SNOPT_summary'+str(test_ind)+'.out'
    prob.driver.opt_settings['Print file'] = './SNOPT_report/SNOPT_print'+str(test_ind)+'.out'
    prob.driver.options['debug_print'] = ['objs']#, 'desvars']
    prob.driver.options['print_results'] = True
elif optimizer.upper() == 'SLSQP':
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1e-12
    prob.driver.options['disp'] = True
    prob.driver.options['debug_print'] = ['objs']#, 'desvars']
    prob.driver.options['maxiter'] = 50000
else:
    raise ValueError("Undefined optimizer: {}".format(optimizer))

# Create a recorder variable
recorder_name = './opt_data/recorder'+str(test_ind)+'.sql'
FFD_data_name = './opt_data/FFD_data'+str(test_ind)+'.npz'

prob.driver.recording_options['includes'] = ['*']
prob.driver.recording_options['record_objectives'] = True
prob.driver.recording_options['record_derivatives'] = True
prob.driver.recording_options['record_constraints'] = True
prob.driver.recording_options['record_desvars'] = True
prob.driver.recording_options['record_inputs'] = True
prob.driver.recording_options['record_outputs'] = True
prob.driver.recording_options['record_residuals'] = True

recorder = om.SqliteRecorder(recorder_name)
prob.driver.add_recorder(recorder)

prob.setup()
# prob.set_solver_print(0)
prob.run_driver()

if mpirank == 0:
    max_F2 = []
    for i in range(num_surfs):
        max_F2 += [np.max(nonmatching_opt_ffd.splines[i].cpFuncs[2]
                   .vector().get_local())]
    print("Maximum F2: {:8.6f} (reference: 5.4779)".
          format(np.max(max_F2)))


major_iter_inds = model.disp_states_comp.func_eval_major_ind
np.savez(FFD_data_name, opt_field=opt_field,
                        major_iter_ind=major_iter_inds,
                        ffd_control=FFD_block.control,
                        ffd_knots=FFD_block.knots,
                        QoI=np.max(max_F2))

# # # Load recorder data
# # Instantiate your CaseReader
# cr = om.CaseReader(recorder_name)
# # Isolate "problem" as your source
# # driver_cases = cr.list_cases('driver', out_stream=None)
# driver_cases = cr.get_cases('driver', recurse=False)

# def update_FFD_block(FFD_block, new_cps, opt_field):
#     init_cpffd_flat = FFD_block.control[:,:,:,0:3].\
#                       transpose(2,1,0,3).reshape(-1,3)
#     new_cpffd_flat = init_cpffd_flat.copy()
#     for i, field in enumerate(opt_field):
#         new_cpffd_flat[:, field] = new_cps[i]
#     new_cpffd = new_cpffd_flat.reshape(FFD_block.control[...,0:3].\
#                 transpose(2,1,0,3).shape)
#     new_cpffd = new_cpffd.transpose(2,1,0,3)
#     new_FFD_block = NURBS(FFD_block.knots, new_cpffd)
#     return new_FFD_block

# ffd_block_name_pre = 'FFD_block_'
# ffd_CP_name_pre = 'FFD_CP_'
# ffd_path = './geometry/'
# wint_list_major = []
# design_var_list_major = []
# FFD_block_list = []
# for i, major_ind in enumerate(SLSQP_major_inds):
#     case = driver_cases[major_ind]
#     wint_list_major += [case.get_objectives()['internal_energy_comp.int_E'][0]]
#     design_val = []
#     for val in case.get_design_vars().values():
#         design_val += [val]
#     design_var_list_major += [design_val]
#     FFD_block_list += [update_FFD_block(FFD_block, design_val, opt_field),]
#     # Save files
#     ffd_name_ind = '0'*(6-len(str(i)))+str(i)
#     ffd_block_name_vtk = ffd_path + ffd_block_name_pre + ffd_name_ind + '.vtk'
#     VTKWriter().write(ffd_block_name_vtk, FFD_block_list[i], ref_level=3)
#     ffd_block_name_vtu = ffd_path + ffd_block_name_pre + ffd_name_ind + '.vtu'
#     ffd_block_vtk_file = meshio.read(ffd_block_name_vtk)
#     meshio.write(ffd_block_name_vtu, ffd_block_vtk_file)
#     ffd_cp_name_vtk = ffd_path + ffd_CP_name_pre + ffd_name_ind + '.vtk'
#     VTKWriter().write_cp(ffd_cp_name_vtk, FFD_block_list[i])
#     ffd_cp_name_vtu = ffd_path + ffd_CP_name_pre + ffd_name_ind + '.vtu'
#     ffd_cp_vtk_file = meshio.read(ffd_cp_name_vtk)
#     meshio.write(ffd_cp_name_vtu, ffd_cp_vtk_file)


# #### Save initial and final shape of FFD block
# VTK().write("./geometry/FFD_block_initial.vtk", FFD_block)
# init_cpffd = FFD_block.control[:,:,:,0:3].transpose(2,1,0,3).reshape(-1,3)
# final_cpffd = init_cpffd.copy()
# final_cpffd2 = prob[model.inputs_comp_name+'.'+model.cpffd_name_list[0]]
# final_cpffd[:,2] = final_cpffd2
# final_cpffd = final_cpffd.reshape(FFD_block.control[:,:,:,0:3]\
#               .transpose(2,1,0,3).shape)
# final_cpffd = final_cpffd.transpose(2,1,0,3)
# final_FFD_block = NURBS(FFD_block.knots, final_cpffd)
# VTK().write('./geometry/FFD_block_final.vtk', final_FFD_block)

# # Save initial discretized geometry
# nonmatching_opt_ffd.create_files(save_path=save_path, 
#                                  folder_name='results_temp/', 
#                                  refine_mesh=False)

# Save intersection curves
for i in range(preprocessor.num_intersections_all):
    mesh_phy = generate_mortar_mesh(preprocessor.intersections_phy_coords[i], num_el=128)
    File('./geometry/int_curve'+str(i)+'.pvd') << mesh_phy

# from PENGoLINS.igakit_utils import *
# vtk_writer = VTKWriter()
# ik_surfs = []
# for i in range(len(occ_surf_list)):
#     ik_surfs += [BSpline_surface2ikNURBS(occ_surf_list[i])]
#     vtk_writer.write('./geometry/surf_geom/surf'+str(i)+'.vtk', ik_surfs[i], ref_level=5)
#     vtk_writer.write_cp('./geometry/surf_geom/surf_cp'+str(i)+'.vtk', ik_surfs[i])