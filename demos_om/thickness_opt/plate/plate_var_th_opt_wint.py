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
        self.options.declare('h_th_ffd_regu_name', default='thickness_FFD_regu')
        self.options.declare('volume_name', default='volume')

    def init_paramters(self):
        self.nonmatching_opt_ffd = self.options['nonmatching_opt_ffd']
        self.h_th_ffd_name = self.options['h_th_ffd_name']
        self.h_th_fe_name = self.options['h_th_fe_name']
        self.h_th_iga_name = self.options['h_th_iga_name']
        self.disp_name = self.options['disp_name']
        self.volume_name = self.options['volume_name']
        self.h_th_ffd_align_name = self.options['h_th_ffd_align_name']
        self.int_energy_name = self.options['int_energy_name']
        self.h_th_ffd_regu_name = self.options['h_th_ffd_regu_name']

        self.design_var_lower = 5e-4
        self.design_var_upper = 5e-1

        self.num_splines = self.nonmatching_opt_ffd.num_splines
        self.nonmatching_opt_ffd.get_init_h_th_FFD()

        self.inputs_comp_name = 'inputs_comp'
        self.h_th_ffd2fe_comp_name = 'h_th_ffd2fe_comp'
        self.h_th_fe2iga_comp_name = 'h_th_fe2iga_comp'
        self.disp_states_comp_name = 'disp_states_comp'
        self.volume_comp_name = 'volume_comp'
        self.int_energy_comp_name = 'int_energy_comp'
        self.h_th_ffd_align_comp_name = 'h_th_ffd_align_comp'
        self.h_th_ffd_regu_comp_name = 'h_th_ffd_regu_comp'

    def setup(self):
        # Add inputs comp
        inputs_comp = om.IndepVarComp()
        inputs_comp.add_output(self.h_th_ffd_name,
                    shape=self.nonmatching_opt_ffd.thopt_cpffd_size, 
                    val=self.nonmatching_opt_ffd.init_h_th_ffd)
        self.add_subsystem(self.inputs_comp_name, inputs_comp)

        # Add h_th FFD2FE comp
        self.h_th_ffd2fe_comp = HthFFD2FEComp(
                        nonmatching_opt_ffd=self.nonmatching_opt_ffd,
                        input_h_th_ffd_name=self.h_th_ffd_name,
                        output_h_th_fe_name=self.h_th_fe_name)
        self.h_th_ffd2fe_comp.init_paramters()
        self.add_subsystem(self.h_th_ffd2fe_comp_name, self.h_th_ffd2fe_comp)

        # Add h_th FE2IGA comp
        self.h_th_fe2iga_comp = HthFE2IGAComp(
                        nonmatching_opt=self.nonmatching_opt_ffd,
                        input_h_th_fe_name=self.h_th_fe_name,
                        output_h_th_iga_name=self.h_th_iga_name)
        self.h_th_fe2iga_comp.init_paramters()
        self.add_subsystem(self.h_th_fe2iga_comp_name, self.h_th_fe2iga_comp)

        # Add disp_states_comp
        self.disp_states_comp = DispStatesComp(
                           nonmatching_opt=self.nonmatching_opt_ffd,
                           input_h_th_name=self.h_th_iga_name,
                           output_u_name=self.disp_name)
        self.disp_states_comp.init_paramters(save_files=True)
        self.add_subsystem(self.disp_states_comp_name, self.disp_states_comp)

        # Add internal energy comp (objective function)
        self.int_energy_comp = IntEnergyComp(
                          nonmatching_opt=self.nonmatching_opt_ffd,
                          input_h_th_name=self.h_th_iga_name,
                          input_u_name=self.disp_name,
                          output_wint_name=self.int_energy_name)
        self.int_energy_comp.init_paramters()
        self.add_subsystem(self.int_energy_comp_name, self.int_energy_comp)

        # Add volume comp (objective function)
        self.volume_comp = VolumeComp(
                           nonmatching_opt=self.nonmatching_opt_ffd,
                           input_h_th_name=self.h_th_iga_name,
                           output_vol_name=self.volume_name)
        self.volume_comp.init_paramters()
        self.add_subsystem(self.volume_comp_name, self.volume_comp)
        self.vol_val = 0
        for s_ind in range(self.nonmatching_opt_ffd.num_splines):
            self.vol_val += assemble(self.nonmatching_opt_ffd.h_th[s_ind]
                            *self.nonmatching_opt_ffd.splines[s_ind].dx)

        # Add thickness FFD align comp (linear constraint)
        self.h_th_ffd_align_comp = HthFFDAlignComp(
            nonmatching_opt_ffd=self.nonmatching_opt_ffd,
            input_h_th_name=self.h_th_ffd_name,
            output_h_th_align_name=self.h_th_ffd_align_name)
        self.h_th_ffd_align_comp.init_paramters()
        self.add_subsystem(self.h_th_ffd_align_comp_name, self.h_th_ffd_align_comp)
        self.cpffd_align_cons_val = \
            np.zeros(self.h_th_ffd_align_comp.output_shape)


        self.h_th_ffd_regu_comp = HthFFDReguComp(
            nonmatching_opt_ffd=self.nonmatching_opt_ffd,
            input_h_th_name=self.h_th_ffd_name,
            output_h_th_regu_name=self.h_th_ffd_regu_name)
        self.h_th_ffd_regu_comp.init_paramters()
        self.add_subsystem(self.h_th_ffd_regu_comp_name, self.h_th_ffd_regu_comp)
        self.cpffd_regu_cons_val = \
            np.ones(self.h_th_ffd_regu_comp.output_shape)*1e-5


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

        self.connect(self.inputs_comp_name+'.'+self.h_th_ffd_name,
                     self.h_th_ffd_align_comp_name+'.'+self.h_th_ffd_name)

        self.connect(self.inputs_comp_name+'.'+self.h_th_ffd_name,
                     self.h_th_ffd_regu_comp_name+'.'+self.h_th_ffd_name)

        # Add design variable, constraints and objective
        self.add_design_var(self.inputs_comp_name+'.'
                            +self.h_th_ffd_name,
                            lower=self.design_var_lower,
                            upper=self.design_var_upper,
                            scaler=1e2)

        self.add_constraint(self.h_th_ffd_align_comp_name+'.'
                            +self.h_th_ffd_align_name,
                            equals=self.cpffd_align_cons_val)

        self.add_constraint(self.h_th_ffd_regu_comp_name+'.'
                            +self.h_th_ffd_regu_name,
                            lower=self.cpffd_regu_cons_val)

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

# optimizer = 'SLSQP'
optimizer = 'SNOPT'

save_path = './'
folder_name = "results/"
p_ffd = 3
ffd_block_num_el = [2,1,1]

geom_scale = 1.  # Convert current length unit to m
E = Constant(68e9)  # Young's modulus, Pa
nu = Constant(0.35)  # Poisson's ratio
h_th_val = Constant(1.0e-2)  # Thickness of surfaces, m

# p = test_ind  # spline order
penalty_coefficient = 1.0e3

print("Importing geometry...")
filename_igs = "./geometry/plate_geometry_cubic.igs"
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
    preprocessor.compute_intersections(rtol=1e-6, mortar_refine=2,) 
                                       # edge_rel_ratio=1e-3)
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
    h_th += [Function(splines[i].V_control)]
    h_th[i].interpolate(Constant(h_val_list[i]))

# Create non-matching problem
nonmatching_opt = NonMatchingOptFFD(splines, E, h_th, nu, opt_shape=False, 
                                 opt_thickness=True, var_thickness=True, 
                                 comm=worldcomm)

nonmatching_opt.create_mortar_meshes(preprocessor.mortar_nels)

if mpirank == 0:
    print("Setting up mortar meshes...")

nonmatching_opt.mortar_meshes_setup(preprocessor.mapping_list, 
                            preprocessor.intersections_para_coords, 
                            penalty_coefficient)

# Define magnitude of load
load = Constant(-100) # The load should be in the unit of N/m^3
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

# Create FFD block in igakit format
cp_ffd_lims = nonmatching_opt.cpsurf_lims
for field in [2]:
    cp_ffd_lims[field][1] = 0.05
    cp_ffd_lims[field][0] = -0.05
FFD_block = create_3D_block(ffd_block_num_el, p_ffd, cp_ffd_lims)

VTK().write("./geometry/ffd_block_init.vtk", FFD_block)
vtk_writer = VTKWriter()
vtk_writer.write_cp("./geometry/ffd_cp_init.vtk", FFD_block)

# Set FFD to non-matching optimization instance
nonmatching_opt.set_thopt_FFD(FFD_block.knots, FFD_block.control)
nonmatching_opt.set_thopt_align_CPFFD(thopt_align_dir=1)
nonmatching_opt.set_thopt_regu_CPFFD([None], [None], None)

# Set up optimization
nonmatching_opt.create_files(save_path=save_path, folder_name=folder_name, 
                             thickness=nonmatching_opt.opt_thickness)
model = VarThOptGroup(nonmatching_opt_ffd=nonmatching_opt)
model.init_paramters()
prob = om.Problem(model=model)

if optimizer.upper() == 'SNOPT':
    prob.driver = om.pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'
    prob.driver.opt_settings['Minor feasibility tolerance'] = 1e-6
    prob.driver.opt_settings['Major feasibility tolerance'] = 1e-6
    prob.driver.opt_settings['Major optimality tolerance'] = 1e-6
    prob.driver.opt_settings['Major iterations limit'] = 50000
    prob.driver.opt_settings['Summary file'] = './SNOPT_report/SNOPT_summary.out'
    prob.driver.opt_settings['Print file'] = './SNOPT_report/SNOPT_print.out'
    prob.driver.options['debug_print'] = ['objs', 'desvars']
    prob.driver.options['print_results'] = True
elif optimizer.upper() == 'SLSQP':
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1e-15
    prob.driver.options['disp'] = True
    prob.driver.options['debug_print'] = ['objs', 'desvars']
    prob.driver.options['maxiter'] = 50000
else:
    raise ValueError("Undefined optimizer: {}".format(optimizer))

prob.setup()
prob.run_driver()

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

import matplotlib.pyplot as plt
plt.figure()
plt.plot(x_array, h_th_norm, '-', label='Kirchhoff–Love shell, y=0.5')
plt.legend()
plt.xlabel("x")
plt.ylabel("Normalized thickness")
plt.show()
