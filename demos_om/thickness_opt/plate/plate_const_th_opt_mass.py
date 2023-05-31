import numpy as np
import matplotlib.pyplot as plt
import openmdao.api as om
from igakit.cad import *
from igakit.io import VTK
from GOLDFISH.nonmatching_opt_om import *

class ThicknessOptGroup(om.Group):

    def initialize(self):
        self.options.declare('nonmatching_opt')
        self.options.declare('h_th_name_design', default='thickness')
        self.options.declare('h_th_name_full', default='thickness_full')
        self.options.declare('disp_name', default='displacements')
        self.options.declare('max_vM_name', default='max_vM_stress')
        self.options.declare('volume_name', default='volume')

    def init_paramters(self):
        self.nonmatching_opt = self.options['nonmatching_opt']
        self.h_th_name_design = self.options['h_th_name_design']
        self.h_th_name_full = self.options['h_th_name_full']
        self.disp_name = self.options['disp_name']
        self.volume_name = self.options['volume_name']
        self.max_vM_name = self.options['max_vM_name']

        self.design_var_lower = 2e-3
        self.design_var_upper = 5e-2

        self.num_splines = self.nonmatching_opt.num_splines
        self.init_h_th = [np.average(h_th_sub_array) for h_th_sub_array
                          in self.nonmatching_opt.init_h_th_list]

        self.inputs_comp_name = 'inputs_comp'
        self.h_th_map_comp_name = 'h_th_map_comp'
        self.disp_states_comp_name = 'disp_states_comp'
        self.volume_comp_name = 'volume_comp'
        self.max_vM_comp_name = 'max_vM_comp'

    def setup(self):
        # Add inputs comp
        inputs_comp = om.IndepVarComp()
        inputs_comp.add_output(self.h_th_name_design,
                    shape=self.num_splines, val=self.init_h_th)
        self.add_subsystem(self.inputs_comp_name, inputs_comp)

        # Add h_th map comp
        self.h_th_map_comp = HthMapComp(
                        nonmatching_opt=self.nonmatching_opt,
                        input_h_th_name_design=self.h_th_name_design,
                        output_h_th_name_full=self.h_th_name_full)
        self.h_th_map_comp.init_paramters()
        self.add_subsystem(self.h_th_map_comp_name, self.h_th_map_comp)

        # Add disp_states_comp
        self.disp_states_comp = DispStatesComp(
                           nonmatching_opt=self.nonmatching_opt,
                           input_h_th_name=self.h_th_name_full,
                           output_u_name=self.disp_name)
        self.disp_states_comp.init_paramters(save_files=True)
        self.add_subsystem(self.disp_states_comp_name, self.disp_states_comp)

        # Add volume comp (objective function)
        self.volume_comp = VolumeComp(
                           nonmatching_opt=self.nonmatching_opt,
                           input_h_th_name=self.h_th_name_full,
                           output_vol_name=self.volume_name)
        self.volume_comp.init_paramters()
        self.add_subsystem(self.volume_comp_name, self.volume_comp)
        self.vol_val = 0
        for s_ind in range(self.nonmatching_opt.num_splines):
            self.vol_val += assemble(self.nonmatching_opt.h_th[s_ind]
                            *self.nonmatching_opt.splines[s_ind].dx)

        # Add max von Mises stress comp (constraint)
        surf = 'top'
        rho = 1e2
        upper_vM = 1e6
        self.max_vM_comp = MaxvMStressComp(nonmatching_opt=nonmatching_opt,
                           rho=rho, alpha=None, m=upper_vM, surf=surf,
                           method='pnorm', linearize_stress=False, 
                           input_u_name=self.disp_name,
                           input_h_th_name=self.h_th_name_full,
                           output_max_vM_name=self.max_vM_name)
        self.max_vM_comp.init_paramters()
        self.add_subsystem(self.max_vM_comp_name, self.max_vM_comp)

        # Connect names between components
        self.connect(self.inputs_comp_name+'.'
                     +self.h_th_name_design,
                     self.h_th_map_comp_name+'.'
                     +self.h_th_name_design)
        self.connect(self.h_th_map_comp_name+'.'
                     +self.h_th_name_full,
                     self.disp_states_comp_name+'.'
                     +self.h_th_name_full)
        self.connect(self.h_th_map_comp_name+'.'
                     +self.h_th_name_full,
                     self.volume_comp_name+'.'
                     +self.h_th_name_full)
        self.connect(self.h_th_map_comp_name+'.'
                     +self.h_th_name_full,
                     self.max_vM_comp_name+'.'
                     +self.h_th_name_full)
        self.connect(self.disp_states_comp_name+'.'+self.disp_name,
                     self.max_vM_comp_name+'.'+self.disp_name)

        # Add design variable, constraints and objective
        self.add_design_var(self.inputs_comp_name+'.'
                            +self.h_th_name_design,
                            lower=self.design_var_lower,
                            upper=self.design_var_upper,
                            scaler=1e2)

        self.add_constraint(self.max_vM_comp_name+'.'
                            +self.max_vM_name,
                            upper=upper_vM,
                            scaler=1e-6)
        # Use scaler 1e10 for SNOPT optimizer, 1e8 for SLSQP
        self.add_objective(self.volume_comp_name+'.'
                           +self.volume_name,
                           scaler=1.e3)


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

optimizer = 'SLSQP'
# optimizer = 'SNOPT'

save_path = './'
folder_name = "results/"

# Define material, gemetric and coupling parameters
geom_scale = 1.  # Convert current length unit to m
E = Constant(68e9)  # Young's modulus, Pa
nu = Constant(0.35)  # Poisson's ratio

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

# Geometry preprocessing and surface--surface intersections computation
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

# Initial thickness in linear function space
h_th = []
h_val_list = [1e-2]*num_surfs
for i in range(num_surfs):
    h_th += [Function(splines[i].V_linear)]
    h_th[i].interpolate(Constant(h_val_list[i]))

# Create non-matching problem
nonmatching_opt = NonMatchingOpt(splines, E, h_th, nu, opt_thickness=True, 
                                 opt_shape=False, comm=worldcomm)
nonmatching_opt.create_mortar_meshes(preprocessor.mortar_nels)

if mpirank == 0:
    print("Setting up mortar meshes...")

nonmatching_opt.mortar_meshes_setup(preprocessor.mapping_list, 
                                    preprocessor.intersections_para_coords, 
                                    penalty_coefficient)

# Define magnitude of load
load = Constant(-100)
f1 = as_vector([Constant(0.0), Constant(0.0), load])

# Distributed downward load
loads = [f1]*num_surfs
source_terms = []
residuals = []
for i in range(num_surfs):
    source_terms += [inner(loads[i], nonmatching_opt.splines[i].rationalize(
        nonmatching_opt.spline_test_funcs[i]))*nonmatching_opt.splines[i].dx]
    residuals += [SVK_residual(nonmatching_opt.splines[i], 
                               nonmatching_opt.spline_funcs[i], 
                               nonmatching_opt.spline_test_funcs[i], 
                               E, nu, h_th[i], source_terms[i])]
nonmatching_opt.set_residuals(residuals)

if mpirank == 0:
    print("Solving linear non-matching problem ...")
nonmatching_opt.solve_linear_nonmatching_problem()

# Set up optimization
nonmatching_opt.create_files(save_path=save_path, folder_name=folder_name, 
                             thickness=nonmatching_opt.opt_thickness)
model = ThicknessOptGroup(nonmatching_opt=nonmatching_opt)
model.init_paramters()
prob = om.Problem(model=model)

if optimizer.upper() == 'SNOPT':
    prob.driver = om.pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'
    prob.driver.opt_settings['Minor feasibility tolerance'] = 1e-6
    prob.driver.opt_settings['Major feasibility tolerance'] = 1e-6
    prob.driver.opt_settings['Major optimality tolerance'] = 1e-5
    prob.driver.opt_settings['Major iterations limit'] = 50000
    prob.driver.opt_settings['Summary file'] = './SNOPT_summary.out'
    prob.driver.opt_settings['Print file'] = './SNOPT_print.out'
    prob.driver.options['debug_print'] = ['objs', 'desvars']
    prob.driver.options['print_results'] = True
elif optimizer.upper() == 'SLSQP':
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1e-6
    prob.driver.options['disp'] = True
    prob.driver.options['debug_print'] = ['objs', 'desvars']
    prob.driver.options['maxiter'] = 50000
else:
    raise ValueError("Undefined optimizer: {}".format(optimizer))

prob.setup()
prob.run_driver()

if mpirank == 0:
    for i in range(num_surfs):
        print("Thickness for patch {:2d}: {:10.6f}".format(i, 
              nonmatching_opt.h_th[i].vector().get_local()[0]))

von_Mises_funcs = []
vM_max_list = []
for i in range(nonmatching_opt.num_splines):
    von_Mises_proj = nonmatching_opt.splines[i].projectScalarOntoLinears(
                     model.max_vM_comp.max_vm_exop.vMstress[i], lumpMass=False)
    von_Mises_funcs += [von_Mises_proj,]
    vM_max_list += [v2p(von_Mises_funcs[i].vector()).max()[1]]

print("True max stress:", np.max(vM_max_list))

if mpirank == 0:
    print("Saving results...")

save_disp = True
save_stress = True
if save_disp:
    for i in range(nonmatching_opt.num_splines):
        save_results(splines[i], nonmatching_opt.spline_funcs[i], i, 
                     save_path=save_path, folder=folder_name, 
                     save_cpfuncs=True, comm=worldcomm)

if save_stress:
    for i in range(nonmatching_opt.num_splines):
        von_Mises_funcs[i].rename("von_Mises_top_"+str(i), 
                                 "von_Mises_top_"+str(i))
        File(save_path+folder_name+"von_Mises_top_"+str(i)+".pvd") \
            << von_Mises_funcs[i]