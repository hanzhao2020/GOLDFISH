"""
Initial geometry can be downloaded from the following link:
https://drive.google.com/file/d/1fRaho_xzmChlgLdrMM9CQ7WTqr9_DItt/view?usp=share_link
"""
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


class ShapeOptModel(Model):

    def initialize(self):
        self.parameters.declare('nonmatching_opt_ffd')
        self.parameters.declare('cpffd_name_pre', default='CP_FFD')
        self.parameters.declare('cpsurf_fe_name_pre', default='CPS_FE')
        self.parameters.declare('cpsurf_iga_name_pre', default='CPS_IGA')
        self.parameters.declare('disp_name', default='displacements')
        self.parameters.declare('int_energy_name', default='int_E')
        self.parameters.declare('cpffd_pin_name_pre', default='CP_FFD_pin')
        self.parameters.declare('cpffd_regu_name_pre', default='CP_FFD_regu')
        self.parameters.declare('volume_name', default='volume')

    def init_parameters(self):
        self.nonmatching_opt_ffd = self.parameters['nonmatching_opt_ffd']
        self.cpffd_name_pre = self.parameters['cpffd_name_pre']
        self.cpsurf_fe_name_pre = self.parameters['cpsurf_fe_name_pre']
        self.cpsurf_iga_name_pre = self.parameters['cpsurf_iga_name_pre']
        self.disp_name = self.parameters['disp_name']
        self.int_energy_name = self.parameters['int_energy_name']
        self.cpffd_pin_name_pre = self.parameters['cpffd_pin_name_pre']
        self.cpffd_regu_name_pre = self.parameters['cpffd_regu_name_pre']
        self.volume_name = self.parameters['volume_name']

        self.opt_field = self.nonmatching_opt_ffd.opt_field
        self.design_var_lower = -10.
        self.design_var_upper = 1.e-4

        self.cpffd_name_list = []
        self.cpsurf_fe_name_list = []
        self.cpsurf_iga_name_list = []
        self.cpffd_pin_name_list = []
        self.cpffd_regu_name_list = []
        for i, field in enumerate(self.opt_field):
            self.cpffd_name_list += [self.cpffd_name_pre+str(field)]
            self.cpsurf_fe_name_list += [self.cpsurf_fe_name_pre+str(field)]
            self.cpsurf_iga_name_list += [self.cpsurf_iga_name_pre+str(field)]
            self.cpffd_pin_name_list += [self.cpffd_pin_name_pre+str(field)]
            self.cpffd_regu_name_list += [self.cpffd_regu_name_pre+str(field)]

        # Create models' names
        self.inputs_model_name = 'inputs_model'
        self.ffd2surf_model_name = 'FFD2Surf_model'
        self.cpfe2iga_model_name = 'CPFE2IGA_model'
        self.disp_states_model_name = 'disp_states_model'
        self.int_energy_model_name = 'internal_energy_model'
        self.cpffd_pin_model_name = 'CPFFD_pin_model'
        self.cpffd_regu_model_name = 'CPFFD_align_model'
        self.volume_model_name = 'volume_model'

    def define(self):
        for i, field in enumerate(self.opt_field):
            self.create_input(self.cpffd_name_list[i], 
                              shape=(self.nonmatching_opt_ffd.cpffd_size),
                              val=self.nonmatching_opt_ffd.cpffd_flat[:,field])

        # Add FFD model
        self.ffd2surf_model = FFD2SurfModel(
                        nonmatching_opt_ffd=self.nonmatching_opt_ffd,
                        input_cpffd_name_pre=self.cpffd_name_pre,
                        output_cpsurf_name_pre=self.cpsurf_fe_name_pre)
        self.ffd2surf_model.init_parameters()
        self.add(self.ffd2surf_model, self.ffd2surf_model_name, promotes=[])

        # Add CPFE2IGA model
        self.cpfe2iga_model = CPFE2IGAModel(
                        nonmatching_opt=self.nonmatching_opt_ffd,
                        input_cp_fe_name_pre=self.cpsurf_fe_name_pre,
                        output_cp_iga_name_pre=self.cpsurf_iga_name_pre)
        self.cpfe2iga_model.init_parameters()
        self.add(self.cpfe2iga_model, self.cpfe2iga_model_name, promotes=[])

        # Add disp_states model
        self.disp_states_model = DispStatesModel(
                        nonmatching_opt=self.nonmatching_opt_ffd,
                        input_cp_iga_name_pre=self.cpsurf_iga_name_pre,
                        output_u_name=self.disp_name)
        self.disp_states_model.init_parameters(save_files=True)
        self.add(self.disp_states_model, self.disp_states_model_name, 
                 promotes=[])

        # Add internal energy model (objective function)
        self.int_energy_model = IntEnergyModel(
                        nonmatching_opt=self.nonmatching_opt_ffd,
                        input_cp_iga_name_pre=self.cpsurf_iga_name_pre,
                        input_u_name=self.disp_name,
                        output_wint_name=self.int_energy_name)
        self.int_energy_model.init_parameters()
        self.add(self.int_energy_model, self.int_energy_model_name, 
                 promotes=[])

        # Add CP FFD pin comp (linear constraint)
        self.cpffd_pin_model = CPFFDPinModel(
                                nonmatching_opt_ffd=self.nonmatching_opt_ffd,
                         input_cpffd_name_pre=self.cpffd_name_pre,
                         output_cppin_name_pre=self.cpffd_pin_name_pre)
        self.cpffd_pin_model.init_parameters()
        self.add(self.cpffd_pin_model, self.cpffd_pin_model_name, promotes=[])
        self.cpffd_pin_cons_val = []
        for i, field in enumerate(self.opt_field):
            self.cpffd_pin_cons_val += [self.nonmatching_opt_ffd.cpffd_flat
                                [:,field][self.nonmatching_opt_ffd.pin_dof]]

        # Add CP FFD regu model (linear constraint)
        self.cpffd_regu_model = CPFFDReguModel(
                           nonmatching_opt_ffd=self.nonmatching_opt_ffd,
                           input_cpffd_name_pre=self.cpffd_name_pre,
                           output_cpregu_name_pre=self.cpffd_regu_name_pre)
        self.cpffd_regu_model.init_parameters()
        self.add(self.cpffd_regu_model, self.cpffd_regu_model_name, promotes=[])
        self.cpffd_regu_lower = [np.ones(self.cpffd_regu_model.op.\
                                 output_shapes[i])*0.4
                                 for i in range(len(self.opt_field))]

        # Add volume model (constraint)
        self.volume_model = VolumeModel(
                           nonmatching_opt=self.nonmatching_opt_ffd,
                           input_cp_iga_name_pre=self.cpsurf_iga_name_pre,
                           output_vol_name=self.volume_name)
        self.volume_model.init_parameters()
        self.add(self.volume_model, self.volume_model_name, promotes=[])
        self.vol_val = 0
        for s_ind in range(self.nonmatching_opt_ffd.num_splines):
            self.vol_val += assemble(self.nonmatching_opt_ffd.h_th[s_ind]
                            *self.nonmatching_opt_ffd.splines[s_ind].dx)

        # Connect names between components
        for i, field in enumerate(self.opt_field):
            # For optimization components
            self.connect(self.cpffd_name_list[i],
                         self.ffd2surf_model_name+'.'
                         +self.cpffd_name_list[i])
            self.connect(self.ffd2surf_model_name+'.'
                         +self.cpsurf_fe_name_list[i],
                         self.cpfe2iga_model_name+'.'
                         +self.cpsurf_fe_name_list[i])
            self.connect(self.cpfe2iga_model_name+'.'
                         +self.cpsurf_iga_name_list[i],
                         self.disp_states_model_name+'.'
                         +self.cpsurf_iga_name_list[i])
            self.connect(self.cpfe2iga_model_name+'.'
                         +self.cpsurf_iga_name_list[i],
                         self.int_energy_model_name+'.'
                         +self.cpsurf_iga_name_list[i])
            self.connect(self.cpfe2iga_model_name+'.'
                         +self.cpsurf_iga_name_list[i],
                         self.volume_model_name+'.'
                         +self.cpsurf_iga_name_list[i])
            # For constraints
            self.connect(self.cpffd_name_list[i],
                         self.cpffd_pin_model_name+'.'
                         +self.cpffd_name_list[i])
            self.connect(self.cpffd_name_list[i],
                         self.cpffd_regu_model_name+'.'
                         +self.cpffd_name_list[i])

        self.connect(self.disp_states_model_name+'.'+self.disp_name,
                     self.int_energy_model_name+'.'+self.disp_name)

        # Add design variable, constraints and objective
        for i, field in enumerate(self.opt_field):
            self.add_design_variable(self.cpffd_name_list[i],
                                lower=self.design_var_lower,
                                upper=self.design_var_upper)
            self.add_constraint(self.cpffd_pin_model_name+'.'
                                +self.cpffd_pin_name_list[i],
                                equals=self.cpffd_pin_cons_val[i])
            self.add_constraint(self.cpffd_regu_model_name+'.'
                                +self.cpffd_regu_name_list[i],
                                lower=self.cpffd_regu_lower[i])
        self.add_constraint(self.volume_model_name+'.'
                            +self.volume_name,
                            equals=self.vol_val)
        self.add_objective(self.int_energy_model_name+'.'
                           +self.int_energy_name,
                           scaler=1e6)

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
opt_field = [2]
ffd_block_num_el = [1,4,1]
save_path = './'
folder_name = "results/"

filename_igs = "./geometry/init_Tbeam_geom.igs"
igs_shapes = read_igs_file(filename_igs, as_compound=False)
occ_surf_list = [topoface2surface(face, BSpline=True) 
                 for face in igs_shapes]
occ_surf_data_list = [BSplineSurfaceData(surf) for surf in occ_surf_list]
num_surfs = len(occ_surf_list)
p = occ_surf_data_list[0].degree[0]

# Define material and geometric parameters
E = Constant(1.0e12)
nu = Constant(0.)
h_th = Constant(0.1)
penalty_coefficient = 1.0e3
pressure = Constant(1.)

fields0 = [None, [[0,1,2]],]
spline_bc0 = SplineBC(directions=[1], sides=[None, [0]],
                     fields=fields0, n_layers=[None, [1]])
spline_bcs = [spline_bc0,]*2

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

# Display B-spline surfaces and intersections using 
# PythonOCC build-in 3D viewer.
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
nonmatching_opt_ffd.create_mortar_meshes(preprocessor.mortar_nels)

if mpirank == 0:
    print("Setting up mortar meshes...")
nonmatching_opt_ffd.mortar_meshes_setup(preprocessor.mapping_list, 
                    preprocessor.intersections_para_coords, 
                    penalty_coefficient)
pressure = -Constant(1.)
f = as_vector([Constant(0.), Constant(0.), pressure])
source_terms = []
residuals = []
for s_ind in range(nonmatching_opt_ffd.num_splines):
    z = nonmatching_opt_ffd.splines[s_ind].rationalize(
        nonmatching_opt_ffd.spline_test_funcs[s_ind])
    source_terms += [inner(f, z)*nonmatching_opt_ffd.splines[s_ind].dx]
    residuals += [SVK_residual(nonmatching_opt_ffd.splines[s_ind], 
                  nonmatching_opt_ffd.spline_funcs[s_ind], 
                  nonmatching_opt_ffd.spline_test_funcs[s_ind], 
                  E, nu, h_th, source_terms[s_ind])]
nonmatching_opt_ffd.set_residuals(residuals)

# Create FFD block in igakit format
cp_ffd_lims = nonmatching_opt_ffd.cpsurf_lims
for field in [2]:
    cp_range = cp_ffd_lims[field][1] - cp_ffd_lims[field][0]
    cp_ffd_lims[field][0] = cp_ffd_lims[field][0] - 0.2*cp_range
FFD_block = create_3D_block(ffd_block_num_el, p, cp_ffd_lims)

# Set FFD to non-matching optimization instance
nonmatching_opt_ffd.set_FFD(FFD_block.knots, FFD_block.control)
# Set constraint info
nonmatching_opt_ffd.set_pin_CPFFD(pin_dir0=2, pin_side0=[1],
                                  pin_dir1=1, pin_side1=[1])
nonmatching_opt_ffd.set_regu_CPFFD(regu_dir=[None], regu_side=[None])

# Set up optimization
nonmatching_opt_ffd.create_files(save_path=save_path, 
                                 folder_name=folder_name)
model = ShapeOptModel(nonmatching_opt_ffd=nonmatching_opt_ffd)
model.init_parameters()
sim = py_simulator(model, analytics=False)
sim.run()
prob = CSDLProblem(problem_name='T-beam-shape-opt', simulator=sim)

# optimizer = SNOPT(prob, Major_iterations = 1000,
#                   Major_optimality = 1e-6, append2file=False)
optimizer = SLSQP(prob, maxiter=1000, ftol=1e-8)

optimizer.solve()
optimizer.print_results()

if mpirank == 0:
    print("Maximum F2: {:8.6f}".
          format(np.max(nonmatching_opt_ffd.splines[0].cpFuncs[2]
                 .vector().get_local())))
    print("Miminum F2: {:8.6f}".
          format(np.min(nonmatching_opt_ffd.splines[1].cpFuncs[2]
                 .vector().get_local())))

#### Save final shape of FFD block
VTK().write("./geometry/FFD_block_initial.vtk", FFD_block)
init_CP_FFD = FFD_block.control[:,:,:,0:3].transpose(2,1,0,3).reshape(-1,3)
final_CP_FFD = init_CP_FFD.copy()
final_FFD_CP2 = sim[model.cpffd_name_list[0]]
final_CP_FFD[:,2] = final_FFD_CP2
final_CP_FFD = final_CP_FFD.reshape(FFD_block.control[:,:,:,0:3]\
               .transpose(2,1,0,3).shape)
final_CP_FFD = final_CP_FFD.transpose(2,1,0,3)
final_FFD_block = NURBS(FFD_block.knots, final_CP_FFD)
VTK().write('./geometry/FFD_block_final.vtk', final_FFD_block)