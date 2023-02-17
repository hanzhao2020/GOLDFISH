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

from cpiga2xi_comp import CPIGA2XiComp
from disp_states_mi_comp import DispMintStatesComp
from pin_cpsurf_comp import CPSurfPinComp
from align_cpsurf_comp import CPSurfAlignComp


class ShapeOptGroup(om.Group):

    def initialize(self):
        self.options.declare('nonmatching_opt')
        self.options.declare('preprocessor')
        self.options.declare('cpsurf_iga_name_pre', default='CPS_IGA')
        self.options.declare('int_name', default='int_para')
        self.options.declare('disp_name', default='displacements')
        self.options.declare('int_energy_name', default='int_E')
        self.options.declare('cpsurf_align_name_pre', default='CP_FFD_align')
        self.options.declare('cpffd_pin_name_pre', default='CP_FFD_pin')
        self.options.declare('volume_name', default='volume')

    def init_paramters(self):
        self.nonmatching_opt = self.options['nonmatching_opt']
        self.preprocessor = self.options['preprocessor']
        self.cpsurf_iga_name_pre = self.options['cpsurf_iga_name_pre']
        self.int_name = self.options['int_name']
        self.disp_name = self.options['disp_name']
        self.int_energy_name = self.options['int_energy_name']
        self.cpsurf_align_name_pre = self.options['cpsurf_align_name_pre']
        self.cpffd_pin_name_pre = self.options['cpffd_pin_name_pre']
        self.volume_name = self.options['volume_name']


        self.opt_field = self.nonmatching_opt.opt_field
        self.init_cp_iga = self.nonmatching_opt.get_init_CPIGA()
        self.input_cp_shape = self.nonmatching_opt.vec_scalar_iga_dof
        # self.design_var_lower = -10.
        # self.design_var_upper = 1.e-4
        self.design_var_lower = -1.
        self.design_var_upper = 1.

        self.cpsurf_iga_name_list = []
        self.cpsurf_pin_name_list = []
        self.cpsurf_align_name_list = []
        for i, field in enumerate(self.opt_field):
            self.cpsurf_iga_name_list += [self.cpsurf_iga_name_pre+str(field)]
            self.cpsurf_align_name_list += [self.cpsurf_align_name_pre+str(field)]
            self.cpsurf_pin_name_list += [self.cpffd_pin_name_pre+str(field)]

        self.inputs_comp_name = 'inputs_comp'
        self.cpiga2xi_comp_name = 'CPIGA2Xi_comp'
        self.disp_states_comp_name = 'disp_states_comp'
        self.int_energy_comp_name = 'internal_energy_comp'
        self.cpsurf_align_comp_name = 'CPS_align_comp'
        self.cpsurf_align_comp2_name = 'CPS_align_comp2'
        self.cpsurf_pin_comp_name = 'CPS_pin_comp'
        self.volume_comp_name = 'volume_comp'


    def setup(self):
        # Add inputs comp
        inputs_comp = om.IndepVarComp()
        for i, field in enumerate(self.opt_field):
            inputs_comp.add_output(self.cpsurf_iga_name_list[i],
                        shape=self.input_cp_shape,
                        val=self.init_cp_iga[:,field])
        self.add_subsystem(self.inputs_comp_name, inputs_comp)

        # Add CPIGA2Xi comp
        self.cpiga2xi_comp = CPIGA2XiComp(
                        preprocessor=self.preprocessor,
                        int_indices_diff=[0],
                        opt_field=self.opt_field,
                        input_cp_iga_name_pre=self.cpsurf_iga_name_pre,
                        output_xi_name=self.int_name)
        self.cpiga2xi_comp.init_paramters()
        self.add_subsystem(self.cpiga2xi_comp_name, self.cpiga2xi_comp)

        # Add disp_states_comp
        self.disp_states_comp = DispMintStatesComp(
                           nonmatching_opt=self.nonmatching_opt,
                           input_cp_iga_name_pre=self.cpsurf_iga_name_pre,
                           input_xi_name=self.int_name,
                           output_u_name=self.disp_name)
        self.disp_states_comp.init_paramters(save_files=True,
                                             nonlinear_solver_rtol=1e-3)
        self.add_subsystem(self.disp_states_comp_name, self.disp_states_comp)

        # Add internal energy comp (objective function)
        self.int_energy_comp = IntEnergyComp(
                          nonmatching_opt=self.nonmatching_opt,
                          input_cp_iga_name_pre=self.cpsurf_iga_name_pre,
                          input_u_name=self.disp_name,
                          output_wint_name=self.int_energy_name)
        self.int_energy_comp.init_paramters()
        self.add_subsystem(self.int_energy_comp_name, self.int_energy_comp)

        # Add CP FFD align comp (linear constraint)
        self.cpsurf_align_comp = CPSurfAlignComp(
            nonmatching_opt=self.nonmatching_opt,
            align_surf_ind=[1],
            align_dir=[1],
            input_cp_iga_name_pre=self.cpsurf_iga_name_pre,
            output_cp_align_name_pre=self.cpsurf_align_name_pre)
        self.cpsurf_align_comp.init_paramters()
        self.add_subsystem(self.cpsurf_align_comp_name, self.cpsurf_align_comp)
        self.cpsurf_align_cons_val = np.zeros(self.cpsurf_align_comp.output_shape)


        # # Add CP FFD align comp (linear constraint)
        # self.cpsurf_align_comp2 = CPSurfAlignComp(
        #     nonmatching_opt=self.nonmatching_opt,
        #     align_surf_ind=[1],
        #     align_dir=[0],
        #     input_cp_iga_name_pre=self.cpsurf_iga_name_pre,
        #     output_cp_align_name_pre=self.cpsurf_align_name_pre)
        # self.cpsurf_align_comp2.init_paramters()
        # self.add_subsystem(self.cpsurf_align_comp2_name, self.cpsurf_align_comp2)
        # self.cpsurf_align_cons_val2 = np.zeros(self.cpsurf_align_comp2.output_shape)

        # Add CP FFD pin comp (linear constraint)
        self.cpsurf_pin_comp = CPSurfPinComp(
                         nonmatching_opt=self.nonmatching_opt,
                         pin_surf_inds=[0],
                         input_cp_iga_name_pre=self.cpsurf_iga_name_pre,
                         output_cp_pin_name_pre=self.cpffd_pin_name_pre)
        self.cpsurf_pin_comp.init_paramters()
        self.add_subsystem(self.cpsurf_pin_comp_name, self.cpsurf_pin_comp)
        self.cpsurf_pin_cons_val = np.zeros(self.cpsurf_pin_comp.output_shape)

        # Add volume comp (constraint)
        self.volume_comp = VolumeComp(
                           nonmatching_opt=self.nonmatching_opt,
                           input_cp_iga_name_pre=self.cpsurf_iga_name_pre,
                           output_vol_name=self.volume_name)
        self.volume_comp.init_paramters()
        self.add_subsystem(self.volume_comp_name, self.volume_comp)
        self.vol_val = 0
        for s_ind in range(self.nonmatching_opt.num_splines):
            self.vol_val += assemble(self.nonmatching_opt.h_th[s_ind]
                            *self.nonmatching_opt.splines[s_ind].dx)

        # Connect names between components
        for i, field in enumerate(self.opt_field):
            # For optimization components
            self.connect(self.inputs_comp_name+'.'
                         +self.cpsurf_iga_name_list[i],
                         self.cpiga2xi_comp_name+'.'
                         +self.cpsurf_iga_name_list[i])

            self.connect(self.inputs_comp_name+'.'
                         +self.cpsurf_iga_name_list[i],
                         self.disp_states_comp_name+'.'
                         +self.cpsurf_iga_name_list[i])

            self.connect(self.inputs_comp_name+'.'
                         +self.cpsurf_iga_name_list[i],
                         self.int_energy_comp_name+'.'
                         +self.cpsurf_iga_name_list[i])

            self.connect(self.inputs_comp_name+'.'
                         +self.cpsurf_iga_name_list[i],
                         self.volume_comp_name+'.'
                         +self.cpsurf_iga_name_list[i])

            # For constraints
            self.connect(self.inputs_comp_name+'.'
                         +self.cpsurf_iga_name_list[i],
                         self.cpsurf_pin_comp_name+'.'
                         +self.cpsurf_iga_name_list[i])

            self.connect(self.inputs_comp_name+'.'
                         +self.cpsurf_iga_name_list[i],
                         self.cpsurf_align_comp_name +'.'
                         +self.cpsurf_iga_name_list[i])

            # self.connect(self.inputs_comp_name+'.'
            #              +self.cpsurf_iga_name_list[i],
            #              self.cpsurf_align_comp2_name +'.'
            #              +self.cpsurf_iga_name_list[i])

        self.connect(self.cpiga2xi_comp_name+'.'+self.int_name,
                     self.disp_states_comp_name+'.'+self.int_name)

        self.connect(self.disp_states_comp_name+'.'+self.disp_name,
                     self.int_energy_comp_name+'.'+self.disp_name)

        # Add design variable, constraints and objective
        for i, field in enumerate(self.opt_field):
            self.add_design_var(self.inputs_comp_name+'.'
                                +self.cpsurf_iga_name_list[i],
                                lower=self.design_var_lower,
                                upper=self.design_var_upper)
            self.add_constraint(self.cpsurf_pin_comp_name+'.'
                                +self.cpsurf_pin_name_list[i],
                                equals=self.cpsurf_pin_cons_val[i])
            self.add_constraint(self.cpsurf_align_comp_name+'.'
                                +self.cpsurf_align_name_list[i],
                                equals=self.cpsurf_align_cons_val[i])

            # self.add_constraint(self.cpsurf_align_comp2_name+'.'
            #                     +self.cpsurf_align_name_list[i],
            #                     equals=self.cpsurf_align_cons_val2[i])

        self.add_constraint(self.cpiga2xi_comp_name+'.'+self.int_name,
                            lower=0., upper=1.)
        self.add_constraint(self.volume_comp_name+'.'
                            +self.volume_name,
                            equals=self.vol_val)
        # Use scaler 1e10 for SNOPT optimizer, 1e8 for SLSQP
        self.add_objective(self.int_energy_comp_name+'.'
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

test_ind = 5
optimizer = 'SNOPT'
opt_field = [0]
# save_path = './'
save_path = '/home/han/Documents/test_results/'
# folder_name = "results/"
folder_name = "results"+str(test_ind)+"/"

filename_igs = "./geometry/init_Tbeam_geom_moved.igs"
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

# cpiga2xi = CPIGA2Xi(preprocessor)


if mpirank == 0:
    print("Creating splines...")
# Create tIGAr extracted spline instances
splines = []
for i in range(num_surfs):
        spline = OCCBSpline2tIGArSpline(preprocessor.BSpline_surfs[i], 
                                        spline_bc=spline_bcs[i], index=i)
        splines += [spline,]

# Create non-matching problem
nonmatching_opt = NonMatchingOpt(splines, E, h_th, nu, 
                                        opt_field=opt_field, comm=worldcomm)
nonmatching_opt.create_mortar_meshes(preprocessor.mortar_nels)

if mpirank == 0:
    print("Setting up mortar meshes...")
nonmatching_opt.mortar_meshes_setup(preprocessor.mapping_list, 
                    preprocessor.intersections_para_coords, 
                    penalty_coefficient, 2)
pressure = -Constant(1.)
f = as_vector([Constant(0.), Constant(0.), pressure])
source_terms = []
residuals = []
for s_ind in range(nonmatching_opt.num_splines):
    z = nonmatching_opt.splines[s_ind].rationalize(
        nonmatching_opt.spline_test_funcs[s_ind])
    source_terms += [inner(f, z)*nonmatching_opt.splines[s_ind].dx]
    residuals += [SVK_residual(nonmatching_opt.splines[s_ind], 
                  nonmatching_opt.spline_funcs[s_ind], 
                  nonmatching_opt.spline_test_funcs[s_ind], 
                  E, nu, h_th, source_terms[s_ind])]
nonmatching_opt.set_residuals(residuals)


#################################
nonmatching_opt.set_xi_diff_info(preprocessor)
#################################

# Set up optimization
nonmatching_opt.create_files(save_path=save_path, 
                             folder_name=folder_name)
model = ShapeOptGroup(nonmatching_opt=nonmatching_opt,
                      preprocessor=preprocessor)
model.init_paramters()
prob = om.Problem(model=model)

if optimizer.upper() == 'SNOPT':
    prob.driver = om.pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'
    prob.driver.opt_settings['Minor feasibility tolerance'] = 1e-5
    prob.driver.opt_settings['Major feasibility tolerance'] = 1e-5
    prob.driver.opt_settings['Major optimality tolerance'] = 1e-5
    prob.driver.opt_settings['Major iterations limit'] = 50000
    prob.driver.opt_settings['Summary file'] = './SNOPT_summary'+str(test_ind)+'.out'
    prob.driver.opt_settings['Print file'] = './SNOPT_print'+str(test_ind)+'.out'
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
#           format(np.max(nonmatching_opt.splines[0].cpFuncs[2]
#                  .vector().get_local())))
#     print("Miminum F2: {:8.6f}".
#           format(np.min(nonmatching_opt.splines[1].cpFuncs[2]
#                  .vector().get_local())))

# #### Save final shape of FFD block
# VTK().write("./geometry/FFD_block_initial.vtk", FFD_block)
# init_CP_FFD = FFD_block.control[:,:,:,0:3].transpose(2,1,0,3).reshape(-1,3)
# final_CP_FFD = init_CP_FFD.copy()
# final_FFD_CP0 = prob[model.inputs_comp_name+'.'+model.cpffd_name_list[0]]
# final_CP_FFD[:,0] = final_FFD_CP0
# final_CP_FFD = final_CP_FFD.reshape(FFD_block.control[:,:,:,0:3]\
#                .transpose(2,1,0,3).shape)
# final_CP_FFD = final_CP_FFD.transpose(2,1,0,3)
# final_FFD_block = NURBS(FFD_block.knots, final_CP_FFD)
# VTK().write('./geometry/FFD_block_final.vtk', final_FFD_block)