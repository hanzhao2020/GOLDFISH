"""
Initial geometry can be downloaded from the following link:
https://drive.google.com/file/d/1fRaho_xzmChlgLdrMM9CQ7WTqr9_DItt/view?usp=share_link
"""
import sys
sys.path.append("./opers/")
sys.path.append("./comps/")

import numpy as np
import matplotlib.pyplot as plt
import openmdao.api as om
from igakit.cad import *
from igakit.io import VTK
from GOLDFISH.nonmatching_opt_om import *

from cpiga2xi_comp import CPIGA2XiComp
from disp_states_mi_comp import DispMintStatesComp
# from pin_cpsurf_comp import CPSurfPinComp
# from align_cpsurf_comp import CPSurfAlignComp
from max_int_xi_comp import MaxIntXiComp
from min_int_xi_comp import MinIntXiComp
from int_xi_edge_comp import IntXiEdgeComp

set_log_active(False)


class ShapeOptGroup(om.Group):

    def initialize(self):
        self.options.declare('nonmatching_opt')
        self.options.declare('cpsurf_iga_design_name_pre', default='CPS_IGA_design')
        self.options.declare('cpsurf_iga_name_pre', default='CPS_IGA')
        self.options.declare('int_name', default='int_para')
        self.options.declare('disp_name', default='displacements')
        self.options.declare('int_energy_name', default='int_E')
        # self.options.declare('cpsurf_align_name_pre', default='CP_FFD_align')
        self.options.declare('cpffd_pin_name_pre', default='CP_FFD_pin')
        self.options.declare('volume_name', default='volume')
        self.options.declare('max_int_xi_name', default='max_int_xi')
        self.options.declare('min_int_xi_name', default='min_int_xi')
        self.options.declare('int_xi_edge_name', default='int_xi_edge')

    def init_parameters(self):
        self.nonmatching_opt = self.options['nonmatching_opt']
        self.cpsurf_iga_design_name_pre = self.options['cpsurf_iga_design_name_pre']
        self.cpsurf_iga_name_pre = self.options['cpsurf_iga_name_pre']
        self.int_name = self.options['int_name']
        self.disp_name = self.options['disp_name']
        self.int_energy_name = self.options['int_energy_name']
        # self.cpsurf_align_name_pre = self.options['cpsurf_align_name_pre']
        self.cpffd_pin_name_pre = self.options['cpffd_pin_name_pre']
        self.volume_name = self.options['volume_name']
        self.max_int_xi_name = self.options['max_int_xi_name']
        self.min_int_xi_name = self.options['min_int_xi_name']
        self.int_xi_edge_name = self.options['int_xi_edge_name']

        self.opt_field = self.nonmatching_opt.opt_field
        
        self.input_cp_shapes = []
        for field_ind, field in enumerate(self.opt_field):        
            self.input_cp_shapes += [len(self.nonmatching_opt.cpdes_iga_dofs[field_ind])]
        # self.design_var_lower = -10.
        # self.design_var_upper = 1.e-4
        self.design_var_lower = -1.
        self.design_var_upper = 1.

        self.cpsurf_iga_design_name_list = []
        self.cpsurf_iga_name_list = []
        self.cpsurf_pin_name_list = []
        # self.cpsurf_align_name_list = []
        for i, field in enumerate(self.opt_field):
            self.cpsurf_iga_name_list += [self.cpsurf_iga_name_pre+str(field)]
            self.cpsurf_iga_design_name_list += [self.cpsurf_iga_design_name_pre+str(field)]
            # self.cpsurf_align_name_list += [self.cpsurf_align_name_pre+str(field)]
            self.cpsurf_pin_name_list += [self.cpffd_pin_name_pre+str(field)]

        self.inputs_comp_name = 'inputs_comp'
        self.cpdesign2full_comp_name = 'CPdesign2surf_comp'
        self.cpiga2xi_comp_name = 'CPIGA2Xi_comp'
        self.disp_states_comp_name = 'disp_states_comp'
        self.int_energy_comp_name = 'internal_energy_comp'
        self.cpsurf_align_comp_name = 'CPS_align_comp'
        self.cpsurf_align_comp2_name = 'CPS_align_comp2'
        self.cpsurf_pin_comp_name = 'CPS_pin_comp'
        self.volume_comp_name = 'volume_comp'
        self.max_int_xi_comp_name = 'max_int_xi_comp'
        self.min_int_xi_comp_name = 'min_int_xi_comp'
        self.int_xi_edge_comp_name = 'int_xi_edge_comp'


    def setup(self):
        # Add inputs comp
        inputs_comp = om.IndepVarComp()
        for i, field in enumerate(self.opt_field):
            inputs_comp.add_output(self.cpsurf_iga_design_name_list[i],
                        shape=self.input_cp_shapes[i],
                        val=self.nonmatching_opt.init_cp_iga_design[field])
        self.add_subsystem(self.inputs_comp_name, inputs_comp)

        self.cpdesign2full_comp = CPSurfDesign2FullComp(
                                  nonmatching_opt=self.nonmatching_opt,
                                  input_cp_design_name=self.cpsurf_iga_design_name_pre,
                                  output_cp_iga_name_pre=self.cpsurf_iga_name_pre)
        self.cpdesign2full_comp.init_parameters()
        self.add_subsystem(self.cpdesign2full_comp_name, self.cpdesign2full_comp)

        # Add CPIGA2Xi comp
        self.cpiga2xi_comp = CPIGA2XiComp(
                        nonmatching_opt=self.nonmatching_opt,
                        input_cp_iga_name_pre=self.cpsurf_iga_name_pre,
                        output_xi_name=self.int_name)
        self.cpiga2xi_comp.init_parameters()
        self.add_subsystem(self.cpiga2xi_comp_name, self.cpiga2xi_comp)

        # Add disp_states_comp
        self.disp_states_comp = DispMintStatesComp(
                           nonmatching_opt=self.nonmatching_opt,
                           input_cp_iga_name_pre=self.cpsurf_iga_name_pre,
                           input_xi_name=self.int_name,
                           output_u_name=self.disp_name)
        self.disp_states_comp.init_parameters(save_files=True,
                                             nonlinear_solver_rtol=1e-3)
        self.add_subsystem(self.disp_states_comp_name, self.disp_states_comp)

        # Add internal energy comp (objective function)
        self.int_energy_comp = IntEnergyComp(
                          nonmatching_opt=self.nonmatching_opt,
                          input_cp_iga_name_pre=self.cpsurf_iga_name_pre,
                          input_u_name=self.disp_name,
                          output_wint_name=self.int_energy_name)
        self.int_energy_comp.init_parameters()
        self.add_subsystem(self.int_energy_comp_name, self.int_energy_comp)

        # # Add CP surf align comp (linear constraint)
        # self.cpsurf_align_comp = CPSurfAlignComp(
        #     nonmatching_opt=self.nonmatching_opt,
        #     align_surf_ind=[1],
        #     align_dir=[1],
        #     input_cp_iga_name_pre=self.cpsurf_iga_name_pre,
        #     output_cp_align_name_pre=self.cpsurf_align_name_pre)
        # self.cpsurf_align_comp.init_parameters()
        # self.add_subsystem(self.cpsurf_align_comp_name, self.cpsurf_align_comp)
        # self.cpsurf_align_cons_val = np.zeros(self.cpsurf_align_comp.output_shape)

        # # Add CP surf pin comp (linear constraint)
        # self.cpsurf_pin_comp = CPSurfPinComp(
        #                  nonmatching_opt=self.nonmatching_opt,
        #                  pin_surf_inds=[0],
        #                  input_cp_iga_name_pre=self.cpsurf_iga_name_pre,
        #                  output_cp_pin_name_pre=self.cpffd_pin_name_pre)
        # self.cpsurf_pin_comp.init_parameters()
        # self.add_subsystem(self.cpsurf_pin_comp_name, self.cpsurf_pin_comp)
        # self.cpsurf_pin_cons_val = np.zeros(self.cpsurf_pin_comp.output_shape)

        # # Add CP surf pin comp (linear constraint)
        # self.cpsurf_pin_comp = CPSurfPinComp(
        #                  nonmatching_opt=self.nonmatching_opt,
        #                  input_cp_iga_name_pre=self.cpsurf_iga_name_pre,
        #                  output_cp_pin_name_pre=self.cpffd_pin_name_pre)
        # self.cpsurf_pin_comp.init_parameters()
        # self.add_subsystem(self.cpsurf_pin_comp_name, self.cpsurf_pin_comp)
        # self.cpsurf_pin_cons_vals = [np.zeros(shape) for shape 
        #                              in self.cpsurf_pin_comp.output_shapes]

        # Add volume comp (constraint)
        self.volume_comp = VolumeComp(
                           nonmatching_opt=self.nonmatching_opt,
                           input_cp_iga_name_pre=self.cpsurf_iga_name_pre,
                           output_vol_name=self.volume_name)
        self.volume_comp.init_parameters()
        self.add_subsystem(self.volume_comp_name, self.volume_comp)
        self.vol_val = 0
        for s_ind in range(self.nonmatching_opt.num_splines):
            self.vol_val += assemble(self.nonmatching_opt.h_th[s_ind]
                            *self.nonmatching_opt.splines[s_ind].dx)

        # Add max int xi comp (constraint)
        self.max_int_xi_comp = MaxIntXiComp(
                               nonmatching_opt=self.nonmatching_opt, rho=1e3,
                               input_xi_name=self.int_name,
                               output_name=self.max_int_xi_name)
        self.max_int_xi_comp.init_parameters()
        self.add_subsystem(self.max_int_xi_comp_name, self.max_int_xi_comp)

        # Add min int xi comp (constraint)
        self.min_int_xi_comp = MinIntXiComp(
                               nonmatching_opt=self.nonmatching_opt, rho=1e3,
                               input_xi_name=self.int_name,
                               output_name=self.min_int_xi_name)
        self.min_int_xi_comp.init_parameters()
        self.add_subsystem(self.min_int_xi_comp_name, self.min_int_xi_comp)

        # Add int xi edge comp (constraint)
        self.int_xi_edge_comp = IntXiEdgeComp(
                               nonmatching_opt=self.nonmatching_opt,
                               input_xi_name=self.int_name,
                               output_name=self.int_xi_edge_name)
        self.int_xi_edge_comp.init_parameters()
        self.add_subsystem(self.int_xi_edge_comp_name, self.int_xi_edge_comp)
        self.int_xi_edge_cons_val = np.zeros(self.int_xi_edge_comp.output_shape)

        # Connect names between components
        for i, field in enumerate(self.opt_field):
            # For optimization components
            self.connect(self.inputs_comp_name+'.'
                         +self.cpsurf_iga_design_name_list[i],
                         self.cpdesign2full_comp_name+'.'
                         +self.cpsurf_iga_design_name_list[i])


            self.connect(self.cpdesign2full_comp_name+'.'
                         +self.cpsurf_iga_name_list[i],
                         self.cpiga2xi_comp_name+'.'
                         +self.cpsurf_iga_name_list[i])

            self.connect(self.cpdesign2full_comp_name+'.'
                         +self.cpsurf_iga_name_list[i],
                         self.disp_states_comp_name+'.'
                         +self.cpsurf_iga_name_list[i])

            self.connect(self.cpdesign2full_comp_name+'.'
                         +self.cpsurf_iga_name_list[i],
                         self.int_energy_comp_name+'.'
                         +self.cpsurf_iga_name_list[i])

            self.connect(self.cpdesign2full_comp_name+'.'
                         +self.cpsurf_iga_name_list[i],
                         self.volume_comp_name+'.'
                         +self.cpsurf_iga_name_list[i])

            # For constraints
            # self.connect(self.cpdesign2full_comp_name+'.'
            #              +self.cpsurf_iga_name_list[i],
            #              self.cpsurf_pin_comp_name+'.'
            #              +self.cpsurf_iga_name_list[i])

            # self.connect(self.inputs_comp_name+'.'
            #              +self.cpsurf_iga_name_list[i],
            #              self.cpsurf_align_comp_name +'.'
            #              +self.cpsurf_iga_name_list[i])

        self.connect(self.cpiga2xi_comp_name+'.'+self.int_name,
                     self.max_int_xi_comp_name+'.'+self.int_name)

        self.connect(self.cpiga2xi_comp_name+'.'+self.int_name,
                     self.min_int_xi_comp_name+'.'+self.int_name)

        self.connect(self.cpiga2xi_comp_name+'.'+self.int_name,
                     self.int_xi_edge_comp_name+'.'+self.int_name)

        self.connect(self.cpiga2xi_comp_name+'.'+self.int_name,
                     self.disp_states_comp_name+'.'+self.int_name)

        self.connect(self.disp_states_comp_name+'.'+self.disp_name,
                     self.int_energy_comp_name+'.'+self.disp_name)

        # Add design variable, constraints and objective
        for i, field in enumerate(self.opt_field):
            self.add_design_var(self.inputs_comp_name+'.'
                                +self.cpsurf_iga_design_name_list[i],
                                lower=self.design_var_lower,
                                upper=self.design_var_upper)
            # self.add_constraint(self.cpsurf_pin_comp_name+'.'
            #                     +self.cpsurf_pin_name_list[i],
            #                     equals=self.cpsurf_pin_cons_val)
            # self.add_constraint(self.cpsurf_align_comp_name+'.'
            #                     +self.cpsurf_align_name_list[i],
            #                     equals=self.cpsurf_align_cons_val[i])

        # self.add_constraint(self.cpiga2xi_comp_name+'.'+self.int_name,
        #                     lower=0., upper=1.)
        self.add_constraint(self.max_int_xi_comp_name+'.'+self.max_int_xi_name,
                            upper=1.)
        self.add_constraint(self.min_int_xi_comp_name+'.'+self.min_int_xi_name,
                            lower=0.)
        self.add_constraint(self.int_xi_edge_comp_name+'.'+self.int_xi_edge_name,
                            equals=self.int_xi_edge_cons_val)

        self.add_constraint(self.volume_comp_name+'.'+self.volume_name,
                            equals=self.vol_val)
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

test_ind = 2
optimizer = 'SNOPT'
# optimizer = 'SLSQP'
opt_field = [0]
# save_path = './'
# save_path = '/home/han/Documents/test_results/'
save_path = '/Users/hanzhao/Documents/test_results/'
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
# int_data_filename = "int_data_2patch.npz"
# if os.path.isfile(int_data_filename):
#     preprocessor.load_intersections_data(int_data_filename)
# else:
#     preprocessor.compute_intersections(mortar_refine=2)
#     preprocessor.save_intersections_data(int_data_filename)

preprocessor.compute_intersections(mortar_refine=1)
if mpirank == 0:
    print("Total DoFs:", preprocessor.total_DoFs)
    print("Number of intersections:", preprocessor.num_intersections_all)

# cpiga2xi = CPIGA2Xi(preprocessor)

# bs_data = [BSplineSurfaceData(bs) for bs in preprocessor.BSpline_surfs]
# cp_shapes = [surf_data.control.shape[0:2]
#              for surf_data in bs_data]

if mpirank == 0:
    print("Creating splines...")
# Create tIGAr extracted spline instances
splines = []
for i in range(num_surfs):
        spline = OCCBSpline2tIGArSpline(preprocessor.BSpline_surfs[i], 
                                        spline_bc=spline_bcs[i], index=i)
        splines += [spline,]

# Create non-matching problem
nonmatching_opt = NonMatchingOpt(splines, E, h_th, nu, comm=worldcomm)
# nonmatching_opt.cp_shapes = cp_shapes

opt_field = [0]
opt_surf_inds = [[1]]

nonmatching_opt.set_geom_preprocessor(preprocessor)


nonmatching_opt.set_shopt_surf_inds(opt_field=opt_field, shopt_surf_inds=opt_surf_inds)
nonmatching_opt.get_init_CPIGA()
nonmatching_opt.set_shopt_align_CP(align_surf_inds=[[1]],
                                   align_dir=[[1]])
                                   
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

# _, uiga_init = nonmatching_opt.solve_nonlinear_nonmatching_problem(iga_dofs=True)

# opt_field = [0,1]
# opt_surf_inds0 = list(range(nonmatching_opt.num_splines))
# opt_surf_inds1 = [1]
# opt_surf_inds = [opt_surf_inds0, opt_surf_inds1]

# nonmatching_opt.set_shape_opt(opt_field=opt_field, opt_surf_inds=opt_surf_inds)
# nonmatching_opt.get_init_CPIGA()
# nonmatching_opt.set_shopt_align_CP(align_surf_inds=[[0,1], [1]],
#                                    align_dir=[[0,1], [0]])
# nonmatching_opt.set_shopt_pin_CP(pin_surf_inds=[[0,1], [1]],
#                                  pin_dir=[[1,0], [1]],
#                                  pin_side=[[0,0], [0]])



#################################
preprocessor.check_intersections_type()
preprocessor.get_diff_intersections()
nonmatching_opt.create_diff_intersections()
#################################

# Set up optimization
nonmatching_opt.create_files(save_path=save_path, 
                             folder_name=folder_name)
model = ShapeOptGroup(nonmatching_opt=nonmatching_opt)
model.init_parameters()
prob = om.Problem(model=model)

if optimizer.upper() == 'SNOPT':
    prob.driver = om.pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'
    prob.driver.opt_settings['Minor feasibility tolerance'] = 1e-6
    prob.driver.opt_settings['Major feasibility tolerance'] = 1e-6
    prob.driver.opt_settings['Major optimality tolerance'] = 1e-5
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
    prob.driver.options['debug_print'] = ['objs']
    prob.driver.options['maxiter'] = 50000
else:
    raise ValueError("Undefined optimizer: {}".format(optimizer))

prob.setup()

# raise RuntimeError

prob.run_driver()

for i in range(num_surfs):
    max_F0 = np.max(nonmatching_opt.splines[i].cpFuncs[0].vector().get_local())
    min_F0 = np.min(nonmatching_opt.splines[i].cpFuncs[0].vector().get_local())
    print("Spline: {:2d}, Max F0: {:8.6f}".format(i, max_F0))
    print("Spline: {:2d}, Min F0: {:8.6f}".format(i, min_F0))