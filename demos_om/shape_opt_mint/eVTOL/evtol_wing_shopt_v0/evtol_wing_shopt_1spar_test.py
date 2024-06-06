"""
Initial geometry can be downloaded from the following link:
https://drive.google.com/file/d/1fRaho_xzmChlgLdrMM9CQ7WTqr9_DItt/view?usp=share_link
"""

import time
from datetime import datetime

import sys
sys.path.append("../T-beam/opers/")
sys.path.append("../T-beam/comps/")

import numpy as np
import matplotlib.pyplot as plt
import openmdao.api as om
from igakit.cad import *
from igakit.io import VTK
from GOLDFISH.nonmatching_opt_om import *

from cpiga2xi_comp import CPIGA2XiComp
from disp_states_mi_comp import DispMintStatesComp
from max_int_xi_comp import MaxIntXiComp
from min_int_xi_comp import MinIntXiComp
# from int_xi_edge_comp_quadratic import IntXiEdgeComp
from int_xi_edge_comp import IntXiEdgeComp
# from int_energy_regu_comp import IntEnergyReguComp

from cpffd_rigid_comp import CPFFRigidComp
from create_geom_evtol_1spar import preprocessor

set_log_active(False)

class ShapeOptGroup(om.Group):

    def initialize(self):
        self.options.declare('nonmatching_opt')
        self.options.declare('cpdesign2analysis')
        self.options.declare('cpsurf_iga_design_name_pre', default='CPS_IGA_design')
        self.options.declare('cpsurf_iga_name_pre', default='CPS_IGA')
        self.options.declare('cpsurf_regu_name_pre', default='CPS_regu')
        self.options.declare('int_name', default='int_para')
        self.options.declare('disp_name', default='displacements')
        self.options.declare('int_energy_name', default='int_E')
        self.options.declare('volume_name', default='volume')
        self.options.declare('max_int_xi_name', default='max_int_xi')
        self.options.declare('min_int_xi_name', default='min_int_xi')
        self.options.declare('int_xi_edge_name', default='int_xi_edge')

    def init_parameters(self):
        self.nonmatching_opt = self.options['nonmatching_opt']
        self.cpdesign2analysis = self.options['cpdesign2analysis']
        self.cpsurf_iga_design_name_pre = self.options['cpsurf_iga_design_name_pre']
        self.cpsurf_iga_name_pre = self.options['cpsurf_iga_name_pre']
        self.cpsurf_regu_name_pre = self.options['cpsurf_regu_name_pre']
        self.int_name = self.options['int_name']
        self.disp_name = self.options['disp_name']
        self.int_energy_name = self.options['int_energy_name']
        self.volume_name = self.options['volume_name']
        self.max_int_xi_name = self.options['max_int_xi_name']
        self.min_int_xi_name = self.options['min_int_xi_name']
        self.int_xi_edge_name = self.options['int_xi_edge_name']

        self.opt_field = self.nonmatching_opt.opt_field

        self.init_cp_design = self.cpdesign2analysis.init_cp_design
        self.input_cp_shapes = [len(cp) for cp in self.init_cp_design]

        self.design_var_lower = [3.8, 2.7] # x min, z min
        self.design_var_upper = [5.6, 3.7] # x max, z max

        self.cpsurf_iga_design_name_list = []
        self.cpsurf_iga_name_list = []
        for i, field in enumerate(self.opt_field):
            self.cpsurf_iga_name_list += [self.cpsurf_iga_name_pre+str(field)]
            self.cpsurf_iga_design_name_list += [self.cpsurf_iga_design_name_pre+str(field)]

        self.regu_field = []
        for i, mat in enumerate(self.cpdesign2analysis.cp_coarse_regu_deriv_list):
            if mat is not None:
                self.regu_field += [self.opt_field[i]]

        self.cpsurf_regu_name_list = []
        for i, field in enumerate(self.regu_field):
            self.cpsurf_regu_name_list += [self.cpsurf_regu_name_pre+str(field)]

        self.inputs_comp_name = 'inputs_comp'
        self.cpdesign2analysis_comp_name = 'CPdesign2analysis_comp'
        self.cpiga2xi_comp_name = 'CPIGA2Xi_comp'
        self.disp_states_comp_name = 'disp_states_comp'
        self.int_energy_comp_name = 'internal_energy_comp'
        self.cpsurf_regu_comp_name = 'CPS_align_comp'
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
                        val=self.init_cp_design[i])
        self.add_subsystem(self.inputs_comp_name, inputs_comp)

        self.cpdesign2analysis_comp = CPSurfDesign2AnalysisComp(
                                  cpdesign2analysis=self.cpdesign2analysis,
                                  input_cp_design_name_pre=self.cpsurf_iga_design_name_pre,
                                  output_cp_analysis_name_pre=self.cpsurf_iga_name_pre)
        self.cpdesign2analysis_comp.init_parameters()
        self.add_subsystem(self.cpdesign2analysis_comp_name, self.cpdesign2analysis_comp)

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
                                             nonlinear_solver_rtol=1e-3,
                                             nonlinear_solver_max_it=10)
        self.add_subsystem(self.disp_states_comp_name, self.disp_states_comp)

        # Add internal energy comp (objective function)
        self.int_energy_comp = IntEnergyComp(
                          nonmatching_opt=self.nonmatching_opt,
                          input_cp_iga_name_pre=self.cpsurf_iga_name_pre,
                          input_u_name=self.disp_name,
                          output_wint_name=self.int_energy_name)
        self.int_energy_comp.init_parameters()
        self.add_subsystem(self.int_energy_comp_name, self.int_energy_comp)

        # Add CP surf pin comp (linear constraint)
        self.cpsurf_regu_comp = CPSurfReguComp(
                         cpdesign2analysis=self.cpdesign2analysis,
                         input_cp_design_name_pre=self.cpsurf_iga_design_name_pre,
                         output_cp_regu_name_pre=self.cpsurf_regu_name_pre)
        self.cpsurf_regu_comp.init_parameters()
        self.add_subsystem(self.cpsurf_regu_comp_name, self.cpsurf_regu_comp)
        self.cpsurf_regu_cons_vals = [np.ones(shape)*2e-2 for shape 
                                     in self.cpsurf_regu_comp.output_shapes]

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

        # # Add max int xi comp (constraint)
        # self.max_int_xi_comp = MaxIntXiComp(
        #                        nonmatching_opt=self.nonmatching_opt, rho=1e3,
        #                        input_xi_name=self.int_name,
        #                        output_name=self.max_int_xi_name)
        # self.max_int_xi_comp.init_parameters()
        # self.add_subsystem(self.max_int_xi_comp_name, self.max_int_xi_comp)

        # # Add min int xi comp (constraint)
        # self.min_int_xi_comp = MinIntXiComp(
        #                        nonmatching_opt=self.nonmatching_opt, rho=1e3,
        #                        input_xi_name=self.int_name,
        #                        output_name=self.min_int_xi_name)
        # self.min_int_xi_comp.init_parameters()
        # self.add_subsystem(self.min_int_xi_comp_name, self.min_int_xi_comp)

        # Add int xi edge comp (constraint)
        self.int_xi_edge_comp = IntXiEdgeComp(
                               nonmatching_opt=self.nonmatching_opt,
                               input_xi_name=self.int_name,
                               output_name=self.int_xi_edge_name)
        self.int_xi_edge_comp.init_parameters()
        self.add_subsystem(self.int_xi_edge_comp_name, self.int_xi_edge_comp)
        self.int_xi_edge_cons_val = np.zeros(self.int_xi_edge_comp.output_shape)
        # self.int_xi_edge_cons_val = self.nonmatching_opt.cpiga2xi.int_edge_cons_vals

        # Connect names between components
        for i, field in enumerate(self.opt_field):
            # For optimization components
            self.connect(self.inputs_comp_name+'.'
                         +self.cpsurf_iga_design_name_list[i],
                         self.cpdesign2analysis_comp_name+'.'
                         +self.cpsurf_iga_design_name_list[i])

            self.connect(self.cpdesign2analysis_comp_name+'.'
                         +self.cpsurf_iga_name_list[i],
                         self.cpiga2xi_comp_name+'.'
                         +self.cpsurf_iga_name_list[i])

            self.connect(self.cpdesign2analysis_comp_name+'.'
                         +self.cpsurf_iga_name_list[i],
                         self.disp_states_comp_name+'.'
                         +self.cpsurf_iga_name_list[i])

            self.connect(self.cpdesign2analysis_comp_name+'.'
                         +self.cpsurf_iga_name_list[i],
                         self.int_energy_comp_name+'.'
                         +self.cpsurf_iga_name_list[i])

            self.connect(self.cpdesign2analysis_comp_name+'.'
                         +self.cpsurf_iga_name_list[i],
                         self.volume_comp_name+'.'
                         +self.cpsurf_iga_name_list[i])

        # self.connect(self.cpiga2xi_comp_name+'.'+self.int_name,
        #              self.max_int_xi_comp_name+'.'+self.int_name)

        # self.connect(self.cpiga2xi_comp_name+'.'+self.int_name,
        #              self.min_int_xi_comp_name+'.'+self.int_name)

        self.connect(self.cpiga2xi_comp_name+'.'+self.int_name,
                     self.int_xi_edge_comp_name+'.'+self.int_name)

        self.connect(self.cpiga2xi_comp_name+'.'+self.int_name,
                     self.disp_states_comp_name+'.'+self.int_name)

        self.connect(self.disp_states_comp_name+'.'+self.disp_name,
                     self.int_energy_comp_name+'.'+self.disp_name)

        # For constraints
        for i, regu_field in enumerate(self.regu_field):
            opt_field_ind = self.opt_field.index(regu_field)
            self.connect(self.inputs_comp_name+'.'
                         +self.cpsurf_iga_design_name_list[opt_field_ind],
                         self.cpsurf_regu_comp_name+'.'
                         +self.cpsurf_iga_design_name_list[opt_field_ind])


        # Add design variable, constraints and objective
        for i, field in enumerate(self.opt_field):
            self.add_design_var(self.inputs_comp_name+'.'
                                +self.cpsurf_iga_design_name_list[i],
                                lower=self.design_var_lower[i],
                                upper=self.design_var_upper[i])

        for i, regu_field in enumerate(self.regu_field):
            self.add_constraint(self.cpsurf_regu_comp_name+'.'
                         +self.cpsurf_regu_name_list[i],
                         lower=self.cpsurf_regu_cons_vals[i])

        self.add_constraint(self.cpiga2xi_comp_name+'.'+self.int_name,
                            lower=0., upper=1.)#, scaler=1e3)
        # self.add_constraint(self.max_int_xi_comp_name+'.'+self.max_int_xi_name,
        #                     upper=1.)
        # self.add_constraint(self.min_int_xi_comp_name+'.'+self.min_int_xi_name,
        #                     lower=0.)
        self.add_constraint(self.int_xi_edge_comp_name+'.'+self.int_xi_edge_name,
                            equals=self.int_xi_edge_cons_val)

        self.add_constraint(self.volume_comp_name+'.'+self.volume_name,
                            lower=self.vol_val*0.5, upper=self.vol_val*2)
        self.add_objective(self.int_energy_comp_name+'.'
                           +self.int_energy_name,
                           scaler=1e7)


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

test_ind = 53
optimizer = 'SNOPT'
# optimizer = 'SLSQP'
# save_path = './'
save_path = '/home/han/Documents/test_results/'
save_path = '/Users/hanzhao/Documents/test_results/'
# folder_name = "results/"
folder_name = "results"+str(test_ind)+"/"

if mpirank == 0:
    print("Test ind:", test_ind)


geom_scale = 2.54e-5  # Convert current length unit to m
E = Constant(68e9)  # Young's modulus, Pa
nu = Constant(0.35)  # Poisson's ratio
penalty_coefficient = 1e3
num_surfs = preprocessor.num_surfs
h_th = [Constant(3.0e-3)]*num_surfs  # Thickness of surfaces, m
# h_th[3:] = [Constant(6.0e-3)]*(num_surfs-3)


if mpirank == 0:
    print("Total DoFs:", preprocessor.total_DoFs)
    print("Number of surfaces:", preprocessor.num_surfs)
    print("Number of intersections:", preprocessor.num_intersections_all)

# # Display B-spline surfaces and intersections using 
# # PythonOCC build-in 3D viewer.
# display, start_display, add_menu, add_function_to_menu = init_display()
# preprocessor.display_surfaces(display, save_fig=False)
# preprocessor.display_intersections(display, color='RED', save_fig=False)

# print(aaa)

if mpirank == 0:
    print("Creating splines...")
# Create tIGAr extracted spline instances
splines = []
for i in range(num_surfs):
    if i in [0, 1, 3, 4]:
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
nonmatching_opt = NonMatchingOpt(splines, E, h_th, nu, comm=worldcomm)

opt_field = [0,2]
shopt_surf_inds = [[3], [3]]

nonmatching_opt.set_shopt_surf_inds(opt_field=opt_field, 
                                    shopt_surf_inds=shopt_surf_inds)
nonmatching_opt.get_init_CPIGA()

nonmatching_opt.set_geom_preprocessor(preprocessor)

nonmatching_opt.create_mortar_meshes(preprocessor.mortar_nels)

if mpirank == 0:
    print("Setting up mortar meshes...")
nonmatching_opt.mortar_meshes_setup(preprocessor.mapping_list, 
                    preprocessor.intersections_para_coords, 
                    penalty_coefficient, 2)
pressure = Constant(1e-1)
f1 = as_vector([Constant(0.), Constant(0.), pressure])
f0 = as_vector([Constant(0.), Constant(0.), Constant(0.)])

# line_force0 = as_vector([Constant(0.), Constant(0.), Constant(1.)])
# xi0 = SpatialCoordinate(nonmatching_opt.splines[0].mesh)
# line_indicator0 = conditional(gt(xi0[1], 1.-1e-3), Constant(1.), Constant(0.))

# line_force1 = as_vector([Constant(0.), Constant(0.), Constant(-1.)])
# xi1 = SpatialCoordinate(nonmatching_opt.splines[1].mesh)
# line_indicator1 = conditional(le(xi1[1], 1.e-3), Constant(1.), Constant(0.))

source_terms = []
residuals = []
for s_ind in range(nonmatching_opt.num_splines):
    if s_ind == 0:
        f = f1
    else:
        f = f0
    z = nonmatching_opt.splines[s_ind].rationalize(
        nonmatching_opt.spline_test_funcs[s_ind])
    source_terms += [inner(f, z)*nonmatching_opt.splines[s_ind].dx]
    residuals += [SVK_residual(nonmatching_opt.splines[s_ind], 
                  nonmatching_opt.spline_funcs[s_ind], 
                  nonmatching_opt.spline_test_funcs[s_ind], 
                  E, nu, h_th[s_ind], source_terms[s_ind])]        
nonmatching_opt.set_residuals(residuals)

cp0_init = nonmatching_opt.splines[-1].cpFuncs[0].vector().get_local()
cp2_init = nonmatching_opt.splines[-1].cpFuncs[2].vector().get_local()



# nonmatching_opt.solve_nonlinear_nonmatching_problem()

# opt_field = [0,1]
# opt_surf_inds0 = list(range(nonmatching_opt.num_splines))
# opt_surf_inds1 = [1]
# opt_surf_inds = [opt_surf_inds0, opt_surf_inds1]


# nonmatching_opt.set_shopt_align_CP(align_surf_inds=[[0,1], [1]],
#                                    align_dir=[[0,1], [0]])
# nonmatching_opt.set_shopt_pin_CP(pin_surf_inds=[[0,1], [1]],
#                                  pin_dir=[[1,0], [1]],
#                                  pin_side=[[0,0], [0]])

# opt_field = [0,2]
# opt_surf_inds = [[3],[3]]
# nonmatching_opt.set_shopt_surf_inds(opt_field=opt_field, shopt_surf_inds=opt_surf_inds)
# # nonmatching_opt.get_init_CPIGA()
# # nonmatching_opt.set_shopt_align_CP(align_surf_inds=[[3],None],
# #                                    align_dir=[[1],None])
# # NonMatchingOpt.set_shopt_regu_CP(regu_surf_inds=[None,[3]],
# #                                  regu_dir=[None,[1]])


#################################
# preprocessor.check_intersections_type()
# preprocessor.get_diff_intersections()
nonmatching_opt.create_diff_intersections()
#################################

# raise RuntimeError


cpsurfd2a = CPSurfDesign2Analysis(preprocessor, opt_field=opt_field, 
            shopt_surf_inds=shopt_surf_inds)
design_degree = [2,1]
p_list = [[design_degree]*len(shopt_surf_inds[0]), 
          [design_degree]*len(shopt_surf_inds[1])]
design_knots = [[0]*(design_degree[0]+1)+[1]*(design_degree[0]+1),
                [0]*(design_degree[1]+1)+[1]*(design_degree[1]+1)]
knots_list = [[design_knots]*len(shopt_surf_inds[0]), 
              [design_knots]*len(shopt_surf_inds[1])]

cpsurfd2a.set_init_knots(p_list, knots_list)
cpsurfd2a.get_init_cp_coarse()
t0 = cpsurfd2a.set_cp_align(0, [1])
t1 = cpsurfd2a.set_cp_regu(2, [1], rev_dir=True)


# nonmatching_opt.cpiga2xi.end_xi_ind = np.array([[70,36], [70,36]],dtype='int32')
# raise RuntimeError


# Set up optimization
nonmatching_opt.create_files(save_path=save_path, 
                             folder_name=folder_name)
model = ShapeOptGroup(nonmatching_opt=nonmatching_opt,
                      cpdesign2analysis=cpsurfd2a)
model.init_parameters()
prob = om.Problem(model=model)


if optimizer.upper() == 'SNOPT':
    prob.driver = om.pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'
    prob.driver.opt_settings['Minor feasibility tolerance'] = 1e-3
    prob.driver.opt_settings['Major feasibility tolerance'] = 1e-3
    prob.driver.opt_settings['Major optimality tolerance'] = 1e-4
    prob.driver.opt_settings['Major iterations limit'] = 50000
    prob.driver.opt_settings['Summary file'] = './SNOPT_report/SNOPT_summary'+str(test_ind)+'.out'
    prob.driver.opt_settings['Print file'] = './SNOPT_report/SNOPT_print'+str(test_ind)+'.out'
    prob.driver.options['debug_print'] = ['objs', 'desvars']
    prob.driver.options['print_results'] = True
elif optimizer.upper() == 'SLSQP':
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1e-9
    prob.driver.options['disp'] = True
    prob.driver.options['debug_print'] = ['objs']
    prob.driver.options['maxiter'] = 50000
else:
    raise ValueError("Undefined optimizer: {}".format(optimizer))


start_time = time.perf_counter()
start_current_time = datetime.now().strftime("%H:%M:%S")
if mpirank == 0:
    print("Start current time:", start_current_time)

prob.setup()
prob.run_driver()