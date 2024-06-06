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
from GOLDFISH.om_comps.cpffd_regu_comp_aggregated import CPFFDReguCompAgg
# from GOLDFISH.om_comps.cpffd_align_comp_quadratic import CPFFDAlignCompQuad
# from GOLDFISH.om_comps.cpffd_pin_comp_quadratic import CPFFDPinCompQuad

from cpiga2xi_comp import CPIGA2XiComp
from cpiga2cpint_comp import CPIGA2CPIntComp
from disp_states_mi_comp import DispMintStatesComp
from pin_cpsurf_comp import CPSurfPinComp
from align_cpsurf_comp import CPSurfAlignComp
from max_int_xi_comp import MaxIntXiComp
from min_int_xi_comp import MinIntXiComp
from int_xi_edge_comp import IntXiEdgeComp
# from int_xi_edge_comp_quadratic import IntXiEdgeComp
from int_energy_regu_comp import IntEnergyReguComp

from create_geom_evtol_wribs import preprocessor


set_log_active(False)


class ShapeOptGroup(om.Group):

    def initialize(self):
        self.options.declare('nonmatching_opt')
        self.options.declare('cpffd_name_pre', default='CP_FFD')
        self.options.declare('cpsurf_fe_name_pre', default='CPS_FE')
        self.options.declare('cpsurf_iga_name_pre', default='CPS_IGA')
        self.options.declare('cpsurf_iga_int_name_pre', default='CPS_IGA_int')
        self.options.declare('int_name', default='int_para')
        self.options.declare('disp_name', default='displacements')
        self.options.declare('int_energy_name', default='int_E')
        # self.options.declare('cpsurf_align_name_pre', default='CP_FFD_align')
        # self.options.declare('cpffd_pin_name_pre', default='CP_FFD_pin')
        self.options.declare('cpffd_align_name_pre', default='CP_FFD_align')
        self.options.declare('cpffd_pin_name_pre', default='CP_FFD_pin')
        self.options.declare('cpffd_regu_name_pre', default='CP_regu_pin')
        self.options.declare('volume_name', default='volume')
        self.options.declare('max_int_xi_name', default='max_int_xi')
        self.options.declare('min_int_xi_name', default='min_int_xi')
        self.options.declare('int_xi_edge_name', default='int_xi_edge')

    def init_parameters(self):
        self.nonmatching_opt = self.options['nonmatching_opt']
        self.cpffd_name_pre = self.options['cpffd_name_pre']
        self.cpsurf_fe_name_pre = self.options['cpsurf_fe_name_pre']
        self.cpsurf_iga_name_pre = self.options['cpsurf_iga_name_pre']
        self.cpsurf_iga_int_name_pre = self.options['cpsurf_iga_int_name_pre']
        self.int_name = self.options['int_name']
        self.disp_name = self.options['disp_name']
        self.int_energy_name = self.options['int_energy_name']
        # self.cpsurf_align_name_pre = self.options['cpsurf_align_name_pre']
        # self.cpffd_pin_name_pre = self.options['cpffd_pin_name_pre']
        self.cpffd_align_name_pre = self.options['cpffd_align_name_pre']
        self.cpffd_pin_name_pre = self.options['cpffd_pin_name_pre']
        self.cpffd_regu_name_pre = self.options['cpffd_regu_name_pre']
        self.volume_name = self.options['volume_name']
        self.max_int_xi_name = self.options['max_int_xi_name']
        self.min_int_xi_name = self.options['min_int_xi_name']
        self.int_xi_edge_name = self.options['int_xi_edge_name']

        self.opt_field = self.nonmatching_opt.opt_field
        self.init_cpffd = []
        for i, field in enumerate(self.opt_field):
            self.init_cpffd += [self.nonmatching_opt.get_init_CPFFD_multiFFD(field)]
        self.input_cpffd_shape = self.init_cpffd[0].size
        # self.init_cp_iga = self.nonmatching_opt.get_init_CPIGA()
        # self.input_cp_shape = self.nonmatching_opt.vec_scalar_iga_dof
        # self.design_var_lower = -10.
        # self.design_var_upper = 1.e-4
        self.design_var_lower = [3., 0.2, 2.9] # x-lower, z-lower
        self.design_var_upper = [5.7, 5.6, 3.7] # x-upper, z-upper

        self.cpffd_name_list = []
        self.cpsurf_fe_name_list = []
        self.cpsurf_iga_name_list = []
        self.cpsurf_iga_int_name_list = []
        self.cpffd_align_name_list = []
        self.cpffd_pin_name_list = []
        self.cpffd_regu_name_list = []
        for i, field in enumerate(self.opt_field):
            self.cpffd_name_list += [self.cpffd_name_pre+str(field)]
            self.cpsurf_fe_name_list += [self.cpsurf_fe_name_pre+str(field)]
            self.cpsurf_iga_name_list += [self.cpsurf_iga_name_pre+str(field)]
            self.cpsurf_iga_int_name_list += [self.cpsurf_iga_int_name_pre+str(field)]
            self.cpffd_align_name_list += [self.cpffd_align_name_pre+str(field)]
            self.cpffd_pin_name_list += [self.cpffd_pin_name_pre+str(field)]
            self.cpffd_regu_name_list += [self.cpffd_regu_name_pre+str(field)]

        self.inputs_comp_name = 'inputs_comp'
        self.cpffd2fe_comp_name = 'CPFFD2FE_comp'
        self.cpfe2iga_comp_name = 'CPFE2IGA_comp'
        self.cpiga2cpigaint_comp_name = 'CPIGA2CPIGAInt_comp'
        self.cpiga2xi_comp_name = 'CPIGA2Xi_comp'
        self.disp_states_comp_name = 'disp_states_comp'
        self.int_energy_comp_name = 'internal_energy_comp'
        self.cpffd_align_comp_name = 'CPFFD_align_comp'
        self.cpffd_pin_comp_name = 'CPFFD_pin_comp'
        self.cpffd_regu_comp_name = 'CPFFD_regu_comp'
        self.volume_comp_name = 'volume_comp'
        self.max_int_xi_comp_name = 'max_int_xi_comp'
        self.min_int_xi_comp_name = 'min_int_xi_comp'
        self.int_xi_edge_comp_name = 'int_xi_edge_comp'


    def setup(self):
        # Add inputs comp
        inputs_comp = om.IndepVarComp()
        for i, field in enumerate(self.opt_field):
            inputs_comp.add_output(self.cpffd_name_list[i],
                        shape=self.input_cpffd_shape,
                        val=self.init_cpffd[i])
        self.add_subsystem(self.inputs_comp_name, inputs_comp)

        # Add CP FFD to FE comp
        self.cpffd2fe_comp = CPFFD2SurfComp(
                        nonmatching_opt_ffd=self.nonmatching_opt,
                        input_cpffd_name_pre=self.cpffd_name_pre,
                        output_cpsurf_name_pre=self.cpsurf_fe_name_pre)
        self.cpffd2fe_comp.init_parameters()
        self.add_subsystem(self.cpffd2fe_comp_name, self.cpffd2fe_comp)

        # Add CP  FE 2 IGA comp
        self.cpfe2iga_comp = CPFE2IGAComp(
                        nonmatching_opt=self.nonmatching_opt,
                        input_cp_fe_name_pre=self.cpsurf_fe_name_pre,
                        output_cp_iga_name_pre=self.cpsurf_iga_name_pre)
        self.cpfe2iga_comp.init_parameters()
        self.add_subsystem(self.cpfe2iga_comp_name, self.cpfe2iga_comp)

        # CPIGA2CPIGAInt
        self.cpiga2cpigaint_comp = CPIGA2CPIntComp(
                            nonmatching_opt=self.nonmatching_opt,
                            input_cp_iga_name_pre=self.cpsurf_iga_name_pre,
                            output_cp_iga_int_name_pre=self.cpsurf_iga_int_name_pre)
        self.cpiga2cpigaint_comp.init_parameters()
        self.add_subsystem(self.cpiga2cpigaint_comp_name, self.cpiga2cpigaint_comp)


        # Add CPIGA2Xi comp
        self.cpiga2xi_comp = CPIGA2XiComp(
                        nonmatching_opt=self.nonmatching_opt,
                        input_cp_iga_name_pre=self.cpsurf_iga_int_name_pre,
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
                                             nonlinear_solver_rtol=1e-4,
                                             nonlinear_solver_max_it=10)
        self.add_subsystem(self.disp_states_comp_name, self.disp_states_comp)

        # Add internal energy comp (objective function)
        # self.int_energy_comp = IntEnergyComp(
        #                   nonmatching_opt=self.nonmatching_opt,
        #                   input_cp_iga_name_pre=self.cpsurf_iga_name_pre,
        #                   input_u_name=self.disp_name,
        #                   output_wint_name=self.int_energy_name)
        # self.int_energy_comp.init_parameters()
        # self.add_subsystem(self.int_energy_comp_name, self.int_energy_comp)
        self.regu_para = 1e-4
        self.int_energy_comp = IntEnergyReguComp(
                          nonmatching_opt=self.nonmatching_opt,
                          input_cp_iga_name_pre=self.cpsurf_iga_name_pre,
                          input_u_name=self.disp_name,
                          output_wint_name=self.int_energy_name,
                          regu_para=self.regu_para)
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

        # Add CP FFD regu comp (linear constraint)
        self.cpffd_regu_comp = CPFFDReguComp(
                           nonmatching_opt_ffd=self.nonmatching_opt,
                           input_cpffd_name_pre=self.cpffd_name_pre,
                           output_cpregu_name_pre=self.cpffd_regu_name_pre)
        self.cpffd_regu_comp.init_parameters()
        self.add_subsystem(self.cpffd_regu_comp_name, self.cpffd_regu_comp)
        self.cpffd_regu_lower_val = 1.e-3
        self.cpffd_regu_lower = [np.ones(self.cpffd_regu_comp.\
                                 output_shapes[i])*self.cpffd_regu_lower_val
                                 for i in range(len(self.opt_field))]

        # # Add CP FFD regu comp using aggregated method
        # self.cpffd_regu_comp = CPFFDReguCompAgg(
        #                    nonmatching_opt_ffd=self.nonmatching_opt,
        #                    input_cpffd_name_pre=self.cpffd_name_pre,
        #                    output_cpregu_name_pre=self.cpffd_regu_name_pre)
        # self.cpffd_regu_comp.init_parameters()
        # self.add_subsystem(self.cpffd_regu_comp_name, self.cpffd_regu_comp)
        # self.cpffd_regu_lower_val = 5.e-2
        # self.cpffd_regu_lower = [self.cpffd_regu_lower_val
        #                          for i in range(len(self.opt_field))]

        self.cpffd_align_comp = CPFFDAlignComp(
                            nonmatching_opt_ffd=self.nonmatching_opt,
                            input_cpffd_name_pre=self.cpffd_name_pre,
                            output_cpalign_name_pre=self.cpffd_align_name_pre)
        self.cpffd_align_comp.init_parameters()
        self.add_subsystem(self.cpffd_align_comp_name, self.cpffd_align_comp)
        self.cpffd_align_cons_val = np.zeros(self.cpffd_align_comp.output_shape)
        # self.cpffd_align_cons_val = 0

        self.cpffd_pin_comp = CPFFDPinComp(
                            nonmatching_opt_ffd=self.nonmatching_opt,
                            input_cpffd_name_pre=self.cpffd_name_pre,
                            output_cppin_name_pre=self.cpffd_pin_name_pre)
        self.cpffd_pin_comp.init_parameters()
        self.add_subsystem(self.cpffd_pin_comp_name, self.cpffd_pin_comp)
        # self.cpffd_pin_cons_val = []
        # for i, field in enumerate(self.opt_field):
        #     self.cpffd_pin_cons_val += [np.zeros(self.cpffd_pin_comp.output_shapes[i])]
        self.cpffd_pin_cons_val =  []
        for i, field in enumerate(self.opt_field):
            self.cpffd_pin_cons_val += [self.init_cpffd[i]\
                                        [self.cpffd_pin_comp.pin_dofs[i]]]
        # self.cpffd_pin_comp.set_pin_constraint_vals(self.cpffd_pin_cons_val)

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

        # Connect names between components
        for i, field in enumerate(self.opt_field):
            # For optimization components
            self.connect(self.inputs_comp_name+'.'
                         +self.cpffd_name_list[i],
                         self.cpffd2fe_comp_name+'.'
                         +self.cpffd_name_list[i])
            self.connect(self.cpffd2fe_comp_name+'.'
                         +self.cpsurf_fe_name_list[i],
                         self.cpfe2iga_comp_name+'.'
                         +self.cpsurf_fe_name_list[i])

            self.connect(self.cpfe2iga_comp_name+'.'
                         +self.cpsurf_iga_name_list[i],
                         self.cpiga2cpigaint_comp_name+'.'
                         +self.cpsurf_iga_name_list[i])

            self.connect(self.cpiga2cpigaint_comp_name+'.'
                         +self.cpsurf_iga_int_name_list[i],
                         self.cpiga2xi_comp_name+'.'
                         +self.cpsurf_iga_int_name_list[i])

            self.connect(self.cpfe2iga_comp_name+'.'
                         +self.cpsurf_iga_name_list[i],
                         self.disp_states_comp_name+'.'
                         +self.cpsurf_iga_name_list[i])

            self.connect(self.cpfe2iga_comp_name+'.'
                         +self.cpsurf_iga_name_list[i],
                         self.int_energy_comp_name+'.'
                         +self.cpsurf_iga_name_list[i])

            self.connect(self.cpfe2iga_comp_name+'.'
                         +self.cpsurf_iga_name_list[i],
                         self.volume_comp_name+'.'
                         +self.cpsurf_iga_name_list[i])

            # For constraints
            self.connect(self.inputs_comp_name+'.'
                         +self.cpffd_name_list[i],
                         self.cpffd_align_comp_name+'.'
                         +self.cpffd_name_list[i])

            self.connect(self.inputs_comp_name+'.'
                         +self.cpffd_name_list[i],
                         self.cpffd_pin_comp_name +'.'
                         +self.cpffd_name_list[i])

            self.connect(self.inputs_comp_name+'.'
                         +self.cpffd_name_list[i],
                         self.cpffd_regu_comp_name +'.'
                         +self.cpffd_name_list[i])

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

        # Add design variable, constraints and objective
        for i, field in enumerate(self.opt_field):
            self.add_design_var(self.inputs_comp_name+'.'
                                +self.cpffd_name_list[i],
                                lower=self.design_var_lower[i],
                                upper=self.design_var_upper[i])

            self.add_constraint(self.cpffd_align_comp_name+'.'
                                +self.cpffd_align_name_list[i],
                                equals=self.cpffd_align_cons_val)
            self.add_constraint(self.cpffd_pin_comp_name+'.'
                                +self.cpffd_pin_name_list[i],
                                equals=self.cpffd_pin_cons_val[i])
                                # equals=0)
            self.add_constraint(self.cpffd_regu_comp_name+'.'
                                +self.cpffd_regu_name_list[i],
                                lower=self.cpffd_regu_lower[i])

        self.add_constraint(self.cpiga2xi_comp_name+'.'+self.int_name,
                            lower=0., upper=1.)
        # self.add_constraint(self.max_int_xi_comp_name+'.'+self.max_int_xi_name,
        #                     upper=1.)
        # self.add_constraint(self.min_int_xi_comp_name+'.'+self.min_int_xi_name,
        #                     lower=0.)
        self.add_constraint(self.int_xi_edge_comp_name+'.'+self.int_xi_edge_name,
                            equals=self.int_xi_edge_cons_val, scaler=1e1)

        self.add_constraint(self.volume_comp_name+'.'+self.volume_name,
                            # equals=self.vol_val)
                            lower=self.vol_val*0.8,
                            upper=self.vol_val*1.2)
        self.add_objective(self.int_energy_comp_name+'.'
                           +self.int_energy_name,
                           scaler=1e8)


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

test_ind = 14
optimizer = 'SNOPT'
# optimizer = 'SLSQP'
opt_field = [0,1,2]
# save_path = './'
save_path = '/home/han/Documents/test_results/'
# save_path = '/Users/hanzhao/Documents/test_results/'
# folder_name = "results/"
folder_name = "results"+str(test_ind)+"/"

if mpirank == 0:
    print("Test ind:", test_ind)


geom_scale = 2.54e-5  # Convert current length unit to m
E = Constant(68e9)  # Young's modulus, Pa
nu = Constant(0.35)  # Poisson's ratio
h_th = Constant(3.0e-3)  # Thickness of surfaces, m
penalty_coefficient = 1e3
num_surfs = preprocessor.num_surfs

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
nonmatching_opt = NonMatchingOptFFD(splines, E, h_th, nu, 
                                    opt_field=opt_field, comm=worldcomm)
nonmatching_opt.create_mortar_meshes(preprocessor.mortar_nels)

if mpirank == 0:
    print("Setting up mortar meshes...")
nonmatching_opt.mortar_meshes_setup(preprocessor.mapping_list, 
                    preprocessor.intersections_para_coords, 
                    penalty_coefficient, 2)
pressure = Constant(-1.e-1)
f = as_vector([Constant(0.), Constant(0.), pressure])

# line_force0 = as_vector([Constant(0.), Constant(0.), Constant(1.)])
# xi0 = SpatialCoordinate(nonmatching_opt.splines[0].mesh)
# line_indicator0 = conditional(gt(xi0[1], 1.-1e-3), Constant(1.), Constant(0.))

# line_force1 = as_vector([Constant(0.), Constant(0.), Constant(-1.)])
# xi1 = SpatialCoordinate(nonmatching_opt.splines[1].mesh)
# line_indicator1 = conditional(le(xi1[1], 1.e-3), Constant(1.), Constant(0.))

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

nonmatching_opt.solve_nonlinear_nonmatching_problem()

shopt_multi_ffd_inds = [[0,1,2], [3,4,5]]
nonmatching_opt.set_shopt_multiFFD_surf_inds(shopt_multi_ffd_inds)

#################################################
num_shopt_ffd = nonmatching_opt.num_shopt_ffd
shopt_ffd_lims_multiffd = nonmatching_opt.shopt_cpsurf_lims_multiffd

shopt_ffd_num_el = [[1,1,1], [1,2,2]]
shopt_ffd_p = [[1,1,1],[2,2,2]]
extrude_dir = [2,1]
extrude_coeff = 0.05

shopt_ffd_block_list = []
for ffd_ind in range(num_shopt_ffd):
    field = extrude_dir[ffd_ind]
    cp_range = shopt_ffd_lims_multiffd[ffd_ind][field][1]\
              -shopt_ffd_lims_multiffd[ffd_ind][field][0]
    shopt_ffd_lims_multiffd[ffd_ind][field][1] = \
        shopt_ffd_lims_multiffd[ffd_ind][field][1] + extrude_coeff*cp_range
    shopt_ffd_lims_multiffd[ffd_ind][field][0] = \
        shopt_ffd_lims_multiffd[ffd_ind][field][0] - extrude_coeff*cp_range
    shopt_ffd_block_list += [create_3D_block(shopt_ffd_num_el[ffd_ind],
                                       shopt_ffd_p[ffd_ind],
                                       shopt_ffd_lims_multiffd[ffd_ind])]

for ffd_ind in range(num_shopt_ffd):
    vtk_writer = VTKWriter()
    vtk_writer.write("./geometry/evtol_shopt_ffd_block_init"+str(ffd_ind)+".vtk", 
                     shopt_ffd_block_list[ffd_ind])
    vtk_writer.write_cp("./geometry/evtol_shopt_ffd_cp_init"+str(ffd_ind)+".vtk", 
                     shopt_ffd_block_list[ffd_ind])

shopt_ffd_knots_list = [ffd_block.knots for ffd_block 
                        in shopt_ffd_block_list]
shopt_ffd_control_list = [ffd_block.control for ffd_block 
                          in shopt_ffd_block_list]
print("Setting multiple shape FFD blocks ...")
nonmatching_opt.set_shopt_multiFFD(shopt_ffd_knots_list, 
                                       shopt_ffd_control_list)

########### Set constraints info #########
a0 = nonmatching_opt.set_shopt_regu_CP_multiFFD(shopt_regu_dir_list=[[None, None, None], 
                                                                [None, None, None]], 
                                           shopt_regu_side_list=[[None, None, None], 
                                                                [None, None, None]])
a1 = nonmatching_opt.set_shopt_pin_CP_multiFFD(0, pin_dir0_list=['whole', None], 
                                          pin_side0_list=None,
                                          pin_dir1_list=None, 
                                          pin_side1_list=None)
a2 = nonmatching_opt.set_shopt_pin_CP_multiFFD(1, pin_dir0_list=['whole', 1], 
                                          pin_side0_list=[None, 0],
                                          pin_dir1_list=None, 
                                          pin_side1_list=None)
a3 = nonmatching_opt.set_shopt_pin_CP_multiFFD(2, pin_dir0_list=['whole', None], 
                                          pin_side0_list=None,
                                          pin_dir1_list=None, 
                                          pin_side1_list=None)

nonmatching_opt.set_shopt_align_CP_multiFFD([None,2])
# a2 = nonmatching_opt.set_shopt_pin_CP_multiFFD(2, pin_dir0_list=['whole', None], 
#                                           pin_side0_list=None,
#                                           pin_dir1_list=None, 
#                                           pin_side1_list=None)
# a2 = nonmatching_opt.set_shopt_pin_CP_multiFFD(2, pin_dir0_list=[0, None], 
#                                           pin_side0_list=[[0,1], None],
#                                           pin_dir1_list=None, 
#                                           pin_side1_list=None)
# a3 = nonmatching_opt.set_shopt_align_CP_multiFFD(shopt_align_dir_list=[1,1,0])

#################################################

# print(aaaa)

#################################
# preprocessor.check_intersections_type()
# preprocessor.get_diff_intersections()
nonmatching_opt.set_diff_intersections(preprocessor)
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


start_time = time.perf_counter()
start_current_time = datetime.now().strftime("%H:%M:%S")
if mpirank == 0:
    print("Start current time:", start_current_time)

prob.setup()
prob.run_driver()

end_time = time.perf_counter()
run_time = end_time - start_time
end_current_time = datetime.now().strftime("%H:%M:%S")

if mpirank == 0:
    print("End current time:", end_current_time)
    print("Simulation run time: {:.2f} s".format(run_time))

# for i in range(1,num_surfs):
#     max_F0 = np.max(nonmatching_opt.splines[i].cpFuncs[0].vector().get_local())
#     min_F0 = np.min(nonmatching_opt.splines[i].cpFuncs[0].vector().get_local())
#     print("Spline: {:2d}, Max F0: {:8.6f}".format(i, max_F0))
#     print("Spline: {:2d}, Min F0: {:8.6f}".format(i, min_F0))

# for i in range(1,num_surfs):
#     max_F0 = np.max(nonmatching_opt.splines[i].cpFuncs[2].vector().get_local())
#     min_F0 = np.min(nonmatching_opt.splines[i].cpFuncs[2].vector().get_local())
#     print("Spline: {:2d}, Max F2: {:8.6f}".format(i, max_F0))
#     print("Spline: {:2d}, Min F2: {:8.6f}".format(i, min_F0))

# for i in range(preprocessor.num_intersections_all):
#     mesh_phy = generate_mortar_mesh(preprocessor.intersections_phy_coords[i], num_el=128)
#     File('./geometry/int_curve'+str(i)+'.pvd') << mesh_phy