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
from disp_states_mi_comp import DispMintStatesComp
from pin_cpsurf_comp import CPSurfPinComp
from align_cpsurf_comp import CPSurfAlignComp
from max_int_xi_comp import MaxIntXiComp
from min_int_xi_comp import MinIntXiComp
# from int_xi_edge_comp import IntXiEdgeComp
from int_xi_edge_comp_quadratic import IntXiEdgeComp


set_log_active(False)


import openmdao.api as om
from openmdao.api import Problem

class XiConsComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('input_xi_name', default='thickness')
        self.options.declare('output_xi_cons_name', default='w_int')

    def init_parameters(self, input_shape):
        self.input_xi_name = self.options['input_xi_name']
        self.output_xi_cons_name = self.options['output_xi_cons_name']

        self.input_shape = input_shape
        self.num_pts = int(self.input_shape/4)
        self.output_shape = 4
        self.deriv = np.zeros((self.output_shape, self.input_shape))
        self.deriv[0,0] = 1
        self.deriv[1,1] = 1
        self.deriv[2,self.num_pts*2] = 1
        self.deriv[3,self.num_pts*2+1] = 1
        self.deriv = coo_matrix(self.deriv)


    def setup(self):
        self.add_input(self.input_xi_name,
                       shape=self.input_shape)
        self.add_output(self.output_xi_cons_name,
                        shape=self.output_shape)
        self.declare_partials(self.output_xi_cons_name,
                              self.input_xi_name,
                              val=self.deriv.data,
                              rows=self.deriv.row,
                              cols=self.deriv.col)

    def compute(self, inputs, outputs):
        outputs[self.output_xi_cons_name] = np.dot(self.deriv.todense(),inputs[self.input_xi_name])



class ShapeOptGroup(om.Group):

    def initialize(self):
        self.options.declare('nonmatching_opt')
        self.options.declare('cpffd_design_name_pre', default='CP_design_FFD')
        self.options.declare('cpffd_full_name_pre', default='CP_FFD')
        self.options.declare('cpsurf_fe_name_pre', default='CPS_FE')
        self.options.declare('cpsurf_iga_name_pre', default='CPS_IGA')
        self.options.declare('int_name', default='int_para')
        self.options.declare('disp_name', default='displacements')
        self.options.declare('int_energy_name', default='int_E')
        # self.options.declare('cpsurf_align_name_pre', default='CP_FFD_align')
        # self.options.declare('cpffd_pin_name_pre', default='CP_FFD_pin')
        # self.options.declare('cpffd_align_name_pre', default='CP_FFD_align')
        self.options.declare('cpffd_pin_name_pre', default='CP_FFD_pin')
        self.options.declare('cpffd_regu_name_pre', default='CP_regu_pin')
        # self.options.declare('volume_name', default='volume')
        self.options.declare('max_int_xi_name', default='max_int_xi')
        self.options.declare('min_int_xi_name', default='min_int_xi')
        self.options.declare('int_xi_edge_name', default='int_xi_edge')

    def init_parameters(self):
        self.nonmatching_opt = self.options['nonmatching_opt']
        self.cpffd_design_name_pre = self.options['cpffd_design_name_pre']
        self.cpffd_full_name_pre = self.options['cpffd_full_name_pre']
        self.cpsurf_fe_name_pre = self.options['cpsurf_fe_name_pre']
        self.cpsurf_iga_name_pre = self.options['cpsurf_iga_name_pre']
        self.int_name = self.options['int_name']
        self.disp_name = self.options['disp_name']
        self.int_energy_name = self.options['int_energy_name']
        # self.cpsurf_align_name_pre = self.options['cpsurf_align_name_pre']
        # self.cpffd_pin_name_pre = self.options['cpffd_pin_name_pre']
        # self.cpffd_align_name_pre = self.options['cpffd_align_name_pre']
        self.cpffd_pin_name_pre = self.options['cpffd_pin_name_pre']
        self.cpffd_regu_name_pre = self.options['cpffd_regu_name_pre']
        # self.volume_name = self.options['volume_name']
        self.max_int_xi_name = self.options['max_int_xi_name']
        self.min_int_xi_name = self.options['min_int_xi_name']
        self.int_xi_edge_name = self.options['int_xi_edge_name']

        self.opt_field = self.nonmatching_opt.opt_field
        self.init_cpffd = self.nonmatching_opt.shopt_init_cp_mffd_design
        self.input_cpffd_shapes = [cpffd.size for cpffd in self.init_cpffd]
        # self.init_cp_iga = self.nonmatching_opt.get_init_CPIGA()
        # self.input_cp_shape = self.nonmatching_opt.vec_scalar_iga_dof
        # self.design_var_lower = -10.
        # self.design_var_upper = 1.e-4
        self.design_var_lower = [-1.e-3, -1.e-3] #z-lower
        self.design_var_upper = [1.6, 1.6] #z-upper

        self.cpffd_design_name_list = []
        self.cpffd_full_name_list = []
        self.cpsurf_fe_name_list = []
        self.cpsurf_iga_name_list = []
        self.cpffd_pin_name_list = []
        self.cpffd_regu_name_list = []
        for i, field in enumerate(self.opt_field):
            self.cpffd_design_name_list += [self.cpffd_design_name_pre+str(field)]
            self.cpffd_full_name_list += [self.cpffd_full_name_pre+str(field)]
            self.cpsurf_fe_name_list += [self.cpsurf_fe_name_pre+str(field)]
            self.cpsurf_iga_name_list += [self.cpsurf_iga_name_pre+str(field)]
            self.cpffd_pin_name_list += [self.cpffd_pin_name_pre+str(field)]
            self.cpffd_regu_name_list += [self.cpffd_regu_name_pre+str(field)]

        self.inputs_comp_name = 'inputs_comp'
        self.cpffd_design2full_comp_name = 'CPFFDDesign2Full_comp'
        self.cpffd2fe_comp_name = 'CPFFD2FE_comp'
        self.cpfe2iga_comp_name = 'CPFE2IGA_comp'
        self.cpiga2xi_comp_name = 'CPIGA2Xi_comp'
        self.disp_states_comp_name = 'disp_states_comp'
        self.int_energy_comp_name = 'internal_energy_comp'
        # self.cpffd_align_comp_name = 'CPFFD_align_comp'
        self.cpffd_pin_comp_name = 'CPFFD_pin_comp'
        self.cpffd_regu_comp_name = 'CPFFD_regu_comp'
        # self.volume_comp_name = 'volume_comp'
        self.max_int_xi_comp_name = 'max_int_xi_comp'
        self.min_int_xi_comp_name = 'min_int_xi_comp'
        # self.int_xi_edge_comp_name = 'int_xi_edge_comp'


    def setup(self):
        # Add inputs comp
        inputs_comp = om.IndepVarComp()
        for i, field in enumerate(self.opt_field):
            inputs_comp.add_output(self.cpffd_design_name_list[i],
                        shape=self.input_cpffd_shapes[i],
                        val=self.init_cpffd[i])
        self.add_subsystem(self.inputs_comp_name, inputs_comp)

        # Add CP FFD design to full comp
        self.cpffd_design2full_comp = CPFFDesign2FullComp(
                        nonmatching_opt_ffd=self.nonmatching_opt,
                        input_cpffd_design_name_pre=self.cpffd_design_name_pre,
                        output_cpffd_full_name_pre=self.cpffd_full_name_pre)
        self.cpffd_design2full_comp.init_parameters()
        self.add_subsystem(self.cpffd_design2full_comp_name, self.cpffd_design2full_comp)

        # Add CP FFD to FE comp
        self.cpffd2fe_comp = CPFFD2SurfComp(
                        nonmatching_opt_ffd=self.nonmatching_opt,
                        input_cpffd_name_pre=self.cpffd_full_name_pre,
                        output_cpsurf_name_pre=self.cpsurf_fe_name_pre)
        self.cpffd2fe_comp.init_parameters()
        self.add_subsystem(self.cpffd2fe_comp_name, self.cpffd2fe_comp)

        # print("Inspection point 1 ....")
        # Add CP  FE 2 IGA comp
        self.cpfe2iga_comp = CPFE2IGAComp(
                        nonmatching_opt=self.nonmatching_opt,
                        input_cp_fe_name_pre=self.cpsurf_fe_name_pre,
                        output_cp_iga_name_pre=self.cpsurf_iga_name_pre)
        self.cpfe2iga_comp.init_parameters()
        self.add_subsystem(self.cpfe2iga_comp_name, self.cpfe2iga_comp)

        # print("Inspection point 2 ....")

        # Add CPIGA2Xi comp
        self.cpiga2xi_comp = CPIGA2XiComp(
                        nonmatching_opt=self.nonmatching_opt,
                        input_cp_iga_name_pre=self.cpsurf_iga_name_pre,
                        output_xi_name=self.int_name)
        self.cpiga2xi_comp.init_parameters()
        self.add_subsystem(self.cpiga2xi_comp_name, self.cpiga2xi_comp)

        # print("Inspection point 3 ....")

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

        # print("Inspection point 4 ....")


        # Add internal energy comp (objective function)
        self.int_energy_comp = IntEnergyComp(
                          nonmatching_opt=self.nonmatching_opt,
                          input_cp_iga_name_pre=self.cpsurf_iga_name_pre,
                          input_u_name=self.disp_name,
                          output_wint_name=self.int_energy_name)
        self.int_energy_comp.init_parameters()
        self.add_subsystem(self.int_energy_comp_name, self.int_energy_comp)

        # Add CP FFD regu comp (linear constraint)
        self.cpffd_regu_comp = CPFFDReguComp(
                           nonmatching_opt_ffd=self.nonmatching_opt,
                           input_cpffd_design_name_pre=self.cpffd_design_name_pre,
                           output_cpregu_name_pre=self.cpffd_regu_name_pre)
        self.cpffd_regu_comp.init_parameters()
        self.add_subsystem(self.cpffd_regu_comp_name, self.cpffd_regu_comp)
        self.cpffd_regu_lower_val = 1.e-2
        self.cpffd_regu_lower = [np.ones(self.cpffd_regu_comp.\
                                 output_shapes[i])*self.cpffd_regu_lower_val
                                 for i in range(len(self.opt_field))]

        self.cpffd_pin_comp = CPFFDPinComp(
                            nonmatching_opt_ffd=self.nonmatching_opt,
                            input_cpffd_design_name_pre=self.cpffd_design_name_pre,
                            output_cppin_name_pre=self.cpffd_pin_name_pre)
        self.cpffd_pin_comp.init_parameters()
        self.add_subsystem(self.cpffd_pin_comp_name, self.cpffd_pin_comp)
        self.cpffd_pin_cons_val = self.nonmatching_opt.shopt_pin_vals


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


        self.int_xi_cons_name = "xi_cons"
        self.xi_cons_comp_name = "xi_cons_comp"
        self.xi_cons_comp = XiConsComp(input_xi_name=self.int_name,
                            output_xi_cons_name=self.int_xi_cons_name)
        self.xi_cons_comp.init_parameters(input_shape=self.nonmatching_opt.xi_size)
        self.add_subsystem(self.xi_cons_comp_name, self.xi_cons_comp)


        # Connect names between components
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
                         self.cpiga2xi_comp_name+'.'
                         +self.cpsurf_iga_name_list[i])

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
                         +self.cpffd_design_name_list[i],
                         self.cpffd_pin_comp_name +'.'
                         +self.cpffd_design_name_list[i])

            self.connect(self.inputs_comp_name+'.'
                         +self.cpffd_design_name_list[i],
                         self.cpffd_regu_comp_name +'.'
                         +self.cpffd_design_name_list[i])

        self.connect(self.cpiga2xi_comp_name+'.'+self.int_name,
                     self.disp_states_comp_name+'.'+self.int_name)

        self.connect(self.disp_states_comp_name+'.'+self.disp_name,
                     self.int_energy_comp_name+'.'+self.disp_name)

        # self.connect(self.cpiga2xi_comp_name+'.'+self.int_name,
        #              self.max_int_xi_comp_name+'.'+self.int_name)

        # self.connect(self.cpiga2xi_comp_name+'.'+self.int_name,
        #              self.min_int_xi_comp_name+'.'+self.int_name)

        self.connect(self.cpiga2xi_comp_name+'.'+self.int_name,
                     self.xi_cons_comp_name+'.'+self.int_name)

        # Add design variable, constraints and objective
        for i, field in enumerate(self.opt_field):
            self.add_design_var(self.inputs_comp_name+'.'
                                +self.cpffd_design_name_list[i],
                                lower=self.design_var_lower[i],
                                upper=self.design_var_upper[i])

            self.add_constraint(self.cpffd_pin_comp_name+'.'
                                +self.cpffd_pin_name_list[i],
                                equals=self.cpffd_pin_cons_val[i])
            self.add_constraint(self.cpffd_regu_comp_name+'.'
                                +self.cpffd_regu_name_list[i],
                                lower=self.cpffd_regu_lower[i])

        # self.add_constraint(self.cpiga2xi_comp_name+'.'+self.int_name,
        #                     lower=0., upper=1.)
        # self.add_constraint(self.max_int_xi_comp_name+'.'+self.max_int_xi_name,
        #                     upper=1.)
        # self.add_constraint(self.min_int_xi_comp_name+'.'+self.min_int_xi_name,
        #                     lower=0.)

        self.add_constraint(self.xi_cons_comp_name+'.'+self.int_xi_cons_name,
                            lower=0., upper=1.)
        self.add_objective(self.int_energy_comp_name+'.'
                           +self.int_energy_name,
                           scaler=1e10)


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

test_ind = 11
optimizer = 'SNOPT'
# optimizer = 'SLSQP'
# opt_field = [0,1]
# save_path = './'
save_path = '/home/han/Documents/test_results/'
# save_path = '/Users/hanzhao/Documents/test_results/'
folder_name = "results"+str(test_ind)+"/"

filename_igs = "./geometry/init_tube_geom_4patch.igs"
igs_shapes = read_igs_file(filename_igs, as_compound=False)
occ_surf_list = [topoface2surface(face, BSpline=True) 
                 for face in igs_shapes]
occ_surf_data_list = [BSplineSurfaceData(surf) for surf in occ_surf_list]
num_surfs = len(occ_surf_list)
p = occ_surf_data_list[0].degree[0]

# Define material and geometric parameters
E = Constant(1.0e9)
nu = Constant(0.)
h_th = Constant(0.01)
penalty_coefficient = 1.0e3

fields0 = [[[0,2]],]
fields1 = [[None,[1,2]],]
spline_bc0 = SplineBC(directions=[0], sides=[[0],],
                     fields=fields0, n_layers=[[1],])
spline_bc1 = SplineBC(directions=[0], sides=[[1],],
                     fields=fields1, n_layers=[[None, 1],])
spline_bcs = [spline_bc0]*2 + [spline_bc1]*2

# Geometry preprocessing and surface-surface intersections computation
preprocessor = OCCPreprocessing(occ_surf_list, reparametrize=False, 
                                refine=False)
print("Computing intersections...")
# int_data_filename = "int_data.npz"
# if os.path.isfile(int_data_filename):
#     preprocessor.load_intersections_data(int_data_filename)
# else:
#     preprocessor.compute_intersections(mortar_refine=2)
#     preprocessor.save_intersections_data(int_data_filename)

preprocessor.compute_intersections(mortar_refine=2)
if mpirank == 0:
    print("Total DoFs:", preprocessor.total_DoFs)
    print("Number of intersections:", preprocessor.num_intersections_all)


if mpirank == 0:
    print("Creating splines...")
# Create tIGAr extracted spline instances
splines = []
for i in range(num_surfs):
    # print("surf ind:", i)
    spline = OCCBSpline2tIGArSpline(preprocessor.BSpline_surfs[i], 
                                    spline_bc=spline_bcs[i], index=i)
    splines += [spline,]

# Create non-matching problem
nonmatching_opt = NonMatchingOptFFD(splines, E, h_th, nu, comm=worldcomm)

opt_field = [[0,1],[0,1]]
opt_surf_inds = [[0,1], [2,3]]
nonmatching_opt.set_shopt_surf_inds_multiFFD(opt_field, opt_surf_inds)

# opt_field = [0,1]
# shopt_surf_inds = [[0,1],[0,1]]

# nonmatching_opt.set_shopt_surf_inds(opt_field=opt_field, 
#                                     shopt_surf_inds=shopt_surf_inds)
nonmatching_opt.get_init_CPIGA()
nonmatching_opt.set_geom_preprocessor(preprocessor)



shopt_ffd_p = [[3,3,3], [3,3,3]]
shopt_ffd_num_el = [[2,2,1], [2,2,1]]
shopt_ffd_lims_multiffd = nonmatching_opt.shopt_cpsurf_lims_mffd

extrude_ratio = 0.35
shopt_num_ffd = nonmatching_opt.shopt_num_ffd

shopt_ffd_block_list = []
for ffd_ind in range(shopt_num_ffd):
    if ffd_ind == 0:
        for field in [0,1]:
            cp_range = shopt_ffd_lims_multiffd[ffd_ind][field][1]\
                    -shopt_ffd_lims_multiffd[ffd_ind][field][0]
            if field == 0:
                coeff = extrude_ratio
                shopt_ffd_lims_multiffd[ffd_ind][field][1] = \
                    shopt_ffd_lims_multiffd[ffd_ind][field][1] + coeff*cp_range
            elif field == 1:
                coeff = -extrude_ratio
                shopt_ffd_lims_multiffd[ffd_ind][field][0] = \
                    shopt_ffd_lims_multiffd[ffd_ind][field][0] + coeff*cp_range
        shopt_ffd_block_list += [create_3D_block(shopt_ffd_num_el[ffd_ind],
                                        shopt_ffd_p[ffd_ind],
                                            shopt_ffd_lims_multiffd[ffd_ind])]
    if ffd_ind == 1:
        for field in [0,1]:
            cp_range = shopt_ffd_lims_multiffd[ffd_ind][field][1]\
                    -shopt_ffd_lims_multiffd[ffd_ind][field][0]
            if field == 0:
                coeff = -extrude_ratio
                shopt_ffd_lims_multiffd[ffd_ind][field][0] = \
                    shopt_ffd_lims_multiffd[ffd_ind][field][0] + coeff*cp_range
            elif field == 1:
                coeff = extrude_ratio
                shopt_ffd_lims_multiffd[ffd_ind][field][1] = \
                    shopt_ffd_lims_multiffd[ffd_ind][field][1] + coeff*cp_range
        shopt_ffd_block_list += [create_3D_block(shopt_ffd_num_el[ffd_ind],
                                        shopt_ffd_p[ffd_ind],
                                            shopt_ffd_lims_multiffd[ffd_ind])]

shopt_ffd_knots_list = [ffd_block.knots for ffd_block 
                        in shopt_ffd_block_list]
shopt_ffd_control_list = [ffd_block.control for ffd_block 
                          in shopt_ffd_block_list]


nonmatching_opt.set_shopt_multiFFD(shopt_ffd_knots_list,
                              shopt_ffd_control_list)

a00 = nonmatching_opt.set_shopt_align_CP_multiFFD(ffd_ind=0, align_dir=[[2],[2]])
a01 = nonmatching_opt.set_shopt_align_CP_multiFFD(ffd_ind=1, align_dir=[[2],[2]])
# a10 = nonmatching_opt.set_shopt_pin_CP_multiFFD(ffd_ind=0, pin_dir0=[0,1], 
#                      pin_side0=[[0],[1]], pin_dir1=None, pin_side1=None)
# a11 = nonmatching_opt.set_shopt_pin_CP_multiFFD(ffd_ind=1, pin_dir0=[0,1], 
#                      pin_side0=[[1],[0]], pin_dir1=None, pin_side1=None)
a10 = nonmatching_opt.set_shopt_pin_CP_multiFFD(ffd_ind=0, pin_dir0=[0,0], 
                     pin_side0=[[0],[0]], pin_dir1=None, pin_side1=None)
a11 = nonmatching_opt.set_shopt_pin_CP_multiFFD(ffd_ind=1, pin_dir0=[1,1], 
                     pin_side0=[[0],[0]], pin_dir1=None, pin_side1=None)
a2 = nonmatching_opt.set_shopt_regu_CP_multiFFD()

if mpirank == 0:
    print("Setting up mortar meshes...")
nonmatching_opt.create_mortar_meshes(preprocessor.mortar_nels)
nonmatching_opt.mortar_meshes_setup(preprocessor.mapping_list, 
                    preprocessor.intersections_para_coords, 
                    penalty_coefficient, 2)

pressure = -Constant(1.)
source_terms = []
residuals = []
for s_ind in range(nonmatching_opt.num_splines):
    X = nonmatching_opt.splines[s_ind].F
    x = X + nonmatching_opt.splines[s_ind].rationalize(
            nonmatching_opt.spline_funcs[s_ind])
    z = nonmatching_opt.splines[s_ind].rationalize(
        nonmatching_opt.spline_test_funcs[s_ind])
    A0,A1,A2,deriv_A2,A,B = surfaceGeometry(
                            nonmatching_opt.splines[s_ind], X)
    a0,a1,a2,deriv_a2,a,b = surfaceGeometry(
                            nonmatching_opt.splines[s_ind], x)
    pressure_dir = sqrt(det(a)/det(A))*a2        
    normal_pressure = pressure*pressure_dir
    source_terms += [inner(normal_pressure, z)\
                     *nonmatching_opt.splines[s_ind].dx]
    residuals += [SVK_residual(nonmatching_opt.splines[s_ind], 
                  nonmatching_opt.spline_funcs[s_ind], 
                  nonmatching_opt.spline_test_funcs[s_ind], 
                  E, nu, h_th, source_terms[s_ind])]
nonmatching_opt.set_residuals(residuals)

nonmatching_opt.solve_nonlinear_nonmatching_problem()


# #################################
# preprocessor.check_intersections_type()
# preprocessor.get_diff_intersections()
# nonmatching_opt.set_diff_intersections(preprocessor)
# #################################

#################################
preprocessor.check_intersections_type()
preprocessor.get_diff_intersections()
nonmatching_opt.create_diff_intersections(num_edge_pts=None)
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
    prob.driver.opt_settings['Major optimality tolerance'] = 1e-2
    prob.driver.opt_settings['Major iterations limit'] = 50000
    prob.driver.opt_settings['Summary file'] = './SNOPT_report/SNOPT_summary'+str(test_ind)+'.out'
    prob.driver.opt_settings['Print file'] = './SNOPT_report/SNOPT_print'+str(test_ind)+'.out'
    prob.driver.options['debug_print'] = ['objs']#, 'desvars']
    prob.driver.options['print_results'] = True
elif optimizer.upper() == 'SLSQP':
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1e-6
    prob.driver.options['disp'] = True
    prob.driver.options['debug_print'] = ['objs']
    prob.driver.options['maxiter'] = 50000
else:
    raise ValueError("Undefined optimizer: {}".format(optimizer))


start_time = time.perf_counter()
start_current_time = datetime.now().strftime("%H:%M:%S")
if mpirank == 0:
    print("Start current time:", start_current_time)

# # Create a recorder variable
# recorder_name = './opt_data/recorder'+str(test_ind)+'.sql'
# FFD_data_name = './opt_data/FFD_data'+str(test_ind)+'.npz'

# prob.driver.recording_options['includes'] = ['*']
# prob.driver.recording_options['record_objectives'] = True
# prob.driver.recording_options['record_derivatives'] = True
# prob.driver.recording_options['record_constraints'] = True
# prob.driver.recording_options['record_desvars'] = True
# prob.driver.recording_options['record_inputs'] = True
# prob.driver.recording_options['record_outputs'] = True
# prob.driver.recording_options['record_residuals'] = True

# recorder = om.SqliteRecorder(recorder_name)
# prob.driver.add_recorder(recorder)

# cp0_shape = 100
FFD_block = shopt_ffd_block_list[0].copy()
VTK().write("./geometry/FFD_block_initial0.vtk", FFD_block)
FFD_block1 = shopt_ffd_block_list[1].copy()
VTK().write("./geometry/FFD_block_initial1.vtk", FFD_block1)

prob.setup()
prob.run_driver()

# major_iter_inds = model.disp_states_comp.func_eval_major_ind
# np.savez(FFD_data_name, opt_field=opt_field,
#                         major_iter_ind=major_iter_inds,
#                         ffd_control=FFD_block.control,
#                         ffd_knots=FFD_block.knots,
#                         QoI=0)

# #### Save initial and final shape of FFD block
# init_cpffd = FFD_block.control[:,:,:,0:3].transpose(2,1,0,3).reshape(-1,3)
# final_cpffd = init_cpffd.copy()

# final_cpffd0 = prob[model.inputs_comp_name+'.'+model.cpffd_full_name_list[0]][0:100]
# final_cpffd[:,0] = final_cpffd0
# final_cpffd1 = prob[model.inputs_comp_name+'.'+model.cpffd_full_name_list[1]][0:100]
# final_cpffd[:,1] = final_cpffd1
# final_cpffd = final_cpffd.reshape(FFD_block.control[:,:,:,0:3]\
#               .transpose(2,1,0,3).shape)
# final_cpffd = final_cpffd.transpose(2,1,0,3)
# final_FFD_block = NURBS(FFD_block.knots, final_cpffd)
# VTK().write('./geometry/FFD_block_final0.vtk', final_FFD_block)

# # Save initial discretized geometry
# nonmatching_opt.create_files(save_path=save_path, 
#                              folder_name='results_temp/', 
#                              refine_mesh=False)

end_time = time.perf_counter()
run_time = end_time - start_time
end_current_time = datetime.now().strftime("%H:%M:%S")

if mpirank == 0:
    print("End current time:", end_current_time)
    print("Simulation run time: {:.2f} s".format(run_time))

for i in range(0,num_surfs):
    max_F2 = np.max(nonmatching_opt.splines[i].cpFuncs[0].vector().get_local())
    min_F2 = np.min(nonmatching_opt.splines[i].cpFuncs[0].vector().get_local())
    print("Spline: {:2d}, Max F0: {:8.6f}".format(i, max_F2))
    print("Spline: {:2d}, Min F0: {:8.6f}".format(i, min_F2))


num_eval_pts = 101
xi0_pts = np.linspace(0,1,num_eval_pts)
xi1_pts = 0.1
s0_x_pts = np.zeros(num_eval_pts)
s0_y_pts = np.zeros(num_eval_pts)
s1_x_pts = np.zeros(num_eval_pts)
s1_y_pts = np.zeros(num_eval_pts)
for i in range(num_eval_pts):
    s0_x_pts[i] = nonmatching_opt.splines[1].cpFuncs[0]([xi0_pts[i], xi1_pts])
    s0_y_pts[i] = nonmatching_opt.splines[1].cpFuncs[1]([xi0_pts[i], xi1_pts])
    s1_x_pts[i] = nonmatching_opt.splines[3].cpFuncs[0]([xi0_pts[i], xi1_pts])
    s1_y_pts[i] = nonmatching_opt.splines[3].cpFuncs[1]([xi0_pts[i], xi1_pts])

y_ana_pts = np.zeros(num_eval_pts)
for i in range(num_eval_pts):
    y_ana_pts[i] = np.sqrt(1-xi0_pts[i]**2)

np.savez('slice_data.npz', 
         s0x=s0_x_pts, s0y=s0_y_pts, s1x=s1_x_pts, 
         s1y=s1_y_pts, xpts=xi0_pts, ypts=y_ana_pts)

plt.figure()
plt.plot(s0_x_pts, s0_y_pts, linewidth=3)
plt.plot(s1_x_pts, s1_y_pts, linewidth=3)
plt.plot(xi0_pts, y_ana_pts, '--', color='gray', linewidth=2)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.show()

# for i in range(preprocessor.num_intersections_all):
#     mesh_phy = generate_mortar_mesh(preprocessor.intersections_phy_coords[i], num_el=128)
#     File('./geometry/int_curve'+str(i)+'.pvd') << mesh_phy