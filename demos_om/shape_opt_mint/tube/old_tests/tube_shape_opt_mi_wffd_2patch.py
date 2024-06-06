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


class ShapeOptGroup(om.Group):

    def initialize(self):
        self.options.declare('nonmatching_opt')
        self.options.declare('cpffd_name_pre', default='CP_FFD')
        self.options.declare('cpsurf_fe_name_pre', default='CPS_FE')
        self.options.declare('cpsurf_iga_name_pre', default='CPS_IGA')
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
        self.design_var_lower = [-1.e-3, -1.e-3] #z-lower
        self.design_var_upper = [2., 2.] #z-upper

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

        self.inputs_comp_name = 'inputs_comp'
        self.cpffd2fe_comp_name = 'CPFFD2FE_comp'
        self.cpfe2iga_comp_name = 'CPFE2IGA_comp'
        self.cpiga2xi_comp_name = 'CPIGA2Xi_comp'
        self.disp_states_comp_name = 'disp_states_comp'
        self.int_energy_comp_name = 'internal_energy_comp'
        self.cpffd_align_comp_name = 'CPFFD_align_comp'
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
                                             nonlinear_solver_rtol=1e-4,
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
        self.cpffd_regu_lower_val = 1.e-2
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

        # # Add volume comp (constraint)
        # self.volume_comp = VolumeComp(
        #                    nonmatching_opt=self.nonmatching_opt,
        #                    input_cp_iga_name_pre=self.cpsurf_iga_name_pre,
        #                    output_vol_name=self.volume_name)
        # self.volume_comp.init_parameters()
        # self.add_subsystem(self.volume_comp_name, self.volume_comp)
        # self.vol_val = 0
        # for s_ind in range(self.nonmatching_opt.num_splines):
        #     self.vol_val += assemble(self.nonmatching_opt.h_th[s_ind]
        #                     *self.nonmatching_opt.splines[s_ind].dx)

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

        # # Add int xi edge comp (constraint)
        # self.int_xi_edge_comp = IntXiEdgeComp(
        #                        nonmatching_opt=self.nonmatching_opt,
        #                        input_xi_name=self.int_name,
        #                        output_name=self.int_xi_edge_name)
        # self.int_xi_edge_comp.init_parameters()
        # self.add_subsystem(self.int_xi_edge_comp_name, self.int_xi_edge_comp)
        # self.int_xi_edge_cons_val = np.zeros(self.int_xi_edge_comp.output_shape)

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

            # self.connect(self.cpfe2iga_comp_name+'.'
            #              +self.cpsurf_iga_name_list[i],
            #              self.volume_comp_name+'.'
            #              +self.cpsurf_iga_name_list[i])

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

        # self.connect(self.cpiga2xi_comp_name+'.'+self.int_name,
        #              self.int_xi_edge_comp_name+'.'+self.int_name)

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
        # self.add_constraint(self.int_xi_edge_comp_name+'.'+self.int_xi_edge_name,
        #                     equals=self.int_xi_edge_cons_val, scaler=1e1)

        # self.add_constraint(self.volume_comp_name+'.'+self.volume_name,
        #                     equals=self.vol_val)
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

test_ind = 4
optimizer = 'SNOPT'
# optimizer = 'SLSQP'
opt_field = [0,1]
# save_path = './'
save_path = '/home/han/Documents/test_results/'
folder_name = "results"+str(test_ind)+"/"

filename_igs = "./geometry/init_tube_geom_2patch.igs"
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

# fields0 = [None, [[0,1,2]],]
# fields1= [None, [None,[0,1,2]]]
# spline_bc0 = SplineBC(directions=[1], sides=[None, [0]],
#                       fields=fields0, n_layers=[None, [1]])
# spline_bc1 = SplineBC(directions=[1], sides=[None, [1]],
#                       fields=fields1, n_layers=[None, [None,1]])
# spline_bcs = [spline_bc0, spline_bc1]

fields0 = [[[0,2]],]
fields1 = [[None,[1,2]],]
spline_bc0 = SplineBC(directions=[0], sides=[[0],],
                     fields=fields0, n_layers=[[1],])
spline_bc1 = SplineBC(directions=[0], sides=[[1],],
                     fields=fields1, n_layers=[[None, 1],])
spline_bcs = [spline_bc0, spline_bc1]*1

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
nonmatching_opt = NonMatchingOptFFD(splines, E, h_th, nu, 
                                    opt_field=opt_field, comm=worldcomm)
nonmatching_opt.create_mortar_meshes(preprocessor.mortar_nels)

if mpirank == 0:
    print("Setting up mortar meshes...")
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

# nonmatching_opt.create_files(save_path=save_path, 
#                              folder_name=folder_name)
# nonmatching_opt.save_files()
# print(aaa)


shopt_multi_ffd_inds = [[0], [1]]
nonmatching_opt.set_shopt_multiFFD_surf_inds(shopt_multi_ffd_inds)

#################################################
num_shopt_ffd = nonmatching_opt.num_shopt_ffd
shopt_ffd_lims_multiffd = nonmatching_opt.shopt_cpsurf_lims_multiffd

shopt_ffd_num_el = [[2,2,1], [2,2,1]]
shopt_ffd_p = [3]*num_shopt_ffd
extrude_dir = [None, None]
extrude_coeff = 0.3

shopt_ffd_block_list = []
for ffd_ind in range(num_shopt_ffd):
    if ffd_ind == 0:
        for field in [0,1]:
            cp_range = shopt_ffd_lims_multiffd[ffd_ind][field][1]\
                    -shopt_ffd_lims_multiffd[ffd_ind][field][0]
            if field == 0:
                coeff = extrude_coeff
                shopt_ffd_lims_multiffd[ffd_ind][field][1] = \
                    shopt_ffd_lims_multiffd[ffd_ind][field][1] + coeff*cp_range
            elif field == 1:
                coeff = -extrude_coeff
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
                coeff = -extrude_coeff
                shopt_ffd_lims_multiffd[ffd_ind][field][0] = \
                    shopt_ffd_lims_multiffd[ffd_ind][field][0] + coeff*cp_range
            elif field == 1:
                coeff = extrude_coeff
                shopt_ffd_lims_multiffd[ffd_ind][field][1] = \
                    shopt_ffd_lims_multiffd[ffd_ind][field][1] + coeff*cp_range
        shopt_ffd_block_list += [create_3D_block(shopt_ffd_num_el[ffd_ind],
                                        shopt_ffd_p[ffd_ind],
                                            shopt_ffd_lims_multiffd[ffd_ind])]

    # field = extrude_dir[ffd_ind]
    # if field is not None:
    #     cp_range = shopt_ffd_lims_multiffd[ffd_ind][field][1]\
    #             -shopt_ffd_lims_multiffd[ffd_ind][field][0]
    #     shopt_ffd_lims_multiffd[ffd_ind][field][1] = \
    #         shopt_ffd_lims_multiffd[ffd_ind][field][1] + 0.1*cp_range
    #     shopt_ffd_lims_multiffd[ffd_ind][field][0] = \
    #         shopt_ffd_lims_multiffd[ffd_ind][field][0] - 0.1*cp_range
    #     shopt_ffd_block_list += [create_3D_block(shopt_ffd_num_el[ffd_ind],
    #                                     shopt_ffd_p[ffd_ind],
    #                                     shopt_ffd_lims_multiffd[ffd_ind])]
    # else:
    #     shopt_ffd_block_list += [create_3D_block(shopt_ffd_num_el[ffd_ind],
    #                                     shopt_ffd_p[ffd_ind],
    #                                     shopt_ffd_lims_multiffd[ffd_ind])]

# for ffd_ind in range(num_shopt_ffd):
#     vtk_writer = VTKWriter()
#     vtk_writer.write("./geometry/arch_shopt_ffd_block_init"+str(ffd_ind)+".vtk", 
#                      shopt_ffd_block_list[ffd_ind])
#     vtk_writer.write_cp("./geometry/arch_shopt_ffd_cp_init"+str(ffd_ind)+".vtk", 
#                      shopt_ffd_block_list[ffd_ind])

shopt_ffd_knots_list = [ffd_block.knots for ffd_block 
                        in shopt_ffd_block_list]
shopt_ffd_control_list = [ffd_block.control for ffd_block 
                          in shopt_ffd_block_list]
print("Setting multiple shape FFD blocks ...")
nonmatching_opt.set_shopt_multiFFD(shopt_ffd_knots_list, 
                                   shopt_ffd_control_list)

########### Set constraints info #########
a0 = nonmatching_opt.set_shopt_regu_CP_multiFFD(shopt_regu_dir_list=[[None, None], 
                                                                [None, None]], 
                                           shopt_regu_side_list=[[None, None], 
                                                                 [None, None]])
a1 = nonmatching_opt.set_shopt_pin_CP_multiFFD(0, pin_dir0_list=[0, 1], 
                                          pin_side0_list=[[0], [0]],
                                          pin_dir1_list=[2, 2], 
                                          pin_side1_list=[0, 0])
a2 = nonmatching_opt.set_shopt_pin_CP_multiFFD(1, pin_dir0_list=[0, 1], 
                                          pin_side0_list=[[0], [0]],
                                          pin_dir1_list=[2, 2], 
                                          pin_side1_list=[0, 0])

a5 = nonmatching_opt.set_shopt_align_CP_multiFFD(shopt_align_dir_list=[2,2])

#################################################

#################################
preprocessor.check_intersections_type()
preprocessor.get_diff_intersections()
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
    prob.driver.opt_settings['Major feasibility tolerance'] = 1e-5
    prob.driver.opt_settings['Major optimality tolerance'] = 1e-4
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

cp0_shape = 100
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

#### Save initial and final shape of FFD block
init_cpffd = FFD_block.control[:,:,:,0:3].transpose(2,1,0,3).reshape(-1,3)
final_cpffd = init_cpffd.copy()

final_cpffd0 = prob[model.inputs_comp_name+'.'+model.cpffd_name_list[0]][0:100]
final_cpffd[:,0] = final_cpffd0
final_cpffd1 = prob[model.inputs_comp_name+'.'+model.cpffd_name_list[1]][0:100]
final_cpffd[:,1] = final_cpffd1
final_cpffd = final_cpffd.reshape(FFD_block.control[:,:,:,0:3]\
              .transpose(2,1,0,3).shape)
final_cpffd = final_cpffd.transpose(2,1,0,3)
final_FFD_block = NURBS(FFD_block.knots, final_cpffd)
VTK().write('./geometry/FFD_block_final0.vtk', final_FFD_block)

# Save initial discretized geometry
nonmatching_opt_ffd.create_files(save_path=save_path, 
                                 folder_name='results_temp/', 
                                 refine_mesh=False)

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