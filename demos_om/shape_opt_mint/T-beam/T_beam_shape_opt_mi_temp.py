"""
Initial geometry can be downloaded from the following link:
https://drive.google.com/file/d/1fRaho_xzmChlgLdrMM9CQ7WTqr9_DItt/view?usp=share_link
"""
import sys
sys.path.append("./opers/")
sys.path.append("./comps/")

# import numpy as np
# import matplotlib.pyplot as plt
# import openmdao.api as om
# from igakit.cad import *
# from igakit.io import VTK
# from GOLDFISH.nonmatching_opt_om import *

# from cpiga2xi_comp import CPIGA2XiComp
# from disp_states_mi_comp import DispMintStatesComp
# # from pin_cpsurf_comp import CPSurfPinComp
# # from align_cpsurf_comp import CPSurfAlignComp
# from max_int_xi_comp import MaxIntXiComp
# from min_int_xi_comp import MinIntXiComp
# from int_xi_edge_comp import IntXiEdgeComp

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
from int_xi_edge_int_comp import IntXiEdgeIntComp

from int_xi_extreme_comp import IntXiExtremeComp
# from int_xi_pin_comp import IntXiPinComp

set_log_active(False)


# class ShapeOptGroup(om.Group):

#     def initialize(self):
#         self.options.declare('nonmatching_opt')
#         self.options.declare('cpdesign2analysis')
#         self.options.declare('cpsurf_iga_design_name_pre', default='CPS_IGA_design')
#         self.options.declare('cpsurf_iga_name_pre', default='CPS_IGA')
#         self.options.declare('cpsurf_regu_name_pre', default='CPS_regu')
#         self.options.declare('int_name', default='int_para')
#         self.options.declare('disp_name', default='displacements')
#         self.options.declare('int_energy_name', default='int_E')
#         # self.options.declare('cpsurf_align_name_pre', default='CP_FFD_align')
#         self.options.declare('cpffd_pin_name_pre', default='CP_FFD_pin')
#         self.options.declare('volume_name', default='volume')
#         self.options.declare('max_int_xi_name', default='max_int_xi')
#         self.options.declare('min_int_xi_name', default='min_int_xi')
#         self.options.declare('int_xi_edge_name', default='int_xi_edge')

#     def init_parameters(self):
#         self.nonmatching_opt = self.options['nonmatching_opt']
#         self.cpdesign2analysis = self.options['cpdesign2analysis']
#         self.cpsurf_iga_design_name_pre = self.options['cpsurf_iga_design_name_pre']
#         self.cpsurf_iga_name_pre = self.options['cpsurf_iga_name_pre']
#         self.cpsurf_regu_name_pre = self.options['cpsurf_regu_name_pre']
#         self.int_name = self.options['int_name']
#         self.disp_name = self.options['disp_name']
#         self.int_energy_name = self.options['int_energy_name']
#         # self.cpsurf_align_name_pre = self.options['cpsurf_align_name_pre']
#         self.cpffd_pin_name_pre = self.options['cpffd_pin_name_pre']
#         self.volume_name = self.options['volume_name']
#         self.max_int_xi_name = self.options['max_int_xi_name']
#         self.min_int_xi_name = self.options['min_int_xi_name']
#         self.int_xi_edge_name = self.options['int_xi_edge_name']

#         self.opt_field = self.nonmatching_opt.opt_field
        
#         # self.input_cp_shapes = []
#         # for field_ind, field in enumerate(self.opt_field):        
#         #     self.input_cp_shapes += [len(self.nonmatching_opt.cpdes_iga_dofs[field_ind])]
#         # self.design_var_lower = -10.
#         # self.design_var_upper = 1.e-4

#         self.init_cp_design = self.cpdesign2analysis.init_cp_design
#         self.input_cp_shapes = [len(cp) for cp in self.init_cp_design]

#         self.design_var_lower = [-1.+1e-3, -2]
#         self.design_var_upper = [1.-1e-3, 0.5]

#         self.cpsurf_iga_design_name_list = []
#         self.cpsurf_iga_name_list = []
#         self.cpsurf_pin_name_list = []
#         # self.cpsurf_align_name_list = []
#         for i, field in enumerate(self.opt_field):
#             self.cpsurf_iga_name_list += [self.cpsurf_iga_name_pre+str(field)]
#             self.cpsurf_iga_design_name_list += [self.cpsurf_iga_design_name_pre+str(field)]
#             # self.cpsurf_align_name_list += [self.cpsurf_align_name_pre+str(field)]
#             self.cpsurf_pin_name_list += [self.cpffd_pin_name_pre+str(field)]

#         self.regu_field = []
#         for i, mat in enumerate(self.cpdesign2analysis.cp_coarse_regu_deriv_list):
#             if mat is not None:
#                 self.regu_field += [self.opt_field[i]]

#         self.cpsurf_regu_name_list = []
#         for i, field in enumerate(self.regu_field):
#             self.cpsurf_regu_name_list += [self.cpsurf_regu_name_pre+str(field)]

#         self.inputs_comp_name = 'inputs_comp'
#         self.cpdesign2full_comp_name = 'CPdesign2surf_comp'
#         self.cpiga2xi_comp_name = 'CPIGA2Xi_comp'
#         self.disp_states_comp_name = 'disp_states_comp'
#         self.int_energy_comp_name = 'internal_energy_comp'
#         # self.cpsurf_align_comp_name = 'CPS_align_comp'
#         # self.cpsurf_align_comp2_name = 'CPS_align_comp2'
#         # self.cpsurf_pin_comp_name = 'CPS_pin_comp'
#         self.volume_comp_name = 'volume_comp'
#         self.max_int_xi_comp_name = 'max_int_xi_comp'
#         self.min_int_xi_comp_name = 'min_int_xi_comp'
#         self.int_xi_edge_comp_name = 'int_xi_edge_comp'

#         self.cpsurf_regu_comp_name = 'CPS_regu_comp'


#     def setup(self):
#         # Add inputs comp
#         inputs_comp = om.IndepVarComp()
#         for i, field in enumerate(self.opt_field):
#             inputs_comp.add_output(self.cpsurf_iga_design_name_list[i],
#                         shape=self.input_cp_shapes[i],
#                         val=self.init_cp_design[i])
#         self.add_subsystem(self.inputs_comp_name, inputs_comp)

#         self.cpdesign2full_comp = CPSurfDesign2AnalysisComp(
#                                   cpdesign2analysis=self.cpdesign2analysis,
#                                   input_cp_design_name_pre=self.cpsurf_iga_design_name_pre,
#                                   output_cp_analysis_name_pre=self.cpsurf_iga_name_pre)
#         self.cpdesign2full_comp.init_parameters()
#         self.add_subsystem(self.cpdesign2full_comp_name, self.cpdesign2full_comp)


#         # Add CP surf pin comp (linear constraint)
#         self.cpsurf_regu_comp = CPSurfReguComp(
#                          cpdesign2analysis=self.cpdesign2analysis,
#                          input_cp_design_name_pre=self.cpsurf_iga_design_name_pre,
#                          output_cp_regu_name_pre=self.cpsurf_regu_name_pre)
#         self.cpsurf_regu_comp.init_parameters()
#         self.add_subsystem(self.cpsurf_regu_comp_name, self.cpsurf_regu_comp)
#         self.cpsurf_regu_cons_vals = [np.ones(shape)*5e-2 for shape 
#                                      in self.cpsurf_regu_comp.output_shapes]

#         # self.cpdesign2full_comp = CPSurfDesign2FullComp(
#         #                           nonmatching_opt=self.nonmatching_opt,
#         #                           input_cp_design_name=self.cpsurf_iga_design_name_pre,
#         #                           output_cp_iga_name_pre=self.cpsurf_iga_name_pre)
#         # self.cpdesign2full_comp.init_parameters()
#         # self.add_subsystem(self.cpdesign2full_comp_name, self.cpdesign2full_comp)

#         # Add CPIGA2Xi comp
#         self.cpiga2xi_comp = CPIGA2XiComp(
#                         nonmatching_opt=self.nonmatching_opt,
#                         input_cp_iga_name_pre=self.cpsurf_iga_name_pre,
#                         output_xi_name=self.int_name)
#         self.cpiga2xi_comp.init_parameters()
#         self.add_subsystem(self.cpiga2xi_comp_name, self.cpiga2xi_comp)

#         # Add disp_states_comp
#         self.disp_states_comp = DispMintStatesComp(
#                            nonmatching_opt=self.nonmatching_opt,
#                            input_cp_iga_name_pre=self.cpsurf_iga_name_pre,
#                            input_xi_name=self.int_name,
#                            output_u_name=self.disp_name)
#         self.disp_states_comp.init_parameters(save_files=True,
#                                              nonlinear_solver_rtol=1e-3)
#         self.add_subsystem(self.disp_states_comp_name, self.disp_states_comp)

#         # Add internal energy comp (objective function)
#         self.int_energy_comp = IntEnergyComp(
#                           nonmatching_opt=self.nonmatching_opt,
#                           input_cp_iga_name_pre=self.cpsurf_iga_name_pre,
#                           input_u_name=self.disp_name,
#                           output_wint_name=self.int_energy_name)
#         self.int_energy_comp.init_parameters()
#         self.add_subsystem(self.int_energy_comp_name, self.int_energy_comp)

#         # # Add CP surf align comp (linear constraint)
#         # self.cpsurf_align_comp = CPSurfAlignComp(
#         #     nonmatching_opt=self.nonmatching_opt,
#         #     align_surf_ind=[1],
#         #     align_dir=[1],
#         #     input_cp_iga_name_pre=self.cpsurf_iga_name_pre,
#         #     output_cp_align_name_pre=self.cpsurf_align_name_pre)
#         # self.cpsurf_align_comp.init_parameters()
#         # self.add_subsystem(self.cpsurf_align_comp_name, self.cpsurf_align_comp)
#         # self.cpsurf_align_cons_val = np.zeros(self.cpsurf_align_comp.output_shape)

#         # # Add CP surf pin comp (linear constraint)
#         # self.cpsurf_pin_comp = CPSurfPinComp(
#         #                  nonmatching_opt=self.nonmatching_opt,
#         #                  pin_surf_inds=[0],
#         #                  input_cp_iga_name_pre=self.cpsurf_iga_name_pre,
#         #                  output_cp_pin_name_pre=self.cpffd_pin_name_pre)
#         # self.cpsurf_pin_comp.init_parameters()
#         # self.add_subsystem(self.cpsurf_pin_comp_name, self.cpsurf_pin_comp)
#         # self.cpsurf_pin_cons_val = np.zeros(self.cpsurf_pin_comp.output_shape)

#         # # Add CP surf pin comp (linear constraint)
#         # self.cpsurf_pin_comp = CPSurfPinComp(
#         #                  nonmatching_opt=self.nonmatching_opt,
#         #                  input_cp_iga_name_pre=self.cpsurf_iga_name_pre,
#         #                  output_cp_pin_name_pre=self.cpffd_pin_name_pre)
#         # self.cpsurf_pin_comp.init_parameters()
#         # self.add_subsystem(self.cpsurf_pin_comp_name, self.cpsurf_pin_comp)
#         # self.cpsurf_pin_cons_vals = [np.zeros(shape) for shape 
#         #                              in self.cpsurf_pin_comp.output_shapes]

#         # Add volume comp (constraint)
#         self.volume_comp = VolumeComp(
#                            nonmatching_opt=self.nonmatching_opt,
#                            input_cp_iga_name_pre=self.cpsurf_iga_name_pre,
#                            output_vol_name=self.volume_name)
#         self.volume_comp.init_parameters()
#         self.add_subsystem(self.volume_comp_name, self.volume_comp)
#         self.vol_val = 0
#         for s_ind in range(self.nonmatching_opt.num_splines):
#             self.vol_val += assemble(self.nonmatching_opt.h_th[s_ind]
#                             *self.nonmatching_opt.splines[s_ind].dx)

#         # # Add max int xi comp (constraint)
#         # self.max_int_xi_comp = MaxIntXiComp(
#         #                        nonmatching_opt=self.nonmatching_opt, rho=1e3,
#         #                        input_xi_name=self.int_name,
#         #                        output_name=self.max_int_xi_name)
#         # self.max_int_xi_comp.init_parameters()
#         # self.add_subsystem(self.max_int_xi_comp_name, self.max_int_xi_comp)

#         # # Add min int xi comp (constraint)
#         # self.min_int_xi_comp = MinIntXiComp(
#         #                        nonmatching_opt=self.nonmatching_opt, rho=1e3,
#         #                        input_xi_name=self.int_name,
#         #                        output_name=self.min_int_xi_name)
#         # self.min_int_xi_comp.init_parameters()
#         # self.add_subsystem(self.min_int_xi_comp_name, self.min_int_xi_comp)

#         # Add int xi edge comp (constraint)
#         self.int_xi_edge_comp = IntXiEdgeComp(
#                                nonmatching_opt=self.nonmatching_opt,
#                                input_xi_name=self.int_name,
#                                output_name=self.int_xi_edge_name)
#         self.int_xi_edge_comp.init_parameters()
#         self.add_subsystem(self.int_xi_edge_comp_name, self.int_xi_edge_comp)
#         self.int_xi_edge_cons_val = np.zeros(self.int_xi_edge_comp.output_shape)

#         # Connect names between components
#         for i, field in enumerate(self.opt_field):
#             # For optimization components
#             self.connect(self.inputs_comp_name+'.'
#                          +self.cpsurf_iga_design_name_list[i],
#                          self.cpdesign2full_comp_name+'.'
#                          +self.cpsurf_iga_design_name_list[i])


#             self.connect(self.cpdesign2full_comp_name+'.'
#                          +self.cpsurf_iga_name_list[i],
#                          self.cpiga2xi_comp_name+'.'
#                          +self.cpsurf_iga_name_list[i])

#             self.connect(self.cpdesign2full_comp_name+'.'
#                          +self.cpsurf_iga_name_list[i],
#                          self.disp_states_comp_name+'.'
#                          +self.cpsurf_iga_name_list[i])

#             self.connect(self.cpdesign2full_comp_name+'.'
#                          +self.cpsurf_iga_name_list[i],
#                          self.int_energy_comp_name+'.'
#                          +self.cpsurf_iga_name_list[i])

#             self.connect(self.cpdesign2full_comp_name+'.'
#                          +self.cpsurf_iga_name_list[i],
#                          self.volume_comp_name+'.'
#                          +self.cpsurf_iga_name_list[i])

#             # For constraints
#             # self.connect(self.cpdesign2full_comp_name+'.'
#             #              +self.cpsurf_iga_name_list[i],
#             #              self.cpsurf_pin_comp_name+'.'
#             #              +self.cpsurf_iga_name_list[i])

#             # self.connect(self.inputs_comp_name+'.'
#             #              +self.cpsurf_iga_name_list[i],
#             #              self.cpsurf_align_comp_name +'.'
#             #              +self.cpsurf_iga_name_list[i])

#         # # For xi constraints
#         # self.connect(self.cpiga2xi_comp_name+'.'+self.int_name,
#         #              self.max_int_xi_comp_name+'.'+self.int_name)

#         # self.connect(self.cpiga2xi_comp_name+'.'+self.int_name,
#         #              self.min_int_xi_comp_name+'.'+self.int_name)

#         self.connect(self.cpiga2xi_comp_name+'.'+self.int_name,
#                      self.int_xi_edge_comp_name+'.'+self.int_name)

#         self.connect(self.cpiga2xi_comp_name+'.'+self.int_name,
#                      self.disp_states_comp_name+'.'+self.int_name)

#         self.connect(self.disp_states_comp_name+'.'+self.disp_name,
#                      self.int_energy_comp_name+'.'+self.disp_name)

#         # For constraints
#         for i, regu_field in enumerate(self.regu_field):
#             opt_field_ind = self.opt_field.index(regu_field)
#             self.connect(self.inputs_comp_name+'.'
#                          +self.cpsurf_iga_design_name_list[opt_field_ind],
#                          self.cpsurf_regu_comp_name+'.'
#                          +self.cpsurf_iga_design_name_list[opt_field_ind])

#         # Add design variable, constraints and objective
#         for i, field in enumerate(self.opt_field):
#             self.add_design_var(self.inputs_comp_name+'.'
#                                 +self.cpsurf_iga_design_name_list[i],
#                                 lower=self.design_var_lower[i],
#                                 upper=self.design_var_upper[i])
#             # self.add_constraint(self.cpsurf_pin_comp_name+'.'
#             #                     +self.cpsurf_pin_name_list[i],
#             #                     equals=self.cpsurf_pin_cons_val)
#             # self.add_constraint(self.cpsurf_align_comp_name+'.'
#             #                     +self.cpsurf_align_name_list[i],
#             #                     equals=self.cpsurf_align_cons_val[i])

#         for i, regu_field in enumerate(self.regu_field):
#             self.add_constraint(self.cpsurf_regu_comp_name+'.'
#                          +self.cpsurf_regu_name_list[i],
#                          lower=self.cpsurf_regu_cons_vals[i])

#         # self.add_constraint(self.cpiga2xi_comp_name+'.'+self.int_name,
#         #                     lower=0., upper=1.)
#         # self.add_constraint(self.max_int_xi_comp_name+'.'+self.max_int_xi_name,
#         #                     upper=1.)
#         # self.add_constraint(self.min_int_xi_comp_name+'.'+self.min_int_xi_name,
#         #                     lower=0.)
#         self.add_constraint(self.int_xi_edge_comp_name+'.'+self.int_xi_edge_name,
#                             equals=self.int_xi_edge_cons_val)

#         self.add_constraint(self.volume_comp_name+'.'+self.volume_name,
#                             equals=self.vol_val)
#         self.add_objective(self.int_energy_comp_name+'.'
#                            +self.int_energy_name,
#                            scaler=1e6)



class ShapeOptGroup(om.Group):

    def initialize(self):
        self.options.declare('nonmatching_opt')
        self.options.declare('cpdesign2analysis')
        self.options.declare('cp_design_name_pre', default='CP_design')
        self.options.declare('cp_coarse_name_pre', default='CP_coarse')
        self.options.declare('cp_order_ele_name_pre', default='CP_order_ele')
        self.options.declare('cp_analysis_name_pre', default='CP_analysis')

        self.options.declare('cp_regu_name_pre', default='CP_regu')
        self.options.declare('cp_pin_name_pre', default='CP_pin')
        self.options.declare('cp_dist_name_pre', default='CP_dist')

        self.options.declare('int_name', default='int_para')
        self.options.declare('disp_name', default='displacements')
        self.options.declare('int_energy_name', default='int_E')
        self.options.declare('volume_name', default='volume')
        self.options.declare('max_int_xi_name', default='max_int_xi')
        self.options.declare('min_int_xi_name', default='min_int_xi')
        self.options.declare('int_xi_edge_name', default='int_xi_edge')
        self.options.declare('int_xi_extreme_name', default='int_xi_extreme')
        self.options.declare('int_xi_pin_name', default='int_xi_pin')


    def init_parameters(self):
        self.nonmatching_opt = self.options['nonmatching_opt']
        self.cpdesign2analysis = self.options['cpdesign2analysis']
        self.cp_design_name_pre = self.options['cp_design_name_pre']
        self.cp_coarse_name_pre = self.options['cp_coarse_name_pre']
        self.cp_order_ele_name_pre = self.options['cp_order_ele_name_pre']
        self.cp_analysis_name_pre = self.options['cp_analysis_name_pre']

        self.cp_regu_name_pre = self.options['cp_regu_name_pre']
        self.cp_pin_name_pre = self.options['cp_pin_name_pre']
        self.cp_dist_name_pre = self.options['cp_dist_name_pre']

        self.int_name = self.options['int_name']
        self.disp_name = self.options['disp_name']
        self.int_energy_name = self.options['int_energy_name']
        self.volume_name = self.options['volume_name']
        self.max_int_xi_name = self.options['max_int_xi_name']
        self.min_int_xi_name = self.options['min_int_xi_name']
        self.int_xi_edge_name = self.options['int_xi_edge_name']
        self.int_xi_extreme_name = self.options['int_xi_extreme_name']
        self.int_xi_pin_name = self.options['int_xi_pin_name']

        self.opt_field = self.nonmatching_opt.opt_field

        self.init_cp_design = self.cpdesign2analysis.init_cp_design
        self.input_cp_shapes = [len(cp) for cp in self.init_cp_design]

        self.design_var_lower = [-1., -2] # x min, z min
        self.design_var_upper = [1., 0.5] # x max, z max

        self.cp_design_name_list = []
        self.cp_coarse_name_list = []
        self.cp_order_ele_name_list = []
        self.cp_analysis_name_list = []
        for field_ind, field in enumerate(self.opt_field):
            self.cp_design_name_list += [self.cp_design_name_pre+str(field)]
            self.cp_coarse_name_list += [self.cp_coarse_name_pre+str(field)]
            self.cp_order_ele_name_list += [self.cp_order_ele_name_pre+str(field)]
            self.cp_analysis_name_list += [self.cp_analysis_name_pre+str(field)]


        self.regu_field = self.cpdesign2analysis.cp_coarse_regu_field
        self.cp_regu_name_list = []
        for field_ind, field in enumerate(self.regu_field):
            self.cp_regu_name_list += [self.cp_regu_name_pre+str(field)]

        self.pin_field = self.cpdesign2analysis.cp_coarse_pin_field
        self.cp_pin_name_list = []
        for field_ind, field in enumerate(self.pin_field):
            self.cp_pin_name_list += [self.cp_pin_name_pre+str(field)]

        self.dist_field = self.cpdesign2analysis.cp_coarse_dist_field
        self.cp_dist_name_list = []
        for field_ind, field in enumerate(self.dist_field):
            self.cp_dist_name_list += [self.cp_dist_name_pre+str(field)]

        self.inputs_comp_name = 'inputs_comp'
        self.cp_align_comp_name = 'CP_design_align_comp'
        self.cp_order_ele_comp_name = 'CP_order_ele_comp'
        self.cp_knot_refine_comp_name = 'CP_knot_refine_comp'

        self.cp_regu_comp_name = 'CP_regu_comp'
        self.cp_pin_comp_name = 'CP_pin_comp'
        self.cp_dist_comp_name = 'CP_dist_comp'

        self.cpiga2xi_comp_name = 'CPIGA2Xi_comp'
        self.disp_states_comp_name = 'disp_states_comp'
        self.int_energy_comp_name = 'internal_energy_comp'
        self.volume_comp_name = 'volume_comp'

        self.max_int_xi_comp_name = 'max_int_xi_comp'
        self.min_int_xi_comp_name = 'min_int_xi_comp'
        self.int_xi_edge_comp_name = 'int_xi_edge_comp'
        self.int_xi_extreme_comp_name = 'int_xi_extreme_comp'
        self.int_xi_pin_comp_name = 'int_xi_pin_comp'

        # self.diff_vec = []
        # for field_ind, field in enumerate(self.opt_field):
        #     if field == 0:
        #         self.diff_vec += [np.asarray(self.cpdesign2analysis.init_cp_coarse[field_ind] \
        #                        - np.dot(self.cpdesign2analysis.cp_coarse_align_deriv_list[field_ind].todense()[:,0:2],
        #                          self.cpdesign2analysis.init_cp_design[field_ind]))[0]]
        #     else:
        #         self.diff_vec += [self.cpdesign2analysis.init_cp_coarse[field_ind] \
        #                        - self.cpdesign2analysis.cp_coarse_align_deriv_list[field_ind] \
        #                         *self.cpdesign2analysis.init_cp_design[field_ind]]
        #     if field in [1,2]:
        #         self.diff_vec[field_ind] = np.zeros(self.diff_vec[field_ind].size)
        
    def setup(self):
        # Add inputs comp
        inputs_comp = om.IndepVarComp()
        for field_ind, field in enumerate(self.opt_field):
            inputs_comp.add_output(self.cp_design_name_list[field_ind],
                        shape=self.input_cp_shapes[field_ind],
                        val=self.init_cp_design[field_ind])
        self.add_subsystem(self.inputs_comp_name, inputs_comp)

        self.cp_align_comp = CPSurfAlignComp(
        # self.cp_align_comp = CPSurfRigidAlignComp(
                             cpdesign2analysis=self.cpdesign2analysis,
                             diff_vec=None, #self.diff_vec,
                             input_cp_design_name_pre=self.cp_design_name_pre,
                             output_cp_coarse_name_pre=self.cp_coarse_name_pre)
        self.cp_align_comp.init_parameters()
        self.add_subsystem(self.cp_align_comp_name, self.cp_align_comp)

        self.cp_order_ele_comp = CPSurfOrderElevationComp(
                                 cpdesign2analysis=self.cpdesign2analysis,
                                 input_cp_coarse_name_pre=self.cp_coarse_name_pre,
                                 output_cp_order_ele_name_pre=self.cp_order_ele_name_pre)
        self.cp_order_ele_comp.init_parameters()
        self.add_subsystem(self.cp_order_ele_comp_name, self.cp_order_ele_comp)


        self.cp_knot_refine_comp = CPSurfKnotRefinementComp(
                                   cpdesign2analysis=self.cpdesign2analysis,
                                   input_cp_order_ele_name_pre=self.cp_order_ele_name_pre,
                                   output_cp_fine_name_pre=self.cp_analysis_name_pre)
        self.cp_knot_refine_comp.init_parameters()
        self.add_subsystem(self.cp_knot_refine_comp_name, self.cp_knot_refine_comp)


        if len(self.regu_field) > 0:
            self.cp_regu_comp = CPSurfReguComp(
                                cpdesign2analysis=self.cpdesign2analysis,
                                input_cp_design_name_pre=self.cp_design_name_pre,
                                output_cp_regu_name_pre=self.cp_regu_name_pre)
            self.cp_regu_comp.init_parameters()
            self.add_subsystem(self.cp_regu_comp_name, self.cp_regu_comp)


        if len(self.pin_field) > 0:
            self.cp_pin_comp = CPSurfPinComp(
                                cpdesign2analysis=self.cpdesign2analysis,
                                input_cp_design_name_pre=self.cp_design_name_pre,
                                output_cp_pin_name_pre=self.cp_pin_name_pre)
            self.cp_pin_comp.init_parameters()
            self.add_subsystem(self.cp_pin_comp_name, self.cp_pin_comp)


        if len(self.dist_field) > 0:
            self.cp_dist_comp = CPSurfDistanceComp(
                                cpdesign2analysis=self.cpdesign2analysis,
                                input_cp_design_name_pre=self.cp_design_name_pre,
                                output_cp_dist_name_pre=self.cp_dist_name_pre)
            self.cp_dist_comp.init_parameters()
            self.add_subsystem(self.cp_dist_comp_name, self.cp_dist_comp)


        # Add CPIGA2Xi comp
        self.cpiga2xi_comp = CPIGA2XiComp(
                        nonmatching_opt=self.nonmatching_opt,
                        input_cp_iga_name_pre=self.cp_analysis_name_pre,
                        output_xi_name=self.int_name)
        self.cpiga2xi_comp.init_parameters()
        self.add_subsystem(self.cpiga2xi_comp_name, self.cpiga2xi_comp)

        # Add disp_states_comp
        self.disp_states_comp = DispMintStatesComp(
                           nonmatching_opt=self.nonmatching_opt,
                           input_cp_iga_name_pre=self.cp_analysis_name_pre,
                           input_xi_name=self.int_name,
                           output_u_name=self.disp_name)
        self.disp_states_comp.init_parameters(save_files=True,
                                             nonlinear_solver_rtol=1e-3,
                                             nonlinear_solver_max_it=10)
        self.add_subsystem(self.disp_states_comp_name, self.disp_states_comp)

        # Add internal energy comp (objective function)
        self.int_energy_comp = IntEnergyComp(
                          nonmatching_opt=self.nonmatching_opt,
                          input_cp_iga_name_pre=self.cp_analysis_name_pre,
                          input_u_name=self.disp_name,
                          output_wint_name=self.int_energy_name)
        self.int_energy_comp.init_parameters()
        self.add_subsystem(self.int_energy_comp_name, self.int_energy_comp)


        # # Add internal energy comp (objective function)
        # self.int_energy_comp = IntEnergyReguComp(
        #                   nonmatching_opt=self.nonmatching_opt,
        #                   input_cp_iga_name_pre=self.cp_analysis_name_pre,
        #                   input_u_name=self.disp_name,
        #                   input_xi_name=self.int_name,
        #                   output_wint_name=self.int_energy_name,
        #                   regu_para=1e3,
        #                   regu_xi_edge=True)
        # self.int_energy_comp.init_parameters()
        # self.add_subsystem(self.int_energy_comp_name, self.int_energy_comp)

        # Add volume comp (constraint)
        self.vol_surf_inds = [1]
        self.volume_comp = VolumeComp(
                           nonmatching_opt=self.nonmatching_opt,
                           vol_surf_inds=self.vol_surf_inds,
                           input_cp_iga_name_pre=self.cp_analysis_name_pre,
                           output_vol_name=self.volume_name)
        self.volume_comp.init_parameters()
        self.add_subsystem(self.volume_comp_name, self.volume_comp)
        self.vol_val = 0
        for s_ind in self.vol_surf_inds:
            self.vol_val += assemble(self.nonmatching_opt.h_th[s_ind]
                            *self.nonmatching_opt.splines[s_ind].dx)

        # self.volume_comps = []
        # self.vol_val_list = []
        # for vol_ind, surf_ind in enumerate(self.vol_surf_inds):
        #     self.volume_comps += [VolumeComp(
        #                    nonmatching_opt=self.nonmatching_opt,
        #                    vol_surf_inds=[surf_ind],
        #                    input_cp_iga_name_pre=self.cp_analysis_name_pre,
        #                    output_vol_name=self.volume_name+str(vol_ind))]
        #     self.volume_comps[vol_ind].init_parameters()
        #     self.add_subsystem(self.volume_comp_name+str(vol_ind), 
        #                       self.volume_comps[vol_ind])
        #     self.vol_val_list += [assemble(self.nonmatching_opt.h_th[surf_ind]
        #                           *self.nonmatching_opt.splines[surf_ind].dx)]

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


        # self.int_xi_edge_comp = IntXiEdgeIntComp(
        #                         nonmatching_opt=self.nonmatching_opt,
        #                         input_xi_name=self.int_name,
        #                         output_name=self.int_xi_edge_name)
        # self.int_xi_edge_comp.init_parameters()
        # self.add_subsystem(self.int_xi_edge_comp_name, self.int_xi_edge_comp)


        # self.int_xi_extreme_comp = IntXiExtremeComp(
        #                            nonmatching_opt=self.nonmatching_opt, 
        #                            num_eval_pts=extreme_cons_eval_pts,
        #                            input_xi_name=self.int_name,
        #                            output_name=self.int_xi_extreme_name)
        # self.int_xi_extreme_comp.init_parameters()
        # self.add_subsystem(self.int_xi_extreme_comp_name, self.int_xi_extreme_comp)

        # self.int_xi_pin_comp = IntXiPinComp(
        #                        nonmatching_opt=self.nonmatching_opt,
        #                        input_xi_name=self.int_name,
        #                        output_name=self.int_xi_pin_name)
        # self.int_xi_pin_comp.init_parameters()
        # self.add_subsystem(self.int_xi_pin_comp_name, self.int_xi_pin_comp)
        # self.int_xi_pin_cons_val = np.zeros(self.int_xi_pin_comp.output_shape)

        # Connect names between components
        for field_ind, field in enumerate(self.opt_field):
            # For optimization components
            self.connect(self.inputs_comp_name+'.'
                         +self.cp_design_name_list[field_ind],
                         self.cp_align_comp_name+'.'
                         +self.cp_design_name_list[field_ind])

            self.connect(self.cp_align_comp_name+'.'
                         +self.cp_coarse_name_list[field_ind],
                         self.cp_order_ele_comp_name+'.'
                         +self.cp_coarse_name_list[field_ind])

            self.connect(self.cp_order_ele_comp_name+'.'
                         +self.cp_order_ele_name_list[field_ind],
                         self.cp_knot_refine_comp_name+'.'
                         +self.cp_order_ele_name_list[field_ind])

            self.connect(self.cp_knot_refine_comp_name+'.'
                         +self.cp_analysis_name_list[field_ind],
                         self.cpiga2xi_comp_name+'.'
                         +self.cp_analysis_name_list[field_ind])

            self.connect(self.cp_knot_refine_comp_name+'.'
                         +self.cp_analysis_name_list[field_ind],
                         self.disp_states_comp_name+'.'
                         +self.cp_analysis_name_list[field_ind])

            self.connect(self.cp_knot_refine_comp_name+'.'
                         +self.cp_analysis_name_list[field_ind],
                         self.int_energy_comp_name+'.'
                         +self.cp_analysis_name_list[field_ind])

            self.connect(self.cp_knot_refine_comp_name+'.'
                         +self.cp_analysis_name_list[field_ind],
                         self.volume_comp_name+'.'
                         +self.cp_analysis_name_list[field_ind])

            # for vol_ind, surf_ind in enumerate(self.vol_surf_inds):
            #     self.connect(self.cp_knot_refine_comp_name+'.'
            #                  +self.cp_analysis_name_list[field_ind],
            #                  self.volume_comp_name+str(vol_ind)+'.'
            #                  +self.cp_analysis_name_list[field_ind])


        # self.connect(self.cpiga2xi_comp_name+'.'+self.int_name,
        #              self.max_int_xi_comp_name+'.'+self.int_name)

        # self.connect(self.cpiga2xi_comp_name+'.'+self.int_name,
        #              self.min_int_xi_comp_name+'.'+self.int_name)

        self.connect(self.cpiga2xi_comp_name+'.'+self.int_name,
                     self.int_xi_edge_comp_name+'.'+self.int_name)

        # self.connect(self.cpiga2xi_comp_name+'.'+self.int_name,
        #              self.int_xi_extreme_comp_name+'.'+self.int_name)

        # self.connect(self.cpiga2xi_comp_name+'.'+self.int_name,
        #              self.int_xi_pin_comp_name+'.'+self.int_name)

        self.connect(self.cpiga2xi_comp_name+'.'+self.int_name,
                     self.disp_states_comp_name+'.'+self.int_name)

        self.connect(self.disp_states_comp_name+'.'+self.disp_name,
                     self.int_energy_comp_name+'.'+self.disp_name)

        # # Int energy with xi edge regu
        # self.connect(self.cpiga2xi_comp_name+'.'+self.int_name,
        #              self.int_energy_comp_name+'.'+self.int_name)

        # For constraints
        if len(self.regu_field) > 0:
            for field_ind, field in enumerate(self.regu_field):
                opt_field_ind = self.opt_field.index(field)
                self.connect(self.inputs_comp_name+'.'
                         +self.cp_design_name_list[opt_field_ind],
                         self.cp_regu_comp_name+'.'
                         +self.cp_design_name_list[opt_field_ind])

        if len(self.pin_field) > 0:
            for field_ind, field in enumerate(self.pin_field):
                opt_field_ind = self.opt_field.index(field)
                self.connect(self.inputs_comp_name+'.'
                         +self.cp_design_name_list[opt_field_ind],
                         self.cp_pin_comp_name+'.'
                         +self.cp_design_name_list[opt_field_ind])

        if len(self.dist_field) > 0:
            for field_ind, field in enumerate(self.dist_field):
                opt_field_ind = self.opt_field.index(field)
                self.connect(self.inputs_comp_name+'.'
                         +self.cp_design_name_list[opt_field_ind],
                         self.cp_dist_comp_name+'.'
                         +self.cp_design_name_list[opt_field_ind])


        ####################################################
        # Add design variable, constraints and objective
        for field_ind, field in enumerate(self.opt_field):
            if field == 1:
                scaler = 1e3
            else:
                scaler = 1
            self.add_design_var(self.inputs_comp_name+'.'
                                +self.cp_design_name_list[field_ind],
                                lower=self.design_var_lower[field_ind],
                                upper=self.design_var_upper[field_ind],
                                scaler=scaler)


        if len(self.regu_field) > 0:
            for field_ind, regu_field in enumerate(self.regu_field):
                self.add_constraint(self.cp_regu_comp_name+'.'
                             +self.cp_regu_name_list[field_ind],
                             upper=0.25, lower=2e-2)

        if len(self.pin_field) > 0:
            for field_ind, pin_field in enumerate(self.pin_field):
                self.add_constraint(self.cp_pin_comp_name+'.'
                             +self.cp_pin_name_list[field_ind],
                             equals=0)

        if len(self.dist_field) > 0:
            for field_ind, dist_field in enumerate(self.dist_field):
                if dist_field == 0:
                    lower_val = 0.1
                else:
                    lower_val = 0.1
                self.add_constraint(self.cp_dist_comp_name+'.'
                             +self.cp_dist_name_list[field_ind],
                             lower=lower_val)


        # self.add_constraint(self.cpiga2xi_comp_name+'.'+self.int_name,
        #                     lower=0., upper=1.)#, scaler=1e3)
        # self.add_constraint(self.max_int_xi_comp_name+'.'+self.max_int_xi_name,
        #                     upper=1.)
        # self.add_constraint(self.min_int_xi_comp_name+'.'+self.min_int_xi_name,
        #                     lower=0.)
        self.add_constraint(self.int_xi_edge_comp_name+'.'+self.int_xi_edge_name,
                            equals=self.int_xi_edge_cons_val, scaler=1)
                            # upper=1e-3, scaler=1)

        # self.add_constraint(self.int_xi_extreme_comp_name+'.'+self.int_xi_extreme_name,
        #                     lower=0., upper=1.)
        # self.add_constraint(self.int_xi_pin_comp_name+'.'+self.int_xi_pin_name,
        #                     equals=self.int_xi_pin_cons_val)


        self.add_constraint(self.volume_comp_name+'.'+self.volume_name,
                            # upper=1.5*self.vol_val)
                            equals=self.vol_val)

        # for vol_ind, surf_ind in enumerate(self.vol_surf_inds):
        #     self.add_constraint(self.volume_comp_name+str(vol_ind)+'.'+self.volume_name+str(vol_ind),
        #                         # upper=1.2*self.vol_val_list[vol_ind])
        #                         equals=self.vol_val[vol_ind])

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

test_ind = 4
optimizer = 'SNOPT'
# optimizer = 'SLSQP'
# opt_field = [0,2]
# save_path = './'
# save_path = '/home/han/Documents/test_results/'
save_path = '/Users/hanzhao/Documents/test_results/'
# folder_name = "results/"
folder_name = "results"+str(test_ind)+"/"

# filename_igs = "./geometry/init_Tbeam_geom_moved.igs"
filename_igs = "./geometry/init_Tbeam_geom_curved.igs"
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
#     preprocessor.compute_intersections(mortar_refine=1)
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

opt_field = [0,2]
shopt_surf_inds = [[1], [1]]

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


# nonmatching_opt.set_geom_preprocessor(preprocessor)


# nonmatching_opt.set_shopt_surf_inds(opt_field=opt_field, shopt_surf_inds=shopt_surf_inds)
# nonmatching_opt.get_init_CPIGA()
# # nonmatching_opt.set_shopt_align_CP(align_surf_inds=[[1], [1]],
# #                                    align_dir=[[1], [1]])
                                   
# nonmatching_opt.create_mortar_meshes(preprocessor.mortar_nels)

# if mpirank == 0:
#     print("Setting up mortar meshes...")
# nonmatching_opt.mortar_meshes_setup(preprocessor.mapping_list, 
#                     preprocessor.intersections_para_coords, 
#                     penalty_coefficient, 2)

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
preprocessor.check_intersections_type()
preprocessor.get_diff_intersections()
num_edge_pts = [1]*1
nonmatching_opt.create_diff_intersections(num_edge_pts=num_edge_pts)
#################################

cpsurfd2a = CPSurfDesign2Analysis(preprocessor, opt_field=opt_field, 
            shopt_surf_inds=shopt_surf_inds)


p_list_x = [[3,1],]*len(shopt_surf_inds[0])
p_list_z = [[3,1],]*len(shopt_surf_inds[1])
init_p_list = [p_list_x,p_list_z]
init_knots_list = []
for field_ind, field in enumerate(opt_field):
    knots_list = []
    for p in init_p_list[field_ind]:
        knots_list += [[[0]*(p[0]+1)+[1]*(p[0]+1), 
                        [0]*(p[1]+1)+[1]*(p[1]+1)]]
    init_knots_list += [knots_list]


p_ele = [3,3]
p_list_ele = [] 
for field_ind, field in enumerate(opt_field):
    p_list = []
    for i, s_ind in enumerate(shopt_surf_inds[field_ind]):
        p_list += [p_ele]
    p_list_ele += [p_list]

knots_list_ele = []
for field_ind, field in enumerate(opt_field):
    knots_list = []
    for p in p_list_ele[field_ind]:
        knots_list += [[[0]*(p[0]+1)+[1]*(p[0]+1), 
                        [0]*(p[1]+1)+[1]*(p[1]+1)]]
    knots_list_ele += [knots_list]

# Optimize x, y and z direction
cpsurfd2a.set_init_knots_by_field(init_p_list, init_knots_list)
a0 = cpsurfd2a.set_order_elevation_by_field(p_list_ele, knots_list_ele)
a1 = cpsurfd2a.set_knot_refinement()
a2 = cpsurfd2a.get_init_cp_coarse()

# t00 = cpsurfd2a.set_cp_align(field=0, align_dir_list=[[0,1]])
t00 = cpsurfd2a.set_cp_align(field=0, align_dir_list=[1])
t00 = cpsurfd2a.set_cp_align(field=2, align_dir_list=[1])





# #################################
# preprocessor.check_intersections_type()
# preprocessor.get_diff_intersections()
# nonmatching_opt.create_diff_intersections()
# #################################


# cpsurfd2a = CPSurfDesign2Analysis(preprocessor, opt_field=opt_field, 
#             shopt_surf_inds=shopt_surf_inds)
# design_degree = [3,1]
# p_list = [[design_degree]*len(shopt_surf_inds[0]), 
#           [design_degree]*len(shopt_surf_inds[1])]
# design_knots = [[0]*(design_degree[0]+1)+[1]*(design_degree[0]+1),
#                 [0]*(design_degree[1]+1)+[1]*(design_degree[1]+1)]
# knots_list = [[design_knots]*len(shopt_surf_inds[0]), 
#               [design_knots]*len(shopt_surf_inds[1])]

# cpsurfd2a.set_init_knots(p_list, knots_list)
# cpsurfd2a.get_init_cp_coarse()
# t0 = cpsurfd2a.set_cp_align(0, [1])
# t0 = cpsurfd2a.set_cp_align(2, [1])
# t1 = cpsurfd2a.set_cp_regu(2, [0], rev_dir=True)

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
    prob.driver.opt_settings['Minor feasibility tolerance'] = 1e-6
    prob.driver.opt_settings['Major feasibility tolerance'] = 1e-6
    prob.driver.opt_settings['Major optimality tolerance'] = 1e-6
    prob.driver.opt_settings['Major iterations limit'] = 50000
    prob.driver.opt_settings['Summary file'] = './SNOPT_report/SNOPT_summary'+str(test_ind)+'.out'
    prob.driver.opt_settings['Print file'] = './SNOPT_report/SNOPT_print'+str(test_ind)+'.out'
    prob.driver.options['debug_print'] = ['objs', 'desvars']
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