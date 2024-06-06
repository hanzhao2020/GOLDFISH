import numpy as np
from GOLDFISH.utils.bsp_utils import *


cp_shape = [4,6]


A0, free_dof0 = surface_cp_align_operator(cp_shape, align_dir=0, coo=False)
A1, free_dof1 = surface_cp_align_operator(cp_shape, align_dir=1, coo=False)
A2, pin_dof0 = surface_cp_pin_operator(cp_shape, pin_dir0=0, pin_side0=[1], coo=False)
A3, pin_dof1 = surface_cp_pin_operator(cp_shape, pin_dir0=1, pin_side0=[1],
                                      pin_dir1=0, pin_side1=[0], coo=False)

A4 = surface_cp_regu_operator(cp_shape, regu_dir=1, coo=False)
A5 = surface_cp_regu_operator(cp_shape, regu_dir=0, rev_dir=True, coo=False)

input_array = np.array([3, 5, 6, 4, 7, 11])
A6 = expand_operator(input_array)
A7 = expand_operator(input_array, 'arithmetic')
A8 = expand_operator(input_array, 'geometric')


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
from int_xi_edge_comp_quadratic import IntXiEdgeComp
from int_energy_regu_comp import IntEnergyReguComp

from cpffd_rigid_comp import CPFFRigidComp
from create_geom_evtol_1spar import preprocessor


opt_field = [0,2]
shopt_surf_inds = [[3], [0,1,3]]
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

cpsurfd2a.get_init_coarse_CP()

t0 = cpsurfd2a.set_cp_align(2, [None, 1, None])
t1 = cpsurfd2a.set_cp_regu(2, [0, 0, 1])
