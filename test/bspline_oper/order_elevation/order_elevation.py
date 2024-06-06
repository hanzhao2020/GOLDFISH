import math
import numpy as np
import os
from PENGoLINS.occ_preprocessing import *
from PENGoLINS.nonmatching_shell import *
from GOLDFISH.utils.opt_utils import *
from GOLDFISH.utils.bsp_utils import *
from igakit.cad import *


# # 1D
# p_input = 1
# p_output = 3
# knots_input = np.array([0.,]*(p_input+1)+[1.,]*(p_input+1))
# knots_output = np.array([0.,]*(p_output+1)+[1.,]*(p_output+1))
# num_cp_input = len(knots_input) - (p_input+1)
# num_cp_output = len(knots_output) - (p_output+1)

# num_eval_pts = 10
# eval_pts = np.linspace(0,1,num_eval_pts)


# bsp_input = BSpline([p_input], [knots_input])
# bsp_output  = BSpline([p_output], [knots_output])
# mat_input = np.zeros((num_eval_pts,num_cp_input))
# mat_output = np.zeros((num_eval_pts,num_cp_output))

# for i in range(num_eval_pts):
#     eval_pt_coord = [eval_pts[i]]
#     nodes_vals_input = bsp_input.getNodesAndEvals(eval_pt_coord)
#     for j in range(len(nodes_vals_input)):
#         node, val = nodes_vals_input[j]
#         mat_input[i, node] = val
#     nodes_vals_output = bsp_output.getNodesAndEvals(eval_pt_coord)
#     for j in range(len(nodes_vals_output)):
#         node, val = nodes_vals_output[j]
#         mat_output[i, node] = val


# LHS_square_mat = np.dot(mat_output.T, mat_output)

# cp_input = np.array([[0.,1.]]).T
# RHS_vec = np.dot(mat_output.T, np.dot(mat_input, cp_input))

# cp_output_solve = np.linalg.solve(LHS_square_mat, RHS_vec)
# cp_output_inv = np.dot(np.linalg.inv(LHS_square_mat), RHS_vec)


# 2D
p_input = [2,1]
p_output = [3,3]
knots_input = [np.array([0.,]*(p_input[0]+1)+[1.,]*(p_input[0]+1)),
               np.array([0.,]*(p_input[1]+1)+[1.,]*(p_input[1]+1))]
knots_output = [np.array([0.,]*(p_output[0]+1)+[1.,]*(p_output[0]+1)),
                np.array([0.,]*(p_output[1]+1)+[1.,]*(p_output[1]+1))]
num_cp_input = (len(knots_input[0])-(p_input[0]+1))*(len(knots_input[1])-(p_input[1]+1))
num_cp_output = (len(knots_output[0])-(p_output[0]+1))*(len(knots_output[1])-(p_output[1]+1))

num_eval_pts_1D = 20
eval_pts_1D = np.linspace(0,1,num_eval_pts_1D)
num_eval_pts = num_eval_pts_1D**2
eval_pts = np.zeros((num_eval_pts, 2))
for i in range(num_eval_pts_1D):
    for j in range(num_eval_pts_1D):
        eval_pts[i*num_eval_pts_1D+j,0] = eval_pts_1D[i]
        eval_pts[i*num_eval_pts_1D+j,1] = eval_pts_1D[j]


bsp_input = BSpline(p_input, knots_input)
bsp_output  = BSpline(p_output, knots_output)
mat_input = np.zeros((num_eval_pts,num_cp_input))
mat_output = np.zeros((num_eval_pts,num_cp_output))

for i in range(num_eval_pts):
    eval_pt_coord = eval_pts[i]
    nodes_vals_input = bsp_input.getNodesAndEvals(eval_pt_coord)
    for j in range(len(nodes_vals_input)):
        node, val = nodes_vals_input[j]
        mat_input[i, node] = val
    nodes_vals_output = bsp_output.getNodesAndEvals(eval_pt_coord)
    for j in range(len(nodes_vals_output)):
        node, val = nodes_vals_output[j]
        mat_output[i, node] = val


LHS_square_mat = np.dot(mat_output.T, mat_output)

cp_input = np.array([[0.,0.5, 1.,0., 0.5, 1.]]).T
RHS_vec = np.dot(mat_output.T, np.dot(mat_input, cp_input))

cp_output_solve = np.linalg.solve(LHS_square_mat, RHS_vec)
cp_output_inv = np.dot(np.linalg.inv(LHS_square_mat), RHS_vec)


order_ele_operator = np.dot(np.dot(np.linalg.inv(LHS_square_mat), mat_output.T), mat_input)
cp_output = np.dot(order_ele_operator, cp_input)