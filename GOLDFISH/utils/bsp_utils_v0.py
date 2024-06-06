import numpy as np
from scipy.sparse import coo_matrix, hstack, vstack

from PENGoLINS.occ_preprocessing import *
from PENGoLINS.nonmatching_shell import *
from GOLDFISH.utils.opt_utils import *
from GOLDFISH.utils.bsp_utils import *
from igakit.cad import *

from scipy.linalg import block_diag
from scipy.sparse import bmat


def ij2dof(l,i,j):
    return i + j*l

def find_mult(knots, u):
    """
    Return the multiplicity of a given knot ``u`` in
    the knot vector ``knots``.

    Parameters
    ----------
    knots : ndarray
    u : float

    Returns
    -------
    res : int
    """
    # return knots.tolist().count(u)
    tol = 1e-6
    u_mult = 0
    for i in range(len(knots)):
        if abs(knots[i] - u) < tol:
            u_mult += 1
    return u_mult

def spline_degree(knots, first_knot=0):
    """
    Return the degree of a spline given its konts vector.

    Parameters
    ----------
    knots : ndarray
    first_knot : float, optional, default is 0

    Returns
    -------
    res : int
    """
    return find_mult(knots, first_knot) - 1

def find_span(knots, u):
    """
    Find the knot span index of a given knot ``u`` in 
    the knot vector ``knots``.

    Parameters
    ----------
    knots : ndarray
    u : float

    Returns
    -------
    ind : int
    """
    degree = find_mult(knots, 0) - 1
    end_ind = len(knots) - 1
    n = end_ind - degree - 1
    ind = 0
    if u >= knots[n]:
        ind = n
    elif u <= knots[degree]:
        ind = degree
    else:
        low = degree
        high = n+1
        mid = int((low+high)/2)
        while u < knots[mid] or u >= knots[mid+1]:
            if u < knots[mid]:
                high = mid
            else:
                low = mid
            mid = int((low+high)/2)
        ind = mid
    return ind

def insert_knot(knots, cp, u, r=1):
    """
    Return the new knot vector and control points after 
    knot insertion.

    Parameters
    ----------
    knots : ndarray
    cp : ndarray
    u : float
    r : int, optional, default is 1

    Returns
    -------
    new_knots : ndarray
    new_cp : ndarray
    """
    degree = find_mult(knots, 0) - 1
    knot_end_ind = len(knots) - 1  # mp in the algorithm
    cp_end_ind = knot_end_ind - degree - 1  # np in the algorithm
    assert cp.shape[0] == cp_end_ind + 1
    num_row = cp.shape[1]
    k = find_span(knots, u)  # Knot span of inserting knot
    s = find_mult(knots, u)  # Initial multiplicity of inserting knot

    new_cp_end_ind = cp_end_ind + r
    new_knots = np.zeros(knot_end_ind + r + 1)
    new_cp = np.zeros((cp_end_ind + r + 1, num_row))
    temp_cp = np.zeros((degree-s+1, num_row))

    # Load new knot vector
    for i in range(0, k+1):
        new_knots[i] = knots[i]
    for i in range(1, r+1):
        new_knots[i+k] = u
    for i in range(k+1, knot_end_ind+1):
        new_knots[i+r] = knots[i]

    # Save unaltered control points
    for i in range(0, k-degree+1):
        new_cp[i] = cp[i]
    for i in range(k-s, cp_end_ind+1):
        new_cp[i+r] = cp[i]
    for i in range(0, degree-s+1):
        temp_cp[i] = cp[k-degree+i]

    # Insert knot
    for j in range(1, r+1):
        L = k-degree+j

        for i in range(0, degree-s-j+1):
            alpha = (u - knots[L+i])/(knots[i+k+1]-knots[L+i])
            temp_cp[i] = alpha*temp_cp[i+1] + (1.-alpha)*temp_cp[i]

        new_cp[L] = temp_cp[0]
        new_cp[k+r-j-s] = temp_cp[degree-j-s]

    # Load remaining control points
    for i in range(L+1, k-s):
        new_cp[i] = temp_cp[i-L]

    return new_knots, new_cp

def insert_knot_mat(knots, u, r=1):
    """
    Compute the linear operator of knot insertion.

    Parameters
    ----------
    knots : ndarray
    u : float
    r : int, optional, default is 1

    Returns
    -------
    A_PQ : ndarray
    """
    degree = find_mult(knots, 0) - 1
    knot_end_ind = len(knots) - 1
    cp_end_ind = knot_end_ind - degree - 1
    k = find_span(knots, u)  # Knot span of inserting knot
    s = find_mult(knots, u)  # Initial multiplicity of inserting knot
    assert r+s <= degree
    new_cp_end_ind = cp_end_ind + r

    # Linear operator from initial control points to temp control points
    A_PR = np.zeros((degree-s+1, cp_end_ind+1))
    # Linear operator form temp control points to new control points
    A_RQ = np.zeros((new_cp_end_ind+1, degree-s+1))
    # Linear operator from temp control points to temp control points
    A_RR = np.eye(degree-s+1)

    for i in range(0, degree-s+1):
        A_PR[i, k-degree+i] = 1

    for j in range(1, r+1):
        L = k-degree+j

        for i in range(0, degree-s-j+1):
            A_RR_temp = np.eye(degree-s+1)
            alpha = (u - knots[L+i])/(knots[i+k+1]-knots[L+i])
            A_RR_temp[i, i:i+2] = np.array([1.0-alpha, alpha])
            A_RR = np.dot(A_RR_temp, A_RR)

        A_RQ[L] = A_RR[0]
        A_RQ[k+r-j-s] = A_RR[degree-j-s]

    for i in range(L+1, k-s):
        A_RQ[i] = A_RR[i-L]

    A_PQ = np.dot(A_RQ, A_PR)

    for i in range(0, k-degree+1):
        A_PQ[i, i] = 1
    for i in range(k-s, cp_end_ind+1):
        A_PQ[i+r, i] = 1

    return A_PQ

def refine_knot_cp(knots, cp, ref_knots):
    """
    Return new knot vector and control points after knot 
    refinement.

    Parameters
    ----------
    knots : ndarray
    cp : ndarray
    ref_knots : ndarray

    Returns
    -------
    new_knots : ndarray
    new_cp : ndarray
    """
    degree = find_mult(knots, 0) - 1
    num_row = cp.shape[1]
    knot_end_ind = len(knots) - 1  # m
    cp_end_ind = knot_end_ind - degree - 1  # n
    r = len(ref_knots) - 1

    new_knots = np.zeros(knot_end_ind+r+2)
    new_cp = np.zeros((cp_end_ind+r+2, num_row))

    a = find_span(knots, ref_knots[0])
    b = find_span(knots, ref_knots[-1]) + 1

    for j in range(0, a-degree+1):
        new_cp[j,:] = cp[j,:]
    for j in range(b-1, cp_end_ind+1):
        new_cp[j+r+1,:] = cp[j,:]

    for j in range(0, a+1):
        new_knots[j] = knots[j]
    for j in range(b+degree, knot_end_ind+1):
        new_knots[j+r+1] = knots[j]

    i = b + degree - 1
    k = b + degree + r

    for j in range(r,-1,-1):

        while ref_knots[j] <= knots[i] and i > a:
            new_cp[k-degree-1] = cp[i-degree-1]
            new_knots[k] = knots[i]
            k = k - 1
            i = i - 1

        new_cp[k-degree-1] = new_cp[k-degree]

        for l in range(1, degree+1):
            L = k - degree + l
            alpha = new_knots[k+l] - ref_knots[j]

            if abs(alpha) < 1e-14:
                new_cp[L-1] = new_cp[L]
            else:
                alpha = alpha/(new_knots[k+l] - knots[i-degree+l])
                new_cp[L-1] = alpha*new_cp[L-1] + (1.0-alpha)*new_cp[L]

        new_knots[k] = ref_knots[j]
        k = k - 1

    return new_knots, new_cp

def refine_knot_mat(knots, ref_knots):
    """
    Return knot refinement linear operator.

    Parameters
    ----------
    knots : ndarray
    ref_knots : ndarray

    Returns
    -------
    A_PQ : ndarray
    """
    degree = find_mult(knots, 0) - 1
    knot_end_ind = len(knots) - 1
    cp_end_ind = knot_end_ind - degree - 1
    r = len(ref_knots) - 1

    new_knots = np.zeros(knot_end_ind+r+2)

    a = find_span(knots, ref_knots[0])
    b = find_span(knots, ref_knots[-1]) + 1

    A_PQ = np.zeros((cp_end_ind+r+2, cp_end_ind+1))
    A_QQ = np.eye(cp_end_ind+r+2)

    for j in range(0, a-degree+1):
        A_PQ[j,j] = 1
    for j in range(b-1, cp_end_ind+1):
        A_PQ[j+r+1,j] = 1

    for j in range(0, a+1):
        new_knots[j] = knots[j]
    for j in range(b+degree, knot_end_ind+1):
        new_knots[j+r+1] = knots[j]

    i = b + degree - 1
    k = b + degree + r

    for j in range(r,-1,-1):
        A_QQ_temp1 = np.eye(cp_end_ind+r+2)

        while ref_knots[j] <= knots[i] and i > a:
            new_knots[k] = knots[i]
            A_PQ[k-degree-1, i-degree-1] = 1
            k = k - 1
            i = i - 1

        A_QQ_temp1[k-degree-1] = 0
        A_QQ_temp1[k-degree-1, k-degree] = 1
        A_QQ = np.dot(A_QQ_temp1, A_QQ)

        for l in range(1, degree+1):
            A_QQ_temp2 = np.eye(cp_end_ind+r+2)
            L = k-degree+l
            alpha = new_knots[k+l] - ref_knots[j]

            if abs(alpha) < 1e-14:
                A_QQ_temp2[L-1] = 0
                A_QQ_temp2[L-1, L] = 1
            else:
                alpha = float(alpha/(new_knots[k+l] - knots[i-degree+l]))
                A_QQ_temp2[L-1, L-1:L+1] = np.array([alpha, 1.0-alpha])

            A_QQ = np.dot(A_QQ_temp2, A_QQ)

        new_knots[k] = ref_knots[j]
        k = k - 1

    A_PQ = np.dot(A_QQ, A_PQ)

    return A_PQ

def repeat_blcok_diag_matrix(A, times=1, coo=True):
    """
    Return a block diagnol matrix whose each block is ``A`` with
    ``times`` blocks.

    Parameters
    ----------
    A : ndarray
    times : int
    coo : bool, optional, default is True

    Return
    ------
    A_block : ndarray or COO matrix
    """
    shape = A.shape
    block_shape = (shape[0]*times, shape[1]*times)

    if coo:
        Acoo = coo_matrix(A)
        data_list = [Acoo.data for i in range(times)]
        row_list = [Acoo.row + i*shape[0] for i in range(times)]
        col_list = [Acoo.col + i*shape[1] for i in range(times)]

        data = np.concatenate(data_list)
        row = np.concatenate(row_list)
        col = np.concatenate(col_list)
        A_block = coo_matrix((data, (row, col)), shape=block_shape)
    else:
        A_block = np.zeros(block_shape)
        for i in range(times):
            A_block[i*shape[0]:(i+1)*shape[0], i*shape[1]:(i+1)*shape[1]] = A

    return A_block

def transpose_operator_row_based(num_row, num_col, coo=True):
    """
    Return a matrix which is able to transpose a matrix flatten
    by row.

    Parameters
    ----------
    num_row : int, number of rows for matrix that is being transposed
    num_col : int, number of columns for matrix that is being transposed
    coo : bool, optional, default is True

    Returns
    -------
    transpose_mat : ndarray or COO matrix
    """
    mat_size = num_row*num_col
    if coo:
        data = np.ones(mat_size)
        row = np.zeros(mat_size)
        col = np.zeros(mat_size)
        for i in range(num_col):
            for j in range(num_row):
                row[i*num_row+j] = i*num_row+j
                col[i*num_row+j] = j*num_col+i
        transpose_mat = coo_matrix((data, (row, col)), 
                        shape=(mat_size, mat_size))
    else:
        transpose_mat = np.zeros((mat_size, mat_size))
        for i in range(num_col):
            for j in range(num_row):
                transpose_mat[i*num_row+j, j*num_col+i] = 1
    return transpose_mat

def dup_rows_operator(num_el, dup_inds, coo=True):
    """
    Duplicate elements of a vector given the indices.

    Parameters
    ----------
    num_el : number of elements of vector
    dup_ints : list or ndarray, indices of elements 
               that are being duplicated
    coo : bool, optional, default is True

    Returns
    -------
    A : ndarray or COO matrix
    """
    A = np.eye(num_el)
    for i, ind in enumerate(dup_inds):
        insert_row = np.zeros(num_el)
        insert_row[ind] = 1.
        A = np.insert(A, ind+i, insert_row, 0)
    if coo:
        return coo_matrix(A)
    else:
        return A

def dup_cols_rows_operator(num_row, num_col, row_dup_inds, 
                           col_dup_inds, coo=True):
    """
    Duplicate rows and columns of a matrix by giving the indices
    of the rows and columns.

    Parameters
    ----------
    num_row : number of rows of the applicable matrix
    num_col : number of columns of the applicable matrix
    row_dup_inds : list or ndarray
    col_dup_inds : list or n darray
    coo : bool, optional, default is True

    Returns
    -------
    res : ndarray or COO matrix
    """
    # Duplicate cols
    col_dup_mat_sub = dup_rows_operator(num_col, col_dup_inds, True)
    col_dup_mat = repeat_blcok_diag_matrix(col_dup_mat_sub, num_row, True)
    # Duplicate rows
    num_col_new = num_col + len(col_dup_inds)
    I = np.eye(num_col_new)
    row_dup_mat_sub_list = []
    for i in range(num_row):
        if i in row_dup_inds:
            row_dup_mat_sub = np.concatenate([I,I])
        else:
            row_dup_mat_sub = I
        row_dup_mat_sub_list += [row_dup_mat_sub,]
    row_dup_mat = coo_matrix(block_diag(*row_dup_mat_sub_list))
    if coo:
        return row_dup_mat*col_dup_mat
    else:
        return (row_dup_mat*col_dup_mat).todense()


# def knot_refine_operator(init_knots, ref_knots0=None, ref_knots1=None, 
#                          coo=True):
#     """
#     Parameters
#     ----------
#     ikNURBS_coarse : igakit NURBS instance
#     ref_knots0 : ndarray, refine knots on parametric direction 0
#     ref_knots1 : ndarray, refine knots on parametric direction 1
#     """
#     # ikNURBS_coarse = ikNURBS_coarse
#     deg0 = spline_degree(init_knots[0])
#     deg1 = spline_degree(init_knots[1])
#     init_shape = [len(init_knots[0])-deg0-1, len(init_knots[1])-deg1-1]

#     ref_knots0 = ref_knots0
#     ref_knots1 = ref_knots1

#     operator = coo_matrix(np.eye(init_shape[0]*init_shape[1]))
#     if ref_knots0 is not None and len(ref_knots0) > 0:
#         A_PQ = refine_knot_mat(init_knots[0], ref_knots0)
#         A_block = repeat_blcok_diag_matrix(A_PQ, init_shape[1], True)
#         operator = (A_block*operator).tocoo()
#         init_shape[0] = init_shape[0]+ref_knots0.shape[0]        
#     if ref_knots1 is not None and len(ref_knots1) > 0:
#         A_PQ = refine_knot_mat(init_knots[1], ref_knots1)
#         A_block = repeat_blcok_diag_matrix(A_PQ, init_shape[0], True)
#         transpose_mat0 = transpose_operator_row_based(init_shape[1], 
#                                                       init_shape[0])
#         transpose_mat1 = transpose_operator_row_based(init_shape[0], 
#                          init_shape[1]+ref_knots1.shape[0])
#         A = (transpose_mat1*A_block*transpose_mat0).tocoo()
#         operator = (A*operator).tocoo()
#     if coo:
#         return operator
#     else:
#         return operator.todense()


def surface_knot_refine_operator(init_knots, ref_knots, coo=True):
    """
    Return knot refinement operator for a spline surface.
    
    Parameters
    ----------
    init_knots : list of ndarray, initial knots
    ref_knots : list of ndarray, refine knots
    coo : bool, optional, default is True

    Returns
    -------
    operator : ndarray or COO matrix
    """
    # ikNURBS_coarse = ikNURBS_coarse
    deg0 = spline_degree(init_knots[0])
    deg1 = spline_degree(init_knots[1])
    init_shape = [len(init_knots[0])-deg0-1, len(init_knots[1])-deg1-1]

    operator = coo_matrix(np.eye(init_shape[0]*init_shape[1]))
    if ref_knots[0] is not None and len(ref_knots[0]) > 0:
        A_PQ = refine_knot_mat(init_knots[0], ref_knots[0])
        A_block = repeat_blcok_diag_matrix(A_PQ, init_shape[1], True)
        operator = (A_block*operator).tocoo()
        init_shape[0] = init_shape[0]+ref_knots[0].shape[0]        
    if ref_knots[1] is not None and len(ref_knots[1]) > 0:
        A_PQ = refine_knot_mat(init_knots[1], ref_knots[1])
        A_block = repeat_blcok_diag_matrix(A_PQ, init_shape[0], True)
        transpose_mat0 = transpose_operator_row_based(init_shape[1], 
                                                      init_shape[0])
        transpose_mat1 = transpose_operator_row_based(init_shape[0], 
                         init_shape[1]+ref_knots[1].shape[0])
        A = (transpose_mat1*A_block*transpose_mat0).tocoo()
        operator = (A*operator).tocoo()
    if coo:
        return operator
    else:
        return operator.todense()


def normalize_knots_vector(knots):
    """
    Return normalized knots vector with range from 0 to 1

    Parameters
    ----------
    knots : list of ndarray

    Returns
    -------
    normalized_knots : ndarray
    """
    knots_range = knots[-1] - knots[0]
    normalized_knots = (np.array(knots)-knots[0])/knots_range
    return normalized_knots


def surface_order_elevation_operator(p_input, knots_input, 
                                     p_output, knots_output, 
                                     num_eval_pts_1D=20, coo=True):
    # p_input = [2,1]
    # p_output = [3,3]
    # knots_input = [np.array([0.,]*(p_input[0]+1)+[1.,]*(p_input[0]+1)),
    #                np.array([0.,]*(p_input[1]+1)+[1.,]*(p_input[1]+1))]
    # knots_output = [np.array([0.,]*(p_output[0]+1)+[1.,]*(p_output[0]+1)),
    #                 np.array([0.,]*(p_output[1]+1)+[1.,]*(p_output[1]+1))]
    num_cp_input = (len(knots_input[0])-(p_input[0]+1))\
                   *(len(knots_input[1])-(p_input[1]+1))
    num_cp_output = (len(knots_output[0])-(p_output[0]+1))\
                    *(len(knots_output[1])-(p_output[1]+1))
    if p_input[0]==p_output[0] and p_input[1]==p_output[1]:
        order_ele_operator = np.eye(num_cp_output)
    else:
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
        order_ele_operator = np.dot(np.dot(np.linalg.inv(LHS_square_mat),
                                    mat_output.T), mat_input)
    if coo:
        order_ele_operator = coo_matrix(order_ele_operator)
    return order_ele_operator


def expand_operator(input_array, progression=None, coo=True):
    """
    progression : str, 'arithmetic' or 'geometric' or None
    """
    if progression is not None:
        if progression == 'arithmetic':
            operator = np.ones((len(input_array),1))
            vec_add = np.zeros((len(input_array),1))
            for i in range(1, len(input_array)):
                vec_add[i,0] = input_array[i]-input_array[0]
            if coo:
                operator = coo_matrix(operator)
            return operator, vec_add
        elif progression == 'geometric':
            operator = np.ones((len(input_array),1))
            for i in range(1,len(input_array)):
                operator[i,0] = input_array[i]/input_array[0]
            if coo:
                operator = coo_matrix(operator)
            return operator
    else:
        operator = np.ones((len(input_array),1))
        if coo:
            operator = coo_matrix(operator)
        return operator

def surface_cp_align_operator(cp_shpae, align_dir, coo=True):
    l, m = cp_shpae
    if align_dir == 0:
        sub_mat_list = []
        for i in range(m):
            sub_mat = np.zeros((l,m))
            sub_mat[:,i] = 1.
            sub_mat_list += [coo_matrix(sub_mat)]
        align_operator = vstack(sub_mat_list).todense()
        free_dofs = list(range(0,l*m,l))
    elif align_dir == 1:
        sub_mat_list = []
        for i in range(m):
            sub_mat_list += [coo_matrix(np.eye(l))]
        align_operator = vstack(sub_mat_list).todense()
        free_dofs = list(range(0,l))
    elif align_dir == [0,1]:
        align_operator = np.ones((l*m,1))
        free_dofs = [0]
    else:
        raise ValueError("Undefined align_dir:", align_dir)
    if coo:
        align_operator = coo_matrix(align_operator)
    return align_operator, free_dofs

def surface_cp_pin_operator(cp_shape, pin_dir0, pin_side0, 
                         pin_dir1=None, pin_side1=[0,1], coo=True):
    """
    pin_dir0 : int, 0, 1, or None
    pin_side0 : list
    """
    l, m = cp_shape
    pin_dof = []
    if pin_dir0 == 0:
        for side in pin_side0:
            pin_dof += list(range(side*(l-1), l*m, l))
    elif pin_dir0 == 1:
        for side in pin_side0:
            pin_dof += list(range(side*l*(m-1), side*l*(m-1)+l, 1))
    else:
        raise ValueError("Undefined pin_dir0:", pin_dir0)

    if pin_dir1 is not None:
        if pin_dir1 == 0:
            for side in pin_side1:
                pin_dof += list(range(side*(l-1), l*m, l))
        elif pin_dir1 == 1:
            for side in pin_side1:
                pin_dof += list(range(side*l*(m-1), side*l*(m-1)+l, 1))
        else:
            raise ValueError("Undefined pin_dir1:", pin_dir1)

    pin_operator = np.zeros((len(pin_dof), l*m))
    for i in range(len(pin_dof)):
        pin_operator[i, pin_dof[i]] = 1.
    if coo:
        pin_operator = coo_matrix(pin_operator)
    return pin_operator, pin_dof

def surface_cp_regu_operator(cp_shape, regu_dir, rev_dir=False, coo=True):
    """
    regu_dir: int, 0 or 1
    """
    if rev_dir:
        coeff = -1
    else:
        coeff = 1
    l, m = cp_shape
    if regu_dir == 0:
        regu_operator = np.zeros(((l-1)*m, l*m))
        for i in range(l-1):
            for j in range(m):
                row_ind = i*m+j
                col_ind0 = ij2dof(l,i,j)
                col_ind1 = ij2dof(l,i+1,j)
                # print(row_ind, col_ind0, col_ind1)
                regu_operator[row_ind, col_ind0] = coeff*(-1.)
                regu_operator[row_ind, col_ind1] = coeff*1.
    elif regu_dir == 1:
        regu_operator = np.zeros((l*(m-1), l*m))
        for i in range(l):
            for j in range(m-1):
                row_ind = i*(m-1)+j
                col_ind0 = ij2dof(l,i,j)
                col_ind1 = ij2dof(l,i,j+1)
                regu_operator[row_ind, col_ind0] = coeff*(-1.)
                regu_operator[row_ind, col_ind1] = coeff*1.
    else:
        raise ValueError("Undefined regu_dir:", regu_dir)
    if coo:
        regu_operator = coo_matrix(regu_operator)
    return regu_operator


def create_surf_regu_operator(cp_size, num_surfs, rev_dir=False, coo=True):
    if rev_dir:
        sign = -1
    else:
        sign = 1
    operator_list = [[np.zeros((cp_size, cp_size)) 
                      for i in range(num_surfs)]
                      for j in range(num_surfs-1)]
    for surf_ind in (range(num_surfs-1)):
        operator_list[surf_ind][surf_ind] = coo_matrix(sign*np.eye(cp_size))
        operator_list[surf_ind][surf_ind+1] = coo_matrix(sign*(-1)*np.eye(cp_size))
    operator = bmat(operator_list)
    if not coo:
        operator = np.array(operator)
    return operator


class CPSurfDesign2Analysis(object):
    def __init__(self, preprocessor, opt_field, shopt_surf_inds, 
                 shopt_surf_inds_explicit=None):
        self.preprocessor = preprocessor
        self.opt_field = opt_field
        self.shopt_surf_inds = shopt_surf_inds
        if shopt_surf_inds_explicit is None:
            self.shopt_surf_inds_explicit = self.shopt_surf_inds
        else:
            self.shopt_surf_inds_explicit = shopt_surf_inds_explicit

        self.analysis_cp_shapes_all = [surf_data.control.shape[0:2] for surf_data in 
                          self.preprocessor.BSpline_surfs_data]
        self.analysis_knots_all = [surf_data.knots for surf_data in 
                               self.preprocessor.BSpline_surfs_data]
        self.analysis_degree_all = [surf_data.degree for surf_data in 
                               self.preprocessor.BSpline_surfs_data]
        self.analysis_cp_all = [surf_data.control[:,:,0:3].transpose(1,0,2).reshape(-1,3) 
                                for surf_data in self.preprocessor.BSpline_surfs_data]
        if self.preprocessor.reparametrize:
            self.analysis_cp_shapes_all = [surf_data.control.shape[0:2] for surf_data in 
                              self.preprocessor.BSpline_surfs_repara_data]
            self.analysis_knots_all = [surf_data.knots for surf_data in 
                                   self.preprocessor.BSpline_surfs_repara_data]
            self.analysis_degree_all = [surf_data.degree for surf_data in 
                               self.preprocessor.BSpline_surfs_repara_data]
            self.analysis_cp_all = [surf_data.control[:,:,0:3].transpose(1,0,2).reshape(-1,3) 
                                    for surf_data in self.preprocessor.BSpline_surfs_repara_data]
        if self.preprocessor.refine:
            self.analysis_cp_shapes_all = [surf_data.control.shape[0:2] for surf_data in 
                              self.preprocessor.BSpline_surfs_refine_data]
            self.analysis_knots_all = [surf_data.knots for surf_data in 
                                   self.preprocessor.BSpline_surfs_refine_data]
            self.analysis_degree_all = [surf_data.degree for surf_data in 
                               self.preprocessor.BSpline_surfs_refine_data]
            self.analysis_cp_all = [surf_data.control[:,:,0:3].transpose(1,0,2).reshape(-1,3) 
                                    for surf_data in self.preprocessor.BSpline_surfs_refine_data]

        self.analysis_cp_shapes = [[] for field in self.opt_field]
        self.analysis_cp_decate = [[] for field in self.opt_field]
        self.analysis_knots = [[] for field in self.opt_field]
        self.analysis_degree = [[] for field in self.opt_field]
        for field_ind, field in enumerate(self.opt_field):
            for i, surf_ind in enumerate(self.shopt_surf_inds[field_ind]):
                self.analysis_knots[field_ind] += \
                    [self.analysis_knots_all[surf_ind]]
                self.analysis_degree[field_ind] += \
                    [self.analysis_degree_all[surf_ind]]
                self.analysis_cp_shapes[field_ind] += \
                    [self.analysis_cp_shapes_all[surf_ind]]
                self.analysis_cp_decate[field_ind] += \
                    [self.analysis_cp_all[surf_ind][:,field]]
        self.init_analysis_cp = [None for field in self.opt_field]
        for field_ind, field in enumerate(self.opt_field):
            self.init_analysis_cp[field_ind] = \
                np.concatenate(self.analysis_cp_decate[field_ind])

    def set_init_knots_by_field(self, p_list, knots_list):
        self.design_degree = p_list
        self.design_knots = knots_list
        self.cp_coarse_sizes = [[] for field in self.opt_field]
        self.cp_coarse_shapes = [[] for field in self.opt_field]
        for field_ind, field in enumerate(self.opt_field):
            for i, degree in enumerate(self.design_degree[field_ind]):
                l = len(self.design_knots[field_ind][i][0])-degree[0]-1
                m = len(self.design_knots[field_ind][i][1])-degree[1]-1
                self.cp_coarse_sizes[field_ind] += [l*m]
                self.cp_coarse_shapes[field_ind] += [[l,m]]

        # Constraint related properties
        self.align_dir_list = [None for field in self.opt_field]
        self.cp_coarse_free_dofs = [np.arange(np.sum(self.cp_coarse_sizes[field_ind])) 
                                for field_ind in range(len(self.opt_field))]
        self.cp_coarse_free_dofs_decate = [[] for field in self.opt_field]

        self.cp_coarse_align_deriv_sub_list = [[] for field in self.opt_field]
        for field_ind, field in enumerate(self.opt_field):
            ind_off = 0
            for i, s_ind in enumerate(self.shopt_surf_inds[field_ind]):
                self.cp_coarse_align_deriv_sub_list[field_ind] += \
                    [np.eye(self.cp_coarse_sizes[field_ind][i])]
                self.cp_coarse_free_dofs_decate[field_ind] += \
                    [np.arange(self.cp_coarse_sizes[field_ind][i])+ind_off]
                ind_off += self.cp_coarse_sizes[field_ind][i]

        # self.cp_design2analysis_deriv_list = [None for field in self.opt_field]
        self.cp_coarse_align_deriv_list = [coo_matrix(block_diag(*mat_list)) 
                                           for mat_list in 
                                           self.cp_coarse_align_deriv_sub_list]

        self.order_ele_operator_list = [None for field in self.opt_field]
        self.knot_refine_operator_list = [None for field in self.opt_field]

        self.cp_coarse_pin_field = []
        self.cp_coarse_regu_field = []
        self.cp_coarse_dist_field = []
        
        self.cp_coarse_dist_deriv_list = [None for field in self.opt_field]
        self.cp_coarse_regu_deriv_list = [None for field in self.opt_field]
        self.cp_coarse_pin_deriv_list = [None for field in self.opt_field]
        self.cp_coarse_pin_dofs = [[] for field in self.opt_field]
        self.cp_coarse_pin_vals = [[] for field in self.opt_field]

    def set_order_elevation_by_field(self, p_list, knots_list):
        self.order_ele_degree = p_list
        self.order_ele_knots = knots_list
        # for field_ind, field in enumerate(self.opt_field):
        #     for i, s_ind in enumerate(self.shopt_surf_inds[field_ind]):
        #         local_ind = surf_inds.index(s_ind)
        #         self.order_ele_degree[field_ind] += [p_list[local_ind]]
        #         self.order_ele_knots[field_ind] += [knots_list[local_ind]]

        for field_ind, field in enumerate(self.opt_field):
            order_ele_operator_list_temp = []
            for i, surf_ind in enumerate(self.shopt_surf_inds[field_ind]):
                order_ele_operator_list_temp += [
                    surface_order_elevation_operator(
                    self.design_degree[field_ind][i], 
                    self.design_knots[field_ind][i], 
                    self.order_ele_degree[field_ind][i], 
                    self.order_ele_knots[field_ind][i], coo=False)]
            self.order_ele_operator_list[field_ind] = coo_matrix(
                                   block_diag(*order_ele_operator_list_temp))
        return self.order_ele_operator_list


    def set_init_knots(self, surf_inds, p_list, knots_list):
        self.design_degree = [[] for field in self.opt_field]
        self.design_knots = [[] for field in self.opt_field]
        self.cp_coarse_sizes = [[] for field in self.opt_field]
        self.cp_coarse_shapes = [[] for field in self.opt_field]
        for field_ind, field in enumerate(self.opt_field):
            for i, s_ind in enumerate(self.shopt_surf_inds[field_ind]):
                local_ind = surf_inds.index(s_ind)
                self.design_degree[field_ind] += [p_list[local_ind]]
                self.design_knots[field_ind] += [knots_list[local_ind]]
                l = len(knots_list[local_ind][0])-p_list[local_ind][0]-1
                m = len(knots_list[local_ind][1])-p_list[local_ind][1]-1
                self.cp_coarse_sizes[field_ind] += [l*m]
                self.cp_coarse_shapes[field_ind] += [[l,m]]

        # Constraint related properties
        self.align_dir_list = [None for field in self.opt_field]
        self.cp_coarse_free_dofs = [np.arange(np.sum(self.cp_coarse_sizes[field_ind])) 
                                for field_ind in range(len(self.opt_field))]
        self.cp_coarse_free_dofs_decate = [[] for field in self.opt_field]

        self.cp_coarse_align_deriv_sub_list = [[] for field in self.opt_field]
        for field_ind, field in enumerate(self.opt_field):
            ind_off = 0
            for i, s_ind in enumerate(self.shopt_surf_inds[field_ind]):
                self.cp_coarse_align_deriv_sub_list[field_ind] += \
                    [np.eye(self.cp_coarse_sizes[field_ind][i])]
                self.cp_coarse_free_dofs_decate[field_ind] += \
                    [np.arange(self.cp_coarse_sizes[field_ind][i])+ind_off]
                ind_off += self.cp_coarse_sizes[field_ind][i]

        # self.cp_design2analysis_deriv_list = [None for field in self.opt_field]
        self.cp_coarse_align_deriv_list = [coo_matrix(block_diag(*mat_list)) 
                                           for mat_list in 
                                           self.cp_coarse_align_deriv_sub_list]

        self.order_ele_operator_list = [None for field in self.opt_field]
        self.knot_refine_operator_list = [None for field in self.opt_field]

        self.cp_coarse_pin_field = []
        self.cp_coarse_regu_field = []
        self.cp_coarse_dist_field = []
        
        self.cp_coarse_dist_deriv_list = [None for field in self.opt_field]
        self.cp_coarse_regu_deriv_list = [None for field in self.opt_field]
        self.cp_coarse_pin_deriv_list = [None for field in self.opt_field]
        self.cp_coarse_pin_dofs = [[] for field in self.opt_field]
        self.cp_coarse_pin_vals = [[] for field in self.opt_field]

    def set_order_elevation(self, surf_inds, p_list, knots_list):
        self.order_ele_degree = [[] for field in self.opt_field]
        self.order_ele_knots = [[] for field in self.opt_field]
        for field_ind, field in enumerate(self.opt_field):
            for i, s_ind in enumerate(self.shopt_surf_inds[field_ind]):
                local_ind = surf_inds.index(s_ind)
                self.order_ele_degree[field_ind] += [p_list[local_ind]]
                self.order_ele_knots[field_ind] += [knots_list[local_ind]]

        for field_ind, field in enumerate(self.opt_field):
            order_ele_operator_list_temp = []
            for i, surf_ind in enumerate(self.shopt_surf_inds[field_ind]):
                order_ele_operator_list_temp += [
                    surface_order_elevation_operator(
                    self.design_degree[field_ind][i], 
                    self.design_knots[field_ind][i], 
                    self.order_ele_degree[field_ind][i], 
                    self.order_ele_knots[field_ind][i], coo=False)]
            self.order_ele_operator_list[field_ind] = coo_matrix(
                                   block_diag(*order_ele_operator_list_temp))
        return self.order_ele_operator_list

    def set_knot_refinement(self):
        self.ref_knots = [[] for field in self.opt_field]
        for field_ind, field in enumerate(self.opt_field):
            for i, surf_ind in enumerate(self.shopt_surf_inds[field_ind]):
                self.ref_knots[field_ind] += [[],]
                for side in range(2):
                    ref_knot_temp = []
                    ana_knot_temp = self.analysis_knots[field_ind][i][side]
                    for k in ana_knot_temp:
                        if k not in self.order_ele_knots[field_ind][i][side]:
                            ref_knot_temp += [k]
                    self.ref_knots[field_ind][i] += [np.array(ref_knot_temp)]

        for field_ind, field in enumerate(self.opt_field):
            knot_refine_operator_list_temp = []
            for i, surf_ind in enumerate(self.shopt_surf_inds[field_ind]):
                knot_refine_operator_list_temp += [
                surface_knot_refine_operator(self.order_ele_knots[field_ind][i], 
                    self.ref_knots[field_ind][i], coo=False)]
            self.knot_refine_operator_list[field_ind] = coo_matrix(
                                block_diag(*knot_refine_operator_list_temp))

        return self.knot_refine_operator_list

    # def set_order_ele(self, order_ele_degree, order_ele_knots):
    #     self.order_ele_degree = order_ele_degree
    #     self.order_ele_knots = order_ele_knots
    #     # self.order_ele_knots = [np.array([0.]*(self.order_ele_degree[0]+1)\
    #     #                                 +[1.]*(self.order_ele_degree[0]+1)),
    #     #                         np.array([0.]*(self.order_ele_degree[1]+1)\
    #     #                                 +[1.]*(self.order_ele_degree[1]+1))]

           
    #     self.order_ele_operator_list = [[] for field in self.opt_field]
    #     for field_ind, field in enumerate(self.opt_field):
    #         for i, surf_ind in enumerate(self.shopt_surf_inds[field_ind]):
    #             self.order_ele_operator_list[field_ind] += [
    #             surface_order_elevation_operator(
    #                 self.design_degree[i], 
    #                 self.design_knots[i], 
    #                 self.order_ele_degree, self.order_ele_knots, coo=False)]
    #     self.order_ele_operator = [coo_matrix(block_diag(*order_ele_list)) for 
    #                                order_ele_list in self.order_ele_operator_list]

    #     self.ref_knots = [[] for field in self.opt_field]
    #     for field_ind, field in enumerate(self.opt_field):
    #         for i, surf_ind in enumerate(self.shopt_surf_inds[field_ind]):
    #             self.ref_knots[field_ind] += [[],]
    #             for side in range(2):
    #                 self.ref_knots[field_ind][i] += \
    #                     [self.analysis_knots[field_ind][i][side]\
    #                     [self.order_ele_degree[side]+1:-(self.order_ele_degree[side]+1)]]

    #     self.knot_refine_operator_list = [[] for field in self.opt_field]
    #     for field_ind, field in enumerate(self.opt_field):
    #         for i, surf_ind in enumerate(self.shopt_surf_inds[field_ind]):
    #             self.knot_refine_operator_list[field_ind] += [
    #             surface_knot_refine_operator(self.order_ele_knots, 
    #                 self.ref_knots[field_ind][i], coo=False)]
    #     self.knot_refine_operator = [coo_matrix(block_diag(*knot_ref_list)) for 
    #                                  knot_ref_list in self.knot_refine_operator_list]

    #     for field_ind, field in enumerate(self.opt_field):
    #         self.cp_design2analysis_deriv_list[field_ind] = coo_matrix(
    #                                     self.knot_refine_operator[field_ind]
    #                                     *self.order_ele_operator[field_ind])

    #     # Constraint related properties
    #     self.align_dir_list = [None for field in self.opt_field]
    #     self.cp_coarse_free_dofs = [np.arange(np.sum(self.cp_coarse_sizes[field_ind])) 
    #                             for field_ind in range(len(self.opt_field))]
    #     self.cp_coarse_free_dofs_decate = [[] for field in self.opt_field]
    #     self.cp_coarse_align_deriv_list = [None for field in self.opt_field]
    #     self.cp_coarse_pin_deriv_list = [None for field in self.opt_field]
    #     self.cp_coarse_regu_deriv_list = [None for field in self.opt_field]

    #     self.cp_coarse_align_deriv_sub_list = [[] for field in self.opt_field]
    #     for field_ind, field in enumerate(self.opt_field):
    #         ind_off = 0
    #         for i, s_ind in enumerate(self.shopt_surf_inds[field_ind]):
    #             self.cp_coarse_align_deriv_sub_list[field_ind] += \
    #                 [np.eye(self.cp_coarse_sizes[field_ind][i])]
    #             self.cp_coarse_free_dofs_decate[field_ind] += \
    #                 [np.arange(self.cp_coarse_sizes[field_ind][i])+ind_off]
    #             ind_off += self.cp_coarse_sizes[field_ind][i]


    def get_init_cp_coarse(self):
        self.init_cp_coarse = [None for field in self.opt_field]
        self.init_cp_design = [None for field in self.opt_field]
        for field_ind, field in enumerate(self.opt_field):
            A_mat = self.knot_refine_operator_list[field_ind]\
                    *self.order_ele_operator_list[field_ind]
            b_vec = self.init_analysis_cp[field_ind]
            init_cp = solve_nonsquare(A_mat.todense(),
                                      np.array([b_vec]).T)
            self.init_cp_coarse[field_ind] = np.asarray(init_cp)[:,0]
            self.init_cp_design[field_ind] = np.asarray(init_cp)[:,0].copy()
        return self.init_cp_coarse

    def set_cp_align(self, field, align_dir_list):
        field_ind = self.opt_field.index(field)
        self.align_dir_list[field_ind] = align_dir_list
        ind_off = 0
        for i, align_dir in enumerate(align_dir_list):
            if align_dir is not None:
                cp_shape = self.cp_coarse_shapes[field_ind][i]
                deriv_mat, free_dofs = surface_cp_align_operator(
                                       cp_shape, align_dir, coo=False)
                self.cp_coarse_align_deriv_sub_list[field_ind][i] = deriv_mat
            else:
                free_dofs = list(range(self.cp_coarse_sizes[field_ind][i]))
            self.cp_coarse_free_dofs_decate[field_ind][i] = np.array(free_dofs)+ind_off
            ind_off += self.cp_coarse_sizes[field_ind][i]

        self.cp_coarse_free_dofs[field_ind] = np.concatenate(
            self.cp_coarse_free_dofs_decate[field_ind])
        self.init_cp_design[field_ind] = self.init_cp_design[field_ind]\
                                        [self.cp_coarse_free_dofs[field_ind]]

        self.cp_coarse_align_deriv_list[field_ind] = coo_matrix(
            block_diag(*self.cp_coarse_align_deriv_sub_list[field_ind]))

        return self.cp_coarse_align_deriv_list[field_ind]

        # self.cp_design2analysis_deriv_list[field_ind] = \
        #     coo_matrix(self.cp_design2analysis_deriv_list[field_ind]*
        #     self.cp_coarse_align_deriv_list[field_ind])

        # self.init_cp_design[field_ind] = self.init_cp_design[field_ind]\
        #                                 [self.cp_coarse_free_dofs[field_ind]]
        # return self.cp_design2analysis_deriv_list[field_ind]


    def set_cp_pin(self, field, pin_dir0_list, pin_side0_list,
                   pin_dir1_list=None, pin_side1_list=None,
                   pin_dofs=None, pin_vals=None):
        if field not in self.cp_coarse_pin_field:
            self.cp_coarse_pin_field += [field]
        field_ind = self.opt_field.index(field)
        cp_coarse_pin_dofs_temp = []
        if pin_dofs is not None:
            cp_coarse_pin_dofs_temp += pin_dofs
        else:
            cp_shapes = self.cp_coarse_shapes[field_ind]
            cp_sizes = self.cp_coarse_sizes[field_ind]
            ind_off = 0
            for i, pin_dir0 in enumerate(pin_dir0_list):
                if pin_dir0 is not None:
                    l, m = cp_shapes[i]
                    pin_dir0_dof = []
                    if pin_dir0 == 0:
                        # for side0 in pin_side0_list[i]:
                        if 0 in pin_side0_list[i]:
                            pin_dir0_dof += list(range(0,l*m,l))
                        if 1 in pin_side0_list[i]:
                            pin_dir0_dof += list(range(l-1,l*m,l))
                    elif pin_dir0 == 1:
                        # for side0 in pin_side0_list[i]:
                        if 0 in pin_side0_list[i]:
                            pin_dir0_dof += list(range(0,l))
                        if 1 in pin_side0_list[i]:
                            pin_dir0_dof += list(range(l*(m-1), l*m))
                    cp_coarse_pin_dofs_temp += [dof+ind_off for dof in pin_dir0_dof]
                ind_off += cp_sizes[i]

            if pin_dir1_list is not None:
                ind_off = 0
                for i, pin_dir1 in enumerate(pin_dir1_list):
                    if pin_dir1 is not None:
                        l, m = cp_shapes[i]
                        pin_dir1_dof = []
                        if pin_dir1 == 0:
                            for side0 in pin_side0_list[i]:
                                if 0 in side0:
                                    pin_dir1_dof += list(range(0,l*m,l))
                                if 1 in side0:
                                    pin_dir1_dof += list(range(l-1,l*m,l))
                        elif pin_dir1 == 1:
                            for side0 in pin_side0_list[i]:
                                if 0 in side0:
                                    pin_dir1_dof += list(range(0,l))
                                if 1 in side0:
                                    pin_dir1_dof += list(range(l*(m-1), l*m))
                        cp_coarse_pin_dofs_temp += [dof+ind_off for dof in pin_dir1_dof]
                    ind_off += cp_sizes[i]
            cp_coarse_pin_dofs_temp = np.unique(cp_coarse_pin_dofs_temp)

        # self.cp_coarse_pin_dofs_temp = cp_coarse_pin_dofs_temp
        self.cp_coarse_pin_dofs[field_ind] += [dof for dof in cp_coarse_pin_dofs_temp 
                                               if dof in self.cp_coarse_free_dofs[field_ind]]
        self.cp_coarse_pin_dofs[field_ind] = list(np.sort(self.cp_coarse_pin_dofs[field_ind]))

        if pin_vals is not None:
            self.cp_coarse_pin_vals[field_ind] = pin_vals
        else:
            self.cp_coarse_pin_vals[field_ind] = list(self.init_cp_coarse[field_ind]\
                                                 [self.cp_coarse_pin_dofs[field_ind]])

        deriv_mat = np.zeros((len(self.cp_coarse_pin_vals[field_ind]),
                              len(self.cp_coarse_free_dofs[field_ind])))
        for i, pin_dof in enumerate(self.cp_coarse_pin_dofs[field_ind]):
            local_pin_dof = np.where(self.cp_coarse_free_dofs[field_ind]==pin_dof)[0][0]
            deriv_mat[i, local_pin_dof] = 1.

        self.cp_coarse_pin_deriv_list[field_ind] = coo_matrix(deriv_mat)
        return self.cp_coarse_pin_deriv_list[field_ind]


    def set_cp_regu(self, field, regu_dir_list, rev_dir=False):
        if field not in self.cp_coarse_regu_field:
            self.cp_coarse_regu_field += [field]
        field_ind = self.opt_field.index(field)
        self.deriv_mat_list = []
        for i, regu_dir in enumerate(regu_dir_list):
            if regu_dir is not None:
                deriv_mat_temp_list = [None for s_ind in self.shopt_surf_inds[field_ind]]
                cp_shape = self.cp_coarse_shapes[field_ind][i]
                if self.align_dir_list[field_ind] is not None:
                    if self.align_dir_list[field_ind][i] == 0:
                        cp_shape[0] = 1
                    elif self.align_dir_list[field_ind][i] == 1:
                        cp_shape[1] = 1
                deriv_mat = surface_cp_regu_operator(cp_shape, regu_dir, 
                                                     rev_dir=rev_dir, coo=False)
                deriv_mat_temp_list[i] = coo_matrix(deriv_mat)
                for j, s_ind in enumerate(self.shopt_surf_inds[field_ind]):
                    if j != i:
                        temp_mat = np.zeros((deriv_mat.shape[0], 
                                   len(self.cp_coarse_free_dofs_decate[field_ind][j])))
                        deriv_mat_temp_list[j] = coo_matrix(temp_mat)
                self.deriv_mat_list += [deriv_mat_temp_list]
        if len(self.deriv_mat_list) == 1 and len(self.deriv_mat_list[0]) == 1:
            self.cp_coarse_regu_deriv_list[field_ind] = coo_matrix(self.deriv_mat_list[0][0])
        else:
            self.cp_coarse_regu_deriv_list[field_ind] = coo_matrix(bmat(self.deriv_mat_list))
        return self.cp_coarse_regu_deriv_list[field_ind]

    def set_cp_dist(self, field, surf_inds, rev_dir=False):
        if field not in self.cp_coarse_dist_field:
            self.cp_coarse_dist_field += [field]
        if rev_dir:
            coeff = -1.
        else:
            coeff = 1.
        field_ind = self.opt_field.index(field)
        design_dofs_decate = self.cp_coarse_free_dofs_decate[field_ind]
        design_dofs = self.cp_coarse_free_dofs[field_ind]
        deriv_mat_list = []
        for i in range(len(surf_inds)-1):
            s_ind0_local = self.shopt_surf_inds[field_ind].index(surf_inds[i])
            s_ind1_local = self.shopt_surf_inds[field_ind].index(surf_inds[i+1])
            assert len(design_dofs_decate[s_ind0_local]) == \
                   len(design_dofs_decate[s_ind1_local])
            deriv_mat_temp = np.zeros((len(design_dofs_decate[s_ind0_local]), 
                                       len(design_dofs)))
            for row_ind in range(len(design_dofs_decate[s_ind0_local])):
                col_ind0 = np.where(design_dofs==design_dofs_decate[s_ind0_local][row_ind])[0][0]
                col_ind1 = np.where(design_dofs==design_dofs_decate[s_ind1_local][row_ind])[0][0]
                deriv_mat_temp[row_ind, col_ind0] = coeff*(-1.)
                deriv_mat_temp[row_ind, col_ind1] = coeff*(1.)
            deriv_mat_list += [coo_matrix(deriv_mat_temp)]

        self.cp_coarse_dist_deriv_list[field_ind] = vstack(deriv_mat_list)
        return self.cp_coarse_dist_deriv_list[field_ind]



# def reorder_surface_cp(num_cp, num_rows, num_cols, 
#                        row_block_inds, col_block_inds, coo=True):
#     """
#     Return linear operator to reorder spline surface's control points
#     num_rows corresponds to the number of control points in the v direction,
#     num_cols corresponds to the number of control points in the u direction.
#     """
#     row_block_inds = np.array(row_block_inds)
#     col_block_inds = np.array(col_block_inds)
#     A_re = np.zeros((num_cp, num_cp))

#     # # Get sizes for rows and columns
#     row_sizes = []
#     col_sizes = []
#     for i in range(len(row_block_inds)):
#         if i == len(row_block_inds)-1:
#             row_sizes += [num_rows-row_block_inds[-1]]
#         else:
#             row_sizes += [row_block_inds[i+1]-row_block_inds[i]]

#     for i in range(len(col_block_inds)):
#         if i == len(col_block_inds)-1:
#             col_sizes += [num_cols-col_block_inds[-1]]
#         else:
#             col_sizes += [col_block_inds[i+1]-col_block_inds[i]]

#     # # Get block sizes
#     block_sizes = np.zeros(len(row_sizes)*len(col_sizes))
#     for i in range(len(row_sizes)):
#         for j in range(len(col_sizes)):
#             block_sizes[i*len(col_sizes)+j] = row_sizes[i]*col_sizes[j]

#     # # Fill in entries for the reordering operator
#     for row_ind in range(num_rows):
#         for col_ind in range(num_cols):
#             row_block_ind = len(np.where(row_block_inds<=row_ind)[0])-1
#             col_block_ind = len(np.where(col_block_inds<=col_ind)[0])-1
#             local_row_ind = row_ind - row_block_inds[row_block_ind]
#             local_col_ind = col_ind - col_block_inds[col_block_ind]
#             ind_off = np.sum(block_sizes[0:row_block_ind
#                                          *len(col_sizes)+col_block_ind])
#             operator_row_ind = int(ind_off+local_row_ind
#                                    *col_sizes[col_block_ind]+local_col_ind)
#             operator_col_ind = int(row_ind*num_cols+col_ind)
#             A_re[operator_row_ind, operator_col_ind] = 1
#     if coo:
#         return coo_matrix(A_re)
#     else:
#         return A_re