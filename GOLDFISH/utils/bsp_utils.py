import numpy as np
from scipy.sparse import coo_matrix, hstack, vstack
from scipy.linalg import block_diag

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

def refine_knot(knots, cp, ref_knots):
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