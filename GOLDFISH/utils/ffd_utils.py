from PENGoLINS.occ_preprocessing import *
from PENGoLINS.nonmatching_shell import *
from IGAOPT.utils.opt_utils import *
from IGAOPT.utils.bsp_utils import *
from igakit.cad import *

def scale_knots(knots, CP):
    """
    Scale input knots vector so it has same limits 
    with input control points

    Parameters
    ----------
    knots : list of ndarray
    CP : ndarray

    Returns
    -------
    new_knots : ndarray
    """
    num_field = len(knots)
    CP_flat = CP.reshape(-1,num_field)
    new_knots = [None]*num_field
    for field in range(num_field):
        knots_temp = np.array(knots[field])
        CP_min = np.min(CP_flat[:,field])
        CP_range = np.max(CP_flat[:,field]) - CP_min
        new_knots_temp = knots_temp*CP_range + CP_min
        new_knots[field] = new_knots_temp
    return new_knots

def CP_FFD_matrix(CP_S, p_V, knots_V, coo=True):
    """
    Return the linear operator for FFD, where ``FFD_mat*CP_V=CP_S``,
    assume ``knots_V`` has the same limits with ``CP_V``, so the 
    geometric mappling of the FFD spline block is identity.

    Parameters
    ----------
    CP_S : ndarray, contorl points of surface
    p_V : int, spline degree of FFD block
    knots_V : ndarray, knots vector of FFD block
    coo : bool, optional, default is True

    Returns
    -------
    FFD_mat : ndarray or COO matrix
    """
    # CP_S is in the tIGAr ordering convention
    bsp_V = BSpline(p_V, knots_V)
    n_CP_S = CP_S.shape[0]
    n_cp_V = bsp_V.getNcp()
    FFD_mat = np.zeros((n_CP_S, n_cp_V))
    for i in range(CP_S.shape[0]):
        nodes_vals = bsp_V.getNodesAndEvals(CP_S[i])
        for j in range(len(nodes_vals)):
            node, val = nodes_vals[j]
            FFD_mat[i, node] = val
    if coo:
        FFD_mat = coo_matrix(FFD_mat)
    return FFD_mat

def create_3D_block(num_els, p, CP_lims):
    """
    Create FFD block in igakit format.

    Parameters
    ----------
    num_els : list of ints, number of elements for FFD block 
              in 3 directions
    p : int, degree of FFD block
    CP_lims : ndarray or list, limits of FFD block control points
              in 3 directions
    """
    ref_knots0 = np.linspace(0,1,num_els[0]+1)[1:-1]
    ref_knots1 = np.linspace(0,1,num_els[1]+1)[1:-1]
    ref_knots2 = np.linspace(0,1,num_els[2]+1)[1:-1]

    pts0 = np.array([[CP_lims[0][0], CP_lims[1][0], CP_lims[2][0]],
                     [CP_lims[0][1], CP_lims[1][0], CP_lims[2][0]]])
    pts1 = np.array([[CP_lims[0][0], CP_lims[1][1], CP_lims[2][0]],
                     [CP_lims[0][1], CP_lims[1][1], CP_lims[2][0]]])
    v_disp = CP_lims[2][1] - CP_lims[2][0]

    L0 = line(pts0[0], pts0[1])
    L1 = line(pts1[0], pts1[1])
    S = ruled(L0, L1)
    V = extrude(S, v_disp, 2)
    V_deg = V.degree
    V.elevate(0, p - V_deg[0])
    V.elevate(1, p - V_deg[1])
    V.elevate(2, p - V_deg[2])
    V.refine(0, ref_knots0)
    V.refine(1, ref_knots1)
    V.refine(2, ref_knots2)
    V_knots = V.knots
    V_control = V.control
    new_knots = []
    for i in range(3):
        new_knots += [V_knots[i]*(CP_lims[i][1]-CP_lims[i][0]) 
                  + CP_lims[i][0]]
    V_new = NURBS(new_knots, V_control)
    return V_new