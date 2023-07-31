import math
import numpy as np
import os
import glob
import natsort
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
        surf_node = CP_S[i][0:3]
        # if surf_node.shape[0] > bsp_V.nvar:
        #     surf_node = surf_node[0:bsp_V.nvar]/surf_node[bsp_V.nvar]
        nodes_vals = bsp_V.getNodesAndEvals(surf_node)
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
    cp_ranges = []
    for i in range(3):
        cp_ranges += [CP_lims[i][1]-CP_lims[i][0]]
        
    for i in range(3):
        if abs(cp_ranges[i]) < 1e-6:
            cp_ranges[i] = np.sort(cp_ranges)[1]*0.1
            CP_lims[i][1] = CP_lims[i][1] + 0.5*cp_ranges[i]
            CP_lims[i][0] = CP_lims[i][0] - 0.5*cp_ranges[i]

    ref_knots0 = np.linspace(0,1,num_els[0]+1)[1:-1]
    ref_knots1 = np.linspace(0,1,num_els[1]+1)[1:-1]
    ref_knots2 = np.linspace(0,1,num_els[2]+1)[1:-1]

    pts0 = np.array([[CP_lims[0][0], CP_lims[1][0], CP_lims[2][0]],
                     [CP_lims[0][1], CP_lims[1][0], CP_lims[2][0]]])
    pts1 = np.array([[CP_lims[0][0], CP_lims[1][1], CP_lims[2][0]],
                     [CP_lims[0][1], CP_lims[1][1], CP_lims[2][0]]])
    # v_disp = CP_lims[2][1] - CP_lims[2][0]

    L0 = line(pts0[0], pts0[1])
    L1 = line(pts1[0], pts1[1])
    S = ruled(L0, L1)
    V = extrude(S, cp_ranges[2], 2)
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
        new_knots += [V_knots[i]*cp_ranges[i] + CP_lims[i][0]]
    V_new = NURBS(new_knots, V_control)
    return V_new

def rationalized_control(nurbs):
    dim = nurbs.dim
    control_hom = nurbs.control
    chom_shape = list(control_hom.shape)
    c_shape = chom_shape
    c_shape[-1] = int(chom_shape[-1]-1)
    control = np.zeros(c_shape)
    if dim == 1:
        for dir0_ind in range(c_shape[0]):
            control[dir0_ind, :] = \
                control_hom[dir0_ind, 0:3]\
                /control_hom[dir0_ind, -1]
    elif dim == 2:
        for dir0_ind in range(c_shape[0]):
            for dir1_ind in range(c_shape[1]):
                control[dir0_ind, dir1_ind, :] = \
                    control_hom[dir0_ind, dir1_ind, 0:3]\
                    /control_hom[dir0_ind, dir1_ind, -1]
    else:
        for dir0_ind in range(c_shape[0]):
            for dir1_ind in range(c_shape[1]):
                for dir2_ind in range(c_shape[2]):
                    control[dir0_ind, dir1_ind, dir2_ind, :] = \
                        control_hom[dir0_ind, dir1_ind, dir2_ind, 0:3]\
                        /control_hom[dir0_ind, dir1_ind, dir2_ind, -1]
    return control


def refine_knot(knot, ref_level=1):
    knot_ref = knot.copy()
    for ref_ind in range(ref_level):
        new_knots = []
        for i in range(len(knot_ref)-1):
            new_knots += [(knot_ref[i]+knot_ref[i+1])*0.5]
        knot_ref = np.sort(np.concatenate([knot_ref, new_knots]))
    return knot_ref


class VTKWriter(object):

    """
    VTK_ Writer

    .. _VTK: http://www.vtk.org/

    """

    title = 'VTK Data'

    def __init__(self):
        pass

    def write_cp(self, filename, nurbs):

        dim  = nurbs.dim
        C = rationalized_control(nurbs)
        F = C[...,0:0]

        dimensions = C.shape[:-1] + (1,)*(3-dim)
        points = np.rollaxis(C, -1).ravel('f')
        points.shape = (-1, 3)
        fields = np.rollaxis(F, -1).ravel('f')
        fields.shape = (len(points), -1)

        fh = open(filename, 'wb')
        fh_write = lambda s: fh.write(s.encode('ascii'))

        header = '# vtk DataFile Version %d.%d'
        version = (2, 0)
        fh_write(header % version)
        fh_write('\n')
        title = self.title
        fh_write(title[:255])
        fh_write('\n')

        format = 'BINARY'
        fh_write(format)
        fh_write('\n')

        dataset_type = 'STRUCTURED_GRID'
        fh_write('DATASET %s' % dataset_type);
        fh_write('\n')
        fh_write('DIMENSIONS %d %d %d' % dimensions)
        fh_write('\n')
        fh_write('POINTS %d %s' % (len(points), 'double'))
        fh_write('\n')
        points.astype('>d').tofile(fh)
        fh_write('\n')
        fh.flush()
        fh.close()


    def write(self, filename, nurbs,
              control=True, fields=None,
              scalars=(), vectors=(),
              sampler=None, ref_level=0):
        """
        Parameters
        ----------
        filename : string
        nurbs : NURBS
        control : bool, optional
        fields : array, optional
        scalars : dict or sequence of 2-tuple, optional
        vectors : dict or sequence or 2-tuple, optional
        sampler : callable, optional

        """
        if sampler is None:
            sampler = lambda u: u
        dim  = nurbs.dim
        uvw = [sampler(u) for u in nurbs.breaks()]
        uvw = [refine_knot(u, ref_level) for u in uvw]
        flag = bool(scalars or vectors)
        if not flag: fields = flag
        elif fields is None: fields = flag
        out = nurbs(*uvw, **dict(fields=fields))
        if flag: C, F = out
        else:    C, F = out, out[..., 0:0]

        self.F = F

        dimensions = C.shape[:-1] + (1,)*(3-dim)
        coordinates = uvw + [np.zeros(1)]*(3-dim)
        points = np.rollaxis(C, -1).ravel('f')
        points.shape = (-1, 3)
        fields = np.rollaxis(F, -1).ravel('f')
        fields.shape = (len(points), -1)

        if isinstance(scalars, dict):
            keys = sorted(scalars.keys())
            scalars = [(k, scalars[k]) for k in keys]
        else:
            scalars = list(scalars)
        for i, (name, index) in enumerate(scalars):
            array = np.zeros((len(points), 1), dtype='d')
            array[:,0] = fields[:,index]
            scalars[i] = (name, array)

        if isinstance(vectors, dict):
            keys = sorted(vectors.keys())
            vectors = [(k, vectors[k]) for k in keys]
        else:
            vectors = list(vectors)
        for i, (name, index) in enumerate(vectors):
            array = np.zeros((len(points), 3), dtype='d')
            array[:,:len(index)] = fields[:,index]
            vectors[i] = (name, array)

        fh = open(filename, 'wb')
        fh_write = lambda s: fh.write(s.encode('ascii'))

        header = '# vtk DataFile Version %d.%d'
        version = (2, 0)
        fh_write(header % version)
        fh_write('\n')
        title = self.title
        fh_write(title[:255])
        fh_write('\n')

        format = 'BINARY'
        fh_write(format)
        fh_write('\n')

        if control:
            dataset_type = 'STRUCTURED_GRID'
            fh_write('DATASET %s' % dataset_type);
            fh_write('\n')
            fh_write('DIMENSIONS %d %d %d' % dimensions)
            fh_write('\n')
            fh_write('POINTS %d %s' % (len(points), 'double'))
            fh_write('\n')
            points.astype('>d').tofile(fh)
            fh_write('\n')
        else:
            dataset_type = 'RECTILINEAR_GRID'
            fh_write('DATASET %s' % dataset_type);
            fh_write('\n')
            fh_write('DIMENSIONS %d %d %d' % dimensions)
            fh_write('\n')
            for X, array in zip("XYZ", coordinates):
                label = X+'_COORDINATES'
                fh_write('%s %s %s' % (label, len(array), 'double'))
                fh_write('\n')
                array.astype('>d').tofile(fh)
                fh_write('\n')

        if (not scalars and
            not vectors):
            fh.flush()
            fh.close()
            return

        data_type = 'POINT_DATA'
        fh_write('%s %d' % (data_type, len(points)))
        fh_write('\n')

        for i, (name, array) in enumerate(scalars):
            attr_type = 'SCALARS'
            attr_name = name or (attr_type.lower() + str(i))
            attr_name = attr_name.replace(' ', '_')
            fh_write('%s %s %s' %(attr_type, attr_name, 'double'))
            fh_write('\n')
            lookup_table = 'default'
            lookup_table = lookup_table.replace(' ', '_')
            fh_write('LOOKUP_TABLE %s' % lookup_table)
            fh_write('\n')
            array.astype('>d').tofile(fh)
            fh_write('\n')

        for i, (name, array) in enumerate(vectors):
            attr_type = 'VECTORS'
            attr_name = name or (attr_type.lower() + str(i))
            attr_name = attr_name.replace(' ', '_')
            fh_write('%s %s %s' %(attr_type, attr_name, 'double'))
            fh_write('\n')
            array.astype('>d').tofile(fh)
            fh_write('\n')

        fh.flush()
        fh.close()

def update_FFD_block(FFD_block, new_cps, opt_field):
    init_cpffd_flat = FFD_block.control[:,:,:,0:3].\
                      transpose(2,1,0,3).reshape(-1,3)
    new_cpffd_flat = init_cpffd_flat.copy()
    for i, field in enumerate(opt_field):
        new_cpffd_flat[:, field] = new_cps[i]
    new_cpffd = new_cpffd_flat.reshape(FFD_block.control[...,0:3].\
                transpose(2,1,0,3).shape)
    new_cpffd = new_cpffd.transpose(2,1,0,3)
    new_FFD_block = NURBS(FFD_block.knots, new_cpffd)
    return new_FFD_block

def create_pvd(filename, vtu_pre, directory='./'):
    vtu_names = natsort.natsorted(glob.glob(directory+vtu_pre+'*.vtu'))
    pvd_file = open(directory+filename, 'w')
    pvd_file.write('<?xml version="1.0"?>\n')
    pvd_file.write('<VTKFile type="Collection" version="0.1">\n')
    pvd_file.write('  <Collection>\n')
    for i in range(len(vtu_names)):
        line = '    <DataSet timestep=\"'+str(i)+'\" part=\"0\" file=\"' \
             + vtu_names[i].split('/')[-1] +'\" />\n'
        pvd_file.write(line)
    pvd_file.write('  </Collection>/n\n')
    pvd_file.write('</VTKFile>\n')
    pvd_file.close()

""" Example to create pvd file
FFD_block_pre = 'FFD_block_'
FFD_CP_pre = 'FFD_CP_'
create_pvd('FFD_block.pvd', vtu_pre=FFD_block_pre, directory='./geometry/')
create_pvd('FFD_CP.pvd', vtu_pre=FFD_CP_pre, directory='./geometry/')
"""