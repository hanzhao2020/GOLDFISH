import numpy as np
# from GOLDFISH.nonmatching_opt import *
from scipy.sparse import coo_matrix, csr_matrix
from scipy.linalg import block_diag
# from scipy.sparse.linalg import spsolve
import scipy.sparse.linalg as splin

from scipy.optimize import fsolve, newton_krylov
from scipy.sparse import coo_matrix, bmat

from igakit.cad import NURBS
from igakit import igalib as _igalib
_bsp = _igalib.bsp

from tIGAr import BSplines
# from PENGoLINS.occ_preprocessing import *
# from PENGoLINS.igakit_utils import *

def np_normalize(vec):
    return vec/np.linalg.norm(vec)

def ij2dof(l,i,j):
    return i+j*l

class BSplineCurveBasis(object):
    def __init__(self, p, knots):
        self.p = p
        self.knots = knots
        self.ncp = len(self.knots)-p-1
    def find_span(self,xi):
        return _bsp.FindSpan(self.p, self.knots, xi)
    def get_nodes(self, xi):
        span = self.find_span(xi)
        nodes = list(range(span-self.p, span+1))
        return nodes
    def eval_basis(self, xi):
        basis_val = _bsp.EvalBasisFuns(self.p, self.knots, xi)
        return basis_val
    def eval_basis_deriv(self, xi, order=1):
        basis_val = _bsp.EvalBasisFunsDers(self.p, self.knots, xi)
        return [basis_val[i] for i in range(order+1)]


class BSplineSurfaceBasis(object):
    def __init__(self, p, knots):
        assert len(p) == len(knots)
        self.p = p
        self.knots = knots
        self.ncp = [len(self.knots[0])-self.p[0]-1, 
                    len(self.knots[1])-self.p[1]-1]
        self.bsc0 = BSplineCurveBasis(p[0], knots[0])
        self.bsc1 = BSplineCurveBasis(p[1], knots[1])
    def get_nodes(self,xi):
        nodes0 = self.bsc0.get_nodes(xi[0])
        nodes1 = self.bsc1.get_nodes(xi[1])
        nodes = []
        for node0 in nodes0:
            for node1 in nodes1:
                nodes += [ij2dof(self.ncp[0], node0, node1)]
        return nodes
    def eval_basis(self, xi):
        basis_val0 = self.bsc0.eval_basis(xi[0])
        basis_val1 = self.bsc1.eval_basis(xi[1])
        basis_val = np.outer(basis_val0, basis_val1).reshape(-1)
        return basis_val
    def eval_basis_deriv(self, xi, order=1):
        basis_val0 = _bsp.EvalBasisFunsDers(self.bsc0.p, self.bsc0.knots, xi[0])
        basis_val1 = _bsp.EvalBasisFunsDers(self.bsc1.p, self.bsc1.knots, xi[1])
        basis_val = np.outer(basis_val0[0], basis_val1[0]).reshape(-1)
        if order == 1:
            basisdxi0 = np.outer(basis_val0[1], basis_val1[0]).reshape(-1)
            basisdxi1 = np.outer(basis_val0[0], basis_val1[1]).reshape(-1)
            return [basis_val, basisdxi0, basisdxi1]
        elif order == 2:
            basisdxi0 = np.outer(basis_val0[1], basis_val1[0]).reshape(-1)
            basisdxi1 = np.outer(basis_val0[0], basis_val1[1]).reshape(-1)
            basisddxi0 = np.outer(basis_val0[2], basis_val1[0]).reshape(-1)
            basisddxi1 = np.outer(basis_val0[0], basis_val1[2]).reshape(-1)
            basisdxi0dxi1 = np.outer(basis_val0[1], basis_val1[1]).reshape(-1)
            return [basis_val, basisdxi0, basisdxi1, 
                    basisddxi0, basisddxi1, basisdxi0dxi1]



class CPIGA2XiCoarse(object):
    def __init__(self, preprocessor, cpsurfd2a,
                 opt_surf_inds, opt_field, num_edge_pts=None):
    # def __init__(self, preprocessor, diff_int_inds=None, opt_field=[0,1,2]):
        """
        preprocessor : PENGoLINS.preprocessor instance
        diff_int_inds : list of ints, the indices of intersections to differentiate
            if None, differentiate all intersections (not recommended)
        """
        # Define basic paramters
        self.num_field = 3
        self.para_dim = 2
        self.num_end_pts = 2
        self.num_sides = 2

        self.preprocessor = preprocessor
        self.num_intersections = self.preprocessor.num_intersections_all
        self.mortar_nels = self.preprocessor.mortar_nels
        self.mortar_pts = [nel+1 for nel in self.mortar_nels]
        self.mapping_list = self.preprocessor.mapping_list

        self.opt_field = opt_field
        self.opt_surf_inds = opt_surf_inds
        self.cpsurfd2a = cpsurfd2a
        self.coarse_knots = self.cpsurfd2a.coarse_knots_all
        self.coarse_degree = self.cpsurfd2a.coarse_degree_all
        self.coarse_cp_all = self.cpsurfd2a.coarse_cp_all

        self.coarse_surf_basis = [BSplineSurfaceBasis(self.coarse_degree[i],
                                                      self.coarse_knots[i]) 
                                  for i in range(self.preprocessor.num_surfs)]

        self.diff_int_inds = self.preprocessor.diff_int_inds
        self.diff_int_types = [self.preprocessor.intersections_type[ind]
                                for ind in self.diff_int_inds]
        int_surf_inds_temp = []
        for i, ind in enumerate(self.diff_int_inds):
            s_ind0, s_ind1 = self.mapping_list[ind]
            if s_ind0 not in int_surf_inds_temp:
                int_surf_inds_temp += [s_ind0]
            if s_ind1 not in int_surf_inds_temp:
                int_surf_inds_temp += [s_ind1]
        self.int_surf_inds = list(np.sort(int_surf_inds_temp))
        self.num_int_surfs = len(self.int_surf_inds)

        #### Control points related properties
        self.int_surf_basis = []
        for i, ind in enumerate(self.int_surf_inds):
            self.int_surf_basis += [self.coarse_surf_basis[ind]]

        # BSpline related properties
        self.cp_shapes = [cp.shape[0:self.para_dim] for cp in self.coarse_cp_all]
        self.cp_sizes = [int(cp_shape[0]*cp_shape[1]) 
                         for cp_shape in self.cp_shapes]
        # The size of all cps (of surfaces of interest)
        self.cp_size_global = int(np.sum(self.cp_sizes))
        # Given a full global flat cps (of surfaces of interest)
        # the start and end indices for each surfaces
        self.cp_flat_inds = []
        for i in range(self.num_int_surfs):
            self.cp_flat_inds += [int(np.sum(self.cp_sizes[0:i]))]
        self.cp_flat_inds += [self.cp_size_global]

        # Initial unflattened control points for all surfaces of interest
        self.cps = [self.coarse_cp_all[ind].copy()[:,:,0:self.num_field]
                    for ind in self.int_surf_inds]
        # Initial flattened control points for all surfaces of interest
        self.cps_flat = [self.coarse_cp_all[ind].copy()[:,:,0:self.num_field]
                         .transpose(1,0,2).reshape(-1,self.num_field)
                         for ind in self.int_surf_inds]
        self.cp_flat_global = np.concatenate(self.cps_flat, axis=0)

        # # create tIGAr BSpline objects to have access to basis functions
        # self.bsplines = [BSplines.BSpline(surf_data.degree, surf_data.knots)
        #                  for surf_data in self.occ_int_surfs_data]

        #### Parametric coordinates related properties
        self.diff_int_num_pts = []
        for int_ind, int_ind_global in enumerate(self.diff_int_inds):
            self.diff_int_num_pts += [self.preprocessor.mortar_nels[int_ind_global]+1]
        # Total number of parametric coordinates for all intersections
        self.xi_size_global = int(np.sum(self.diff_int_num_pts)
                                 *self.num_sides*self.para_dim)
        self.xis = [] # Intersection's parametric coordinates in shape of (x,2)
        self.xis_flat = [] # Intersection's parametric coordinates in shape of (x,1)
        self.xi_sizes = [] # Coordinates sizes for all intersections
        for int_ind, int_ind_global in enumerate(self.diff_int_inds):
            self.xis += [[],]
            for side in range(self.num_sides):
                # print("*"*50, int_ind_global, "--", side)
                self.xis[int_ind] += [self.preprocessor.intersections_para_coords[int_ind_global][side]]
            self.xis_flat += [np.concatenate([self.xis[int_ind][0].reshape(-1,1),
                              self.xis[int_ind][1].reshape(-1,1)], axis=0)[:,0]]
            self.xi_sizes += [self.xis_flat[int_ind].size]
        # All flattened intersection's parametric coordinates
        self.xi_flat_global = np.concatenate(self.xis_flat)
        # # assert self.xi_size_global == self.xi_flat_global.size
        # Start and end indices for flattened intersection's parametric coordinates
        self.xi_flat_inds = []
        for i in range(len(self.diff_int_inds)):
            self.xi_flat_inds += [int(np.sum(self.xi_sizes[0:i]))]
        self.xi_flat_inds += [self.xi_size_global]

        # Find end points and values
        self.end_xi_ind = np.zeros((len(self.diff_int_inds), self.num_end_pts), dtype='int32')
        self.end_xi_val = np.zeros((len(self.diff_int_inds), self.num_end_pts))

        # # Find end points and values
        # self.end_xi_ind = np.zeros((len(self.diff_int_inds), self.num_end_pts), dtype='int32')
        # self.end_xi_val = np.zeros((len(self.diff_int_inds), self.num_end_pts))

        for int_ind, int_ind_global in enumerate(self.diff_int_inds):
            num_pts = self.diff_int_num_pts[int_ind]
            init_para_coord = self.preprocessor.intersections_para_coords[
                          self.diff_int_inds[int_ind]]
            # int_surf_ind0, int_surf_ind1 = self.local_int_surf_inds(int_ind)
            # Check which end points are on edges
            int_type = self.diff_int_types[int_ind]
            side_list = [0,1]
            end_pts_list = [0,1]
            if int_type[0] == 'surf-edge':
                para_dir_list0 = [0,1]
                edge_para_dir_ind = int_type[1].index('.')-1
                edge_para_dir = int(int_type[1][edge_para_dir_ind])
                if edge_para_dir == 0:
                    para_dir_list1 = [1]
                elif edge_para_dir == 1:
                    para_dir_list1 = [0]
            elif int_type[0] == 'edge-surf':
                para_dir_list1 = [0,1]
                edge_para_dir_ind = int_type[1].index('.')-1
                edge_para_dir = int(int_type[1][edge_para_dir_ind])
                if edge_para_dir == 0:
                    para_dir_list0 = [1]
                elif edge_para_dir == 1:
                    para_dir_list0 = [0]
            else:
                para_dir_list0 = [0,1]
                para_dir_list1 = [0,1]
            para_dir_list = [para_dir_list0, para_dir_list1]

            
            for end_ind in [0,1]:
                end0_xi_err_list = []
                end1_xi_err_list = []
                end_xi_ind_list = []
                for side in side_list:
                    for para_dir in para_dir_list[side]:
                        xi_coord = init_para_coord[side][end_ind*(-1)][para_dir]
                        end_xi_ind_temp = side*2*num_pts + \
                                          end_ind*2*(num_pts-1) + para_dir
                        end0_xi_err_list += [xi_coord]
                        end1_xi_err_list += [1-xi_coord]
                        end_xi_ind_list += [end_xi_ind_temp]
                end0_xi_err_min = np.min(end0_xi_err_list)
                end1_xi_err_min = np.min(end1_xi_err_list)
                if end0_xi_err_min < end1_xi_err_min:
                    min_ind_local = end0_xi_err_list.index(end0_xi_err_min)
                    min_ind = end_xi_ind_list[min_ind_local]
                    self.end_xi_ind[int_ind, end_ind] = min_ind
                    self.end_xi_val[int_ind, end_ind] = 0.
                else:
                    min_ind_local = end1_xi_err_list.index(end1_xi_err_min)
                    min_ind = end_xi_ind_list[min_ind_local]
                    self.end_xi_ind[int_ind, end_ind] = min_ind
                    self.end_xi_val[int_ind, end_ind] = 1.

        self.num_edge_pts = num_edge_pts
        self.get_diff_intersections_edge_cons_info(num_edge_pts=self.num_edge_pts)
        self.get_surf_avg_normal_dir()


    def get_surf_avg_normal_dir(self):
        self.int_surf_avg_normal_dir = [None for ind in self.int_surf_inds]
        self.int_surf_avg_normal_list = [None for ind in self.int_surf_inds]
        num_eval_pts_1D = 17
        sample_pts = np.linspace(0,1,num_eval_pts_1D)

        eval_pts = np.zeros((num_eval_pts_1D**2,2))
        for i, val0 in enumerate(sample_pts):
            for j, val1 in enumerate(sample_pts):
                eval_pts[i*num_eval_pts_1D+j,0] = val0
                eval_pts[i*num_eval_pts_1D+j,1] = val1

        for int_surf_ind, surf_basis in enumerate(self.int_surf_basis):
            normal_list = []
            for eval_pt_ind, eval_pt in enumerate(eval_pts):
                _, v = self.dFdxi(int_surf_ind, eval_pt)
                a0, a1 = v[:,0], v[:,1]
                a2 = np_normalize(np.cross(a0,a1))
                normal_list += [a2]
            self.int_surf_avg_normal_list[int_surf_ind] = \
                np.average(normal_list, axis=0)          

        for int_surf_ind, avg_normal in enumerate(self.int_surf_avg_normal_list):
            self.int_surf_avg_normal_dir[int_surf_ind] = np.argmax(np.abs(avg_normal))


    def get_diff_intersections_edge_cons_info(self, num_edge_pts):
        """
        Return the dofs and values for edge intersections.
        """
        self.int_edge_cons_local_dofs_list_full = [] #None for i in self.diff_int_inds]
        self.int_edge_cons_local_dofs_list = [] #None for i in self.diff_int_inds]
        self.int_edge_cons_dofs_list_full = [] #None for i in self.diff_int_inds]
        self.int_edge_cons_vals_list_full = [] #None for i in self.diff_int_inds]
        self.int_edge_cons_dofs_list = [] #None for i in self.diff_int_inds]
        self.int_edge_cons_vals_list = [] #None for i in self.diff_int_inds]
        for i, diff_int_ind in enumerate(self.diff_int_inds):
            int_type = self.preprocessor.intersections_type[diff_int_ind]
            # print("int_type:", int_type)
            # if 'surf' not in int_type:
            if int_type[0] == 'surf-edge' or int_type[0] == 'edge-surf':
                edge_indicator = self.preprocessor.diff_int_edge_cons[i]
                side = int(edge_indicator[edge_indicator.index('-')-1])
                para_dir = int(edge_indicator[edge_indicator.index('-')+1])
                edge_val = int(edge_indicator[edge_indicator.index('.')+1])
                if side == 0:
                    start_ind = self.xi_flat_inds[i]
                    end_ind = int((self.xi_flat_inds[i]
                              +self.xi_flat_inds[i+1])/2)
                    start_ind_local = 0
                    end_ind_local = int(self.xi_sizes[i]/2)
                elif side == 1:
                    start_ind = int((self.xi_flat_inds[i]
                              +self.xi_flat_inds[i+1])/2)
                    end_ind = self.xi_flat_inds[i+1]
                    start_ind_local = int(self.xi_sizes[i]/2)
                    end_ind_local = self.xi_sizes[i]
                else:
                    raise ValueError("Unknown side value: {}".format(side))
                if para_dir == 1:
                    start_ind += 1
                    start_ind_local += 1
                cons_dofs_temp = np.arange(start_ind, end_ind, self.para_dim, dtype='int32')
                cons_local_dofs_temp = np.arange(start_ind_local, end_ind_local, 
                                       self.para_dim)
                cons_vals_temp = np.ones(cons_dofs_temp.size)*edge_val
                self.int_edge_cons_dofs_list_full += [cons_dofs_temp]
                self.int_edge_cons_vals_list_full += [cons_vals_temp]
                self.int_edge_cons_local_dofs_list_full += [cons_local_dofs_temp]
                if num_edge_pts is not None:
                    if num_edge_pts > cons_dofs_temp.size:
                        num_edge_pts = cons_dofs_temp.size
                    local_cons_ind = np.linspace(0,cons_dofs_temp.size-1,num_edge_pts, dtype='int32')
                else:
                    local_cons_ind = np.arange(0,cons_dofs_temp.size, dtype='int32')
                self.int_edge_cons_dofs_list += [self.int_edge_cons_dofs_list_full[-1][local_cons_ind]]
                self.int_edge_cons_vals_list += [self.int_edge_cons_vals_list_full[-1][local_cons_ind]]
                self.int_edge_cons_local_dofs_list += [cons_local_dofs_temp[local_cons_ind]]

        # print('self.int_edge_cons_dofs:', self.int_edge_cons_dofs_list)
        self.int_edge_cons_dofs = np.concatenate(self.int_edge_cons_dofs_list, 
                                                 dtype='int32')
        self.int_edge_cons_vals = np.concatenate(self.int_edge_cons_vals_list)

        self.int_xi_free_dofs = []
        for i in range(self.xi_size_global):
            if i not in self.int_edge_cons_dofs:
                self.int_xi_free_dofs += [i]
                
        return self.int_edge_cons_dofs, self.int_edge_cons_vals

    def reshape_CPflat(self, int_surf_ind, cp_flat_sub):
        """
        Given global surface index, and flat control points, give
        to the control points that can be used to create new igakit NURBS
        """
        cp_shape = self.cp_shapes[int_surf_ind]
        cp_sub = cp_flat_sub.reshape(cp_shape[1], cp_shape[0], -1)\
                            .transpose(1,0,2)
        return cp_sub

    def update_CPs(self, cp_flat_single_field, field):
        opt_surf_inds = self.opt_surf_inds[self.opt_field.index(field)]
        updated_cp_flat = self.cp_flat_global[:,field].copy()
        ind_off_int = 0
        ind_off_opt = 0
        for i, s_ind in enumerate(self.int_surf_inds):
            if s_ind in opt_surf_inds:
                updated_cp_flat[ind_off_int:ind_off_int+self.cp_sizes[i]] = \
                    cp_flat_single_field[ind_off_opt:ind_off_opt+self.cp_sizes[i]]
                ind_off_opt += self.cp_sizes[i]
            ind_off_int += self.cp_sizes[i]

        self.cp_flat_global[:,field] = updated_cp_flat
        for i, surf_ind in enumerate(self.int_surf_inds):
            self.cps_flat[i] = self.cp_flat_global[self.cp_flat_inds[i]:
                               self.cp_flat_inds[i+1],:]
            self.cps[i] = self.reshape_CPflat(i, self.cps_flat[i])

    # def update_occ_surf(self, int_surf_ind):
    #     knots = self.occ_int_surfs_data[int_surf_ind].knots
    #     cp_sub = self.cps[int_surf_ind]
    #     ik_surf = NURBS(knots, cp_sub)
    #     occ_surf = ikNURBS2BSpline_surface(ik_surf)
    #     self.occ_int_surfs[int_surf_ind] = occ_surf

    # def update_occ_surfs(self):
    #     # Update occ surfs, use this function before computing 
    #     # derivatives dRdxi, which needs updated occ surfs
    #     for int_surf_ind, surf_ind in enumerate(self.int_surf_inds):
    #         self.update_occ_surf(int_surf_ind)

    # def F_occ(self, int_surf_ind, xi):
    #     phy_pt = gp_Pnt()
    #     self.occ_int_surfs[int_surf_ind].D0(xi[0], xi[1], phy_pt)
    #     return np.array(phy_pt.Coord())

    def F(self, int_surf_ind, xi, cp_flat_sub=None):
        if cp_flat_sub is None:
            cp_flat_sub = self.cps_flat[int_surf_ind]

        # nodes_evals = self.bsplines[int_surf_ind].getNodesAndEvals(xi)
        # # phy_pt = [0.,0.,0.]
        # # for i in range(len(nodes_evals)):
        # #     for j in range(self.num_field):
        # #         phy_pt[j] += cp[nodes_evals[i][0],j] * nodes_evals[i][1]
        # nodes = [item[0] for item in nodes_evals]
        # evals = [item[1] for item in nodes_evals]

        nodes = self.int_surf_basis[int_surf_ind].get_nodes(xi)
        evals = self.int_surf_basis[int_surf_ind].eval_basis(xi)
        cp_sub = cp_flat_sub[nodes]

        phy_pt = np.dot(cp_sub.T, evals)
        return phy_pt[:]

    def dFdxi(self, int_surf_ind, xi, cp_flat_sub=None):
        # phy_pt = gp_Pnt()
        # dFdxi1_vec = gp_Vec()
        # dFdxi2_vec = gp_Vec()
        # self.occ_int_surfs[int_surf_ind].D1(
        #      xi[0], xi[1], phy_pt, dFdxi1_vec, dFdxi2_vec)
        # phy_coord = np.array(phy_pt.Coord())
        # dFdxi = np.zeros((3, self.para_dim))
        # dFdxi[:,0] = dFdxi1_vec.Coord()
        # dFdxi[:,1] = dFdxi2_vec.Coord()
        if cp_flat_sub is None:
            cp_flat_sub = self.cps_flat[int_surf_ind]

        nodes = self.int_surf_basis[int_surf_ind].get_nodes(xi)
        basis_eval = self.int_surf_basis[int_surf_ind].eval_basis_deriv(xi, 1)
        cp_sub = cp_flat_sub[nodes]

        phy_coord = np.dot(cp_sub.T, basis_eval[0])
        dFdxi = np.zeros((3, self.para_dim))
        dFdxi[:,0] = np.dot(cp_sub.T, basis_eval[1])
        dFdxi[:,1] = np.dot(cp_sub.T, basis_eval[2])
        return phy_coord, dFdxi

    def dFdCP(self, int_surf_ind, xi, field):
        deriv_mat = np.zeros((self.num_field, self.cp_sizes[int_surf_ind]))
        # nodes_evals = self.bsplines[int_surf_ind].getNodesAndEvals(xi)
        # # for i in range(len(nodes_evals)):
        # #     deriv_mat[field,nodes_evals[i][0]] = nodes_evals[i][1]
        # nodes = [item[0] for item in nodes_evals]
        # evals = [item[1] for item in nodes_evals]
        nodes = self.int_surf_basis[int_surf_ind].get_nodes(xi)
        evals = self.int_surf_basis[int_surf_ind].eval_basis(xi)
        deriv_mat[field, nodes] = evals
        return deriv_mat

    def local_int_surf_inds(self, int_ind):
        """
        Given local intersection index (intersections that are differentiated)
        return the local intersecting surface indices.
        Similar to s_ind0, s_ind1 = mappling_list[int_ind], but
        the s_ind0 and s_ind1 are the indices from local interseting 
        surface indices list: self.int_surf_inds
        """
        int_ind_global = self.diff_int_inds[int_ind]
        s_ind0, s_ind1 = self.mapping_list[int_ind_global]
        int_surf_ind0 = self.int_surf_inds.index(s_ind0)
        int_surf_ind1 = self.int_surf_inds.index(s_ind1)
        return int_surf_ind0, int_surf_ind1

    def residual_sub(self, int_ind, xi_flat_sub):
        """
        Returns residuals of coupled system when solving interior
        parametric coordinates for intersection curve

        Parameters
        ----------
        xi_flat_sub : ndarray, size: num_pts*4
            xi_flat_sub = [xiA_11, xiA_12, xiA_21, xiA_22, ..., xiA_(n-1)1, xiA_(n-1)2,
                  xiB_11, xiB_12, xiB_21, xiB_22, ..., xiB_(n-1)1, xiB_(n-1)2]
            First subscript is interior point index, {1, n-1}
            Second subscript is component index, {1, 2}
        int_ind : local intersection index, int

        Returns
        -------
        res : ndarray: size: num_pts*4
        """
        # print("=="*20, int_ind)

        num_pts = self.diff_int_num_pts[int_ind]
        xi_coords = xi_flat_sub.reshape(-1, self.para_dim)
        res = np.zeros(xi_flat_sub.size)
        init_para_coord = self.preprocessor.intersections_para_coords[
                          self.diff_int_inds[int_ind]]
        int_surf_ind0, int_surf_ind1 = self.local_int_surf_inds(int_ind)
        cp_flat_sub0 = self.cps_flat[int_surf_ind0]
        cp_flat_sub1 = self.cps_flat[int_surf_ind1]

        int_type = self.diff_int_types[int_ind]
        if int_type[0] == 'surf-edge':
            surf_normal_dir = self.int_surf_avg_normal_dir[int_surf_ind0]
            edge_cons_dof = self.int_edge_cons_local_dofs_list_full[int_ind]
            edge_cons_val = self.int_edge_cons_vals_list_full[int_ind]
        elif int_type[0] == 'edge-surf':
            surf_normal_dir = self.int_surf_avg_normal_dir[int_surf_ind1]
            edge_cons_dof = self.int_edge_cons_local_dofs_list_full[int_ind]
            edge_cons_val = self.int_edge_cons_vals_list_full[int_ind]
        else:
            surf_normal_dir = None

        # Enforce each pair of parametric points from two surfaces
        # have the same physical location.
        for i in range(num_pts):
            # print("*"*20, i)
            # print("res:", res)
            res[i*self.num_field:(i+1)*self.num_field] = self.F(int_surf_ind0, 
                xi_coords[i,0:self.para_dim], cp_flat_sub0) \
                - self.F(int_surf_ind1, 
                xi_coords[i+num_pts,0:self.para_dim], cp_flat_sub1)

            if surf_normal_dir is not None:
                res[i*self.num_field+surf_normal_dir] = \
                    xi_flat_sub[edge_cons_dof[i]] - edge_cons_val[i]

        # Enforce two adjacent elements has the same magnitude 
        # in physical space for surface 1.
        # int_type = self.diff_int_types[int_ind]
        if int_type[0] == 'surf-edge':
            side = 0
        elif int_type[0] == 'edge-surf':
            side = 1
        else:
            side = 0

        if side == 0:
            for i in range(1, num_pts-1):
                phy_coord1 = self.F(int_surf_ind0, 
                             xi_coords[i-1,0:self.para_dim], cp_flat_sub0)
                phy_coord2 = self.F(int_surf_ind0, 
                             xi_coords[i,0:self.para_dim], cp_flat_sub0)
                phy_coord3 = self.F(int_surf_ind0, 
                             xi_coords[i+1,0:self.para_dim], cp_flat_sub0)
                diff1 = phy_coord2 - phy_coord1
                diff2 = phy_coord3 - phy_coord2
                res[i+num_pts*3-1] = np.dot(diff2, diff2) \
                                        - np.dot(diff1, diff1)
        else:
            for i in range(1, num_pts-1):
                phy_coord1 = self.F(int_surf_ind1, 
                             xi_coords[i-1+num_pts,0:self.para_dim], cp_flat_sub1)
                phy_coord2 = self.F(int_surf_ind1, 
                             xi_coords[i+num_pts,0:self.para_dim], cp_flat_sub1)
                phy_coord3 = self.F(int_surf_ind1, 
                             xi_coords[i+1+num_pts,0:self.para_dim], cp_flat_sub1)
                diff1 = phy_coord2 - phy_coord1
                diff2 = phy_coord3 - phy_coord2
                res[i+num_pts*3-1] = np.dot(diff2, diff2) \
                                        - np.dot(diff1, diff1)

        res[-2] = xi_flat_sub[self.end_xi_ind[int_ind,0]] - self.end_xi_val[int_ind,0]
        res[-1] = xi_flat_sub[self.end_xi_ind[int_ind,1]] - self.end_xi_val[int_ind,1]
        return res

    def residual(self, xi_flat):
        res_list = []
        for int_ind, int_ind_global in enumerate(self.diff_int_inds):
            xi_flat_sub = xi_flat[self.xi_flat_inds[int_ind]:
                                  self.xi_flat_inds[int_ind+1]]
            res_list += [self.residual_sub(int_ind, xi_flat_sub)]
        res_full = np.concatenate(res_list)
        return res_full

    def solve_xi(self, xi_flat_init, rtol=1e-5, max_iter=200):
        print("Solving intersections parametric coordinates ...")
        solver = 'fsolve'
        # num_pts = 9
        # num_ints = 2
        # xi_init = xi_flat_init.reshape(-1,2)
        # xi_init_list = [[] for i in range(num_ints)]
        # for i in range(num_ints):
        #     for side in range(2):
        #         xi_init_list[i] += [xi_init[i*num_pts*2+side*num_pts:
        #                               i*num_pts*2+(side+1)*num_pts,:]]

        # import matplotlib.pyplot as plt
        # plt.figure()
        # for i in range(num_ints):
        #     for side in range(2):
        #         if side == 0:
        #             line_style = '-*'
        #         else:
        #             line_style = '-^'
        #         plt.plot(xi_init_list[i][side][:,0], xi_init_list[i][side][:,1], 
        #                 line_style, label=f'int: {i}, sdie: {side}')
        # plt.legend()
        # plt.xlabel('xi0')
        # plt.xlabel('xi1')
        # plt.title('xi before solve')
        # # plt.show()
        if solver == 'fsolve':
            xi_root = fsolve(self.residual, x0=xi_flat_init, fprime=self.dRdxi)
        else:
            xi_root = xi_flat_init.copy()
            iter_ind = 0
            while True:
                # print(xi_root)
                r = self.residual(xi_root)
                r_norm = np.linalg.norm(r)

                if iter_ind == 0:
                    init_r_norm = r_norm

                rel_r_norm = r_norm/init_r_norm

                # print("Newton solver iteration: {:4d}, "
                #       "rel err: {:6.4e}, true_err: {:6.4e}"
                #       .format(iter_ind, rel_r_norm, r_norm))

                if iter_ind > max_iter:
                    print("Max number of iterations {:4d} exceeded, "
                          "rel err: {:6.4e}, true_err: {:6.4e}"
                          .format(max_iter, rel_r_norm, r_norm))
                    break
                if rel_r_norm < rtol or r_norm < 1e-6:
                    print("Newton solver converged with {:4d} iterations, "
                          "rel err: {:6.4e}, true_err: {:6.4e}"
                          .format(iter_ind, rel_r_norm, r_norm))
                    break

                drdxi = self.dRdxi(xi_root, coo=False)
                self.drdxi_csr = csr_matrix(drdxi)
                dxi = splin.spsolve(self.drdxi_csr, -r)

                xi_root += dxi
                iter_ind += 1


        # print("xi after solve:", xi_root.reshape(-1,2))

        # xi_root_temp = xi_root.reshape(-1,2)
        # xi_root_list = [[] for i in range(num_ints)]
        # for i in range(num_ints):
        #     for side in range(2):
        #         xi_root_list[i] += [xi_root_temp[i*num_pts*2+side*num_pts:
        #                               i*num_pts*2+(side+1)*num_pts,:]]

        # import matplotlib.pyplot as plt
        # plt.figure()
        # for i in range(num_ints):
        #     for side in range(2):
        #         if side == 0:
        #             line_style = '-*'
        #         else:
        #             line_style = '-^'
        #         plt.plot(xi_root_list[i][side][:,0], xi_root_list[i][side][:,1], 
        #                 line_style, label=f'int: {i}, sdie: {side}')
        # plt.legend()
        # plt.xlabel('xi0')
        # plt.xlabel('xi1')
        # plt.title('xi after solve')
        # plt.show()
        return xi_root


    def dRdxi_sub(self, int_ind, xi_flat_sub, coo=True):
        init_para_coord = self.preprocessor.intersections_para_coords[
                          self.diff_int_inds[int_ind]]
        num_pts = self.diff_int_num_pts[int_ind]
        int_surf_ind0, int_surf_ind1 = self.local_int_surf_inds(int_ind)

        int_type = self.diff_int_types[int_ind]
        if int_type[0] == 'surf-edge':
            surf_normal_dir = self.int_surf_avg_normal_dir[int_surf_ind0]
            edge_cons_dof = self.int_edge_cons_local_dofs_list_full[int_ind]
            # edge_cons_val = self.int_edge_cons_vals_list[int_ind]
        elif int_type[0] == 'edge-surf':
            surf_normal_dir = self.int_surf_avg_normal_dir[int_surf_ind1]
            edge_cons_dof = self.int_edge_cons_local_dofs_list_full[int_ind]
            # edge_cons_val = self.int_edge_cons_vals_list[int_ind]
        else:
            surf_normal_dir = None

        xi_coords = xi_flat_sub.reshape(-1, self.para_dim)
        deriv_xi = np.zeros((xi_flat_sub.size, xi_flat_sub.size))

        # Upper sectin block size
        ub_size = [self.num_field, self.para_dim]
        # Lower section block size
        lb_size = [1, self.para_dim]
        # Number of upper section rows
        ur = ub_size[0]*num_pts
        # Number of left section columns
        lc = ub_size[1]*num_pts

        for i in range(num_pts):
            if i == 0:
                # For left end point
                FAi, dFAidxi = \
                    self.dFdxi(int_surf_ind0, xi_coords[i, 0:self.para_dim])
                FBi, dFBidxi = self.dFdxi(int_surf_ind1, 
                     xi_coords[i+num_pts, 0:self.para_dim])
                # 1st point and derivative
                FAir, dFAirdxi = \
                    self.dFdxi(int_surf_ind0, xi_coords[i+1, 0:self.para_dim])
                FBir, dFBirdxi = self.dFdxi(int_surf_ind1, 
                     xi_coords[i+1+num_pts, 0:self.para_dim])
            elif i == num_pts-1:
                # for (n-1)-th point and derivative
                FAil, dFAildxi = FAi, dFAidxi
                FBil, dFBildxi = FBi, dFBidxi
                # For n-th point and derivative
                FAi, dFAidxi = FAir, dFAirdxi
                FBi, dFBidxi = FBir, dFBirdxi
            else:
                # For (i-1)-th point and derivative
                FAil, dFAildxi = FAi, dFAidxi 
                FBil, dFBildxi = FBi, dFBidxi
                # For i-th point and derivative
                FAi, dFAidxi = FAir, dFAirdxi
                FBi, dFBidxi = FBir, dFBirdxi
                # For (i+1)-th point and derivative
                FAir, dFAirdxi = \
                    self.dFdxi(int_surf_ind0, xi_coords[i+1, 0:self.para_dim])
                FBir, dFBirdxi = self.dFdxi(int_surf_ind1, 
                     xi_coords[i+1+num_pts, 0:self.para_dim])

            # For upper section
            deriv_xi[i*ub_size[0]:(i+1)*ub_size[0], 
                     i*ub_size[1]:(i+1)*ub_size[1]] = dFAidxi
            deriv_xi[i*ub_size[0]:(i+1)*ub_size[0], 
                lc+i*ub_size[1]:lc+(i+1)*ub_size[1]] = -dFBidxi

            if surf_normal_dir is not None:
                deriv_xi[i*self.num_field+surf_normal_dir] = 0.
                deriv_xi[i*self.num_field+surf_normal_dir,
                         edge_cons_dof[i]] = 1.

            # For lower section:
            if i > 0 and i < num_pts-1:
                # deriv_xi[i+ur-1, (i-1)*lb_size[1]:i*lb_size[1]] = \
                #     2*np.dot(FAi-FAil, dFAildxi)
                # deriv_xi[i+ur-1, i*lb_size[1]:(i+1)*lb_size[1]] = \
                #     -2*np.dot(FAir-FAil, dFAidxi)
                # deriv_xi[i+ur-1, (i+1)*lb_size[1]:(i+2)*lb_size[1]] = \
                #     2*np.dot(FAir-FAi, dFAirdxi)
                # int_type = self.diff_int_types[int_ind]
                if int_type[0] == 'surf-edge':
                    side = 0
                elif int_type[0] == 'edge-surf':
                    side = 1
                else:
                    side = 0

                if side == 0:
                    deriv_xi[i+ur-1, (i-1)*lb_size[1]:i*lb_size[1]] = \
                        2*np.dot(FAi-FAil, dFAildxi)
                    deriv_xi[i+ur-1, i*lb_size[1]:(i+1)*lb_size[1]] = \
                        -2*np.dot(FAir-FAil, dFAidxi)
                    deriv_xi[i+ur-1, (i+1)*lb_size[1]:(i+2)*lb_size[1]] = \
                        2*np.dot(FAir-FAi, dFAirdxi)
                else:
                    # deriv_xi[i+ur-1, (i-1+num_pts)*lb_size[1]:(i+num_pts)*lb_size[1]] = \
                    #     2*np.dot(FBi-FBil, dFBildxi)
                    # deriv_xi[i+ur-1, (i+num_pts)*lb_size[1]:(i+1+num_pts)*lb_size[1]] = \
                    #     -2*np.dot(FBir-FBil, dFBidxi)
                    # deriv_xi[i+ur-1, (i+1+num_pts)*lb_size[1]:(i+2+num_pts)*lb_size[1]] = \
                    #     2*np.dot(FBir-FBi, dFBirdxi)
                    deriv_xi[i+ur-1, lc+(i-1)*lb_size[1]:lc+(i)*lb_size[1]] = \
                        2*np.dot(FBi-FBil, dFBildxi)
                    deriv_xi[i+ur-1, lc+(i)*lb_size[1]:lc+(i+1)*lb_size[1]] = \
                        -2*np.dot(FBir-FBil, dFBidxi)
                    deriv_xi[i+ur-1, lc+(i+1)*lb_size[1]:lc+(i+2)*lb_size[1]] = \
                        2*np.dot(FBir-FBi, dFBirdxi)
            elif i == num_pts-1:
                deriv_xi[i+ur-1, self.end_xi_ind[int_ind,0]] = 1.
                deriv_xi[i+ur, self.end_xi_ind[int_ind,1]] = 1.

        if coo:
            deriv_xi = coo_matrix(deriv_xi)
        return deriv_xi

    def dRdxi(self, xi_flat, coo=False):
        self.dRdxi_list = [[None for i in range(len(self.diff_int_inds))]
                      for j in range(len(self.diff_int_inds))]
        for int_ind, int_ind_global in enumerate(self.diff_int_inds):
            xi_flat_sub = xi_flat[self.xi_flat_inds[int_ind]:
                                  self.xi_flat_inds[int_ind+1]]
            # print("*"*50, int_ind)
            # print("*"*50, len(xi_flat_sub))
            # print("*"*50, self.end_xi_ind)
            # if len(self.int_edge_cons_local_dofs_list[int_ind]) > 0:
            #     xi_flat_sub[self.int_edge_cons_local_dofs_list[int_ind]] = \
            #         self.int_edge_cons_vals_list[int_ind]
            self.dRdxi_list[int_ind][int_ind] = self.dRdxi_sub(int_ind, 
                                           xi_flat_sub, coo=True)
        dRdxi_full = bmat(self.dRdxi_list, format='coo')
        if not coo:
            dRdxi_full = dRdxi_full.todense()
        return dRdxi_full

    def dRdCP_sub(self, int_ind, xi_flat_sub, field, coo=True):
        # deriv_cp = np.zeros((xi_flat_sub.size, self.cp_size_global))
        xi_coords = xi_flat_sub.reshape(-1, self.para_dim)
        num_pts = self.diff_int_num_pts[int_ind]
        int_surf_inds = self.local_int_surf_inds(int_ind)

        int_type = self.diff_int_types[int_ind]
        if int_type[0] == 'surf-edge':
            surf_normal_dir = self.int_surf_avg_normal_dir[int_surf_inds[0]]
            # edge_cons_dof = self.int_edge_cons_local_dofs_list[int_ind]
            # edge_cons_val = self.int_edge_cons_vals_list[int_ind]
        elif int_type[0] == 'edge-surf':
            surf_normal_dir = self.int_surf_avg_normal_dir[int_surf_inds[1]]
            # edge_cons_dof = self.int_edge_cons_local_dofs_list[int_ind]
            # edge_cons_val = self.int_edge_cons_vals_list[int_ind]
        else:
            surf_normal_dir = None

        u_size = self.num_field*num_pts
        deriv_cp = [np.zeros((xi_flat_sub.size, self.cp_sizes[int_surf_inds[0]])),
                    np.zeros((xi_flat_sub.size, self.cp_sizes[int_surf_inds[1]]))]

        for i in range(num_pts):
            int_type = self.diff_int_types[int_ind]
            if int_type[0] == 'surf-edge':
                deriv_side = 0
            elif int_type[0] == 'edge-surf':
                deriv_side = 1
            else:
                deriv_side = 0

            for side in range(self.num_sides):
                surf_ind = int_surf_inds[side]
                xi_coords_temp = xi_coords[int(i+num_pts*side), 
                                           0:self.para_dim]

                if side == 0:
                    sign = 1.
                else:
                    sign = -1.

                # if side == deriv_side and i > 1:
                if side == deriv_side and i > 1:
                    dFdcp_temp = dFdcpir
                else:
                    dFdcp_temp = self.dFdCP(surf_ind, 
                                 xi_coords_temp, field)
                
                deriv_cp[side][i*self.num_field:(i+1)*self.num_field,:] = \
                    dFdcp_temp[:]*sign

                if side == deriv_side and i == 0:
                    dFdcpi = dFdcp_temp
                elif side == deriv_side and i > 0:
                    dFdcpil = dFdcpi
                    dFdcpi = dFdcp_temp

                if surf_normal_dir is not None:
                    deriv_cp[side][i*self.num_field+surf_normal_dir] = 0.
                    
            if i > 0 and i < num_pts-1:
                surf_ind = int_surf_inds[deriv_side]
                xi_coords_il = xi_coords[i-1, 0:self.para_dim]
                xi_coords_i = xi_coords[i, 0:self.para_dim]
                xi_coords_ir = xi_coords[i+1, 0:self.para_dim]
                dFdcpir = self.dFdCP(surf_ind, xi_coords_ir, field)
                Fil = self.F(surf_ind, xi_coords_il)
                Fi = self.F(surf_ind, xi_coords_i)
                Fir = self.F(surf_ind, xi_coords_ir)
                # res_vec = 2*(np.dot(Fir, dFdcpir) - np.dot(Fir, dFdcpi) 
                #            - np.dot(Fi, dFdcpir) + np.dot(Fi, dFdcpil) 
                #            + np.dot(Fil, dFdcpi) - np.dot(Fil, dFdcpil))
                res_vec = 2*(np.dot(Fir-Fi, dFdcpir-dFdcpi) 
                           - np.dot(Fi-Fil, dFdcpi-dFdcpil))
                deriv_cp[deriv_side][u_size+i-1, 0:self.cp_sizes[surf_ind]] = res_vec
        if coo:
            deriv_cp = [coo_matrix(mat) for mat in deriv_cp]
        return deriv_cp

    def dRdCP(self, xi_flat, field, coo=True):
        field_ind = self.opt_field.index(field)
        dRdCP_list = [[None for i in range(len(self.opt_surf_inds[field_ind]))]
                      for j in range(len(self.diff_int_inds))]
        dRdCP_sub_row_sizes = []
        dRdCP_sub_col_sizes = [None for i in range(len(self.opt_surf_inds[field_ind]))]
        for int_ind, int_ind_global in enumerate(self.diff_int_inds):
            xi_flat_sub = xi_flat[self.xi_flat_inds[int_ind]:
                                  self.xi_flat_inds[int_ind+1]]
            int_surf_ind0, int_surf_ind1 = self.local_int_surf_inds(int_ind)
            #################################################################
            s_ind0 = self.int_surf_inds[int_surf_ind0]
            s_ind1 = self.int_surf_inds[int_surf_ind1]
            dRdCP_sub_temp = self.dRdCP_sub(int_ind, xi_flat_sub, field, coo=True)
            dRdCP_sub_row_sizes += [dRdCP_sub_temp[0].shape[0]]
            if s_ind0 in self.opt_surf_inds[field_ind]:
                col_ind0 = self.opt_surf_inds[field_ind].index(s_ind0)
                dRdCP_list[int_ind][col_ind0] = dRdCP_sub_temp[0]
                dRdCP_sub_col_sizes[col_ind0] = dRdCP_sub_temp[0].shape[1]
            if s_ind1 in self.opt_surf_inds[field_ind]:
                col_ind1 = self.opt_surf_inds[field_ind].index(s_ind1)
                dRdCP_list[int_ind][col_ind1] = dRdCP_sub_temp[1]
                dRdCP_sub_col_sizes[col_ind1] = dRdCP_sub_temp[1].shape[1]

        for int_ind, int_ind_global in enumerate(self.diff_int_inds):
            if dRdCP_list[int_ind].count(None) == len(dRdCP_list[int_ind]):
                temp_mat = coo_matrix(np.zeros((dRdCP_sub_row_sizes[int_ind],
                                     dRdCP_sub_col_sizes[0])))
                dRdCP_list[int_ind][0] = temp_mat

        dRdCP_full = bmat(dRdCP_list, format='coo')
        if not coo:
            dRdCP_full = dRdCP_full.todense()
        return dRdCP_full


    # Implicit edge DoFs solve
    def implicit_edge_residual_sub(self, xi_edge, cp_edge, basis_edge,
                                   xi_mid, cp_mid, basis_mid):
        assert len(xi_edge) == len(xi_mid)


        res_vec = np.zeros(len(xi_edge))
        for xi_ind in range(len(xi_edge)):
            basis_edge_nodes = basis_edge.get_nodes(xi_edge[xi_ind])
            basis_edge_val = basis_edge.eval_basis(xi_edge[xi_ind])
            edge_val = np.dot(basis_edge_val, cp_edge[basis_edge_nodes])

            basis_mid_nodes = basis_mid.get_nodes(xi_mid[xi_ind])
            basis_mid_val = basis_edge.eval_basis(xi_mid[xi_ind])
            mid_val = np.dot(basis_mid_val, cp_mid[basis_mid_nodes])

            res_vec[xi_ind] = edge_val - mid_val
        return res_vec




    def implicit_edge_residual(self, xi_flat, cp_edges, cp_mids=None):

        res_vec = np.zeros(cp_edge.size)
        edge_int_ind = 0
        for i, diff_int_ind in enumerate(self.diff_int_inds):
            int_type = self.preprocessor.intersections_type[diff_int_ind]
            if int_type[0] == 'surf-edge' or int_type[0] == 'edge-surf':
                int_surf_inds = self.local_int_surf_inds(i)
                edge_indicator = self.preprocessor.diff_int_edge_cons[i]
                side = int(edge_indicator[edge_indicator.index('-')-1])
                para_dir = int(edge_indicator[edge_indicator.index('-')+1])
                edge_val = int(edge_indicator[edge_indicator.index('.')+1])

                xi_flat_sub = xi_flat[self.xi_flat_inds[i]:self.xi_flat_inds[i+1]]
                xi_size = self.xi_sizes[i]
                local_cons_dof = self.int_edge_cons_local_dofs_list[edge_int_ind]

                if para_dir == 0:
                    edge_surf_xi_eval_dofs = np.sort(np.concatenate([local_cons_dof, local_cons_dof+1]))
                else:
                    edge_surf_xi_eval_dofs = np.sort(np.concatenate([local_cons_dof, local_cons_dof-1]))

                if side == 0:
                    mid_surf_xi_eval_dofs = edge_surf_xi_eval_dofs+int(xi_sizes/2)
                else:
                    mid_surf_xi_eval_dofs = edge_surf_xi_eval_dofs-int(xi_sizes/2)


                edge_surf_xi_eval = xi_flat_sub[edge_surf_xi_eval_dofs].reshape(-1,2)
                mid_surf_xi_eval = xi_flat_sub[mid_surf_xi_eval_dofs].reshape(-1,2)

                edge_surf_ind = int_surf_inds[side]
                mid_surf_ind = int_surf_inds[int(1-side)]

                edge_surf_basis = self.int_surf_basis[edge_surf_ind]
                mid_surf_basis = self.int_surf_basis[mid_surf_ind]

                edge_surf_cp = cp_edges[self.cp_flat_inds[edge_surf_ind]:
                                        self.cp_flat_inds[edge_surf_ind+1]]

                if cp_mids is None:
                    mid_surf_cp = self.cp_flat_global\
                              [self.cp_flat_inds[mid_surf_ind]:
                               self.cp_flat_inds[mid_surf_ind+1],
                               self.int_surf_avg_normal_dir[mid_surf_ind]]
                else:
                    mid_surf_cp = cp_mids[self.cp_flat_inds[mid_surf_ind]:
                                          self.cp_flat_inds[mid_surf_ind+1]]

                res_vec[self.cp_flat_inds[mid_surf_ind]:
                        self.cp_flat_inds[mid_surf_ind+1]] += \
                        self.implicit_edge_residual_sub(
                            edge_surf_xi_eval, edge_surf_cp, edge_surf_basis,
                            mid_surf_xi_eval, mid_surf_cp, mid_surf_basis)

                edge_int_ind += 1


    def implicit_edge_dresdxi(self):
        pass

    def implicit_edge_dresdcp(self):
        pass


if __name__ == '__main__':
    # from GOLDFISH.tests.test_tbeam import nonmatching_opt
    # from GOLDFISH.tests.test_slr import nonmatching_opt
    from PENGoLINS.occ_preprocessing import *

    # filename_igs = "./tests/geometry/init_Tbeam_geom_moved.igs"
    # filename_igs = "./tests/geometry/init_Tbeam_geom_curved_4patch.igs"
    filename_igs = "/home/han/OneDrive/github/GOLDFISH/demos_om/shape_opt_mint/rib_test/geometry/box_geom_init_1ribs.igs"
    igs_shapes = read_igs_file(filename_igs, as_compound=False)
    occ_surf_list = [topoface2surface(face, BSpline=True) 
                     for face in igs_shapes]
    occ_surf_data_list = [BSplineSurfaceData(surf) for surf in occ_surf_list]
    num_surfs = len(occ_surf_list)
    p = occ_surf_data_list[0].degree[0]

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

    preprocessor.check_intersections_type()
    preprocessor.get_diff_intersections()

    cpiga2xi = CPIGA2XiCoarse(preprocessor)

    # For 5 patches
    # int_surf_ind0 = 0
    # int_surf_ind1 = 1
    # int_ind = 0
    # field = 1
    # xi = [0.55, 0.66]

    # aa = cpiga2xi.F_occ(int_surf_ind0, xi)
    # ab = cpiga2xi.F(int_surf_ind0, xi)
    # ac = cpiga2xi.dFdxi(int_surf_ind1, xi)
    # ad = cpiga2xi.dFdCP(int_surf_ind1, xi, field)
    # xi_flat_sub = cpiga2xi.xis_flat[int_ind] + np.random.random(cpiga2xi.xi_sizes[int_ind])*1e-2
    # xi_flat = xi_flat_sub
    # ae = cpiga2xi.residual_sub(int_ind, xi_flat_sub)
    # af = cpiga2xi.residual(xi_flat)
    # ag = cpiga2xi.dRdxi_sub(int_ind, xi_flat_sub)
    # ah = cpiga2xi.dRdCP_sub(int_ind, xi_flat_sub, field)
    # dRdxi = cpiga2xi.dRdxi(xi_flat)
    # dRdCP = cpiga2xi.dRdCP(xi_flat, field)
    # xi_root = cpiga2xi.solve_xi(xi_flat)


    # For 4 patches
    int_surf_ind0 = 0
    int_surf_ind1 = 2
    int_ind = 1
    int_diff_ind = 0
    field = 1
    xi = [0.55, 0.66]

    # aa = cpiga2xi.F_occ(int_surf_ind0, xi)
    # ab = cpiga2xi.F(int_surf_ind0, xi)
    # ac = cpiga2xi.dFdxi(int_surf_ind1, xi)
    # ad = cpiga2xi.dFdCP(int_surf_ind1, xi, field)
    # xi_flat_sub = cpiga2xi.xis_flat[int_diff_ind] + np.random.random(cpiga2xi.xi_sizes[int_diff_ind])*1e-2
    xi_flat = np.concatenate(cpiga2xi.xis_flat) #+ np.random.random(np.sum(cpiga2xi.xi_sizes))*1e-2
    # ae = cpiga2xi.residual_sub(int_ind, xi_flat_sub)
    af = cpiga2xi.residual(xi_flat)
    # ag = cpiga2xi.dRdxi_sub(int_ind, xi_flat_sub)
    # ah = cpiga2xi.dRdCP_sub(int_ind, xi_flat_sub, field)
    dRdxi = cpiga2xi.dRdxi(xi_flat)
    dRdCP = cpiga2xi.dRdCP(xi_flat, field)
    # xi_root = cpiga2xi.solve_xi(xi_flat)


# class SplineBC(object):
#     """
#     Setting Dirichlet boundary condition to tIGAr spline generator.
#     """
#     def __init__(self, directions=[0,1], sides=[[0,1],[0,1]], 
#                  fields=[[[0,1,2],[0,1,2]],[[0,1,2],[0,1,2]]],
#                  n_layers=[[1,1],[1,1]]):
#         self.fields = fields
#         self.directions = directions
#         self.sides = sides
#         self.n_layers = n_layers

#     def set_bc(self, spline_generator):
#         for direction in self.directions:
#             for side in self.sides[direction]:
#                 for field in self.fields[direction][side]:
#                     scalar_spline = spline_generator.getScalarSpline(field)
#                     side_dofs = scalar_spline.getSideDofs(direction,
#                                 side, nLayers=self.n_layers[direction][side])
#                     spline_generator.addZeroDofs(field, side_dofs)

# def OCCBSpline2tIGArSpline(surface, num_field=3, quad_deg_const=3, 
#                            spline_bc=None, index=0):
#     """
#     Generate ExtractedBSpline from OCC B-spline surface.
#     """
#     quad_deg = surface.UDegree()*quad_deg_const
#     # DIR = SAVE_PATH+"spline_data/extraction_"+str(index)+"_init"
#     # spline = ExtractedSpline(DIR, quad_deg)
#     spline_mesh = NURBSControlMesh4OCC(surface, useRect=False)
#     spline_generator = EqualOrderSpline(worldcomm, num_field, spline_mesh)
#     if spline_bc is not None:
#         spline_bc.set_bc(spline_generator)
#     # spline_generator.writeExtraction(DIR)
#     spline = ExtractedSpline(spline_generator, quad_deg)
#     return spline

# test_ind = 3
# optimizer = 'SNOPT'
# # optimizer = 'SLSQP'
# opt_field = [0]
# # save_path = './'
# save_path = '/home/han/Documents/test_results/'
# # save_path = '/Users/hanzhao/Documents/test_results/'
# # folder_name = "results/"
# folder_name = "results"+str(test_ind)+"/"

# filename_igs = "./geometry/box_geom_init_1ribs.igs"
# igs_shapes = read_igs_file(filename_igs, as_compound=False)
# occ_surf_list_all = [topoface2surface(face, BSpline=True) 
#                  for face in igs_shapes]
# occ_surf_list = [occ_surf_list_all[i] for i in range(len(occ_surf_list_all))]
# occ_surf_data_list = [BSplineSurfaceData(surf) for surf in occ_surf_list]
# num_surfs = len(occ_surf_list)
# p = occ_surf_data_list[0].degree[0]

# # Define material and geometric parameters
# E = Constant(1.0e12)
# nu = Constant(0.)
# h_th = Constant(0.1)
# penalty_coefficient = 1.0e3
# pressure = Constant(1.)

# fields0 = [[[0,1,2]], None,]
# spline_bc0 = SplineBC(directions=[0], sides=[[0], None],
#                       fields=fields0, n_layers=[[1], None])
# spline_bcs = [spline_bc0]*4+[None]*4

# # Geometry preprocessing and surface-surface intersections computation
# preprocessor = OCCPreprocessing(occ_surf_list, reparametrize=False, 
#                                 refine=False)
# print("Computing intersections...")
# # int_data_filename = "int_data.npz"
# # if os.path.isfile(int_data_filename):
# #     preprocessor.load_intersections_data(int_data_filename)
# # else:
# #     preprocessor.compute_intersections(mortar_refine=2)
# #     preprocessor.save_intersections_data(int_data_filename)

# preprocessor.compute_intersections(mortar_refine=1)
# if mpirank == 0:
#     print("Total DoFs:", preprocessor.total_DoFs)
#     print("Number of intersections:", preprocessor.num_intersections_all)

# # # Display B-spline surfaces and intersections using 
# # # PythonOCC build-in 3D viewer.
# # display, start_display, add_menu, add_function_to_menu = init_display()
# # preprocessor.display_surfaces(display, save_fig=False)
# # preprocessor.display_intersections(display, color='RED', save_fig=False)

# # print(aaa)


# if mpirank == 0:
#     print("Creating splines...")
# # Create tIGAr extracted spline instances
# splines = []
# for i in range(num_surfs):
#         spline = OCCBSpline2tIGArSpline(preprocessor.BSpline_surfs[i], 
#                                         spline_bc=spline_bcs[i], index=i)
#         splines += [spline,]

# # Create non-matching problem
# nonmatching_opt = NonMatchingOptFFD(splines, E, h_th, nu, 
#                                     opt_field=opt_field, comm=worldcomm)
# nonmatching_opt.create_mortar_meshes(preprocessor.mortar_nels)

# if mpirank == 0:
#     print("Setting up mortar meshes...")
# nonmatching_opt.mortar_meshes_setup(preprocessor.mapping_list, 
#                     preprocessor.intersections_para_coords, 
#                     penalty_coefficient, 2)
# pressure = -Constant(1.)
# f = as_vector([Constant(0.), Constant(0.), pressure])
# source_terms = []
# residuals = []
# for s_ind in range(nonmatching_opt.num_splines):
#     z = nonmatching_opt.splines[s_ind].rationalize(
#         nonmatching_opt.spline_test_funcs[s_ind])
#     source_terms += [inner(f, z)*nonmatching_opt.splines[s_ind].dx]
#     residuals += [SVK_residual(nonmatching_opt.splines[s_ind], 
#                   nonmatching_opt.spline_funcs[s_ind], 
#                   nonmatching_opt.spline_test_funcs[s_ind], 
#                   E, nu, h_th, source_terms[s_ind])]
# nonmatching_opt.set_residuals(residuals)

# nonmatching_opt.solve_nonlinear_nonmatching_problem()
# # print(aaa)

# # shopt_multi_ffd_inds = [[0,1,2,3,4], [5,6,7]]
# shopt_multi_ffd_inds = [[0,1,2,3,4], [5]]
# nonmatching_opt.set_shopt_multiFFD_surf_inds(shopt_multi_ffd_inds)

# #################################################
# num_shopt_ffd = nonmatching_opt.num_shopt_ffd
# shopt_ffd_lims_multiffd = nonmatching_opt.shopt_cpsurf_lims_multiffd

# shopt_ffd_num_el = [[1,1,1], [2,1,1]]
# shopt_ffd_p = [2]*num_shopt_ffd
# extrude_dir = [1,0]

# shopt_ffd_block_list = []
# for ffd_ind in range(num_shopt_ffd):
#     field = extrude_dir[ffd_ind]
#     cp_range = shopt_ffd_lims_multiffd[ffd_ind][field][1]\
#               -shopt_ffd_lims_multiffd[ffd_ind][field][0]
#     shopt_ffd_lims_multiffd[ffd_ind][field][1] = \
#         shopt_ffd_lims_multiffd[ffd_ind][field][1] + 0.1*cp_range
#     shopt_ffd_lims_multiffd[ffd_ind][field][0] = \
#         shopt_ffd_lims_multiffd[ffd_ind][field][0] - 0.1*cp_range
#     shopt_ffd_block_list += [create_3D_block(shopt_ffd_num_el[ffd_ind],
#                                        shopt_ffd_p[ffd_ind],
#                                        shopt_ffd_lims_multiffd[ffd_ind])]

# for ffd_ind in range(num_shopt_ffd):
#     vtk_writer = VTKWriter()
#     vtk_writer.write("./geometry/tbeam_shopt_ffd_block_init"+str(ffd_ind)+".vtk", 
#                      shopt_ffd_block_list[ffd_ind])
#     vtk_writer.write_cp("./geometry/tbeam_shopt_ffd_cp_init"+str(ffd_ind)+".vtk", 
#                      shopt_ffd_block_list[ffd_ind])

# shopt_ffd_knots_list = [ffd_block.knots for ffd_block 
#                         in shopt_ffd_block_list]
# shopt_ffd_control_list = [ffd_block.control for ffd_block 
#                           in shopt_ffd_block_list]
# print("Setting multiple shape FFD blocks ...")
# nonmatching_opt.set_shopt_multiFFD(shopt_ffd_knots_list, 
#                                        shopt_ffd_control_list)

# ########### Set constraints info #########
# a0 = nonmatching_opt.set_shopt_regu_CP_multiFFD(shopt_regu_dir_list=[[None, None], 
#                                                                 [None, None]], 
#                                            shopt_regu_side_list=[[None, None], 
#                                                                  [None, None]])
# a1 = nonmatching_opt.set_shopt_pin_CP_multiFFD(0, pin_dir0_list=['whole', None], 
#                                           pin_side0_list=None,
#                                           pin_dir1_list=None, 
#                                           pin_side1_list=None)
# # a2 = nonmatching_opt.set_shopt_pin_CP_multiFFD(2, pin_dir0_list=['whole', None], 
# #                                           pin_side0_list=None,
# #                                           pin_dir1_list=None, 
# #                                           pin_side1_list=None)
# # a2 = nonmatching_opt.set_shopt_pin_CP_multiFFD(2, pin_dir0_list=[0, None], 
# #                                           pin_side0_list=[[0,1], None],
# #                                           pin_dir1_list=None, 
# #                                           pin_side1_list=None)
# a3 = nonmatching_opt.set_shopt_align_CP_multiFFD(shopt_align_dir_list=[1,1])

# #################################################

# #################################
# preprocessor.check_intersections_type()
# preprocessor.get_diff_intersections()
# nonmatching_opt.set_diff_intersections(preprocessor)
# #################################
