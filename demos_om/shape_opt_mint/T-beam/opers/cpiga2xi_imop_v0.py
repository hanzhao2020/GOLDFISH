# import sys
# sys.path.append("../")
from GOLDFISH.nonmatching_opt_ffd import *
from scipy.sparse import coo_matrix, csr_matrix
from scipy.linalg import block_diag
from tIGAr import BSplines
# from cpiga2xi import CPIGA2Xi
from scipy.sparse.linalg import splu

from GOLDFISH.nonmatching_opt_ffd import *
from scipy.sparse import coo_matrix, csr_matrix
from scipy.linalg import block_diag
from tIGAr import BSplines
from igakit.cad import NURBS
from scipy.optimize import fsolve, newton_krylov
from scipy.sparse import coo_matrix, bmat
# from scipy.linalg import block_diag

from PENGoLINS.occ_preprocessing import *
from PENGoLINS.igakit_utils import *

class CPIGA2Xiv0(object):
    def __init__(self, preprocessor, int_indices_diff=None, opt_field=[0,1,2]):
        """
        preprocessor : PENGoLINS.preprocessor instance
        int_indices_diff : list of ints, the indices of intersections to differentiate
            if None, differentiate all intersections (not recommended)
        """
        self.preprocessor = preprocessor
        if self.preprocessor.reparametrize is True:
            self.occ_surfs_all = self.preprocessor.BSpline_surfs_repara
            self.occ_surfs_all_data = self.preprocessor.BSpline_surfs_repara_data
        else:
            self.occ_surfs_all = self.preprocessor.BSpline_surfs
            self.occ_surfs_all_data = self.preprocessor.BSpline_surfs_data

        # Define basic paramters
        self.num_field = 3
        self.para_dim = 2
        self.num_end_pts = 2
        self.num_sides = 2
        self.num_intersections = self.preprocessor.num_intersections_all
        self.mortar_nels = self.preprocessor.mortar_nels
        self.mortar_pts = [nel+1 for nel in self.mortar_nels]
        self.mapping_list = self.preprocessor.mapping_list

        # Get indices of intersections to differentiate
        if int_indices_diff is None:
            self.int_indices_diff = list(range(self.num_intersections))
        else:
            self.int_indices_diff = int_indices_diff

        self.opt_field = opt_field

        int_surf_inds_temp = []
        for i, ind in enumerate(self.int_indices_diff):
            s_ind0, s_ind1 = self.mapping_list[ind]
            if s_ind0 not in int_surf_inds_temp:
                int_surf_inds_temp += [s_ind0]
            if s_ind1 not in int_surf_inds_temp:
                int_surf_inds_temp += [s_ind1]
        self.int_surf_inds = list(np.sort(int_surf_inds_temp))
        self.num_int_surfs = len(self.int_surf_inds)

        #### Control points related properties

        # Get B-spline surfaces and data of interest
        self.occ_surfs_int = []
        self.occ_surfs_int_data = []
        for i, ind in enumerate(self.int_surf_inds):
            self.occ_surfs_int += [self.occ_surfs_all[ind]]
            self.occ_surfs_int_data += [self.occ_surfs_all_data[ind]]

        # BSpline related properties
        self.cp_shapes = [surf_data.control.shape[0:self.para_dim]
                          for surf_data in self.occ_surfs_int_data]
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
        self.cps = [surf_data.control[:,:,0:self.num_field]
                    for surf_data in self.occ_surfs_int_data]
        # Initial flattened control points for all surfaces of interest
        self.cps_flat = [surf_data.control[:,:,0:self.num_field]
                         .transpose(1,0,2).reshape(-1,self.num_field)
                         for surf_data in self.occ_surfs_int_data]
        self.cp_flat_global = np.concatenate(self.cps_flat, axis=0)

        # create tIGAr BSpline objects to have access to basis functions
        self.bsplines = [BSplines.BSpline(surf_data.degree, surf_data.knots)
                         for surf_data in self.occ_surfs_int_data]

        #### Parametric coordinates related properties
        self.int_num_pts = []
        for int_ind, int_ind_global in enumerate(self.int_indices_diff):
            self.int_num_pts += [self.preprocessor.mortar_nels[int_ind_global]+1]
        # Total number of parametric coordinates for all intersections
        self.xi_size_global = int(np.sum(self.int_num_pts)
                                 *self.num_sides*self.para_dim)
        self.xis = [] # Intersection's parametric coordinates in shape of (x,2)
        self.xis_flat = [] # Intersection's parametric coordinates in shape of (x,1)
        self.xi_sizes = [] # Coordinates sizes for all intersections
        for int_ind, int_ind_global in enumerate(self.int_indices_diff):
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
        for i in range(len(self.int_indices_diff)):
            self.xi_flat_inds += [int(np.sum(self.xi_sizes[0:i]))]
        self.xi_flat_inds += [self.xi_size_global]

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
        self.cp_flat_global[:,field] = cp_flat_single_field
        for i, surf_ind in enumerate(self.int_surf_inds):
            self.cps_flat[i] = self.cp_flat_global[self.cp_flat_inds[i]:
                               self.cp_flat_inds[i+1],:]
            self.cps[i] = self.reshape_CPflat(i, self.cps_flat[i])

    def update_occ_surf(self, int_surf_ind):
        knots = self.occ_surfs_int_data[int_surf_ind].knots
        cp_sub = self.cps[int_surf_ind]
        ik_surf = NURBS(knots, cp_sub)
        occ_surf = ikNURBS2BSpline_surface(ik_surf)
        self.occ_surfs_int[int_surf_ind] = occ_surf

    def update_occ_surfs(self):
        # Update occ surfs, use this function before computing 
        # derivatives dRdxi, which needs updated occ surfs
        for int_surf_ind, surf_ind in enumerate(self.int_surf_inds):
            self.update_occ_surf(int_surf_ind)

    def F_occ(self, int_surf_ind, xi):
        phy_pt = gp_Pnt()
        self.occ_surfs_int[int_surf_ind].D0(xi[0], xi[1], phy_pt)
        return np.array(phy_pt.Coord())

    def F(self, int_surf_ind, xi, cp_flat_sub=None):
        if cp_flat_sub is None:
            cp_flat_sub = self.cps_flat[int_surf_ind]
        nodes_evals = self.bsplines[int_surf_ind].getNodesAndEvals(xi)
        # phy_pt = [0.,0.,0.]
        # for i in range(len(nodes_evals)):
        #     for j in range(self.num_field):
        #         phy_pt[j] += cp[nodes_evals[i][0],j] * nodes_evals[i][1]
        nodes = [item[0] for item in nodes_evals]
        evals = [item[1] for item in nodes_evals]
        cp_sub = cp_flat_sub[nodes]
        phy_pt = np.dot(cp_sub.T, np.array(evals))
        return phy_pt[:]

    def dFdxi(self, int_surf_ind, xi):
        phy_pt = gp_Pnt()
        dFdxi1_vec = gp_Vec()
        dFdxi2_vec = gp_Vec()
        self.occ_surfs_int[int_surf_ind].D1(
             xi[0], xi[1], phy_pt, dFdxi1_vec, dFdxi2_vec)
        phy_coord = np.array(phy_pt.Coord())
        dFdxi = np.zeros((3, self.para_dim))
        dFdxi[:,0] = dFdxi1_vec.Coord()
        dFdxi[:,1] = dFdxi2_vec.Coord()
        return phy_coord, dFdxi

    def dFdCP(self, int_surf_ind, xi, field):
        deriv_mat = np.zeros((self.num_field, self.cp_sizes[int_surf_ind]))
        nodes_evals = self.bsplines[int_surf_ind].getNodesAndEvals(xi)
        # for i in range(len(nodes_evals)):
        #     deriv_mat[field,nodes_evals[i][0]] = nodes_evals[i][1]
        nodes = [item[0] for item in nodes_evals]
        evals = [item[1] for item in nodes_evals]
        deriv_mat[field, nodes] = np.array(evals)
        return deriv_mat

    def local_int_surf_inds(self, int_ind):
        """
        Given local intersection index (intersections that are differentiated)
        return the local intersecting surface indices.
        Similar to s_ind0, s_ind1 = mappling_list[int_ind], but
        the s_ind0 and s_ind1 are the indices from local interseting 
        surface indices list: self.int_surf_inds
        """
        int_ind_global = self.int_indices_diff[int_ind]
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
        num_pts = self.int_num_pts[int_ind]
        edge_tol = 1e-6
        xi_coords = xi_flat_sub.reshape(-1, self.para_dim)
        res = np.zeros(xi_flat_sub.size)
        init_para_coord = self.preprocessor.intersections_para_coords[
                          self.int_indices_diff[int_ind]]
        int_surf_ind0, int_surf_ind1 = self.local_int_surf_inds(int_ind)
        cp_flat_sub0 = self.cps_flat[int_surf_ind0]
        cp_flat_sub1 = self.cps_flat[int_surf_ind1]             

        # Check which end points are on edges
        self.end_xi_ind = np.zeros(self.num_end_pts, dtype='int32')
        self.end_xi_val = np.zeros(self.num_end_pts)
        end_ind_count = 0
        for side in range(self.num_sides):
            for end_ind in range(self.num_end_pts):
                for para_dir in range(self.para_dim):
                    end_xi_ind_temp = side*2*num_pts + \
                                      end_ind*2*(num_pts-1) + para_dir
                    if abs(init_para_coord[side][end_ind*(-1)][para_dir] 
                           - 0.) < edge_tol:
                        self.end_xi_ind[end_ind_count] = int(end_xi_ind_temp)
                        self.end_xi_val[end_ind_count] = 0.
                        end_ind_count += 1
                        break
                    elif abs(init_para_coord[side][end_ind*(-1)][para_dir] 
                             - 1.) < edge_tol:
                        self.end_xi_ind[end_ind_count] = int(end_xi_ind_temp)
                        self.end_xi_val[end_ind_count] = 1.
                        end_ind_count += 1
                        break
            if end_ind_count > 1:
                break

        # Enforce each pair of parametric points from two surfaces
        # have the same physical location.
        for i in range(num_pts):
            res[i*3:(i+1)*3] = self.F(int_surf_ind0, 
                xi_coords[i,0:self.para_dim], cp_flat_sub0) \
                - self.F(int_surf_ind1, 
                xi_coords[i+num_pts,0:self.para_dim], cp_flat_sub1)

        # Enforce two adjacent elements has the same magnitude 
        # in physical space for surface 1.
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

        res[-2] = xi_flat_sub[self.end_xi_ind[0]] - self.end_xi_val[0]
        res[-1] = xi_flat_sub[self.end_xi_ind[1]] - self.end_xi_val[1]
        return res

    def residual(self, xi_flat):
        res_list = []
        for int_ind, int_ind_global in enumerate(self.int_indices_diff):
            xi_flat_sub = xi_flat[self.xi_flat_inds[int_ind]:
                                  self.xi_flat_inds[int_ind+1]]
            res_list += [self.residual_sub(int_ind, xi_flat_sub)]
        res_full = np.concatenate(res_list)
        return res_full

    def solve_xi(self, xi_flat_init):
        xi_root = fsolve(self.residual, x0=xi_flat_init, fprime=self.dRdxi)
        return xi_root

    def dRdxi_sub(self, int_ind, xi_flat_sub, coo=True):
        init_para_coord = self.preprocessor.intersections_para_coords[
                          self.int_indices_diff[int_ind]]
        num_pts = self.int_num_pts[int_ind]
        int_surf_ind0, int_surf_ind1 = self.local_int_surf_inds(int_ind)

        xiA_0 = init_para_coord[0][0]
        xiA_n = init_para_coord[0][-1]

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

            # For lower section:
            if i > 0 and i < num_pts-1:
                deriv_xi[i+ur-1, (i-1)*lb_size[1]:i*lb_size[1]] = \
                    2*np.dot(FAi-FAil, dFAildxi)
                deriv_xi[i+ur-1, i*lb_size[1]:(i+1)*lb_size[1]] = \
                    -2*np.dot(FAir-FAil, dFAidxi)
                deriv_xi[i+ur-1, (i+1)*lb_size[1]:(i+2)*lb_size[1]] = \
                    2*np.dot(FAir-FAi, dFAirdxi)
            elif i == num_pts-1:
                deriv_xi[i+ur-1, self.end_xi_ind[0]] = 1.
                deriv_xi[i+ur, self.end_xi_ind[1]] = 1.
        if coo:
            deriv_xi = coo_matrix(deriv_xi)
        return deriv_xi

    def dRdxi(self, xi_flat, coo=False):
        dRdxi_list = [[None for i in range(len(self.int_indices_diff))]
                      for j in range(len(self.int_indices_diff))]
        for int_ind, int_ind_global in enumerate(self.int_indices_diff):
            xi_flat_sub = xi_flat[self.xi_flat_inds[int_ind]:
                                  self.xi_flat_inds[int_ind+1]]
            dRdxi_list[int_ind][int_ind] = self.dRdxi_sub(int_ind, 
                                           xi_flat_sub, coo=True)
        dRdxi_full = bmat(dRdxi_list, format='coo')
        if not coo:
            dRdxi_full = dRdxi_full.todense()
        return dRdxi_full

    def dRdCP_sub(self, int_ind, xi_flat_sub, field, coo=True):
        # deriv_cp = np.zeros((xi_flat_sub.size, self.cp_size_global))
        xi_coords = xi_flat_sub.reshape(-1, self.para_dim)
        num_pts = self.int_num_pts[int_ind]
        int_surf_inds = self.local_int_surf_inds(int_ind)
        u_size = self.num_field*num_pts
        deriv_cp = [np.zeros((xi_flat_sub.size, self.cp_sizes[int_surf_inds[0]])),
                    np.zeros((xi_flat_sub.size, self.cp_sizes[int_surf_inds[1]]))]

        for i in range(num_pts):
            for side in range(self.num_sides):
                xi_coords_temp = xi_coords[int(i+num_pts*side), 
                                           0:self.para_dim]
                if side == 0:
                    sign = 1.
                elif side == 1:
                    sign = -1.

                if side == 0 and i > 1:
                    dFdcp_temp = dFdcpir
                else:
                    dFdcp_temp = self.dFdCP(int_surf_inds[side], 
                                 xi_coords_temp, field)
                
                deriv_cp[side][i*self.num_field:(i+1)*self.num_field,:] = \
                    dFdcp_temp[:]*sign

                if side == 0 and i == 0:
                    dFdcpi = dFdcp_temp
                elif side == 0 and i > 0:
                    dFdcpil = dFdcpi
                    dFdcpi = dFdcp_temp
                    
            if i > 0 and i < num_pts-1:
                side = 0
                surf_ind = int_surf_inds[side]
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
                deriv_cp[side][u_size+i-1, 0:self.cp_sizes[surf_ind]] = res_vec
        if coo:
            deriv_cp = [coo_matrix(mat) for mat in deriv_cp]
        return deriv_cp

    def dRdCP(self, xi_flat, field, coo=True):
        dRdCP_list = [[None for i in range(self.num_int_surfs)]
                      for j in range(len(self.int_indices_diff))]
        for int_ind, int_ind_global in enumerate(self.int_indices_diff):
            xi_flat_sub = xi_flat[self.xi_flat_inds[int_ind]:
                                  self.xi_flat_inds[int_ind+1]]
            int_surf_ind0, int_surf_ind1 = self.local_int_surf_inds(int_ind)
            dRdCP_sub_temp = self.dRdCP_sub(int_ind, xi_flat_sub, field, coo=True)
            dRdCP_list[int_ind][int_surf_ind0] = dRdCP_sub_temp[0]
            dRdCP_list[int_ind][int_surf_ind1] = dRdCP_sub_temp[1]
        dRdCP_full = bmat(dRdCP_list, format='coo')
        if not coo:
            dRdCP_full = dRdCP_full.todense()
        return dRdCP_full


class CPIGA2XiImOperation(CPIGA2Xi):
    def __init__(self, preprocessor, int_indices_diff=None, opt_field=[0,1,2]):
        super().__init__(preprocessor, int_indices_diff, opt_field)

    def apply_nonlinear(self, xi_flat):
        # print("Running apply nonlinear ...")
        return self.residual(xi_flat)

    def solve_nonlinear(self, xi_flat_init):
        # print("Running solve nonlinear ...")
        return self.solve_xi(xi_flat_init)

    def linearize(self, xi_flat, coo=True):
        # print("Running linearize...")
        self.update_occ_surfs()
        self.dRdxi_mat = self.dRdxi(xi_flat, coo=coo)
        self.dRdCP_mat_list = []
        for i, field in enumerate(self.opt_field):
            self.dRdCP_mat_list += [self.dRdCP(xi_flat, field, coo=coo)]

        # print('dRdxi norm:', np.linalg.norm(self.dRdxi_mat.todense()))
        # print('dRdxi det:', np.linalg.det(self.dRdxi_mat.todense()))
        self.lu_fwd = splu(self.dRdxi_mat.tocsc())
        self.lu_rev = splu(self.dRdxi_mat.T.tocsc())
        return self.dRdxi_mat, self.dRdCP_mat_list

    def apply_linear_fwd(self, d_inputs_array_list=None, 
                         d_outputs_array=None, 
                         d_residuals_array=None):
        """
        ``d_inputs_array_list`` is the list of control points in IGA DoFs
        ``d_outputs_array`` is the intersections' parametric coordinates
        ``d_residuals_array`` is the implicit residuals beteen CP and xi
        """
        if d_residuals_array is not None:
            if d_outputs_array is not None:
                dres_array = self.dRdxi_mat*d_outputs_array
                d_residuals_array[:] += dres_array
            if d_inputs_array_list is not None:
                for i, field in enumerate(self.opt_field):
                    dres_array = self.dRdCP_mat_list[i]*d_inputs_array_list[i]
                    d_residuals_array[:] += dres_array
        return d_residuals_array

    def apply_linear_rev(self, d_inputs_array_list=None, 
                         d_outputs_array=None,
                         d_residuals_array=None):
        """
        ``d_inputs_array_list`` is the list of control points in IGA DoFs
        ``d_outputs_array`` is the intersections' parametric coordinates
        ``d_residuals_array`` is the implicit residuals beteen CP and xi
        """
        if d_residuals_array is not None:
            if d_outputs_array is not None:
                dxi_array = self.dRdxi_mat.T*d_residuals_array
                d_outputs_array[:] += dxi_array
            if d_inputs_array_list is not None:
                for i, field in enumerate(self.opt_field):
                    dcp_iga_array = self.dRdCP_mat_list[i].T*d_residuals_array
                    d_inputs_array_list[i][:] += dcp_iga_array
        return d_inputs_array_list, d_outputs_array

    def solve_linear_fwd(self, d_outputs_array, d_residuals_array):
        d_outputs_array[:] = self.lu_fwd.solve(d_residuals_array)
        return d_outputs_array

    def solve_linear_rev(self, d_outputs_array, d_residuals_array):
        d_residuals_array[:] = self.lu_rev.solve(d_outputs_array)
        return d_residuals_array


if __name__ == '__main__':
    # from GOLDFISH.tests.test_tbeam import nonmatching_opt
    # from GOLDFISH.tests.test_slr import nonmatching_opt
    from PENGoLINS.occ_preprocessing import *

    filename_igs = "../geometry/init_Tbeam_geom_moved.igs"
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
    int_data_filename = "int_data.npz"
    if os.path.isfile(int_data_filename):
        preprocessor.load_intersections_data(int_data_filename)
    else:
        preprocessor.compute_intersections(mortar_refine=2)
        preprocessor.save_intersections_data(int_data_filename)

    if mpirank == 0:
        print("Total DoFs:", preprocessor.total_DoFs)
        print("Number of intersections:", preprocessor.num_intersections_all)

    cpiga2xi_imop = CPIGA2XiImOperation(preprocessor)

    int_ind = 0
    xi_flat = cpiga2xi_imop.xis_flat[int_ind]
    cpiga2xi_imop.apply_nonlinear(xi_flat)
    cpiga2xi_imop.solve_nonlinear(xi_flat)
    cpiga2xi_imop.linearize(xi_flat)