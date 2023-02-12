import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, newton_krylov
from PENGoLINS.occ_utils import *
from PENGoLINS.igakit_utils import *
from tIGAr import BSplines
from time import perf_counter
np.set_printoptions(precision=4)

p = 3  # Spline degree
L = 10  # Length of spline surface

#### Surface 1 ####
pts1 = [[-L,0,-L/2], [0,0,-L/2],
        [-L,0,L/2], [0,0,L/2]]

num_el = 8
knots = np.linspace(0,1,num_el+1)[1:-1]
# Create distorted NURBS surface
distort_mag1 = 3
c1 = line(pts1[0], pts1[1])
c1.elevate(0,p-c1.degree[0])
c1.control[1][1] = -distort_mag1
c1.control[3][1] = distort_mag1
c1.refine(0, knots)
t1 = line(pts1[1], pts1[3])
t1.elevate(0,p-t1.degree[0])
t1.control[1][1] = -distort_mag1
t1.control[3][1] = distort_mag1
t1.refine(0, knots)
ik_surf1 = sweep(c1, t1)
ik_surf1.move(L/2, axis=2)

#### Surface 2 ####
pts2 = [[-L/2,-L/2,0], [L/2,-L/2,0],
        [-L/2,L/2,0], [L/2,L/2,0]]
distort_mag2 = 2
c2 = line(pts2[0], pts2[1])
c2.elevate(0,p-c2.degree[0])
c2.control[1][2] = -distort_mag2
c2.control[3][2] = distort_mag2
c2.refine(0, knots)
t2 = line(pts2[1], pts2[3])
t2.elevate(0,p-t2.degree[0])
t2.control[1][2] = -distort_mag2
t2.control[3][2] = distort_mag2
t2.refine(0, knots)
ik_surf2 = sweep(c2, t2)
ik_surf2.move(-8, axis=0)
ik_surf2.move(6, axis=1)

# Convert igakit NURBS instance to OCC BSplineSurface
surf1 = ikNURBS2BSpline_surface(ik_surf1)
surf2 = ikNURBS2BSpline_surface(ik_surf2)

class SurfSurfInt(object):

    def __init__(self, occ_surf1, occ_surf2, num_pts=20, tol=1e-6):
        self.occ_surf1 = occ_surf1
        self.occ_surf2 = occ_surf2
        self.occ_surfs = [self.occ_surf1, self.occ_surf2]
        self.num_pts = num_pts
        self.tol = tol
        self.num_sides = 2
        self.num_end_pts = 2
        self.num_para_dir = 2
        self.num_phy_dir = 3
        # self.surf1_data = BSplineSurfaceData(self.occ_surf1)
        # self.surf2_data = BSplineSurfaceData(self.occ_surf2)
        # self.bspline1 = BSplines.BSpline(self.surf1_data.degree, 
        #                                  self.surf1_data.knots)
        # self.bspline2 = BSplines.BSpline(self.surf2_data.degree, 
        #                                  self.surf2_data.knots)

        # self.surf1_cp = self.surf1_data.control[:,:,0:3]
        # self.surf2_cp = self.surf2_data.control[:,:,0:3]
        # self.cp_shapes = [self.surf1_cp.shape[0:2], self.surf2_cp.shape[0:2]]
        # self.cp_sizes = [self.surf1_cp.shape[0]*self.surf1_cp.shape[1],
        #                  self.surf2_cp.shape[0]*self.surf2_cp.shape[1]]
        # self.cp_size_global = np.sum(self.cp_sizes)

        self.surfs_data = [BSplineSurfaceData(surf) for surf in self.occ_surfs]
        self.bsplines = [BSplines.BSpline(surf_data.degree, surf_data.knots)
                         for surf_data in self.surfs_data]
        self.cp_shapes = [surf_data.control.shape[0:self.num_para_dir]
                          for surf_data in self.surfs_data]
        self.cp_sizes = [cp_shape[0]*cp_shape[1] for cp_shape in self.cp_shapes]
        self.cp_size_global = np.sum(self.cp_sizes)

        self.cps_flat = [surf_data.control[:,:,0:self.num_phy_dir]
                         .transpose(1,0,2).reshape(-1,self.num_phy_dir)
                         for surf_data in self.surfs_data]
        self.cp_flat_global = np.concatenate(self.cps_flat, axis=0)

        surf_int = GeomAPI_IntSS(self.occ_surf1, self.occ_surf2, self.tol)
        self.num_int = surf_int.NbLines()
        self.int_curves = [surf_int.Line(i+1) for i in range(self.num_int)]

        self.ints_phy_coord_equidist_para = \
            self._intersections_phy_coord_equidist_para()
        self.ints_para_coord_equidist_para = \
            self._intersections_para_coord_equidist_para()

    def _intersections_phy_coord_equidist_para(self):
        phy_coord_list = []
        for i in range(self.num_int):
            s0 = self.int_curves[i].FirstParameter()
            sn = self.int_curves[i].LastParameter()
            para_coord = np.linspace(s0, sn, self.num_pts)
            phy_coord = np.zeros((self.num_pts, 3))
            pt_temp = gp_Pnt()
            for j in range(self.num_pts):
                self.int_curves[i].D0(para_coord[j], pt_temp)
                phy_coord[j] = pt_temp.Coord()
            phy_coord_list += [phy_coord,]
        return phy_coord_list

    def _intersections_para_coord_equidist_para(self):
        phy_coord_list = self.ints_phy_coord_equidist_para
        para_coord_list = []
        for i in range(self.num_int):
            para_coord1 = parametric_coord(phy_coord_list[i], self.occ_surf1)
            para_coord2 = parametric_coord(phy_coord_list[i], self.occ_surf2)
            para_coord_list += [[para_coord1, para_coord2],]
        return para_coord_list

    def coupled_residual(self, xi, int_ind, cp_global=None):
        """
        Returns residuals of coupled system when solving interior
        parametric coordinates for intersection curve

        Parameters
        ----------
        xi : ndarray, size: num_pts*4
            xi = [xiA_11, xiA_12, xiA_21, xiA_22, ..., xiA_(n-1)1, xiA_(n-1)2,
                  xiB_11, xiB_12, xiB_21, xiB_22, ..., xiB_(n-1)1, xiB_(n-1)2]
            First subscript is interior point index, {1, n-1}
            Second subscript is component index, {1, 2}
        int_ind : intersection index, int

        Returns
        -------
        res : ndarray: size: num_pts*4
        """
        edge_tol = 1e-8
        # Check which end points are on edges
        self.end_xi_ind = np.zeros(self.num_end_pts, dtype='int32')
        self.end_xi_val = np.zeros(self.num_end_pts)
        end_ind_count = 0
        for side in range(self.num_sides):
            for end_ind in range(self.num_end_pts):
                for para_dir in range(self.num_para_dir):
                    end_xi_ind_temp = side*2*self.num_pts + \
                                      end_ind*2*(self.num_pts-1) + para_dir
                    if abs(self.ints_para_coord_equidist_para[int_ind]
                           [side][end_ind*(-1)][para_dir] - 0.) < edge_tol:
                        self.end_xi_ind[end_ind_count] = int(end_xi_ind_temp)
                        self.end_xi_val[end_ind_count] = 0.
                        end_ind_count += 1
                        break
                    elif abs(self.ints_para_coord_equidist_para[int_ind]
                             [side][end_ind*(-1)][para_dir] - 1.) < edge_tol:
                        self.end_xi_ind[end_ind_count] = int(end_xi_ind_temp)
                        self.end_xi_val[end_ind_count] = 1.
                        end_ind_count += 1
                        break
            if end_ind_count > 1:
                break

        xi_coords = xi.reshape(-1, self.num_para_dir)
        res = np.zeros(xi.size)

        if cp_global is not None:
            cp1 = cp_global[0:self.cp_sizes[0], :]
            cp2 = cp_global[self.cp_sizes[0]:int(np.sum(self.cp_sizes)), :]
        else:
            cp1 = None
            cp2 = None

        # Enforce each pair of parametric points from two surfaces
        # have the same physical location.
        for i in range(self.num_pts):
            res[i*3:(i+1)*3] = self.F(0, xi_coords[i,0:self.num_para_dir], cp=cp1) \
                - self.F(1, xi_coords[i+self.num_pts,0:self.num_para_dir], cp=cp2)

        # Enforce two adjacent elements has the same magnitude 
        # in physical space for surface 1.
        for i in range(1, self.num_pts-1):
            phy_coord1 = self.F(0, xi_coords[i-1,0:self.num_para_dir], cp=cp1)
            phy_coord2 = self.F(0, xi_coords[i,0:self.num_para_dir], cp=cp1)
            phy_coord3 = self.F(0, xi_coords[i+1,0:self.num_para_dir], cp=cp1)
            diff1 = phy_coord2 - phy_coord1
            diff2 = phy_coord3 - phy_coord2
            res[i+self.num_pts*3-1] = np.dot(diff2, diff2) \
                                    - np.dot(diff1, diff1)

        res[-2] = xi[self.end_xi_ind[0]] - self.end_xi_val[0]
        res[-1] = xi[self.end_xi_ind[1]] - self.end_xi_val[1]
        return res

    def F_occ(self, surf_ind, xi):
        phy_pt = gp_Pnt()
        self.occ_surfs[surf_ind].D0(xi[0], xi[1], phy_pt)
        return np.array(phy_pt.Coord())

    def F(self, surf_ind, xi, cp=None):
        # phy_pt = [0.,0.,0.]
        if cp is None:
            cp = self.cps_flat[surf_ind]
        nodes_evals = self.bsplines[surf_ind].getNodesAndEvals(xi)
        # for i in range(len(nodes_evals)):
        #     for j in range(self.num_phy_dir):
        #         phy_pt[j] += cp[nodes_evals[i][0],j] * nodes_evals[i][1]
        nodes = [item[0] for item in nodes_evals]
        evals = [item[1] for item in nodes_evals]
        cp_sub = cp[nodes]
        phy_pt = np.dot(cp_sub.T, np.array(evals))
        return phy_pt[:]
        # return np.array(phy_pt)

    def dFdxi(self, surf_ind, xi):
        phy_pt = gp_Pnt()
        dFdxi1_vec = gp_Vec()
        dFdxi2_vec = gp_Vec()
        self.occ_surfs[surf_ind].D1(xi[0], xi[1], phy_pt, 
                                    dFdxi1_vec, dFdxi2_vec)
        phy_coord = np.array(phy_pt.Coord())
        dFdxi = np.zeros((3, self.num_para_dir))
        dFdxi[:,0] = dFdxi1_vec.Coord()
        dFdxi[:,1] = dFdxi2_vec.Coord()
        return phy_coord, dFdxi

    def dFdcp(self, surf_ind, xi, field):
        res_mat = np.zeros((self.num_phy_dir, self.cp_sizes[surf_ind]))
        nodes_evals = self.bsplines[surf_ind].getNodesAndEvals(xi)
        # for i in range(len(nodes_evals)):
        #     res_mat[field,nodes_evals[i][0]] = nodes_evals[i][1]
        nodes = [item[0] for item in nodes_evals]
        evals = [item[1] for item in nodes_evals]
        res_mat[field, nodes] = np.array(evals)
        return res_mat

    def dFdcp_FD(self, surf_ind, xi, field, h=1e-9):
        res_mat = np.zeros((self.num_phy_dir, self.cp_sizes[surf_ind]))
        cp_init = self.cps_flat[surf_ind]
        F_init = self.F(surf_ind, xi, cp=cp_init)
        for i in range(self.cp_sizes[surf_ind]):
            perturb = np.zeros(self.cp_sizes[surf_ind])
            perturb[i] = h
            cp_perturb = cp_init.copy()
            cp_perturb[:,field] = cp_perturb[:,field] + perturb
            F_perturb = self.F(surf_ind, xi, cp=cp_perturb)
            res_mat[:,i] = (F_perturb - F_init)/h
        return res_mat


    def dRdxi(self, xi, int_ind):
        xiA_0 = self.ints_para_coord_equidist_para[int_ind][0][0]
        xiA_n = self.ints_para_coord_equidist_para[int_ind][0][-1]

        xi_coords = xi.reshape(-1, self.num_para_dir)
        deriv_xi = np.zeros((xi.size, xi.size))

        # Upper sectin block size
        ub_size = [self.num_phy_dir, self.num_para_dir]
        # Lower section block size
        lb_size = [1, self.num_para_dir]
        # Number of upper section rows
        ur = ub_size[0]*self.num_pts
        # Number of left section columns
        lc = ub_size[1]*self.num_pts


        for i in range(self.num_pts):
            if i == 0:
                # For left end point
                FAi, dFAidxi = \
                    self.dFdxi(0, xi_coords[i, 0:self.num_para_dir])
                FBi, dFBidxi = self.dFdxi(1, 
                     xi_coords[i+self.num_pts, 0:self.num_para_dir])
                # 1st point and derivative
                FAir, dFAirdxi = \
                    self.dFdxi(0, xi_coords[i+1, 0:self.num_para_dir])
                FBir, dFBirdxi = self.dFdxi(1, 
                     xi_coords[i+1+self.num_pts, 0:self.num_para_dir])
            elif i == self.num_pts-1:
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
                    self.dFdxi(0, xi_coords[i+1, 0:self.num_para_dir])
                FBir, dFBirdxi = self.dFdxi(1, 
                     xi_coords[i+1+self.num_pts, 0:self.num_para_dir])

            # For upper section
            deriv_xi[i*ub_size[0]:(i+1)*ub_size[0], 
                     i*ub_size[1]:(i+1)*ub_size[1]] = dFAidxi
            deriv_xi[i*ub_size[0]:(i+1)*ub_size[0], 
                lc+i*ub_size[1]:lc+(i+1)*ub_size[1]] = -dFBidxi

            # For lower section:
            if i > 0 and i < self.num_pts-1:
                deriv_xi[i+ur-1, (i-1)*lb_size[1]:i*lb_size[1]] = \
                    2*np.dot(FAi-FAil, dFAildxi)
                deriv_xi[i+ur-1, i*lb_size[1]:(i+1)*lb_size[1]] = \
                    -2*np.dot(FAir-FAil, dFAidxi)
                deriv_xi[i+ur-1, (i+1)*lb_size[1]:(i+2)*lb_size[1]] = \
                    2*np.dot(FAir-FAi, dFAirdxi)
            elif i == self.num_pts-1:
                deriv_xi[i+ur-1, self.end_xi_ind[0]] = 1.
                deriv_xi[i+ur, self.end_xi_ind[1]] = 1.
        return deriv_xi

    def dRdxi_FD(self, xi, int_ind, h=1e-9):
        cou_res = self.coupled_residual(xi, int_ind)
        deriv_xi_FD = np.zeros((xi.size, xi.size))
        for i in range(self.num_pts*4):
            perturb = np.zeros(xi.size)
            perturb[i] = h
            xi_perturb = xi + perturb
            cou_res_perturb = self.coupled_residual(xi_perturb, int_ind)
            deriv_xi_FD[:,i] = (cou_res_perturb - cou_res)/h
        return deriv_xi_FD

    def dRdcp(self, xi, ind, field):
        deriv_cp = np.zeros((xi.size, self.cp_size_global))

        xi_coords = xi.reshape(-1, self.num_para_dir)
        u_size = self.num_phy_dir*self.num_pts

        for i in range(self.num_pts):
            for side in range(self.num_sides):
            # for side, surf_ind in enumerate(self.mapping_list[int_ind])
                xi_coords_temp = xi_coords[int(i+self.num_pts*side), 0:self.num_para_dir]
                num_row_shift = int(np.sum(self.cp_sizes[0:side]))

                if side == 0:
                    sign = 1.
                elif side == 1:
                    sign = -1.

                if side == 0 and i > 1:
                    dFdcp_temp = dFdcpir
                else:
                    dFdcp_temp = self.dFdcp(side, xi_coords_temp, field)
                
                deriv_cp[i*self.num_phy_dir:(i+1)*self.num_phy_dir,
                         num_row_shift:num_row_shift+self.cp_sizes[side]] = \
                         dFdcp_temp[:]*sign

                if side == 0 and i == 0:
                    dFdcpi = dFdcp_temp
                elif side == 0 and i > 0:
                    dFdcpil = dFdcpi
                    dFdcpi = dFdcp_temp
                    
            if i > 0 and i < self.num_pts-1:
                surf_ind = 0
                # surf_ind = self.mapping_list[int_ind][0]
                xi_coords_il = xi_coords[i-1, 0:self.num_para_dir]
                xi_coords_i = xi_coords[i, 0:self.num_para_dir]
                xi_coords_ir = xi_coords[i+1, 0:self.num_para_dir]
                dFdcpir = self.dFdcp(surf_ind, xi_coords_ir, field)
                Fil = self.F(surf_ind, xi_coords_il)
                Fi = self.F(surf_ind, xi_coords_i)
                Fir = self.F(surf_ind, xi_coords_ir)
                # res_vec = 2*(np.dot(Fir, dFdcpir) - np.dot(Fir, dFdcpi) 
                #            - np.dot(Fi, dFdcpir) + np.dot(Fi, dFdcpil) 
                #            + np.dot(Fil, dFdcpi) - np.dot(Fil, dFdcpil))
                res_vec = 2*(np.dot(Fir-Fi, dFdcpir-dFdcpi) 
                           - np.dot(Fi-Fil, dFdcpi-dFdcpil))
                deriv_cp[u_size+i-1, 0:self.cp_sizes[surf_ind]] = res_vec
        return deriv_cp

    def dRdcp_FD(self, xi, int_ind, field, h=1e-9):
        cp_init = self.cp_flat_global
        cou_res = self.coupled_residual(xi, int_ind, cp_global=cp_init)
        deriv_cp_FD = np.zeros((xi.size, self.cp_size_global))
        for i in range(self.cp_size_global):
            perturb = np.zeros(self.cp_size_global)
            perturb[i] = h
            cp_perturb = cp_init.copy()
            cp_perturb[:,field] = cp_perturb[:,field] + perturb
            cou_res_perturb = self.coupled_residual(xi, int_ind, cp_global=cp_perturb)
            deriv_cp_FD[:,i] = (cou_res_perturb - cou_res)/h
        return deriv_cp_FD

# Test for SurfSurfInt class
num_pts = 64
int_ind = 0
int_ss = SurfSurfInt(surf1, surf2, num_pts)
xi0 = np.concatenate([int_ss.ints_para_coord_equidist_para[int_ind][0], 
                      int_ss.ints_para_coord_equidist_para[int_ind][1]], 
                     axis=0)
xi0_flat = xi0.reshape(-1,1)[:,0]

xi0_disturb = xi0.copy()
xi0_disturb = xi0_disturb \
            + np.random.random(xi0_disturb.shape)*5e-2
xi0_disturb_flat = xi0_disturb.reshape(-1,1)[:,0]

print("Solving nonlinear residual...")
xi_root = fsolve(int_ss.coupled_residual, x0=xi0_disturb_flat, args=(int_ind),
                 fprime=int_ss.dRdxi)

# Check function residual after solve
res_before = int_ss.coupled_residual(xi0_disturb_flat, int_ind)
res_after = int_ss.coupled_residual(xi_root, int_ind)
res0 = int_ss.coupled_residual(xi0.reshape(-1,1)[:,0], int_ind)
print("Residual before solve:", np.linalg.norm(res_before))
print("Residual after solve:", np.linalg.norm(res_after))
print("Residual initial from OPENCASCADE:", np.linalg.norm(res0))

# Compute derivative w.r.t. xi

time0 = perf_counter()
dRdxi = int_ss.dRdxi(xi_root, int_ind)
time1 = perf_counter()
dRdxi_FD = int_ss.dRdxi_FD(xi_root, int_ind)
time2 = perf_counter()

time_ana = time1 - time0
time_fd = time2 - time1
print("Time of dRdxi analytical: {:8.4f}, Time of dRdxi FD: {:8.4f}"
      .format(time_ana, time_fd))

dRdxi_diff_norm = np.linalg.norm(dRdxi-dRdxi_FD)
# print("Error norm of derivative w.r.t. xi compared to forward FD:", 
#       dRdxi_diff_norm)
print("Relative error norm of derivative w.r.t. xi compared to forward FD:", 
      dRdxi_diff_norm/np.linalg.norm(dRdxi_FD))

# # For checking derivatives
# dRdxi_diff = np.abs(dRdxi_FD - dRdxi)
# for i in range(num_pts*4):
#     for j in range(num_pts*4):
#         if dRdxi_diff[i][j] > 1e-5:
#             print("i:", i, ", j:", j, ", diff:", dRdxi_diff[i,j])
#             print("        dRdxi value:", dRdxi[i,j])
#             print("        dRdxi_FD value:", dRdxi_FD[i,j])


# # Check dRdcp
time0 = perf_counter()
dRdcp = int_ss.dRdcp(xi_root, int_ind, field=1)
time1 = perf_counter()
dRdcp_FD = int_ss.dRdcp_FD(xi_root, int_ind, field=1)
time2 = perf_counter()

time_ana = time1 - time0
time_fd = time2 - time1
print("Time of dRdcp analytical: {:8.4f}, Time of dRdcp FD: {:8.4f}"
      .format(time_ana, time_fd))

err = dRdcp - dRdcp_FD
dRdcp_diff_norm = np.linalg.norm(err)
# print("Error norm of derivative w.r.t. CP compared to forward FD:", 
#       dRdcp_diff_norm)
print("Relative error norm of derivative w.r.t. CP compared to forward FD:", 
      dRdcp_diff_norm/np.linalg.norm(dRdcp_FD))


# # Check dFdcp
# xi = xi_root[2:4]
# surf_ind = 0
# dFdcp00 = int_ss.dFdcp(surf_ind, xi, field=1)
# dFdcp00_FD = int_ss.dFdcp_FD(surf_ind, xi, field=1)
# err = dFdcp00 - dFdcp00_FD
# dFdcp_diff_norm = np.linalg.norm(err)

# print("Error norm of F w.r.t. CP compared to forward FD:", 
#       dFdcp_diff_norm)
# print("Relative error norm of F w.r.t. CP compared to forward FD:", 
#       dFdcp_diff_norm/np.linalg.norm(dFdcp00_FD))

# ######################################################
# xi = [0.82, 0.66]
# surf1data = BSplineSurfaceData(surf1)
# bs0 = BSplines.BSpline(surf1data.degree, surf1data.knots)
# ne = bs0.getNodesAndEvals(xi)
# s0cp = surf1data.control[:,:,0:3]
# s0cpf = s0cp.transpose(1,0,2).reshape(-1,3)

# t0 = [0.,0.,0.]
# for i in range(len(ne)):
#     for j in range(3):
#         t0[j] += s0cpf[ne[i][0],j] * ne[i][1]

# t2pt = gp_Pnt()
# surf1.D0(xi[0], xi[1], t2pt)
# t2 = np.array(t2pt.Coord())

# print("t0:", t0)
# print("t2:", t2)
# print(aaa)
# ######################################################