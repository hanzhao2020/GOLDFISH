import numpy as np
from scipy.optimize import fsolve, newton_krylov
from PENGoLINS.occ_utils import *
from PENGoLINS.igakit_utils import *

p = 4  # Spline degree
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

    def __init__(self, occ_surf1, occ_surf2, num_interior_pts=20, tol=1e-6):
        self.occ_surf1 = occ_surf1
        self.occ_surf2 = occ_surf2
        self.occ_surfs = [self.occ_surf1, self.occ_surf2]
        self.num_interior_pts = num_interior_pts
        self.tol = tol

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
            para_coord = np.linspace(s0, sn, self.num_interior_pts+2)
            phy_coord = np.zeros((self.num_interior_pts+2, 3))
            pt_temp = gp_Pnt()
            for j in range(self.num_interior_pts+2):
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

    def coupled_residual(self, xi, int_ind):
        """
        Returns residuals of coupled system when solving interior
        parametric coordinates for intersection curve

        Parameters
        ----------
        xi : ndarray, size: num_interior_pts*4
            xi = [xiA_11, xiA_12, xiB_11, xiB_12, 
                  xiA_21, xiA_22, xiB_21, xiB_22, 
                  ..., 
                  xiA_(n-1)1, xiA_(n-1)2, xiB_(n-1)1, xiB_(n-1)2]
            First subscript is interior point index, {1, n-1}
            Second subscript is component index, {1, 2}
        int_ind : intersection index, int

        Returns
        -------
        res : ndarray: size: num_interior_pts*4
        """
        xiA_0 = self.ints_para_coord_equidist_para[int_ind][0][0]
        xiA_n = self.ints_para_coord_equidist_para[int_ind][0][-1]

        xi_coords = xi.reshape(-1,4)
        res = np.zeros(xi.size)

        # Enforce each pair of parametric points from two surfaces
        # have the same physical location.
        for i in range(self.num_interior_pts):
            res[i*3:(i+1)*3] = self.F(0, xi_coords[i,0:2]) \
                             - self.F(1, xi_coords[i,2:4])

        # Enforce two adjacent elements has the same magnitude 
        # in physical space for surface 1.
        for i in range(self.num_interior_pts):
            if i == 0:
                phy_coord1 = self.F(0, xiA_0)
                phy_coord2 = self.F(0, xi_coords[i,0:2])
                phy_coord3 = self.F(0, xi_coords[i+1,0:2])
            elif i == self.num_interior_pts-1:
                phy_coord1 = self.F(0, xi_coords[i-1,0:2])
                phy_coord2 = self.F(0, xi_coords[i,0:2])
                phy_coord3 = self.F(0, xiA_n)
            else:
                phy_coord1 = self.F(0, xi_coords[i-1,0:2])
                phy_coord2 = self.F(0, xi_coords[i,0:2])
                phy_coord3 = self.F(0, xi_coords[i+1,0:2])

            diff1 = phy_coord2 - phy_coord1
            diff2 = phy_coord3 - phy_coord2
            res[i+self.num_interior_pts*3] = np.dot(diff2, diff2) \
                                           - np.dot(diff1, diff1) 
        return res

    def F(self, surf_ind, xi):
        phy_pt = gp_Pnt()
        self.occ_surfs[surf_ind].D0(xi[0], xi[1], phy_pt)
        return np.array(phy_pt.Coord())

    def dFdxi(self, surf_ind, xi):
        phy_pt = gp_Pnt()
        dFdxi1_vec = gp_Vec()
        dFdxi2_vec = gp_Vec()
        self.occ_surfs[surf_ind].D1(xi[0], xi[1], phy_pt, 
                                    dFdxi1_vec, dFdxi2_vec)
        phy_coord = np.array(phy_pt.Coord())
        dFdxi = np.zeros((3,2))
        dFdxi[:,0] = dFdxi1_vec.Coord()
        dFdxi[:,1] = dFdxi2_vec.Coord()
        return phy_coord, dFdxi

    def dRdxi(self, xi, int_ind):
        xiA_0 = self.ints_para_coord_equidist_para[int_ind][0][0]
        xiA_n = self.ints_para_coord_equidist_para[int_ind][0][-1]

        xi_coords = xi.reshape(-1,4)
        deriv_xi = np.zeros((xi.size, xi.size))

        # Number of upper section rows
        ur = 3*self.num_interior_pts
        # Upper sectin block size
        ub_size = [3,4]
        # Lower section block size
        lb_size = [1,4]

        for i in range(self.num_interior_pts):
            # Frist compute derivative of geometric mapping
            if i == 0:
                # For left end point
                FAil = self.F(0, xiA_0)
                # 0-th point and derivative
                FAi, dFAidxi = self.dFdxi(0, xi_coords[i, 0:2])
                FBi, dFBidxi = self.dFdxi(1, xi_coords[i, 2:4])
                # 1-th point and derivative
                FAir, dFAirdxi = self.dFdxi(0, xi_coords[i+1, 0:2])
                FBir, dFBirdxi = self.dFdxi(1, xi_coords[i+1, 2:4])

            elif i == self.num_interior_pts-1:
                # for (n-1)-th point and derivative
                FAil = FAi 
                dFAildxi = dFAidxi 
                FBil = FBi
                dFBildxi = dFBidxi
                # For n-th point and derivative
                FAi = FAir
                dFAidxi = dFAirdxi
                FBi = FBir
                dFBidxi = dFBirdxi
                # For right end point
                FAir = self.F(0, xiA_n)
            else:
                # For (i-1)-th point and derivative
                FAil = FAi 
                dFAildxi = dFAidxi 
                FBil = FBi
                dFBildxi = dFBidxi
                # For i-th point and derivative
                FAi = FAir
                dFAidxi = dFAirdxi
                FBi = FBir
                dFBidxi = dFBirdxi
                # For (i+1)-th point and derivative
                FAir, dFAirdxi = self.dFdxi(0, xi_coords[i+1, 0:2])
                FBir, dFBirdxi = self.dFdxi(1, xi_coords[i+1, 2:4])

            # For upper section
            deriv_xi[i*ub_size[0]:(i+1)*ub_size[0], 
                     i*ub_size[1]:(i+1)*ub_size[1]] = \
                np.concatenate([dFAidxi, -dFBidxi], axis=1)

            # For lower section
            if i == 0:
                deriv_xi[i+ur, i*lb_size[1]:(i+1)*lb_size[1]][0:2] = \
                    -2*np.dot(FAir-FAil, dFAidxi)
                deriv_xi[i+ur, (i+1)*lb_size[1]:(i+2)*lb_size[1]][0:2] = \
                    2*np.dot(FAir-FAi, dFAirdxi)
            elif i == self.num_interior_pts-1:
                deriv_xi[i+ur, (i-1)*lb_size[1]:i*lb_size[1]][0:2] = \
                    2*np.dot(FAi-FAil, dFAildxi)
                deriv_xi[i+ur, i*lb_size[1]:(i+2)*lb_size[1]][0:2] = \
                    -2*np.dot(FAir-FAil, dFAidxi)
            else:
                deriv_xi[i+ur, (i-1)*lb_size[1]:i*lb_size[1]][0:2] = \
                    2*np.dot(FAi-FAil, dFAildxi)
                deriv_xi[i+ur, i*lb_size[1]:(i+2)*lb_size[1]][0:2] = \
                    -2*np.dot(FAir-FAil, dFAidxi)
                deriv_xi[i+ur, (i+1)*lb_size[1]:(i+2)*lb_size[1]][0:2] = \
                    2*np.dot(FAir-FAi, dFAirdxi)

        return deriv_xi

    def dRdxi_FD(self, xi, int_ind, h=1e-9):
        cou_res = self.coupled_residual(xi, int_ind)
        deriv_xi_FD = np.zeros((xi.size, xi.size))

        for i in range(self.num_interior_pts*4):
            perturb = np.zeros(xi.size)
            perturb[i] = h
            xi_perturb = xi + perturb
            cou_res_perturb = self.coupled_residual(xi_perturb, int_ind)
            deriv_xi_FD[:,i] = (cou_res_perturb - cou_res)/h

        return deriv_xi_FD

    def dRdcp(self):
        pass

    def dRdcp_FD(self):
        pass

# Test for SurfSurfInt class
num_interior_pts = 200
int_ind = 0
int_ss = SurfSurfInt(surf1, surf2, num_interior_pts)
xi0 = np.concatenate([int_ss.ints_para_coord_equidist_para[int_ind][0][1:-1], 
                      int_ss.ints_para_coord_equidist_para[int_ind][1][1:-1]], 
                     axis=1)
xi0_flat = xi0.reshape(-1,1)[:,0]

xi0_disturb = xi0.copy()
xi0_disturb = xi0_disturb \
            + np.random.random(xi0_disturb.shape)*2e-2
xi0_disturb_flat = xi0_disturb.reshape(-1,1)[:,0]

print("Solving nonlinear residual...")
xi_root = fsolve(int_ss.coupled_residual, x0=xi0_disturb_flat, args=(int_ind),
                 fprime=int_ss.dRdxi)

# Check function residual after solve
res_before = int_ss.coupled_residual(xi0_disturb_flat, int_ind)
res_after = int_ss.coupled_residual(xi_root, int_ind)
print("Residual before solve:", np.linalg.norm(res_before))
print("Residual after solve:", np.linalg.norm(res_after))

# Compute derivative w.r.t. xi
dRdxi = int_ss.dRdxi(xi_root, int_ind)
dRdxi_FD = int_ss.dRdxi_FD(xi_root, int_ind)
dRdxi_diff_norm = np.linalg.norm(dRdxi-dRdxi_FD)
print("Error norm of derivative w.r.t. xi compared to forward FD:", 
      dRdxi_diff_norm)

# For checking derivatives
dRdxi_diff = np.abs(dRdxi_FD - dRdxi)
for i in range(num_interior_pts*4):
    for j in range(num_interior_pts*4):
        if dRdxi_diff[i][j] > 1e-5:
            print("i:", i, ", j:", j, ", diff:", dRdxi_diff[i,j])
            print("        dRdxi value:", dRdxi[i,j])
            print("        dRdxi_FD value:", dRdxi_FD[i,j])