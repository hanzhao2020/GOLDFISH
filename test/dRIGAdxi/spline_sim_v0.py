from tIGAr.BSplines import *
from tIGAr.NURBS import *
from ShNAPr.SVK import *
import igakit.cad as ikcad

from PENGoLINS.occ_preprocessing import *
from PENGoLINS.nonmatching_coupling import *
from PENGoLINS.nurbs4occ import *
from PENGoLINS.nonmatching_shell import *

from GOLDFISH.utils.opt_utils import *

import matplotlib.pyplot as plt

# from pe4opt import *

np.set_printoptions(precision=3)

def Lambda_tilde(R, num_pts, phy_dim, para_dim, order=0):
    """
    order = 0 or 1
    """
    R_mat = zero_petsc_mat(num_pts*para_dim, 
            num_pts*phy_dim*para_dim**(order+1), PREALLOC=R.size)

    if order == 0:
        for i in range(num_pts):
            for j in range(phy_dim):
                row_ind0 = i*para_dim
                row_ind1 = i*para_dim+1
                col_ind0 = i*(phy_dim*para_dim**(order+1))+j*para_dim**(order+1)
                col_ind1 = i*(phy_dim*para_dim**(order+1))+j*para_dim**(order+1)+1
                vec_ind0 = i*phy_dim*para_dim**order + j*para_dim**order
                R_mat.setValue(row_ind0, col_ind0, R[vec_ind0])
                R_mat.setValue(row_ind1, col_ind1, R[vec_ind0])
    elif order == 1:
        for i in range(num_pts):
            for j in range(phy_dim):
                row_ind0 = i*para_dim
                row_ind1 = i*para_dim+1
                col_ind00 = i*(phy_dim*para_dim**(order+1))+j*para_dim**(order+1)
                col_ind01 = i*(phy_dim*para_dim**(order+1))+j*para_dim**(order+1)+1
                col_ind10 = i*(phy_dim*para_dim**(order+1))+j*para_dim**(order+1)+2
                col_ind11 = i*(phy_dim*para_dim**(order+1))+j*para_dim**(order+1)+3
                vec_ind0 = i*phy_dim*para_dim**order + j*para_dim**order
                vec_ind1 = i*phy_dim*para_dim**order + j*para_dim**order+1
                R_mat.setValue(row_ind0, col_ind00, R[vec_ind0])
                R_mat.setValue(row_ind0, col_ind01, R[vec_ind1])
                R_mat.setValue(row_ind1, col_ind10, R[vec_ind0])
                R_mat.setValue(row_ind1, col_ind11, R[vec_ind1])
    else:
        raise RuntimeError("Order {} is not supported.".format(order))
    R_mat.assemble()
    return R_mat

def Lambda(Au, num_pts, phy_dim, para_dim, order=1):
    """
    order = 1 or 2
    """
    if order == 1:
        Au_mat = zero_petsc_mat(num_pts*phy_dim, num_pts*para_dim, 
                                PREALLOC=Au.size)
        for i in range(num_pts):
            for j in range(phy_dim):
                row_ind = i*phy_dim+j
                col_ind0 = i*para_dim
                col_ind1 = i*para_dim+1
                vec_ind0 = i*para_dim*phy_dim+j*para_dim
                vec_ind1 = i*para_dim*phy_dim+j*para_dim+1
                Au_mat.setValue(row_ind, col_ind0, Au[vec_ind0])
                Au_mat.setValue(row_ind, col_ind1, Au[vec_ind1])
    elif order == 2:
        Au_mat = zero_petsc_mat(num_pts*phy_dim*para_dim, num_pts*para_dim, 
                                PREALLOC=Au.size)
        for i in range(num_pts):
            for j in range(phy_dim):
                for k in range(para_dim):
                    row_ind = i*phy_dim*para_dim+j*para_dim+k
                    col_ind0 = i*para_dim
                    col_ind1 = i*para_dim + 1
                    vec_ind0 = i*phy_dim*para_dim**order+j*para_dim**order+k*para_dim
                    vec_ind1 = i*phy_dim*para_dim**order+j*para_dim**order+k*para_dim+1
                    Au_mat.setValue(row_ind, col_ind0, Au[vec_ind0])
                    Au_mat.setValue(row_ind, col_ind1, Au[vec_ind1])
    else:
        raise RuntimeError("Order {} is not supported.".format(order))
    Au_mat.assemble()
    return Au_mat


class SplineBC(object):
    """
    Setting Dirichlet boundary condition to tIGAr spline generator.
    """
    def __init__(self, directions=[0,1], sides=[[0,1],[0,1]], 
                 fields=[[[0,1,2],[0,1,2]],[[0,1,2],[0,1,2]]],
                 n_layers=[[1,1],[1,1]]):
        self.fields = fields
        self.directions = directions
        self.sides = sides
        self.n_layers = n_layers

    def set_bc(self, spline_generator):

        for direction in self.directions:
            for side in self.sides[direction]:
                for field in self.fields[direction][side]:
                    scalar_spline = spline_generator.getScalarSpline(field)
                    side_dofs = scalar_spline.getSideDofs(direction,
                                side, nLayers=self.n_layers[direction][side])
                    spline_generator.addZeroDofs(field, side_dofs)

class SplineSim(object):

    def __init__(self, surfs, E, nu, h_th, pressure, quad_deg_const=3,
                 spline_bcs=None, opt_field=[0,1,2], comm=worldcomm):
        self.surfs = surfs
        self.E = E
        self.nu = nu
        self.h_th = h_th
        self.pressure = pressure

        self.quad_deg_const = quad_deg_const
        self.spline_bcs = spline_bcs
        self.opt_field = opt_field
        self.comm = comm

        self.phy_dim = 3
        self.para_dim = 2

        self.num_splines = len(surfs)
        self.surfs_data = [BSplineSurfaceData(surf) for surf in self.surfs]

        # Create tIGAr extracted splines
        self.spline_meshes = []
        self.spline_generators = []
        self.splines = []
        for s_ind in range(self.num_splines):
            # print("s_ind:", s_ind)
            if isinstance(surfs[s_ind], Geom_BSplineSurface):
                spline_mesh = NURBSControlMesh4OCC(self.surfs[s_ind], 
                                                   useRect=False)
                surf_deg = self.surfs[s_ind].UDegree()
            elif isinstance(surfs[s_ind], ikcad.NURBS):
                spline_mesh = NURBSControlMesh(self.surfs[s_ind], 
                                               useRect=False)
                surf_deg = self.surfs[s_ind].degree[0]
            self.spline_meshes += [spline_mesh,]

            spline_generator = EqualOrderSpline(self.comm, 3, spline_mesh)
            if self.spline_bcs[s_ind] is not None:
                self.spline_bcs[s_ind].set_bc(spline_generator)
            self.spline_generators += [spline_generator,]

            quad_deg = self.quad_deg_const*surf_deg
            spline = ExtractedSpline(spline_generator, quad_deg)
            self.splines += [spline]

        # Create nested vectors in IGA DoFs
        self.u_iga_list = []
        self.u_scalar_iga_list = []
        for s_ind in range(self.num_splines):
            self.u_iga_list += [zero_petsc_vec(
                                self.splines[s_ind].M.size(1),
                                comm=self.comm)]
            self.u_scalar_iga_list += [zero_petsc_vec(
                                 self.splines[s_ind].M_control.size(1),
                                 comm=self.comm)]
        self.u_iga_nest = create_nest_PETScVec(self.u_iga_list, 
                                               comm=self.comm)
        self.u_scalar_iga_nest = create_nest_PETScVec(self.u_scalar_iga_list, 
                                                      comm=self.comm)

        self.u_iga_dof = self.u_iga_nest.getSizes()[1]
        self.u_scalar_iga_dof = self.u_scalar_iga_nest.getSizes()[1]

        # Create nested nevtors in FE DoFs
        self.u_fe_list = []
        self.u_scalar_fe_list = []
        for s_ind in range(self.num_splines):
            self.u_fe_list += [zero_petsc_vec(
                               self.splines[s_ind].M.size(0),
                               comm=self.comm)]
            self.u_scalar_fe_list += [zero_petsc_vec(
                                      self.splines[s_ind].M_control.size(0),
                                      comm=self.comm)]
        self.u_fe_nest = create_nest_PETScVec(self.u_fe_list, comm=self.comm)
        self.u_scalar_fe_nest = create_nest_PETScVec(self.u_scalar_fe_list,
                                                     comm=self.comm)
        self.u_fe_dof = self.u_fe_nest.getSizes()[1]
        self.u_scalar_fe_dof = self.u_scalar_fe_nest.getSizes()[1]

        # Create nested cpFuncs vectors
        self.cp_funcs_list = [[] for i in range(3)]
        self.cp_funcs_nest = [None for i in range(3)]
        for field in self.opt_field:
            for s_ind in range(self.num_splines):
                self.cp_funcs_list[field] += [v2p(self.splines[s_ind].
                                               cpFuncs[field].vector())]
            self.cp_funcs_nest[field] = create_nest_PETScVec(
                                        self.cp_funcs_list[field], 
                                        comm=self.comm)

        # Create control points shape related attributes
        self.CP_shape_list = []
        self.CP_length_list = []
        for s_ind in range(self.num_splines):
            CP_shape0 = self.surfs_data[s_ind].control.shape[0]
            CP_shape1 = self.surfs_data[s_ind].control.shape[0]
            self.CP_shape_list += [[CP_shape0, CP_shape1]]
            self.CP_length_list += [CP_shape0*CP_shape1]

        # Initialize non-matching problem
        self.nonmatching = NonMatchingCoupling(self.splines, self.E, 
                                               self.h_th, self.nu, 
                                               comm=self.comm)

        self.spline_funcs = self.nonmatching.spline_funcs
        self.spline_test_funcs = self.nonmatching.spline_test_funcs

        self.f1 = as_vector([Constant(0.0), Constant(0.0), self.pressure])
        self.f0 = as_vector([Constant(0.0), Constant(0.0), Constant(0.0)])

        self.nonmatching_setup_is_done = False

    def nonmatching_setup(self, mapping_list, mortar_mesh_locations, 
                          penalty_coefficient=1.0e3, mortar_nels=None):

        if mortar_nels is None:
            mortar_nels = []
            for int_ind in range(len(mapping_list)):
                s_ind0, s_ind1 = mapping_list[int_ind]
                num_el0 = np.max(self.ikNURBS_list[s_ind0].control.shape[0],
                                 self.ikNURBS_list[s_ind0].control.shape[1])
                num_el1 = np.max(self.ikNURBS_list[s_ind1].control.shape[0],
                                 self.ikNURBS_list[s_ind1].control.shape[1])
                mortar_nels += [np.max(num_el0, num_el1)*2,]

        self.nonmatching.create_mortar_meshes(mortar_nels)
        self.nonmatching.mortar_meshes_setup(mapping_list, 
                         mortar_mesh_locations, penalty_coefficient)
        self.nonmatching_setup_is_done = True

    def SVK_residual(self, s_ind):
        X = self.splines[s_ind].F
        x = X + self.splines[s_ind].rationalize(
                self.spline_funcs[s_ind])
        z = self.splines[s_ind].rationalize(
            self.spline_test_funcs[s_ind])

        Wint = surfaceEnergyDensitySVK(self.splines[s_ind], X, x, 
               self.E, self.nu, self.h_th)*self.splines[s_ind].dx
        dWint = derivative(Wint, self.spline_funcs[s_ind], 
                           self.spline_test_funcs[s_ind])
        dWext = inner(self.f1, z)*self.splines[s_ind].dx
        res = dWint - dWext
        return res

    def SVK_residuals(self):
        res_list = []
        for s_ind in range(self.num_splines):
            res_list += [self.SVK_residual(s_ind),]
        return res_list

    def iga2fe_dofs_u(self, u_iga_petsc, s_ind):
        u_fe_petsc = v2p(self.spline_funcs[s_ind].vector())
        M_petsc = m2p(self.splines[s_ind].M)
        M_petsc.mult(u_iga_petsc, u_fe_petsc)
        u_fe_petsc.ghostUpdate()
        u_fe_petsc.assemble()

    def iga2fe_dofs_CP(self, CP_iga_petsc, s_ind, field=0):
        CP_fe_petsc = v2p(self.splines[s_ind].cpFuncs[field].vector())
        M_control_petsc = m2p(self.splines[s_ind].M_control)
        M_control_petsc.mult(CP_iga_petsc, CP_fe_petsc)
        CP_fe_petsc.ghostUpdate()
        CP_fe_petsc.assemble()

    def update_us(self, u_array):
        update_nest_vec(u_array, self.u_iga_nest, comm=self.comm)
        u_iga_sub = self.u_iga_nest.getNestSubVecs()
        for s_ind in range(self.num_splines):
            self.iga2fe_dofs_u(u_iga_sub[s_ind], s_ind)

        for i in range(len(self.nonmatching.transfer_matrices_list)):
            for j in range(len(self.nonmatching.transfer_matrices_list[i])):
                for k in range(len(self.nonmatching.transfer_matrices_list[i][j])):
                    A_x_b(self.nonmatching.transfer_matrices_list[i][j][k], 
                        self.nonmatching.spline_funcs[
                            self.nonmatching.mapping_list[i][j]].vector(), 
                        self.nonmatching.mortar_vars[i][j][k].vector())

    def update_CPs(self, CP_array, field=0):
        update_nest_vec(CP_array, self.u_scalar_iga_nest, comm=self.comm)
        cp_iga_sub = self.u_scalar_iga_nest.getNestSubVecs()
        for s_ind in range(self.num_splines):
            self.iga2fe_dofs_CP(cp_iga_sub[s_ind], s_ind, field)

    def update_cp_funcs(self, cf_array, field=0):
        update_nest_vec(cf_array, self.cp_funcs_nest[field], comm=self.comm)

    def update_transfer_matrix(self, mesh_coord, index=0, side=0):
        move_mortar_mesh(self.nonmatching.mortar_meshes[index], mesh_coord)
        self.nonmatching.transfer_matrices_list[index][side] = \
            create_transfer_matrix_list(self.nonmatching.splines[
            self.nonmatching.mapping_list[index][side]].V, 
            self.nonmatching.Vms[index], 2)
        self.nonmatching.transfer_matrices_control_list[index][side] = \
            create_transfer_matrix_list(self.nonmatching.splines[
            self.nonmatching.mapping_list[index][side]].V_control, 
            self.nonmatching.Vms_control[index], 2)

        for i in range(len(self.nonmatching.mortar_funcs[index][side])):
            A_x_b(self.nonmatching.transfer_matrices_list[index][side][i], 
                self.nonmatching.spline_funcs[
                    self.nonmatching.mapping_list[index][side]].vector(), 
                self.nonmatching.mortar_funcs[index][side][i].vector())
        for i in range(len(self.nonmatching.mortar_funcs[index][side])):
            for j in range(3+1):
                A_x_b(self.nonmatching.transfer_matrices_control_list[index][side][i], 
                    self.nonmatching.splines[
                        self.nonmatching.mapping_list[index][side]].cpFuncs[j].vector(), 
                    self.nonmatching.mortar_cpfuncs[index][side][i][j].vector())

    def dRdu(self):
        residuals = self.SVK_residuals()
        self.nonmatching.set_residuals(residuals)
        dRtdut_FE, Rt_FE = self.nonmatching.assemble_nonmatching()
        dRtdut_IGA, Rt_IGA = self.nonmatching.extract_nonmatching_system(
                        Rt_FE, dRtdut_FE)

        if MPI.size(self.comm) == 1:
            dRtdut_IGA.convert('seqaij')
        else:
            dRtdut_IGA = create_aijmat_from_nestmat(dRtdut_IGA, 
                         self.nonmatching.A_list, comm=self.comm)

        return dRtdut_IGA

    def dRdCP(self, field=0):
        residuals = self.SVK_residuals()
        self.dRtdCP_IGA_list = []
        for i in range(self.num_splines):
            self.dRtdCP_IGA_list += [[],]
            for j in range(self.num_splines):
                if i == j:
                    dRdCP_FE = derivative(residuals[i], 
                                         self.splines[i].cpFuncs[field])
                    dRdCP_FE_mat = m2p(assemble(dRdCP_FE))
                    M_petmat = m2p(self.splines[i].M)
                    M_control_petmat = m2p(self.splines[i].M_control)
                    dRdCP_IGA_mat = AT_R_B(M_petmat, dRdCP_FE_mat, 
                                           M_control_petmat)
                else:
                    dRdCP_IGA_mat = None
                self.dRtdCP_IGA_list[i] += [dRdCP_IGA_mat,]
        dRtdCP_IGA = create_nest_PETScMat(self.dRtdCP_IGA_list, comm=self.comm)

        if MPI.size(self.comm) == 1:
            dRtdCP_IGA.convert('seqaij')
        else:
            dRtdCP_IGA = create_aijmat_from_nestmat(dRtdCP_IGA, 
                        self.dRtdCP_IGA_list, comm=self.comm)

        return dRtdCP_IGA

    def dRdcf(self, field=0):
        residuals = self.SVK_residuals()
        self.dRtdcf_IGA_list = []
        for i in range(self.num_splines):
            self.dRtdcf_IGA_list += [[],]
            for j in range(self.num_splines):
                if i == j:
                    dRdcf_FE = derivative(residuals[i], 
                                          self.splines[i].cpFuncs[field])
                    dRdcf_FE_mat = m2p(assemble(dRdcf_FE))
                    M_petmat = m2p(self.splines[i].M)
                    dRdcf_IGA_mat = M_petmat.transposeMatMult(dRdcf_FE_mat)
                    dRdcf_IGA_mat.assemblyBegin()
                    dRdcf_IGA_mat.assemblyEnd()
                else:
                    dRdcf_IGA_mat = None
                self.dRtdcf_IGA_list[i] += [dRdcf_IGA_mat,]
        dRtdcf_IGA = create_nest_PETScMat(self.dRtdcf_IGA_list, comm=self.comm)

        if MPI.size(self.comm) == 1:
            dRtdcf_IGA.convert('seqaij')
        else:
            dRtdcf_IGA = create_aijmat_from_nestmat(dRtdcf_IGA, 
                         self.dRtdcf_IGA_list, comm=self.comm)

        return dRtdcf_IGA

    def dRIGAdxi_diag(self, index=0, side=0):
        """
        This function is for one pair of dRIGAdxi in diagonal blocks.
        index : index of intersetion
        side : side of mortar mesh for two intersecting surfaces {0,1}
        """
        self.num_pts = self.nonmatching.mortar_meshes[index].coordinates().shape[0]
        # dx_m = dx(metadata=self.nonmatching.int_dx_metadata)
        mapping_list = self.nonmatching.mapping_list
        s_ind0, s_ind1 = self.nonmatching.mapping_list[index]
        # self.der_mat = zero_petsc_mat(self.nonmatching.splines[s_ind0].M.size(1), self.num_pts*self.para_dim)
        if side == 0:
            proj_tan = False
        else:
            proj_tan = True

        self.PE = penalty_energy(self.nonmatching.splines[s_ind0], self.nonmatching.splines[s_ind1], 
            self.nonmatching.spline_funcs[s_ind0], self.nonmatching.spline_funcs[s_ind1], 
            self.nonmatching.mortar_meshes[index], 
            self.nonmatching.mortar_funcs[index], self.nonmatching.mortar_cpfuncs[index], 
            self.nonmatching.transfer_matrices_list[index],
            self.nonmatching.transfer_matrices_control_list[index],
            self.nonmatching.alpha_d_list[index], self.nonmatching.alpha_r_list[index], 
            proj_tan=proj_tan)

        print("Step 1 ", "*"*30)
        self.M = m2p(self.nonmatching.splines[mapping_list[index][side]].M)
        self.A0 = m2p(self.nonmatching.transfer_matrices_list[index][side][0])
        self.A1 = m2p(self.nonmatching.transfer_matrices_list[index][side][1])
        self.A2 = m2p(self.nonmatching.transfer_matrices_list[index][side][2])
        # self.A0c = m2p(self.nonmatching.transfer_matrices_control_list[index][side][0])
        self.A1c = m2p(self.nonmatching.transfer_matrices_control_list[index][side][1])
        self.A2c = m2p(self.nonmatching.transfer_matrices_control_list[index][side][2])

        self.u0M = self.nonmatching.mortar_funcs[index][side][0]
        self.u1M = self.nonmatching.mortar_funcs[index][side][1]
        self.P0M = self.nonmatching.mortar_cpfuncs[index][side][0]
        self.P1M = self.nonmatching.mortar_cpfuncs[index][side][1]
        self.uFE = self.nonmatching.spline_funcs[mapping_list[index][side]]
        self.PFE = self.nonmatching.splines[mapping_list[index][side]].cpFuncs

        ###################################
        print("Step 2 ", "*"*30)
        self.R_pen0M = derivative(self.PE, self.u0M)
        self.R_pen1M = derivative(self.PE, self.u1M)
        self.R_pen0M_vec = v2p(assemble(self.R_pen0M))
        self.R_pen1M_vec = v2p(assemble(self.R_pen1M))
        self.R_pen0M_mat = Lambda_tilde(self.R_pen0M_vec, self.num_pts, self.phy_dim, self.para_dim, 0)
        self.R_pen1M_mat = Lambda_tilde(self.R_pen1M_vec, self.num_pts, self.phy_dim, self.para_dim, 1)

        self.der_mat_FE = self.A1.transposeMatMult(self.R_pen0M_mat.transpose())
        self.der_mat_FE += self.A2.transposeMatMult(self.R_pen1M_mat.transpose())

        #############################
        print("Step 3 ", "*"*30)
        self.dR0Mdu0M = derivative(self.R_pen0M, self.u0M)
        self.dR0Mdu1M = derivative(self.R_pen0M, self.u1M)
        self.dR1Mdu0M = derivative(self.R_pen1M, self.u0M)
        self.dR1Mdu1M = derivative(self.R_pen1M, self.u1M)
        self.dR0Mdu0M_mat = m2p(assemble(self.dR0Mdu0M))
        self.dR0Mdu1M_mat = m2p(assemble(self.dR0Mdu1M))
        self.dR1Mdu0M_mat = m2p(assemble(self.dR1Mdu0M))
        self.dR1Mdu1M_mat = m2p(assemble(self.dR1Mdu1M))
        self.A1u_vec = A_x(self.A1, self.uFE)
        self.A2u_vec = A_x(self.A2, self.uFE)
        self.A1u_mat = Lambda(self.A1u_vec, self.num_pts, self.phy_dim, self.para_dim, order=1)
        self.A2u_mat = Lambda(self.A2u_vec, self.num_pts, self.phy_dim, self.para_dim, order=2)

        temp_mat = self.A0.transposeMatMult(self.dR0Mdu0M_mat) + \
                   self.A1.transposeMatMult(self.dR1Mdu0M_mat)
        self.der_mat_FE += temp_mat.matMult(self.A1u_mat)

        temp_mat = self.A0.transposeMatMult(self.dR0Mdu1M_mat) + \
                   self.A1.transposeMatMult(self.dR1Mdu1M_mat)
        self.der_mat_FE += temp_mat.matMult(self.A2u_mat)

        ###############################
        print("Step 4 ", "*"*30)
        self.dR0MdP0M_list = []
        self.dR0MdP1M_list = []
        self.dR1MdP0M_list = []
        self.dR1MdP1M_list = []
        self.dR0MdP0M_mat_list = []
        self.dR0MdP1M_mat_list = []
        self.dR1MdP0M_mat_list = []
        self.dR1MdP1M_mat_list = []
        self.A1cP_vec_list = []
        self.A2cP_vec_list = []
        self.A1cP_mat_list = []
        self.A2cP_mat_list = []
        self.temp_mat1_list = []
        self.temp_mat2_list = []
        for i in range(len(self.PFE)):
            self.dR0MdP0M_list += [derivative(self.R_pen0M, self.P0M[i])]
            self.dR0MdP1M_list += [derivative(self.R_pen0M, self.P1M[i])]
            self.dR1MdP0M_list += [derivative(self.R_pen1M, self.P0M[i])]
            self.dR1MdP1M_list += [derivative(self.R_pen1M, self.P1M[i])]
            self.dR0MdP0M_mat_list += [m2p(assemble(self.dR0MdP0M_list[i]))]
            self.dR0MdP1M_mat_list += [m2p(assemble(self.dR0MdP1M_list[i]))]
            self.dR1MdP0M_mat_list += [m2p(assemble(self.dR1MdP0M_list[i]))]
            self.dR1MdP1M_mat_list += [m2p(assemble(self.dR1MdP1M_list[i]))]
            self.temp_mat1_list += [self.A0.transposeMatMult(self.dR0MdP0M_mat_list[i]) 
                               + self.A1.transposeMatMult(self.dR1MdP0M_mat_list[i])]
            self.temp_mat2_list += [self.A0.transposeMatMult(self.dR0MdP1M_mat_list[i]) 
                               + self.A1.transposeMatMult(self.dR1MdP1M_mat_list[i])]
            self.A1cP_vec_list += [A_x(self.A1c, self.PFE[i])]
            self.A1cP_mat_list += [Lambda(self.A1cP_vec_list[i], self.num_pts, 
                                          1, self.para_dim, 1)]
            self.A2cP_vec_list += [A_x(self.A2c, self.PFE[i])]
            self.A2cP_mat_list += [Lambda(self.A2cP_vec_list[i], self.num_pts, 
                                          1, self.para_dim, 2)]
            self.der_mat_FE += self.temp_mat1_list[i].matMult(self.A1cP_mat_list[i])
            self.der_mat_FE += self.temp_mat2_list[i].matMult(self.A2cP_mat_list[i])

        #####################
        print("Step 5 ", "*"*30)
        self.xi_m = SpatialCoordinate(self.nonmatching.mortar_meshes[index])
        self.dR0Mdxi_m = derivative(self.R_pen0M, self.xi_m)
        self.dR1Mdxi_m = derivative(self.R_pen1M, self.xi_m)
        self.dR0Mdxi_m_mat = m2p(assemble(self.dR0Mdxi_m))
        self.dR1Mdxi_m_mat = m2p(assemble(self.dR1Mdxi_m))

        self.der_mat_FE += self.A0.transposeMatMult(self.dR0Mdxi_m_mat)
        self.der_mat_FE += self.A1.transposeMatMult(self.dR1Mdxi_m_mat)

        #######################
        print("Step 6 ", "*"*30)
        self.der_mat_FE.assemble()

        #######################
        print("Step 7 ", "*"*30)
        self.der_mat_IGA_rev = self.M.transposeMatMult(self.der_mat_FE)
        self.der_mat_IGA_rev.assemble()

        #######################
        print("Step 8 ", "*"*30)
        self.switch_col_mat = zero_petsc_mat(self.num_pts*self.para_dim, self.num_pts*self.para_dim, 
                                             PREALLOC=self.num_pts*self.para_dim)
        for i in range(self.num_pts):
            self.switch_col_mat.setValue(i*self.para_dim, self.num_pts*self.para_dim-i*self.para_dim-2, 1.)
            self.switch_col_mat.setValue(i*self.para_dim+1, self.num_pts*self.para_dim-i*self.para_dim-1, 1.)
        self.switch_col_mat.assemble()
        self.der_mat_IGA = self.der_mat_IGA_rev.matMult(self.switch_col_mat)

        return self.der_mat_IGA


    def dRIGAdxi_offdiag(self, index=0, surf_side=1, side=0):
        """
        This function is for one pair of dRIGAdxi in diagonal blocks.
        index : index of intersetion
        side : side of mortar mesh for two intersecting surfaces {0,1}
        """
        self.num_pts = self.nonmatching.mortar_meshes[index].coordinates().shape[0]
        # dx_m = dx(metadata=self.nonmatching.int_dx_metadata)
        mapping_list = self.nonmatching.mapping_list
        s_ind0, s_ind1 = self.nonmatching.mapping_list[index]
        # self.der_mat = zero_petsc_mat(self.nonmatching.splines[s_ind0].M.size(1), self.num_pts*self.para_dim)
        if side == 0:
            proj_tan = False
        else:
            proj_tan = True

        self.PE = penalty_energy(self.nonmatching.splines[s_ind0], self.nonmatching.splines[s_ind1], 
            self.nonmatching.spline_funcs[s_ind0], self.nonmatching.spline_funcs[s_ind1], 
            self.nonmatching.mortar_meshes[index], 
            self.nonmatching.mortar_funcs[index], self.nonmatching.mortar_cpfuncs[index], 
            self.nonmatching.transfer_matrices_list[index],
            self.nonmatching.transfer_matrices_control_list[index],
            self.nonmatching.alpha_d_list[index], self.nonmatching.alpha_r_list[index], 
            proj_tan=proj_tan)

        print("Step 1 ", "*"*30)
        other_side = int(1 - side)
        self.Mo = m2p(self.nonmatching.splines[mapping_list[index][other_side]].M)
        self.A0o = m2p(self.nonmatching.transfer_matrices_list[index][other_side][0])
        self.A1o = m2p(self.nonmatching.transfer_matrices_list[index][other_side][1])

        self.A1 = m2p(self.nonmatching.transfer_matrices_list[index][side][1])
        self.A2 = m2p(self.nonmatching.transfer_matrices_list[index][side][2])
        self.A1c = m2p(self.nonmatching.transfer_matrices_control_list[index][side][1])
        self.A2c = m2p(self.nonmatching.transfer_matrices_control_list[index][side][2])

        self.u0Mo = self.nonmatching.mortar_funcs[index][other_side][0]
        self.u1Mo = self.nonmatching.mortar_funcs[index][other_side][1]
        self.u0M = self.nonmatching.mortar_funcs[index][side][0]
        self.u1M = self.nonmatching.mortar_funcs[index][side][1]
        self.P0M = self.nonmatching.mortar_cpfuncs[index][side][0]
        self.P1M = self.nonmatching.mortar_cpfuncs[index][side][1]
        self.uFE = self.nonmatching.spline_funcs[mapping_list[index][side]]
        self.PFE = self.nonmatching.splines[mapping_list[index][side]].cpFuncs

        ###################################
        print("Step 2 ", "*"*30)
        self.R_pen0Mo = derivative(self.PE, self.u0Mo)
        self.R_pen1Mo = derivative(self.PE, self.u1Mo)
        self.R_pen0Mo_vec = v2p(assemble(self.R_pen0Mo))
        self.R_pen1Mo_vec = v2p(assemble(self.R_pen1Mo))

        #############################
        print("Step 3 ", "*"*30)
        self.dR0Modu0M = derivative(self.R_pen0Mo, self.u0M)
        self.dR0Modu1M = derivative(self.R_pen0Mo, self.u1M)
        self.dR1Modu0M = derivative(self.R_pen1Mo, self.u0M)
        self.dR1Modu1M = derivative(self.R_pen1Mo, self.u1M)
        self.dR0Modu0M_mat = m2p(assemble(self.dR0Modu0M))
        self.dR0Modu1M_mat = m2p(assemble(self.dR0Modu1M))
        self.dR1Modu0M_mat = m2p(assemble(self.dR1Modu0M))
        self.dR1Modu1M_mat = m2p(assemble(self.dR1Modu1M))
        self.A1u_vec = A_x(self.A1, self.uFE)
        self.A2u_vec = A_x(self.A2, self.uFE)
        self.A1u_mat = Lambda(self.A1u_vec, self.num_pts, self.phy_dim, self.para_dim, order=1)
        self.A2u_mat = Lambda(self.A2u_vec, self.num_pts, self.phy_dim, self.para_dim, order=2)

        temp_mat = self.A0o.transposeMatMult(self.dR0Modu0M_mat) + \
                   self.A1o.transposeMatMult(self.dR1Modu0M_mat)
        self.der_mat_FE = temp_mat.matMult(self.A1u_mat)

        temp_mat = self.A0o.transposeMatMult(self.dR0Modu1M_mat) + \
                   self.A1o.transposeMatMult(self.dR1Modu1M_mat)
        self.der_mat_FE += temp_mat.matMult(self.A2u_mat)

        ###############################
        print("Step 4 ", "*"*30)
        self.dR0ModP0M_list = []
        self.dR0ModP1M_list = []
        self.dR1ModP0M_list = []
        self.dR1ModP1M_list = []
        self.dR0ModP0M_mat_list = []
        self.dR0ModP1M_mat_list = []
        self.dR1ModP0M_mat_list = []
        self.dR1ModP1M_mat_list = []
        self.A1cP_vec_list = []
        self.A2cP_vec_list = []
        self.A1cP_mat_list = []
        self.A2cP_mat_list = []
        self.temp_mat1_list = []
        self.temp_mat2_list = []
        for i in range(len(self.PFE)):
            self.dR0ModP0M_list += [derivative(self.R_pen0Mo, self.P0M[i])]
            self.dR0ModP1M_list += [derivative(self.R_pen0Mo, self.P1M[i])]
            self.dR1ModP0M_list += [derivative(self.R_pen1Mo, self.P0M[i])]
            self.dR1ModP1M_list += [derivative(self.R_pen1Mo, self.P1M[i])]
            self.dR0ModP0M_mat_list += [m2p(assemble(self.dR0ModP0M_list[i]))]
            self.dR0ModP1M_mat_list += [m2p(assemble(self.dR0ModP1M_list[i]))]
            self.dR1ModP0M_mat_list += [m2p(assemble(self.dR1ModP0M_list[i]))]
            self.dR1ModP1M_mat_list += [m2p(assemble(self.dR1ModP1M_list[i]))]
            self.temp_mat1_list += [self.A0o.transposeMatMult(self.dR0ModP0M_mat_list[i]) 
                               + self.A1o.transposeMatMult(self.dR1ModP0M_mat_list[i])]
            self.temp_mat2_list += [self.A0o.transposeMatMult(self.dR0ModP1M_mat_list[i]) 
                               + self.A1o.transposeMatMult(self.dR1ModP1M_mat_list[i])]
            self.A1cP_vec_list += [A_x(self.A1c, self.PFE[i])]
            self.A1cP_mat_list += [Lambda(self.A1cP_vec_list[i], self.num_pts, 
                                          1, self.para_dim, 1)]
            self.A2cP_vec_list += [A_x(self.A2c, self.PFE[i])]
            self.A2cP_mat_list += [Lambda(self.A2cP_vec_list[i], self.num_pts, 
                                          1, self.para_dim, 2)]
            self.der_mat_FE += self.temp_mat1_list[i].matMult(self.A1cP_mat_list[i])
            self.der_mat_FE += self.temp_mat2_list[i].matMult(self.A2cP_mat_list[i])

        #####################
        print("Step 5 ", "*"*30)
        self.xi_m = SpatialCoordinate(self.nonmatching.mortar_meshes[index])
        self.dR0Modxi_m = derivative(self.R_pen0Mo, self.xi_m)
        self.dR1Modxi_m = derivative(self.R_pen1Mo, self.xi_m)
        self.dR0Modxi_m_mat = m2p(assemble(self.dR0Modxi_m))
        self.dR1Modxi_m_mat = m2p(assemble(self.dR1Modxi_m))

        self.der_mat_FE += self.A0o.transposeMatMult(self.dR0Modxi_m_mat)
        self.der_mat_FE += self.A1o.transposeMatMult(self.dR1Modxi_m_mat)

        #######################
        print("Step 6 ", "*"*30)
        self.der_mat_FE.assemble()

        #######################
        print("Step 7 ", "*"*30)
        self.der_mat_IGA_rev = self.Mo.transposeMatMult(self.der_mat_FE)
        self.der_mat_IGA_rev.assemble()


        #######################
        print("Step 8 ", "*"*30)
        self.switch_col_mat = zero_petsc_mat(self.num_pts*self.para_dim, self.num_pts*self.para_dim, 
                                             PREALLOC=self.num_pts*self.para_dim)
        for i in range(self.num_pts):
            self.switch_col_mat.setValue(i*self.para_dim, self.num_pts*self.para_dim-i*self.para_dim-2, 1.)
            self.switch_col_mat.setValue(i*self.para_dim+1, self.num_pts*self.para_dim-i*self.para_dim-1, 1.)
        self.switch_col_mat.assemble()
        self.der_mat_IGA = self.der_mat_IGA_rev.matMult(self.switch_col_mat)

        return self.der_mat_IGA


    def dRIGAdxi_FD_diag(self, index=0, side=0, h=1e-8):
        """
        index : index of intersetion
        side : side of mortar mesh for two intersecting surfaces {0,1}
        """

        if side == 0:
            proj_tan = False
        else:
            proj_tan = True

        residuals = self.SVK_residuals()
        self.num_pts = self.nonmatching.mortar_meshes[index].coordinates().shape[0]
        dx_m = dx(metadata=self.nonmatching.int_dx_metadata)
        mapping_list = self.nonmatching.mapping_list
        s_ind0, s_ind1 = self.nonmatching.mapping_list[index]
        self.PE = penalty_energy(self.nonmatching.splines[s_ind0], self.nonmatching.splines[s_ind1], 
            self.nonmatching.spline_funcs[s_ind0], self.nonmatching.spline_funcs[s_ind1], 
            self.nonmatching.mortar_meshes[index], 
            self.nonmatching.mortar_funcs[index], self.nonmatching.mortar_cpfuncs[index], 
            self.nonmatching.transfer_matrices_list[index],
            self.nonmatching.transfer_matrices_control_list[index],
            self.nonmatching.alpha_d_list[index], self.nonmatching.alpha_r_list[index], 
            proj_tan=proj_tan)
        mortar_mesh = self.nonmatching.mortar_meshes[index]

        def R():
            self.RSFE = v2p(assemble(residuals[self.nonmatching.mapping_list[index][side]]))
            self.M = m2p(self.nonmatching.splines[mapping_list[index][side]].M)
            self.A0 = m2p(self.nonmatching.transfer_matrices_list[index][side][0])
            self.A1 = m2p(self.nonmatching.transfer_matrices_list[index][side][1])
            self.u0M = self.nonmatching.mortar_funcs[index][side][0]
            self.u1M = self.nonmatching.mortar_funcs[index][side][1]
            self.R_pen0M = derivative(self.PE, self.u0M)
            self.R_pen1M = derivative(self.PE, self.u1M)
            self.R_pen0M_vec = v2p(assemble(self.R_pen0M))
            self.R_pen1M_vec = v2p(assemble(self.R_pen1M))
            self.RMFE = AT_x(self.A0, self.R_pen0M_vec) + AT_x(self.A1, self.R_pen1M_vec)
            self.RFE = self.RSFE + self.RMFE
            self.RIGA = AT_x(self.M, self.RFE)
            self.RIGA.assemble()
            return self.RIGA

        self.num_pts = mortar_mesh.coordinates().shape[0]
        self.mesh_coord_init = mortar_mesh.coordinates().reshape(-1).copy()
        self.der_mat_IGA = np.zeros((self.splines[mapping_list[index][side]].M.size(1),
                               self.mesh_coord_init.size))

        R_init = R()[:]

        # print("self.mesh_coord_init:", self.mesh_coord_init)
        # print("R_init:", R_init)

        for i in range(self.num_pts*self.para_dim):
            # print("FD index:", i)
            perturb = np.zeros(self.mesh_coord_init.size)
            perturb[i] = h
            meshm_coord_peturb = self.mesh_coord_init + perturb
            self.update_transfer_matrix(meshm_coord_peturb.reshape(-1,2), index, side)
            R_perturb = R()[:]
            R_diff = R_perturb - R_init
            # if i < 5:
            #     print("i:", i, ", R_diff:", R_diff)
            self.der_mat_IGA[:,i] = R_diff/h
        return self.der_mat_IGA


    def dRIGAdxi_FD_offdiag(self, index=0, side=0, h=1e-8):
        """
        index : index of intersetion
        side : side of mortar mesh for two intersecting surfaces {0,1}
        """

        if side == 0:
            proj_tan = False
        else:
            proj_tan = True

        other_side = int(1 - side)
        residuals = self.SVK_residuals()
        self.num_pts = self.nonmatching.mortar_meshes[index].coordinates().shape[0]
        # dx_m = dx(metadata=self.nonmatching.int_dx_metadata)
        mapping_list = self.nonmatching.mapping_list
        s_ind0, s_ind1 = self.nonmatching.mapping_list[index]
        self.PE = penalty_energy(self.nonmatching.splines[s_ind0], self.nonmatching.splines[s_ind1], 
            self.nonmatching.spline_funcs[s_ind0], self.nonmatching.spline_funcs[s_ind1], 
            self.nonmatching.mortar_meshes[index], 
            self.nonmatching.mortar_funcs[index], self.nonmatching.mortar_cpfuncs[index], 
            self.nonmatching.transfer_matrices_list[index],
            self.nonmatching.transfer_matrices_control_list[index],
            self.nonmatching.alpha_d_list[index], self.nonmatching.alpha_r_list[index], 
            proj_tan=proj_tan)
        mortar_mesh = self.nonmatching.mortar_meshes[index]

        def R():
            self.RSFE = v2p(assemble(residuals[self.nonmatching.mapping_list[index][other_side]]))
            self.M = m2p(self.nonmatching.splines[mapping_list[index][other_side]].M)
            self.A0 = m2p(self.nonmatching.transfer_matrices_list[index][other_side][0])
            self.A1 = m2p(self.nonmatching.transfer_matrices_list[index][other_side][1])
            self.u0M = self.nonmatching.mortar_funcs[index][other_side][0]
            self.u1M = self.nonmatching.mortar_funcs[index][other_side][1]
            self.R_pen0M = derivative(self.PE, self.u0M)
            self.R_pen1M = derivative(self.PE, self.u1M)
            self.R_pen0M_vec = v2p(assemble(self.R_pen0M))
            self.R_pen1M_vec = v2p(assemble(self.R_pen1M))
            self.RMFE = AT_x(self.A0, self.R_pen0M_vec) + AT_x(self.A1, self.R_pen1M_vec)
            self.RFE = self.RSFE + self.RMFE
            self.RIGA = AT_x(self.M, self.RFE)
            self.RIGA.assemble()
            return self.RIGA

        self.num_pts = mortar_mesh.coordinates().shape[0]
        self.mesh_coord_init = mortar_mesh.coordinates().reshape(-1).copy()
        self.der_mat_IGA = np.zeros((self.splines[mapping_list[index][other_side]].M.size(1),
                               self.mesh_coord_init.size))

        R_init = R()[:]

        # print("self.mesh_coord_init:", self.mesh_coord_init)
        # print("R_init:", R_init)

        for i in range(self.num_pts*self.para_dim):
            # print("FD index:", i)
            perturb = np.zeros(self.mesh_coord_init.size)
            perturb[i] = h
            meshm_coord_peturb = self.mesh_coord_init + perturb
            self.update_transfer_matrix(meshm_coord_peturb.reshape(-1,2), index, side)
            R_perturb = R()[:]
            R_diff = R_perturb - R_init
            # if i < 5:
            #     print("i:", i, ", R_diff:", R_diff)
            self.der_mat_IGA[:,i] = R_diff/h
        return self.der_mat_IGA

    def compute_RIGA(self, index=0, side=0):

        residuals = self.SVK_residuals()
        self.num_pts = self.nonmatching.mortar_meshes[index].coordinates().shape[0]
        dx_m = dx(metadata=self.nonmatching.int_dx_metadata)
        mapping_list = self.nonmatching.mapping_list
        s_ind0, s_ind1 = self.nonmatching.mapping_list[index]
        self.PE = penalty_energy(self.nonmatching.splines[s_ind0], self.nonmatching.splines[s_ind1], 
            self.nonmatching.spline_funcs[s_ind0], self.nonmatching.spline_funcs[s_ind1], 
            self.nonmatching.mortar_meshes[index], 
            self.nonmatching.mortar_funcs[index], self.nonmatching.mortar_cpfuncs[index], 
            self.nonmatching.transfer_matrices_list[index],
            self.nonmatching.transfer_matrices_control_list[index],
            self.nonmatching.alpha_d_list[index], self.nonmatching.alpha_r_list[index], 
            dx_m=dx_m, side=side)
        mortar_mesh = self.nonmatching.mortar_meshes[index]

        self.RSFE = v2p(assemble(residuals[self.nonmatching.mapping_list[index][side]]))
        self.M = m2p(self.nonmatching.splines[mapping_list[index][side]].M)
        self.A0 = m2p(self.nonmatching.transfer_matrices_list[index][side][0])
        self.A1 = m2p(self.nonmatching.transfer_matrices_list[index][side][1])
        self.u0M = self.nonmatching.mortar_funcs[index][side][0]
        self.u1M = self.nonmatching.mortar_funcs[index][side][1]
        self.R_pen0M = derivative(self.PE, self.u0M)
        self.R_pen1M = derivative(self.PE, self.u1M)
        self.R_pen0M_vec = v2p(assemble(self.R_pen0M))
        self.R_pen1M_vec = v2p(assemble(self.R_pen1M))
        self.RMFE = AT_x(self.A0, self.R_pen0M_vec) + AT_x(self.A1, self.R_pen1M_vec)
        self.RFE = self.RSFE + self.RMFE
        self.RIGA = AT_x(self.M, self.RFE)
        self.RIGA.assemble()
        return self.RIGA


    def solve_Ax_b(self, A, b, array=False):
        x = b.copy()
        x.zeroEntries()

        # if mpirank == 0:
        #     print("**** Solving ATx=b ...")

        solve_nonmatching_mat(A, x, b, solver='direct')
        x.assemble()

        if array:
            return get_petsc_vec_array(x, self.comm)
        else:
            return x

    def solve_ATx_b(self, A, b, array=False):
        AT = A.transpose()
        x = b.copy()
        x.zeroEntries()

        # if mpirank == 0:
        #     print("**** Solving ATx=b ...")

        solve_nonmatching_mat(AT, x, b, solver='direct')
        x.assemble()
        
        if array:
            return get_petsc_vec_array(x, self.comm)
        else:
            return x

    def create_files(self, save_path="./", folder_name="results/", 
                     thickness=False):
        self.u_file_names = []
        self.u_files = []
        self.F_file_names = []
        self.F_files = []
        if thickness:
            self.t_file_names = []
            self.t_files = []
        for i in range(self.num_splines):
            self.u_file_names += [[],]
            self.u_files += [[],]
            self.F_file_names += [[],]
            self.F_files += [[],]
            for j in range(3):
                self.u_file_names[i] += [save_path+folder_name+'u'+str(i)
                                         +'_'+str(j)+'_file.pvd',]
                self.u_files[i] += [File(self.comm, self.u_file_names[i][j]),]
                self.F_file_names[i] += [save_path+folder_name+'F'+str(i)
                                         +'_'+str(j)+'_file.pvd',]
                self.F_files[i] += [File(self.comm, self.F_file_names[i][j]),]
                if j==2:
                    self.F_file_names[i] += [save_path+folder_name+'F'+str(i)
                                             +'_'+str(j+1)+'_file.pvd',]
                    self.F_files[i] += [File(self.comm, self.F_file_names[i][j+1]),]
            if thickness:
                self.t_file_names += [save_path+folder_name+'t'+str(i)
                                      +'_file.pvd',]
                self.t_files += [File(self.comm, self.t_file_names[i]),]

    def save_files(self, thickness=False):
        for i in range(self.num_splines):
            u_split = self.spline_funcs[i].split()
            for j in range(3):
                u_split[j].rename('u'+str(i)+'_'+str(j),
                                  'u'+str(i)+'_'+str(j))
                self.u_files[i][j] << u_split[j]
                self.splines[i].cpFuncs[j].rename('F'+str(i)+'_'+str(j),
                                                  'F'+str(i)+'_'+str(j))
                self.F_files[i][j] << self.splines[i].cpFuncs[j]
                if j==2:
                    self.splines[i].cpFuncs[j+1].rename('F'+str(i)+'_'+str(j+1),
                                                        'F'+str(i)+'_'+str(j+1))
                    self.F_files[i][j+1] << self.splines[i].cpFuncs[j+1]
            if thickness:
                self.h_ths[i].rename('t'+str(i), 't'+str(i))
                self.t_files[i] << self.h_ths[i]


if __name__ == "__main__":
    import os
    E = Constant(10e12)
    nu = Constant(0.)
    h_th = Constant(0.01)
    penalty_coefficient = 1.0e3
    pressure = -Constant(1.)

    fields0 = [[],[[0,1,2]]]
    spline_bc0 = SplineBC(directions=[1], sides=[[],[0]],
                         fields=fields0, n_layers=[[],[1]])
    spline_bcs = [spline_bc0]*2

    filename_igs = "./geometry/T_beam_geom.igs"
    igs_shapes = read_igs_file(filename_igs, as_compound=False)
    occ_surf_list = [topoface2surface(face, BSpline=True) 
                     for face in igs_shapes]

    # Geometry preprocessing and surface-surface intersections computation
    preprocessor = OCCPreprocessing(occ_surf_list, reparametrize=False, 
                                    refine=False)
    print("Computing intersections...")
    int_data_filename = "int_data.npz"
    if os.path.isfile(int_data_filename):
        preprocessor.load_intersections_data(int_data_filename)
    else:
        # preprocessor.compute_intersections(mortar_refine=1)
        preprocessor.compute_intersections(mortar_nels=[14])
        # preprocessor.save_intersections_data(int_data_filename)

    if mpirank == 0:
        print("Total DoFs:", preprocessor.total_DoFs)
        print("Number of intersections:", preprocessor.num_intersections_all)

    # # Display B-spline surfaces and intersections using 
    # # PythonOCC build-in 3D viewer.
    # display, start_display, add_menu, add_function_to_menu = init_display()
    # preprocessor.display_surfaces(display, save_fig=False)
    # preprocessor.display_intersections(display, save_fig=False)

    spline_sim = SplineSim(occ_surf_list, E, nu, h_th, 
                           pressure, spline_bcs=spline_bcs)
    spline_sim.nonmatching_setup(preprocessor.mapping_list,
                                  preprocessor.intersections_para_coords,
                                  penalty_coefficient,
                                  preprocessor.mortar_nels)

    # ######## Test for SplineSim functions ########
    # dresdu = spline_sim.dRdu()
    # dresdCP = spline_sim.dRdCP()
    # dresdcf = spline_sim.dRdcf()


    ####### Test for derivatives of mortar mesh's locations ######
    # vec0_old = spline_sim.nonmatching.mortar_cpfuncs[0][0][0][0].vector().get_local()
    # vec1_old = spline_sim.nonmatching.mortar_cpfuncs[0][0][0][1].vector().get_local()
    # vec2_old = spline_sim.nonmatching.mortar_cpfuncs[0][0][0][2].vector().get_local()
    # ax0 = plt.figure(1).add_subplot(projection='3d')
    # ax0.plot(vec0_old, vec1_old, vec2_old, "*")
    # ax0.set_xlim([-2,2])
    # ax0.set_ylim([0,20])
    # ax0.set_zlim([-0.5,0.5])

    # mortar_loc = np.array([[0.6, 0.], [0.7, 1.]])
    # spline_sim.update_transfer_matrix(mortar_loc, index=0, side=0)
    # dRIGAdxi = spline_sim.dRIGAdxi()

    # vec0_new = spline_sim.nonmatching.mortar_cpfuncs[0][0][0][0].vector().get_local()
    # vec1_new = spline_sim.nonmatching.mortar_cpfuncs[0][0][0][1].vector().get_local()
    # vec2_new = spline_sim.nonmatching.mortar_cpfuncs[0][0][0][2].vector().get_local()
    # ax0.plot(vec0_new, vec1_new, vec2_new, "+", color='tab:green')
    # plt.show()


    ######## Solve for nonlinear non-matching problem ########
    residuals = spline_sim.SVK_residuals()
    spline_sim.nonmatching.set_residuals(residuals)
    spline_sim.nonmatching.solve_linear_nonmatching_problem()

    index = 0
    side = 0
    mortar_loc = np.array([[0.4, 0.01], [0.7, 0.99]])
    spline_sim.update_transfer_matrix(mortar_loc, index=index, side=side)

    # dRIGAdxi = spline_sim.dRIGAdxi_diag(index=index, side=side)
    # dRIGAdxi_norm = dRIGAdxi.norm()
    # dRIGAdxi_FD = spline_sim.dRIGAdxi_FD_diag(index=index, side=side)
    # dRIGAdxi_FD_norm = np.linalg.norm(dRIGAdxi_FD)
    # diff = (dRIGAdxi_FD - dRIGAdxi[:,:])
    # rel_err = np.linalg.norm(diff)/dRIGAdxi_FD_norm
    # print("rel_err:", rel_err)

    dRIGAdxi_off = spline_sim.dRIGAdxi_offdiag(index=index, side=side)
    dRIGAdxi_off_FD = spline_sim.dRIGAdxi_FD_offdiag(index=index, side=side)
    dRIGAdxi_off_FD_norm = np.linalg.norm(dRIGAdxi_off_FD)
    diff = dRIGAdxi_off_FD - dRIGAdxi_off[:,:]
    rel_err = np.linalg.norm(diff)/dRIGAdxi_off_FD_norm
    print("rel_err:", rel_err)

    # SAVE_PATH = "./"
    # spline_sim.create_files(save_path=SAVE_PATH, folder_name="results/")
    # spline_sim.save_files()