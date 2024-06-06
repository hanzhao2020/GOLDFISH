from PENGoLINS.nonmatching_coupling import *
from ufl import Jacobian

# def lumped_project(to_project, V):
#     u = Function(V)
#     v = TestFunction(V)
#     lhs = assemble(inner(Constant(1.), v)*dx)
#     rhs = assemble(inner(to_project, v)*dx)
#     as_backend_type(u.vector()).vec().\
#         pointwiseDivide(as_backend_type(rhs).vec(),
#                         as_backend_type(lhs).vec())
#     return u

# def penalty_displacement_integrand(alpha_d, u0m_hom, u1m_hom):
#     """
#     Penalization of displacements on the non-matching 
#     interface between two splines.
    
#     Parameters
#     ----------
#     alpha_d : ufl.algebra.Division
#     u0m_hom : dolfin Function
#     u1m_hom : dolfin Function
#     line_Jacobian : dolfin Function or None, optional
#     dx_m : ufl Measure

#     Return
#     ------
#     W_pd : ufl Form
#     """
#     W_pd = 0.5*alpha_d*((u0m_hom-u1m_hom)**2)
#     return W_pd

# def penalty_rotation_integrand(mortar_mesh, alpha_r, dX0dxi, dx0dxi, 
#                                dX1dxi, dx1dxi, symbolic_t=True, t_func=None):
#     """
#     Penalization of rotation on the non-matching interface 
#     between two splines.

#     Parameters
#     ----------
#     mortar_mesh : dolfin Mesh
#     alpha_r : ufl.algebra.Division
#     dX0dxi : dolfin ListTensor
#     dx0dxi : dolfin ListTensor
#     dX1dxi : dolfin ListTensor
#     dx1dxi : dolfin ListTensor
#     line_Jacobian : dolfin Function or None, optional
#     dx_m : ufl Measure or None, optional

#     Return
#     ------
#     W_pr : ufl Form
#     """
#     # Orthonormal basis for patch 0
#     a00, a01, a02 = interface_geometry(dx0dxi)
#     A00, A01, A02 = interface_geometry(dX0dxi)

#     # Orthonormal basis for patch 1
#     a10, a11, a12 = interface_geometry(dx1dxi)
#     A10, A11, A12 = interface_geometry(dX1dxi)

#     if symbolic_t:
#         xi = SpatialCoordinate(mortar_mesh)
#         t = Jacobian(xi)

#         t0, t1 = t[0,0], t[1,0]
#         # Vm = FunctionSpace(mortar_mesh, 'CG', 1)
#         # t0 = lumped_project(t[0,0], Vm)
#         # t1 = lumped_project(t[1,0], Vm)

#         at0 = t0*a00 + t1*a01
#         At0 = t0*A00 + t1*A01

#         # at0 = t[0,0]*a00 + t[1,0]*a01
#         # At0 = t[0,0]*A00 + t[1,0]*A01
#     else:
#         at0 = t_func[0]*a00 + t_func[1]*a01
#         At0 = t_func[0]*A00 + t_func[1]*A01

#     an1 = cross(a02, at0)/sqrt(inner(at0, at0))
#     An1 = cross(A02, At0)/sqrt(inner(At0, At0))

#     W_pr = 0.5*alpha_r*((inner(a02, a12) - inner(A02, A12))**2 
#          + (inner(an1, a12) - inner(An1, A12))**2)

#     return W_pr

# def penalty_energy(spline0, spline1, u0, u1,
#                    mortar_mesh, mortar_funcs, mortar_cpfuncs, 
#                    A, A_control, alpha_d, alpha_r, 
#                    dx_m=None, metadata=None, side=0):
#     """
#     Penalization of displacement and rotation of non-matching interface 
#     between two extracted splines.
    
#     Parameters
#     ----------
#     spline0 : ExtractedSpline
#     spline1 : ExtractedSpline
#     u0 : dolfin Function, displacement of spline0
#     u1 : dolfin Function, displacement of splint1
#     mortar_mesh : dolfin Mesh
#     mortar_funcs : list of dolfin Functions, mortar mesh's displacement
#         and first derivatives on two sides
#     mortar_cpfuncs : list of dolfin Functions, mortar mesh's control
#         point functions and first derivatives on two sides
#     A : list of dolfin PETScMatrices, transfer matrices for displacements
#     A_control : list of dolfin PETScMatrices, transfer matrices
#         for control point functions
#     alpha_d : ufl.algebra.Division
#     alpha_r : ufl.algebra.Division
#     dx_m : ufl Measure or None
#     quadrature_degree : int, default is 2

#     Returns
#     -------
#     W_p : ufl Form
#     """
#     splines = [spline0, spline1]
#     spline_u = [u0, u1]
#     mortar_X = []
#     mortar_dXdxi = []
#     mortar_x = []
#     mortar_dxdxi = []
#     for side_m in range(len(mortar_funcs)):
#         transfer_mortar_u(spline_u[side_m], mortar_funcs[side_m], A[side_m])
#         transfer_mortar_cpfuns(splines[side_m], mortar_cpfuncs[side_m], 
#                                A_control[side_m])
#         X_temp, dXdxi_temp = create_geometrical_mapping(splines[side_m], 
#                              mortar_cpfuncs[side_m])
#         mortar_X += [X_temp]
#         mortar_dXdxi += [dXdxi_temp]
#         x_temp, dxdxi_temp = physical_configuration(X_temp, dXdxi_temp, 
#                              mortar_cpfuncs[side_m], mortar_funcs[side_m])
#         mortar_x += [x_temp]
#         mortar_dxdxi += [dxdxi_temp]

#     if dx_m is not None:
#         dx_m = dx_m
#     else:
#         if metadata is not None:
#             dx_m = dx(domain=mortar_mesh, metadata=metadata)
#         else:
#             dx_m = dx(domain=mortar_mesh, metadata={'quadrature_degree': 0, 
#                                           'quadrature_scheme': 'vertex'})

#     line_Jacobian = compute_line_Jacobian(mortar_X[1])
#     if line_Jacobian == 0.:
#         line_Jacobian = sqrt(tr(mortar_dXdxi[1]*mortar_dXdxi[1].T))

#     xi_m = SpatialCoordinate(mortar_mesh)
#     # # Don't need to write it explicitly
#     # para_Jacbian = det(grad(xi_m))

#     # Penalty of displacement
#     W_pd = penalty_displacement_integrand(alpha_d, mortar_funcs[0][0], 
#                                 mortar_funcs[1][0])
#     # print("side:", side)
#     if side == 0:
#         symbolic_t = True
#         t_func = None
#     elif side == 1:
#         symbolic_t = False
#         t_func = Function(mortar_cpfuncs[0][1][0].function_space())
#         t = Jacobian(xi_m)
#         t_func.assign(project(t[:,0], t_func.function_space()))

#     # print("side:", side)
#     # print("t_func:", t_func)

#     # Penalty of rotation
#     W_pr = penalty_rotation_integrand(mortar_mesh, alpha_r, 
#                             mortar_dXdxi[0], mortar_dxdxi[0], 
#                             mortar_dXdxi[1], mortar_dxdxi[1],
#                             symbolic_t, t_func)
#     # W_p = (W_pd + W_pr)*para_Jacbian*line_Jacobian*dx_m
#     W_p = (W_pd + W_pr)*line_Jacobian*dx_m
#     return W_p


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