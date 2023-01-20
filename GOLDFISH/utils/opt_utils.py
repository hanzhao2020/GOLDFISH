from dolfin import *
from tIGAr import *
from tIGAr.BSplines import *
from tIGAr.NURBS import *
from PENGoLINS.nonmatching_utils import *

DEFAULT_COMM = worldcomm

def array2petsc_vec(ndarray, comm=DEFAULT_COMM):
    """
    Convert numpy array to petsc vector.

    Parameters
    ----------
    ndarray : ndarray
    comm : mpi4py.MPI.Intracomm, optional

    Returns
    -------
    petsc_vec : petsc4py.PETSc.Vec
    """
    size = ndarray.size
    petsc_vec = zero_petsc_vec(size, 'mpi', comm=comm)
    petsc_vec.setValues(np.arange(size, dtype='int32'), ndarray)
    petsc_vec.assemble()
    return petsc_vec

def get_petsc_vec_array(petsc_vec, comm=DEFAULT_COMM):
    """
    Get global values from a PETSc vector

    Parameters
    ----------
    petsc_vec : petsc4py.PETSc.Vec
    comm : mpi4py.MPI.Intracomm, optional

    Returns
    -------
    array : ndarray
    """
    if MPI.size(comm) > 1:
        if petsc_vec.type == 'nest':
            sub_vecs = petsc_vec.getNestSubVecs()
            num_sub_vecs = len(sub_vecs)
            sub_vecs_array_list = []
            for i in range(num_sub_vecs):
                sub_vecs_array_list += [np.concatenate(
                    comm.allgather(sub_vecs[i].array))]
            array = np.concatenate(sub_vecs_array_list)
        else:
            array = np.concatenate(comm.allgather(petsc_vec.array))
    else:
        array = petsc_vec.array
    return array

def update_func(u, u_array):
    """
    Update values in a dolfin function.

    Parameters
    ----------
    u : dolfin Function
    u_array : ndarray
    """
    u_petsc = v2p(u.vector())
    u_petsc.setValues(np.arange(u_array.size, dtype='int32'), u_array)
    u_petsc.assemble()
    u_petsc.ghostUpdate()

def update_nest_vec(vec_array, nest_vec, comm=DEFAULT_COMM):
    """
    Assign values from a numpy array to a nest petsc vector.

    Parameters
    ----------
    vec_array : ndarray
    nest_vec : petsc4py.PETSc.Vec
    comm : mpi4py.MPI.Intracomm, optional
    """
    if nest_vec.type != 'nest':
        if MPI.rank(comm) == 0:
            raise TypeError("Type of PETSc vector is not nest.")

    sub_vecs = nest_vec.getNestSubVecs()
    num_sub_vecs = len(sub_vecs)

    sub_vecs_range = []
    sub_vecs_size = []
    for i in range(num_sub_vecs):
        sub_vecs_range += [sub_vecs[i].getOwnershipRange(),]
        sub_vecs_size += [sub_vecs[i].size,]

    sub_array_list = []
    array_ind_off = 0
    for i in range(num_sub_vecs):
        sub_array_list += [vec_array[array_ind_off+sub_vecs_range[i][0]: 
                                     array_ind_off+sub_vecs_range[i][1]],]
        array_ind_off += sub_vecs_size[i]
    sub_array = np.concatenate(sub_array_list)
    nest_vec.setArray(sub_array)
    nest_vec.assemble()


def PETSc_ksp_solve(A, x, b, ksp_type='cg', pc_type ='jacobi', 
                    max_it=10000, rtol=1e-15):
    """
    PETSc KSP solver to solve "Ax=b".

    Parameters
    ----------
    A : petsc4py.PETSc.Mat
    x : petsc4py.PETSc.Vec
    b : petsc4py.PETSc.Vec
    ksp_type : str, optional, default is 'cg'
    pc_type : str, optional, default is 'jacobi'
    max_it : int, optional, default is 10000
    rtol : float, optional, default is 1e-15
    """
    ksp = PETSc.KSP().create()
    ksp.setType(ksp_type)
    pc = ksp.getPC()
    PETScOptions.set('pc_type', pc_type)
    pc.setType(pc_type)
    ksp.setTolerances(rtol=rtol)
    ksp.max_it = max_it
    ksp.setOperators(A)
    ksp.setFromOptions()
    ksp.solve(b, x)
    x.ghostUpdate()
    x.assemble()
    return x

def Newton_solve(A, x, b, max_it=10):
    """
    Solve the linear non-matching system ``Ax=b`` using Newton's iteration.
    A : petsc4py.PETSc.Mat
    x : petsc4py.PETSc.Vec
    b : petsc4py.PETSc.Vec
    max_it : int, optional, default is 10
    """
    ref_norm = b.norm()
    b_solve = b.copy()
    b_solve.zeroEntries()
    dx = x.copy()
    dx.zeroEntries()
    for i in range(max_it):
        A.mult(x, b_solve)
        b_diff = b_solve - b
        rel_norm = b_diff.norm()/ref_norm
        print("Iteration: {}, relative norm: {}".format(i, rel_norm))
        solve_nonmatching_mat(A, dx, -b_diff, solver='direct')
        x += dx

def solve_Ax_b(A, b, array=False, comm=DEFAULT_COMM):
    """
    Solve the linear non-matching system ``Ax=b`` using direct solver.

    Parameters
    ----------
    A : petsc4py.PETSc.Mat
    b : petsc4py.PETSc.Vec
    array : bool, optional, default is False
        if True, return solution as ndarray
    comm : mpi4py.MPI.Intracomm, optional

    Returns
    -------
    x : petsc4py.PETSc.Vec or ndarray
    """
    x = b.copy()
    x.zeroEntries()
    # if mpirank == 0:
    #     print("**** Solving Ax=b ...")
    solve_nonmatching_mat(A, x, b, solver='direct')
    x.assemble()
    if array:
        return get_petsc_vec_array(x, comm)
    else:
        return x

def solve_ATx_b(A, b, array=False, comm=DEFAULT_COMM):
    """
    Solve the linear non-matching system ``ATx=b`` using direct solver.

    Parameters
    ----------
    A : petsc4py.PETSc.Mat
    b : petsc4py.PETSc.Vec
    array : bool, optional, default is False
        if True, return solution as ndarray
    comm : mpi4py.MPI.Intracomm, optional

    Returns
    -------
    x : petsc4py.PETSc.Vec or ndarray
    """
    AT = A.transpose()
    x = b.copy()
    x.zeroEntries()
    # if mpirank == 0:
    #     print("**** Solving ATx=b ...")
    solve_nonmatching_mat(AT, x, b, solver='direct')
    x.assemble()
    if array:
        return get_petsc_vec_array(x, comm)
    else:
        return x


def dRmdcpm_sub(Rm_list, mortar_cpfuncs, field):
    """
    Compute the derivatives of the non-matching residual w.r.t.
    ``mortar_cpfuncs``
    """
    dRm_dcpm_list = [[[[None for l in range(len(mortar_cpfuncs[k]))]
                     for j in range(len(Rm_list[i]))]
                     for k in range(len(mortar_cpfuncs))]
                     for i in range(len(Rm_list))]
    for i in range(len(Rm_list)):
        for k in range(len(mortar_cpfuncs)):
            for j in range(len(Rm_list[i])):
                for l in range(len(mortar_cpfuncs[k])):
                    dRm_dcpm_list[i][k][j][l] = \
                        derivative(Rm_list[i][j], 
                                   mortar_cpfuncs[k][l][field])
    return dRm_dcpm_list


def transfer_dRmdcpm_sub(dRm_dcpm_list, Al_list, Ar_list):
    """
    Transfer ``dRmdcpm`` for each pair of non-matching shells
    from mortar mesh function space to spline FE function space.
    """
    dRm_dcpm = [[[[None for l in range(len(dRm_dcpm_list[i][j][k]))]
                       for k in range(len(dRm_dcpm_list[i][j]))]
                       for j in range(len(dRm_dcpm_list[i]))]
                       for i in range(len(dRm_dcpm_list))]
    dR_dcp_FE = [[None for j in range(len(dRm_dcpm_list[i]))]
                       for i in range(len(dRm_dcpm_list))]

    for i in range(len(dRm_dcpm_list)):
        for j in range(len(dRm_dcpm_list[i])):
            for k in range(len(dRm_dcpm_list[i][j])):
                for l in range(len(dRm_dcpm_list[i][j][k])):
                    dRm_dcpm[i][j][k][l] = m2p(
                        assemble(dRm_dcpm_list[i][j][k][l]))

    for i in range(len(dRm_dcpm_list)):
        for j in range(len(dRm_dcpm_list)):
            for k in range(len(dRm_dcpm_list[i][j])):
                for l in range(len(dRm_dcpm_list[i][j][k])):
                    dR_du_temp = AT_R_B(m2p(Al_list[i][k]), 
                        dRm_dcpm[i][j][k][l], m2p(Ar_list[j][l]))
                    if dR_dcp_FE[i][j] is None: 
                        dR_dcp_FE[i][j] = dR_du_temp
                    else:
                        dR_dcp_FE[i][j] += dR_du_temp
    return dR_dcp_FE

# def interface_angle(int_ind, nonmatching_problem, array=True):
#     """
#     Return the sin value of the angle of the intersection between
#     two patches.
#     """
#     s1_ind, s2_ind = nonmatching_problem.mapping_list[int_ind]

#     cpfuncs1, cpfuncs1_dxi1, cpfuncs1_dxi2 = transfer_cpfuns(
#         nonmatching_problem.splines[s1_ind],
#         nonmatching_problem.Vms_control[int_ind],
#         nonmatching_problem.dVms_control[int_ind],
#         nonmatching_problem.transfer_matrices_control_list[int_ind][0])

#     X1, dX1dxi = create_geometrical_mapping(
#         nonmatching_problem.splines[s1_ind],
#         cpfuncs1, cpfuncs1_dxi1, cpfuncs1_dxi2)

#     A11, A21, A31 = interface_geometry(dX1dxi)

#     cpfuncs2, cpfuncs2_dxi1, cpfuncs2_dxi2 = transfer_cpfuns(
#         nonmatching_problem.splines[s2_ind],
#         nonmatching_problem.Vms_control[int_ind],
#         nonmatching_problem.dVms_control[int_ind],
#         nonmatching_problem.transfer_matrices_control_list[int_ind][1])

#     X2, dX2dxi = create_geometrical_mapping(
#         nonmatching_problem.splines[s2_ind],
#         cpfuncs2, cpfuncs2_dxi1, cpfuncs2_dxi2)

#     A12, A22, A32 = interface_geometry(dX2dxi)

#     A3_cross = cross(A31, A32)
#     sin_angle = sqrt(inner(A3_cross, A3_cross))
    
#     sin_angle_vec = project(sin_angle, 
#                             nonmatching_problem.Vms_control[int_ind])

#     if array:
#         return get_petsc_vec_array(v2p(sin_angle_vec.vector()))
#     else:
#         return sin_angle_vec