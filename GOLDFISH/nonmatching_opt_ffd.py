from GOLDFISH.nonmatching_opt import *
from GOLDFISH.utils.ffd_utils import *

def ijk2dof(i, j, k, l, m):
    return i + j*l + k*(l*m)

class NonMatchingOptFFD(NonMatchingOpt):
    """
    Subclass of NonmatchingOpt which serves as the base class
    to setup optimization problem for non-matching structures.
    """
    def __init__(self, splines, E, h_th, nu, num_field=3, 
                 int_V_family='CG', int_V_degree=1, 
                 int_dx_metadata=None, contact=None, opt_field=[0,1,2],
                 comm=None):
        """
        Parameters
        ----------
        splines : list of ExtractedSplines
        E : ufl Constant or list, Young's modulus
        h_th : ufl Constant or list, thickness of the splines
        nu : ufl Constant or list, Poisson's ratio
        num_field : int, optional
            Number of field of the unknowns. Default is 3.
        int_V_family : str, optional, element family for 
            mortar meshes. Default is 'CG'.
        int_V_degree : int, optional, default is 1.
        int_dx_metadata : dict, optional
            Metadata information for integration measure of 
            intersection curves. Default is vertex quadrature
            with degree 0.
        contact : ShNAPr.contact.ShellContactContext, optional
        opt_field : list of ints, optional, default is [0,1,2]
            The fields of the splines' control points to be optimized.
        comm : mpi4py.MPI.Intracomm, optional, default is None.
        """
        super().__init__(splines, E, h_th, nu, num_field, 
                         int_V_family, int_V_degree,
                         int_dx_metadata, contact, opt_field, comm)

        # Bspline surfaces' control points in FE DoFs
        cpsurf_fe_temp = [[] for i in range(self.nsd)]
        self.cpsurf_lims = [None for i in range(self.nsd)]
        self.cpsurf_fe_list = [None for i in range(self.nsd)]
        for field in range(self.nsd):
            for s_ind in range(self.num_splines):
                cpsurf_fe_temp[field] += [get_petsc_vec_array(
                                         v2p(self.splines[s_ind].
                                         cpFuncs[field].vector()))]
            self.cpsurf_fe_list[field] = np.concatenate(cpsurf_fe_temp[field])
            self.cpsurf_lims[field] = [np.min(self.cpsurf_fe_list[field]),
                                      np.max(self.cpsurf_fe_list[field])]
        self.cpsurf_fe_list = np.array(self.cpsurf_fe_list).transpose(1,0)

        self.pin_dof = np.array([], dtype='int32')
        self.cppin_size = 0

    def set_FFD(self, knotsffd, cpffd):
        """
        ``cpffd`` is in the igakit order convention
        Assume FFD block has identity geometric mapping.

        Parameters
        ----------
        knotsffd : list of ndarray, ndarray, knots of FFD block
        cpffd : ndarray, control points of FFD block
        """
        self.knotsffd = knotsffd
        self.cpffd = cpffd
        self.cpffd_flat = self.cpffd[...,0:3].transpose(2,1,0,3).reshape(-1,3)
        self.ffd_degree = spline_degree(self.knotsffd[0], 
                                        self.knotsffd[0][0])
        self.cpffd_shape = self.cpffd.shape[0:3]
        self.cpffd_size = self.cpffd_shape[0]*self.cpffd_shape[1]\
                          *self.cpffd_shape[2]

        self.dcpsurf_fedcpffd = CP_FFD_matrix(self.cpsurf_fe_list,
                               [self.ffd_degree]*self.nsd, self.knotsffd)
        return self.dcpsurf_fedcpffd

    def set_regu_CPFFD(self, regu_dir, regu_side):
        """
        ``regu_dir`` is a list that has the same length with ``opt_field``.
        If the entry is None, that means regularize all layers minus one 
        of control points along the direction in ``opt_field``. If it's not
        None, we need the correspond value in ``regu_side`` to determine
        which side to regularize along the direction ``regu_dir``, the 
        value of ``regu_dir`` should be dirrection from ``opt_field``.
        Note: For i-th entry in ``regu_dir``, ``regu_dir[i]`` cannot be
        equal to ``opt_field[i]``, otherwise, this constraint would 
        be trivial and this function returns a zero Jacobian matrix.
        For example, when optimizing vectical coordinates of control points,
        ``opt_field=[2]``, if ``regu_dir=[None]``, that means all vertical
        coordinates are regularized to prevent self-penetration. If 
        ``regu_dir=[1]`` and ``regu_side=[0]``, that means only the control
        points on the surface along direction 1 side 0 is regularized.

        Parameters
        ----------
        regu_dir : list of ints
        regu_side : list of ints
        """
        self.regu_dir = regu_dir
        self.regu_side = regu_side

        self.cpregu_sizes = []
        for i, field in enumerate(self.opt_field):
            cpregu_size_temp = 1
            for j in range(self.nsd):
                if j == field:
                    cpregu_size_temp *= self.cpffd_shape[j]-1
                else:
                    cpregu_size_temp *= self.cpffd_shape[j]
            if self.regu_dir[i] is not None:
                cpregu_size_temp /= self.cpffd_shape[self.regu_dir[i]]
            self.cpregu_sizes += [int(cpregu_size_temp),]

        self.dcpregudcpffd_list = self.dCPregudCPFFD()
        return self.dcpregudcpffd_list

    def set_pin_CPFFD(self, pin_dir0, pin_side0=[0,1], 
                      pin_dir1=None, pin_side1=[0,1]):
        """
        Pin the control points with specific values.
        If pin_dir1 is None, a surface of DoFs will be pinned, if not, 
        a line of DoFs will be pinned.

        Parameters
        ----------
        pin_dir0 : int, {0,1,2}
        pin_side0 : list, default is [0, 1], pin both sides
        pin_dir1 : int or None, {0,1,2}, default is None
        pin_side1 : list, default is [0, 1] 
        """
        self.pin_dir0 = pin_dir0
        self.pin_dir1 = pin_dir1
        if isinstance(pin_side0, list):
            self.pin_side0 = pin_side0
        else:
            self.pin_side0 = [pin_side0]
        if isinstance(pin_side1, list):
            self.pin_side1 = pin_side1
        else:
            self.pin_side1 = [pin_side1]

        self.pin_dof = np.concatenate([self.pin_dof, self.CPpinDoFs()], 
                                      dtype='int32')
        self.pin_dof = np.unique(self.pin_dof)
        self.cppin_size = self.pin_dof.size
        self.dcppindcpffd = self.dCPpindCPFFD()
        return self.dcppindcpffd

    def set_align_CPFFD(self, align_dir):
        """
        Set the direction of the FFD block to be alignment so that the 
        control points have the same coordinates along ``align_dir``

        Parameters
        ----------
        align_dir : int
        """
        if not isinstance(align_dir, list):
            self.align_dir = [align_dir]
        else:
            self.align_dir = align_dir
        self.cp_align_size = 0
        for direction in self.align_dir:
            cp_align_size_sub = 1
            for i in range(self.nsd):
                if direction == i:
                    cp_align_size_sub *= self.cpffd_shape[i]-1
                else:
                    cp_align_size_sub *= self.cpffd_shape[i]
            self.cp_align_size += cp_align_size_sub
        self.dcpaligndcpffd = self.dCPaligndCPFFD()
        return self.dcpaligndcpffd

    def dCPregudCPFFD(self):
        derivs = [np.zeros((self.cpregu_sizes[i], self.cpffd_size)) 
                  for i in range(len(self.opt_field))]
        l, m = self.cpffd_shape[0], self.cpffd_shape[1]
        for field_ind, field in enumerate(self.opt_field):
            row_ind = 0
            if field == 0:
                if self.regu_dir[field_ind] is None:
                    for i in range(self.cpffd_shape[0]-1):
                        for j in range(self.cpffd_shape[1]):
                            for k in range(self.cpffd_shape[2]):
                                col_ind0 = ijk2dof(i, j, k, l, m) 
                                col_ind1 = ijk2dof(i+1, j, k, l, m)
                                derivs[field_ind][row_ind, col_ind0] = -1.
                                derivs[field_ind][row_ind, col_ind1] = 1.
                                row_ind += 1
                elif self.regu_dir[field_ind] == 1:
                    for i in range(self.cpffd_shape[0]-1):
                        for k in range(self.cpffd_shape[2]):
                            if self.regu_side[field_ind] == 0:
                                j = 0
                            elif self.regu_side[field_ind] == 1:
                                j = self.cpffd_shape[1]-1
                            col_ind0 = ijk2dof(i, j, k, l, m) 
                            col_ind1 = ijk2dof(i+1, j, k, l, m)
                            derivs[field_ind][row_ind, col_ind0] = -1.
                            derivs[field_ind][row_ind, col_ind1] = 1.
                            row_ind += 1
                elif self.regu_dir[field_ind] == 2:
                    for i in range(self.cpffd_shape[0]-1):
                        for j in range(self.cpffd_shape[1]):
                            if self.regu_side[field_ind] == 0:
                                k = 0
                            elif self.regu_side[field_ind] == 1:
                                k = self.cpffd_shape[2]-1
                            col_ind0 = ijk2dof(i, j, k, l, m) 
                            col_ind1 = ijk2dof(i+1, j, k, l, m)
                            derivs[field_ind][row_ind, col_ind0] = -1.
                            derivs[field_ind][row_ind, col_ind1] = 1.
                            row_ind += 1
            elif field == 1:
                if self.regu_dir[field_ind] is None:
                    for j in range(self.cpffd_shape[1]-1):
                        for i in range(self.cpffd_shape[0]):
                            for k in range(self.cpffd_shape[2]):
                                col_ind0 = ijk2dof(i, j, k, l, m) 
                                col_ind1 = ijk2dof(i, j+1, k, l, m)
                                derivs[field_ind][row_ind, col_ind0] = -1.
                                derivs[field_ind][row_ind, col_ind1] = 1.
                                row_ind += 1
                elif self.regu_dir[field_ind] == 0:
                    for j in range(self.cpffd_shape[1]-1):
                        for k in range(self.cpffd_shape[2]):
                            if self.regu_side[field_ind] == 0:
                                i = 0
                            elif self.regu_side[field_ind] == 1:
                                i = self.cpffd_shape[0]-1
                            col_ind0 = ijk2dof(i, j, k, l, m) 
                            col_ind1 = ijk2dof(i, j+1, k, l, m)
                            derivs[field_ind][row_ind, col_ind0] = -1.
                            derivs[field_ind][row_ind, col_ind1] = 1.
                            row_ind += 1
                elif self.regu_dir[field_ind] == 2:
                    for j in range(self.cpffd_shape[1]-1):
                        for i in range(self.cpffd_shape[0]):
                            if self.regu_side[field_ind] == 0:
                                k = 0
                            elif self.regu_side[field_ind] == 1:
                                k = self.cpffd_shape[2]-1
                            col_ind0 = ijk2dof(i, j, k, l, m) 
                            col_ind1 = ijk2dof(i, j+1, k, l, m)
                            derivs[field_ind][row_ind, col_ind0] = -1.
                            derivs[field_ind][row_ind, col_ind1] = 1.
                            row_ind += 1
            elif field == 2:
                if self.regu_dir[field_ind] is None:
                    for k in range(self.cpffd_shape[2]-1):
                        for i in range(self.cpffd_shape[0]):
                            for j in range(self.cpffd_shape[1]):
                                col_ind0 = ijk2dof(i, j, k, l, m) 
                                col_ind1 = ijk2dof(i, j, k+1, l, m)
                                derivs[field_ind][row_ind, col_ind0] = -1.
                                derivs[field_ind][row_ind, col_ind1] = 1.
                                row_ind += 1
                elif self.regu_dir[field_ind] == 0:
                    for k in range(self.cpffd_shape[2]-1):
                        for j in range(self.cpffd_shape[1]):
                            if self.regu_side[field_ind] == 0:
                                i = 0
                            elif self.regu_side[field_ind] == 1:
                                i = self.cpffd_shape[0] - 1
                            col_ind0 = ijk2dof(i, j, k, l, m) 
                            col_ind1 = ijk2dof(i, j, k+1, l, m)
                            derivs[field_ind][row_ind, col_ind0] = -1.
                            derivs[field_ind][row_ind, col_ind1] = 1.
                            row_ind += 1
                elif self.regu_dir[field_ind] == 1:
                    for k in range(self.cpffd_shape[2]-1):
                        for i in range(self.cpffd_shape[0]):
                            if self.regu_side[field_ind] == 0:
                                j = 0
                            elif self.regu_side[field_ind] == 1:
                                j = self.cpffd_shape[1] - 1
                            col_ind0 = ijk2dof(i, j, k, l, m) 
                            col_ind1 = ijk2dof(i, j, k+1, l, m)
                            derivs[field_ind][row_ind, col_ind0] = -1.
                            derivs[field_ind][row_ind, col_ind1] = 1.
                            row_ind += 1
        derivs_coo = [coo_matrix(A) for A in derivs]
        return derivs_coo

    def CPpinDoFsOld(self):
        pin_dof = []
        l, m = self.cpffd_shape[0], self.cpffd_shape[1]
        for side in self.pin_side:
            if self.pin_dir0 == 0:
                if self.pin_dir1 is not None:
                    if self.pin_dir1 == 1:
                        for k in range(self.cpffd_shape[2]):
                            pin_ijk = [int(side*(self.cpffd_shape[0]-1)), 
                                       0, k]
                            pin_dof += [ijk2dof(*pin_ijk, l, m)]
                    elif self.pin_dir1 == 2:
                        for j in range(self.cpffd_shape[1]):
                            pin_ijk = [int(side*(self.cpffd_shape[0]-1)), 
                                       j, 0]
                            pin_dof += [ijk2dof(*pin_ijk, l, m)]
                    else:
                        raise ValueError("Unsupported pin_dir1 {}".
                                         format(self.pin_dir1))
                else:
                    for k in range(self.cpffd_shape[2]):
                        for j in range(self.cpffd_shape[1]):
                            pin_ijk = [int(side*(self.cpffd_shape[0]-1)), 
                                       j, k]
                            pin_dof += [ijk2dof(*pin_ijk, l, m)]
            elif self.pin_dir0 == 1:
                if self.pin_dir1 is not None:
                    if self.pin_dir1 == 0:
                        for k in range(self.cpffd_shape[2]):
                            pin_ijk = [0, int(side*(self.cpffd_shape[1]-1)), 
                                       k]
                            pin_dof += [ijk2dof(*pin_ijk, l, m)]
                    elif self.pin_dir1 == 2:
                        for i in range(self.cpffd_shape[0]):
                            pin_ijk = [i, int(side*(self.cpffd_shape[1]-1)), 
                                       0]
                            pin_dof += [ijk2dof(*pin_ijk, l, m)]
                    else:
                        raise ValueError("Unsupported pin_dir1 {}".
                                         format(self.pin_dir1))
                else:
                    for k in range(self.cpffd_shape[2]):
                        for i in range(self.cpffd_shape[0]):
                            pin_ijk = [i, int(side*(self.cpffd_shape[1]-1)), 
                                       k]
                            pin_dof += [ijk2dof(*pin_ijk, l, m)]
            elif self.pin_dir0 == 2:
                if self.pin_dir1 is not None:
                    if self.pin_dir1 == 0:
                        for j in range(self.cpffd_shape[1]):
                            pin_ijk = [0, j, 
                                       int(side*(self.cpffd_shape[2]-1))]
                            pin_dof += [ijk2dof(*pin_ijk, l, m)]
                    elif self.pin_dir1 == 1:
                        for i in range(self.cpffd_shape[0]):
                            pin_ijk = [i, 0, 
                                       int(side*(self.cpffd_shape[2]-1))]
                            pin_dof += [ijk2dof(*pin_ijk, l, m)]
                    else:
                        raise ValueError("Unsupported pin_dir1 {}".
                                         format(self.pin_dir1))
                else:
                    for j in range(self.cpffd_shape[1]):
                        for i in range(self.cpffd_shape[0]):
                            pin_ijk = [i, j, 
                                       int(side*(self.cpffd_shape[2]-1))]
                            pin_dof += [ijk2dof(*pin_ijk, l, m)]
            else:
                raise ValueError("Unsupported pin_dir0 {}".
                                 format(self.pin_dir0))
        pin_dof = np.array(pin_dof)
        return pin_dof

    def CPpinDoFs(self):
        pin_dof = []
        l, m = self.cpffd_shape[0], self.cpffd_shape[1]
        for side0 in self.pin_side0:
            if self.pin_dir0 == 0:
                if self.pin_dir1 is not None:
                    for side1 in self.pin_side1:
                        if self.pin_dir1 == 1:
                            for k in range(self.cpffd_shape[2]):
                                pin_ijk = [int(side0*(self.cpffd_shape[0]-1)), 
                                           int(side1*(self.cpffd_shape[1]-1)), 
                                           k]
                                pin_dof += [ijk2dof(*pin_ijk, l, m)]
                        elif self.pin_dir1 == 2:
                            for j in range(self.cpffd_shape[1]):
                                pin_ijk = [int(side0*(self.cpffd_shape[0]-1)), 
                                           j, int(side1*(self.cpffd_shape[2]-1))]
                                pin_dof += [ijk2dof(*pin_ijk, l, m)]
                        else:
                            raise ValueError("Unsupported pin_dir1 {}".
                                             format(self.pin_dir1))
                else:
                    for k in range(self.cpffd_shape[2]):
                        for j in range(self.cpffd_shape[1]):
                            pin_ijk = [int(side0*(self.cpffd_shape[0]-1)), 
                                       j, k]
                            pin_dof += [ijk2dof(*pin_ijk, l, m)]
            elif self.pin_dir0 == 1:
                if self.pin_dir1 is not None:
                    for side1 in self.pin_side1:
                        if self.pin_dir1 == 0:
                            for k in range(self.cpffd_shape[2]):
                                pin_ijk = [int(side1*(self.cpffd_shape[0]-1)), 
                                           int(side0*(self.cpffd_shape[1]-1)), 
                                           k]
                                pin_dof += [ijk2dof(*pin_ijk, l, m)]
                        elif self.pin_dir1 == 2:
                            for i in range(self.cpffd_shape[0]):
                                pin_ijk = [i, int(side0*(self.cpffd_shape[1]-1)), 
                                           int(side1*(self.cpffd_shape[2]-1))]
                                pin_dof += [ijk2dof(*pin_ijk, l, m)]
                        else:
                            raise ValueError("Unsupported pin_dir1 {}".
                                             format(self.pin_dir1))
                else:
                    for k in range(self.cpffd_shape[2]):
                        for i in range(self.cpffd_shape[0]):
                            pin_ijk = [i, int(side0*(self.cpffd_shape[1]-1)), 
                                       k]
                            pin_dof += [ijk2dof(*pin_ijk, l, m)]
            elif self.pin_dir0 == 2:
                if self.pin_dir1 is not None:
                    for side1 in self.pin_side1:
                        if self.pin_dir1 == 0:
                            for j in range(self.cpffd_shape[1]):
                                pin_ijk = [int(side1*(self.cpffd_shape[0]-1)), j, 
                                           int(side0*(self.cpffd_shape[2]-1))]
                                pin_dof += [ijk2dof(*pin_ijk, l, m)]
                        elif self.pin_dir1 == 1:
                            for i in range(self.cpffd_shape[0]):
                                pin_ijk = [i, int(side1*(self.cpffd_shape[1]-1)), 
                                           int(side0*(self.cpffd_shape[2]-1))]
                                pin_dof += [ijk2dof(*pin_ijk, l, m)]
                        else:
                            raise ValueError("Unsupported pin_dir1 {}".
                                             format(self.pin_dir1))
                else:
                    for j in range(self.cpffd_shape[1]):
                        for i in range(self.cpffd_shape[0]):
                            pin_ijk = [i, j, 
                                       int(side0*(self.cpffd_shape[2]-1))]
                            pin_dof += [ijk2dof(*pin_ijk, l, m)]
            else:
                raise ValueError("Unsupported pin_dir0 {}".
                                 format(self.pin_dir0))
        pin_dof = np.array(pin_dof)
        return pin_dof

    def dCPpindCPFFD(self):
        deriv = np.zeros((self.pin_dof.size, self.cpffd_size))
        for i in range(self.pin_dof.size):
            deriv[i, self.pin_dof[i]] = 1
        deriv_coo = coo_matrix(deriv)
        return deriv_coo

    def dCPaligndCPFFD(self):
        deriv = np.zeros((self.cp_align_size, self.cpffd_size))
        row_ind, l, m = 0, self.cpffd_shape[0], self.cpffd_shape[1]
        for direction in self.align_dir:
            if direction == 0:
                for k in range(self.cpffd_shape[2]):
                    for j in range(self.cpffd_shape[1]):
                        for i in range(1, self.cpffd_shape[0]):
                            col_ind0 = ijk2dof(0, j, k, l, m)
                            col_ind1 = ijk2dof(i, j, k, l, m)
                            deriv[row_ind, col_ind0] = 1.
                            deriv[row_ind, col_ind1] = -1.
                            row_ind += 1
            elif direction == 1:
                for k in range(self.cpffd_shape[2]):
                    for i in range(self.cpffd_shape[0]):
                        for j in range(1, self.cpffd_shape[1]):
                            col_ind0 = ijk2dof(i, 0, k, l, m)
                            col_ind1 = ijk2dof(i, j, k, l, m)
                            deriv[row_ind, col_ind0] = 1.
                            deriv[row_ind, col_ind1] = -1.
                            row_ind += 1
            elif direction == 2:
                for j in range(self.cpffd_shape[1]):    
                    for i in range(self.cpffd_shape[0]):
                        for k in range(1,self.cpffd_shape[2]):
                            col_ind0 = ijk2dof(i, j, 0, l, m)
                            col_ind1 = ijk2dof(i, j, k, l, m)
                            deriv[row_ind, col_ind0] = 1.
                            deriv[row_ind, col_ind1] = -1.
                            row_ind += 1
        deriv_coo = coo_matrix(deriv)
        return deriv_coo

if __name__ == '__main__':
    pass