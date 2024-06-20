from GOLDFISH.nonmatching_opt import *
from GOLDFISH.utils.ffd_utils import *
from scipy.sparse import block_diag, bmat, identity

def ijk2dof(i, j, k, l, m):
    return i + j*l + k*(l*m)

class NonMatchingOptFFD(NonMatchingOpt):
    """
    Subclass of NonmatchingOpt which serves as the base class
    to setup optimization problem for non-matching structures.
    """
    def __init__(self, splines, E, h_th, nu, num_field=3, 
                 int_V_family='CG', int_V_degree=1,
                 int_dx_metadata=None, contact=None, 
                 opt_shape=True, opt_field=[0,1,2], 
                 opt_thickness=False, var_thickness=False, 
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
                         int_V_family, int_V_degree, int_dx_metadata, 
                         contact, opt_shape, opt_field, opt_thickness, 
                         var_thickness, comm)

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

        if opt_thickness and var_thickness:
            self.init_h_th_ffd = None

    def set_FFD(self, knotsffd, cpffd):
        """
        ``cpffd`` is in the igakit order convention
        Assume FFD block has identity geometric mapping.

        Parameters
        ----------
        knotsffd : list of ndarray, ndarray, knots of FFD block
        cpffd : ndarray, control points of FFD block
        """
        self.multiffd = False
        self.knotsffd = knotsffd
        self.cpffd = cpffd
        self.cpffd_flat = self.cpffd[...,0:3].transpose(2,1,0,3).reshape(-1,3)
        self.ffd_degree = spline_degree(self.knotsffd[0], 
                                        self.knotsffd[0][0])
        self.cpffd_shape = self.cpffd.shape[0:3]
        self.cpffd_size = self.cpffd_shape[0]*self.cpffd_shape[1]\
                          *self.cpffd_shape[2]
        self.cpffd_design_size = self.cpffd_size

        self.dcpsurf_fedcpffd = CP_FFD_matrix(self.cpsurf_fe_list,
                               [self.ffd_degree]*self.nsd, self.knotsffd)
        return self.dcpsurf_fedcpffd

    def get_init_h_th_FFD(self):
        if self.init_h_th_ffd is None:
            dfedffd_dense = self.dcpsurf_fedcpffd.todense()
            h_th_fe = self.init_h_th_fe.reshape(dfedffd_dense.shape[0], 1)
            init_h_th_ffd = solve_nonsquare(dfedffd_dense, h_th_fe)
            self.init_h_th_ffd = np.asarray(init_h_th_ffd).reshape(-1)
        return self.init_h_th_ffd

    def set_multiFFD_surf_inds(self, multiffd_surf_ind_list):
        self.multiffd = True
        self.multiffd_surf_ind_list = multiffd_surf_ind_list
        self.num_ffd_blocks = len(self.multiffd_surf_ind_list)
        
        # Sort shell indices based on ffd and non-ffd 
        shell_inds = list(range(self.num_splines))
        self.multiffd_shell_inds = np.concatenate(self.multiffd_surf_ind_list)
        self.nonffd_shell_inds = [s_ind for s_ind in shell_inds
                                 if s_ind not in self.multiffd_shell_inds]
        self.num_nonffd_shells = len(self.nonffd_shell_inds)
        if self.num_nonffd_shells > 0:
            self.mixedffd_shell_inds = np.concatenate([self.multiffd_shell_inds,
                                       self.nonffd_shell_inds])
        else:
            self.mixedffd_shell_inds = self.multiffd_shell_inds            

        # Get control point functions and lims for each FFD blocks
        cpsurf_fe_multiffd_temp = [[[] for i in range(self.nsd)]
                                    for j in range(self.num_ffd_blocks)]
        self.cpsurf_lims_multiffd = [[None for i in range(self.nsd)]
                                      for j in range(self.num_ffd_blocks)]
        self.cpsurf_fe_list_multiffd = [[None for i in range(self.nsd)]
                                        for j in range(self.num_ffd_blocks)]
        for ffd_ind in range(self.num_ffd_blocks):
            for field in range(self.nsd):
                for s_ind in self.multiffd_surf_ind_list[ffd_ind]:
                    cpsurf_fe_multiffd_temp[ffd_ind][field] += \
                        [get_petsc_vec_array(v2p(self.splines[s_ind].
                         cpFuncs[field].vector()))]
                self.cpsurf_fe_list_multiffd[ffd_ind][field] = \
                    np.concatenate(cpsurf_fe_multiffd_temp[ffd_ind][field])
                self.cpsurf_lims_multiffd[ffd_ind][field] = \
                    [np.min(self.cpsurf_fe_list_multiffd[ffd_ind][field]),
                     np.max(self.cpsurf_fe_list_multiffd[ffd_ind][field])]
            self.cpsurf_fe_list_multiffd[ffd_ind] = \
                np.array(self.cpsurf_fe_list_multiffd[ffd_ind]).transpose(1,0)

    def h_th_FE_reorder(self):
        diag_vecs = []
        for s_ind in range(self.num_splines):
            diag_vecs += [np.eye((self.h_th_sizes[s_ind]))]
        block_mat_list = [[] for s_ind in range(self.num_splines)]
        for s_ind in range(self.num_splines):
            for ffd_ind in self.mixedffd_shell_inds:
                if s_ind == ffd_ind:
                    block_mat_list[s_ind] += [identity(self.h_th_sizes[s_ind],
                                                       format='coo')]
                else:
                    block_mat_list[s_ind] += [coo_matrix((self.h_th_sizes[s_ind],
                                                          self.h_th_sizes[ffd_ind]))]
        deriv_mat = bmat(block_mat_list, format='coo')
        return deriv_mat

    def set_multiFFD(self, knotsffd_list, cpffd_list):
        self.knotsffd_list = knotsffd_list
        self.cpffd_list = cpffd_list

        self.cpffd_flat_list = [cpffd[...,0:3].transpose(2,1,0,3).reshape(-1,3)
                                for cpffd in self.cpffd_list]
        self.cpffd_degree_list = [spline_degree(knotsffd[0], knotsffd[0][0])
                                  for knotsffd in self.knotsffd_list]
        self.cpffd_shape_list = [cpffd.shape[0:3] for cpffd in self.cpffd_list]
        self.cpffd_size_list = [cpffd_shape[0]*cpffd_shape[1]*cpffd_shape[2]
                                for cpffd_shape in self.cpffd_shape_list]
        self.cpffd_design_size = np.sum(self.cpffd_size_list)
        self.dcpsurf_fedcpffd_list = []
        for ffd_ind in range(self.num_ffd_blocks):
            self.dcpsurf_fedcpffd_list += [CP_FFD_matrix(
                                    self.cpsurf_fe_list_multiffd[ffd_ind],
                                    [self.cpffd_degree_list[ffd_ind]]*self.nsd,
                                    self.knotsffd_list[ffd_ind])]
        self.dcpsurf_fedcpffd_mat = block_diag(self.dcpsurf_fedcpffd_list)

        # Get derivative of surface control points in FE function space
        # w.r.t. FFD blocks and constant thickness if any
        if self.num_nonffd_shells > 0:
            self.dcpsurf_fedcpconst_list = [np.ones((self.h_th_sizes[ind], 1))
                                            for ind in self.nonffd_shell_inds]
            self.dcpsurf_fedcpconst_mat = block_diag(self.dcpsurf_fedcpconst_list)
            self.dcpsurf_fedcpmultiffd = block_diag([self.dcpsurf_fedcpffd_mat,
                                                     self.dcpsurf_fedcpconst_mat])
            self.cpffd_design_size += self.num_nonffd_shells
        else:
            self.dcpsurf_fedcpmultiffd = self.dcpsurf_fedcpffd_mat
        self.h_th_fe_reorder_mat = self.h_th_FE_reorder()
        self.dcpsurf_fedcpmultiffd = coo_matrix(self.h_th_fe_reorder_mat
                                                *self.dcpsurf_fedcpmultiffd)
        return self.dcpsurf_fedcpmultiffd

    def get_init_h_th_multiFFD(self):
        if self.init_h_th_ffd is None:
            init_h_th_ffd_list = []
            for i in range(len(self.dcpsurf_fedcpffd_list)):
                dfedffd_dense = self.dcpsurf_fedcpffd_list[i].todense()
                h_th_fe = [get_petsc_vec_array(self.h_th_fe_list[ind]) for
                           ind in self.multiffd_surf_ind_list[i]]
                h_th_fe = np.concatenate(h_th_fe).reshape(-1,1)
                init_h_th_ffd = solve_nonsquare(dfedffd_dense, h_th_fe)
                init_h_th_ffd = np.asarray(init_h_th_ffd).reshape(-1)
                init_h_th_ffd_list += [init_h_th_ffd]
            self.init_h_th_ffd = np.concatenate(init_h_th_ffd_list)
            if self.num_nonffd_shells > 0:
                self.init_h_th_const = [self.h_th_fe_list[ind][0]
                                        for ind in self.nonffd_shell_inds]
                self.init_h_th_multiffd = np.concatenate([self.init_h_th_ffd,
                                          self.init_h_th_const])
            else:
                self.init_h_th_multiffd = self.init_h_th_ffd
        return self.init_h_th_multiffd

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
        This function is used as a linear inequality constraint.

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
        This function is used as a linear equality constraint.

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
        control points have the same coordinates along ``align_dir``.
        This function is used as a linear equality constraint.

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
        self.dcpaligndcpffd = self.dCPaligndCPFFD(self.align_dir, 
                              self.cp_align_size, self.cpffd_size, 
                              self.cpffd_shape)
        return self.dcpaligndcpffd

    def set_align_CP_multiFFD(self, align_dir_list):
        assert len(align_dir_list) == self.num_ffd_blocks
        self.align_dir_list = align_dir_list
        self.dcpaligndcpffd_list = []
        for ffd_ind in range(self.num_ffd_blocks):
            align_dir = align_dir_list[ffd_ind]
            if not isinstance(align_dir, list):
                align_dir = [align_dir]
            cp_align_size = 0
            for direction in align_dir:
                cp_align_size_sub = 1
                for i in range(self.nsd):
                    if direction == i:
                        cp_align_size_sub *= self.cpffd_shape_list[ffd_ind][i]-1
                    else:
                        cp_align_size_sub *= self.cpffd_shape_list[ffd_ind][i]
                cp_align_size += cp_align_size_sub
            self.dcpaligndcpffd_list += [self.dCPaligndCPFFD(align_dir, 
                                  cp_align_size, self.cpffd_size_list[ffd_ind], 
                                  self.cpffd_shape_list[ffd_ind])]
        self.dcpaligndcpffd = block_diag(self.dcpaligndcpffd_list)

        if self.num_nonffd_shells > 0:
            temp_block = coo_matrix((self.dcpaligndcpffd.shape[0],
                                    self.num_nonffd_shells))
            self.dcpaligndcpmultiffd = bmat([[self.dcpaligndcpffd, temp_block]])
        else:
            self.dcpaligndcpmultiffd = self.dcpaligndcpffd
        return self.dcpaligndcpmultiffd

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

    def dCPaligndCPFFD(self, align_dir, cp_align_size, 
                       cpffd_size, cpffd_shape):
        deriv = np.zeros((cp_align_size, cpffd_size))
        row_ind, l, m = 0, cpffd_shape[0], cpffd_shape[1]
        for direction in align_dir:
            if direction == 0:
                for k in range(cpffd_shape[2]):
                    for j in range(cpffd_shape[1]):
                        for i in range(1, cpffd_shape[0]):
                            col_ind0 = ijk2dof(0, j, k, l, m)
                            col_ind1 = ijk2dof(i, j, k, l, m)
                            deriv[row_ind, col_ind0] = 1.
                            deriv[row_ind, col_ind1] = -1.
                            row_ind += 1
            elif direction == 1:
                for k in range(cpffd_shape[2]):
                    for i in range(cpffd_shape[0]):
                        for j in range(1, cpffd_shape[1]):
                            col_ind0 = ijk2dof(i, 0, k, l, m)
                            col_ind1 = ijk2dof(i, j, k, l, m)
                            deriv[row_ind, col_ind0] = 1.
                            deriv[row_ind, col_ind1] = -1.
                            row_ind += 1
            elif direction == 2:
                for j in range(cpffd_shape[1]):    
                    for i in range(cpffd_shape[0]):
                        for k in range(1,cpffd_shape[2]):
                            col_ind0 = ijk2dof(i, j, 0, l, m)
                            col_ind1 = ijk2dof(i, j, k, l, m)
                            deriv[row_ind, col_ind0] = 1.
                            deriv[row_ind, col_ind1] = -1.
                            row_ind += 1
        deriv_coo = coo_matrix(deriv)
        return deriv_coo

    def save_FFD_block(self, ind):
        # Save FFD block for visualization
        CP2 = inputs[self.input_CP_name_list[0]]
        FFD_shape = self.FFD_block.shape
        CP_FFD = self.FFD_block.control[...,0:3].transpose(2,1,0,3).\
                    reshape(-1,3)
        CP_FFD[:,2] = CP2
        CP_FFD_temp = CP_FFD.reshape(self.FFD_block.control[:,:,:,0:3]\
                        .transpose(2,1,0,3).shape).transpose(2,1,0,3)
        FFD_temp = NURBS(self.FFD_block.knots, CP_FFD_temp)

        FFD_path = './geometry/'
        FFD_vtk_name_pre = 'FFD_block_opt'
        FFD_names = glob.glob(FFD_path+FFD_vtk_name_pre+'*.vtu')
        name_ind = len(FFD_names)
        name_ind_str = '0'*(6-len(str(name_ind)))+str(name_ind)
        FFD_vtk_name = FFD_vtk_name_pre+name_ind_str+'.vtk'
        FFD_vtu_name = FFD_vtk_name_pre+name_ind_str+'.vtu'

        print("FFD_vtk_name:", FFD_vtk_name)
        print("FFD_vtu_name:", FFD_vtu_name)

        VTK().write(FFD_path+FFD_vtk_name, FFD_temp)
        vtk_file = meshio.read(FFD_path+FFD_vtk_name)
        meshio.write(FFD_path+FFD_vtu_name, vtk_file)


if __name__ == '__main__':
    pass