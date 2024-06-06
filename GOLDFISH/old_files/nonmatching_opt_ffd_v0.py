from GOLDFISH.nonmatching_opt import *
from GOLDFISH.utils.ffd_utils import *
from scipy.sparse import block_diag, bmat, identity
import meshio

def ijk2dof(i, j, k, l, m):
    return i + j*l + k*(l*m)

class NonMatchingOptFFD(NonMatchingOpt):
    """
    Subclass of NonmatchingOpt which serves as the base class
    to setup optimization problem for non-matching structures.
    """
    def __init__(self, splines, E, h_th, nu, 
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
        super().__init__(splines, E, h_th, nu, 
                         int_V_family, int_V_degree, int_dx_metadata, 
                         contact, opt_shape, opt_field, opt_thickness, 
                         var_thickness, comm)

        # Bspline surfaces' control points in FE DoFs
        cpsurf_fe_temp = [[] for i in range(self.nsd+1)]
        self.cpsurf_fe_hom_list = [None for i in range(self.nsd+1)]
        for field in range(self.nsd+1):
            for s_ind in range(self.num_splines):
                cpsurf_fe_temp[field] += [get_petsc_vec_array(
                                         v2p(self.splines[s_ind].
                                         cpFuncs[field].vector()))]
            self.cpsurf_fe_hom_list[field] = np.concatenate(
                                             cpsurf_fe_temp[field])
        self.cpsurf_fe_hom_list = np.array(self.cpsurf_fe_hom_list).transpose(1,0)
 
        self.cpsurf_fe_list = [None for i in range(self.nsd)]
        self.cpsurf_lims = [None for i in range(self.nsd)]
        for field in range(self.nsd):
                self.cpsurf_fe_list[field] = self.cpsurf_fe_hom_list[:,field]\
                                             /self.cpsurf_fe_hom_list[:,-1]
                self.cpsurf_lims[field] = [np.min(self.cpsurf_fe_list[field]),
                                           np.max(self.cpsurf_fe_list[field])]
        self.cpsurf_fe_list = np.array(self.cpsurf_fe_list).transpose(1,0)

        self.shopt_pin_dof = np.array([], dtype='int32')
        self.shopt_cppin_size = 0
        self.shopt_align_dir = None

        if opt_thickness and var_thickness:
            self.init_h_th_ffd = None

    ##################################
    # Methods for shape optimization #
    ##################################

    # def set_FFD(self, knotsffd, cpffd):
    #     """
    #     ``cpffd`` is in the igakit order convention
    #     Assume FFD block has identity geometric mapping.

    #     Parameters
    #     ----------
    #     knotsffd : list of ndarray, ndarray, knots of FFD block
    #     cpffd : ndarray, control points of FFD block
    #     """
    #     self.multiffd = False
    #     self.knotsffd = knotsffd
    #     self.cpffd = cpffd
    #     self.cpffd_flat = self.cpffd[...,0:3].transpose(2,1,0,3).reshape(-1,3)
    #     self.ffd_degree = spline_degree(self.knotsffd[0], 
    #                                     self.knotsffd[0][0])
    #     self.cpffd_shape = self.cpffd.shape[0:3]
    #     self.cpffd_size = self.cpffd_shape[0]*self.cpffd_shape[1]\
    #                       *self.cpffd_shape[2]
    #     self.cpffd_design_size = self.cpffd_size

    #     self.dcpsurf_fedcpffd = CP_FFD_matrix(self.cpsurf_fe_hom_list,
    #                            [self.ffd_degree]*self.nsd, self.knotsffd)
    #     return self.dcpsurf_fedcpffd

    def set_shopt_FFD(self, shopt_knotsffd, shopt_cpffd):
        """
        ``shopt_cpffd`` is in the igakit order convention
        Assume FFD block has identity geometric mapping.

        Parameters
        ----------
        shopt_knotsffd : list of ndarray, ndarray, knots of FFD block
        shopt_cpffd : ndarray, control points of FFD block
        """
        self.shopt_multiffd = False
        self.shopt_knotsffd = shopt_knotsffd
        self.shopt_cpffd = shopt_cpffd
        self.shopt_cpffd_flat = self.shopt_cpffd[...,0:3].transpose(2,1,0,3)\
                                .reshape(-1,3)
        self.shopt_ffd_degree = spline_degree(self.shopt_knotsffd[0], 
                                              self.shopt_knotsffd[0][0])
        self.shopt_cpffd_shape = self.shopt_cpffd.shape[0:3]
        self.shopt_cpffd_size = np.prod(self.shopt_cpffd_shape)
        self.shopt_cpffd_design_size = self.shopt_cpffd_size

        self.shopt_dcpsurf_fedcpffd = CP_FFD_matrix(self.cpsurf_fe_hom_list,
                    [self.shopt_ffd_degree]*self.nsd, self.shopt_knotsffd)
        return self.shopt_dcpsurf_fedcpffd

    def set_shopt_multiFFD_surf_inds(self, shopt_multiffd_surf_ind_list):
        """
        Place holder for shape optimization with multiple FFD blocks,
        this involves change of parametric location of surface--surface
        intersections.
        """
        pass

    def set_shopt_multiFFD(self, shopt_knotsffd_list, shopt_cpffd_list):
        """
        Place holder for shape optimization with multiple FFD blocks,
        this involves change of parametric location of surface--surface
        intersections.
        """
        pass


    ######################################
    # Methods for thickness optimization #
    ######################################

    def set_thopt_FFD(self, thopt_knotsffd, thopt_cpffd):
        """
        ``thopt_cpffd`` is in the igakit order convention
        Assume FFD block has identity geometric mapping.

        Parameters
        ----------
        thopt_knotsffd : list of ndarray, ndarray, knots of FFD block
        thopt_cpffd : ndarray, control points of FFD block
        """
        self.thopt_multiffd = False
        self.thopt_knotsffd = thopt_knotsffd
        self.thopt_cpffd = thopt_cpffd
        self.thopt_cpffd_flat = self.thopt_cpffd[...,0:3].transpose(2,1,0,3)\
                                .reshape(-1,3)
        self.thopt_ffd_degree = spline_degree(self.thopt_knotsffd[0], 
                                              self.thopt_knotsffd[0][0])
        self.thopt_cpffd_shape = self.thopt_cpffd.shape[0:3]
        self.thopt_cpffd_size = np.prod(self.thopt_cpffd_shape)
        self.thopt_cpffd_design_size = self.thopt_cpffd_size

        self.thopt_dcpsurf_fedcpffd = CP_FFD_matrix(self.cpsurf_fe_hom_list,
                    [self.thopt_ffd_degree]*self.nsd, self.thopt_knotsffd)
        return self.thopt_dcpsurf_fedcpffd

    def get_init_h_th_FFD(self):
        """
        Get initial shell thickness values in IGA DoFs
        """
        if self.init_h_th_ffd is None:
            dfedffd_dense = self.thopt_dcpsurf_fedcpffd.todense()
            h_th_fe = self.init_h_th_fe.reshape(dfedffd_dense.shape[0], 1)
            init_h_th_ffd = solve_nonsquare(dfedffd_dense, h_th_fe)
            self.init_h_th_ffd = np.asarray(init_h_th_ffd).reshape(-1)
        return self.init_h_th_ffd

    def set_thopt_multiFFD_surf_inds(self, thopt_multiffd_surf_ind_list):
        """
        ``thopt_multiffd_surf_ind_list`` is a list contains ``num_thopt_ffd``
        elements, each element is a list contains (one or more) shell indices 
        in corresponding FFD block. Shell indices not in this list will
        have constant thickness distribution.

        Parameters
        ----------
        thopt_multiffd_surf_ind_list : list of lists
        """
        self.thopt_multiffd = True
        self.thopt_multiffd_surf_ind_list = thopt_multiffd_surf_ind_list
        self.num_thopt_ffd = len(self.thopt_multiffd_surf_ind_list)
        
        # Sort shell indices based on ffd and non-ffd 
        shell_inds = list(range(self.num_splines))
        # Shell indices in FFD blocks
        self.thopt_ffd_shell_inds = np.concatenate(
                                    self.thopt_multiffd_surf_ind_list)
        # Shell indices not in FFD blocks (these shell would have 
        # constant thickness distribution)
        self.thopt_nonffd_shell_inds = [s_ind for s_ind in shell_inds
                            if s_ind not in self.thopt_ffd_shell_inds]
        self.num_thopt_nonffd_shells = len(self.thopt_nonffd_shell_inds)
        # Get mixed shell indices for thickness optimization
        if self.num_thopt_nonffd_shells > 0:
            self.thopt_mixedffd_shell_inds = np.concatenate(
                                             [self.thopt_ffd_shell_inds,
                                              self.thopt_nonffd_shell_inds])
        else:
            self.thopt_mixedffd_shell_inds = self.thopt_ffd_shell_inds            

        # # Get control point functions and lims for each FFD blocks
        # thopt_cpsurf_fe_multiffd_temp = [[[] for i in range(self.nsd)]
        #                                  for j in range(self.num_thopt_ffd)]
        # self.thopt_cpsurf_lims_multiffd = [[None for i in range(self.nsd)]
        #                                    for j in range(self.num_thopt_ffd)]
        # self.thopt_cpsurf_fe_list_multiffd = [[None for i in range(self.nsd)]
        #                                    for j in range(self.num_thopt_ffd)]
        # for ffd_ind in range(self.num_thopt_ffd):
        #     for field in range(self.nsd):
        #         for s_ind in self.thopt_multiffd_surf_ind_list[ffd_ind]:
        #             thopt_cpsurf_fe_multiffd_temp[ffd_ind][field] += \
        #                 [get_petsc_vec_array(v2p(self.splines[s_ind].
        #                  cpFuncs[field].vector()))]
        #         self.thopt_cpsurf_fe_list_multiffd[ffd_ind][field] = \
        #             np.concatenate(thopt_cpsurf_fe_multiffd_temp\
        #                            [ffd_ind][field])
        #         self.thopt_cpsurf_lims_multiffd[ffd_ind][field] = \
        #             [np.min(self.thopt_cpsurf_fe_list_multiffd\
        #                     [ffd_ind][field]),
        #              np.max(self.thopt_cpsurf_fe_list_multiffd\
        #                     [ffd_ind][field])]
        #     self.thopt_cpsurf_fe_list_multiffd[ffd_ind] = \
        #         np.array(self.thopt_cpsurf_fe_list_multiffd[ffd_ind])\
        #         .transpose(1,0)

        # Get control point functions and lims for each FFD blocks
        thopt_cpsurf_fe_multiffd_temp = [[[] for i in range(self.nsd+1)]
                                         for j in range(self.num_thopt_ffd)]
        self.thopt_cpsurf_fe_hom_list_multiffd = [[None for i in range(self.nsd+1)]
                                           for j in range(self.num_thopt_ffd)]
        for ffd_ind in range(self.num_thopt_ffd):
            for field in range(self.nsd+1):
                for s_ind in self.thopt_multiffd_surf_ind_list[ffd_ind]:
                    thopt_cpsurf_fe_multiffd_temp[ffd_ind][field] += \
                        [get_petsc_vec_array(v2p(self.splines[s_ind].
                         cpFuncs[field].vector()))]
                self.thopt_cpsurf_fe_hom_list_multiffd[ffd_ind][field] = \
                    np.concatenate(thopt_cpsurf_fe_multiffd_temp\
                                   [ffd_ind][field])
            self.thopt_cpsurf_fe_hom_list_multiffd[ffd_ind] = \
                np.array(self.thopt_cpsurf_fe_hom_list_multiffd[ffd_ind])\
                .transpose(1,0)

        self.thopt_cpsurf_fe_list_multiffd = [[None for i in range(self.nsd)]
                                           for j in range(self.num_thopt_ffd)]
        self.thopt_cpsurf_lims_multiffd = [[None for i in range(self.nsd)]
                                           for j in range(self.num_thopt_ffd)]
        for ffd_ind in range(self.num_thopt_ffd):
            for field in range(self.nsd):
                self.thopt_cpsurf_fe_list_multiffd[ffd_ind][field] = \
                    self.thopt_cpsurf_fe_hom_list_multiffd[ffd_ind][:,field]\
                    /self.thopt_cpsurf_fe_hom_list_multiffd[ffd_ind][:,-1]
                self.thopt_cpsurf_lims_multiffd[ffd_ind][field] = \
                    [np.min(self.thopt_cpsurf_fe_list_multiffd\
                            [ffd_ind][field]),
                     np.max(self.thopt_cpsurf_fe_list_multiffd\
                            [ffd_ind][field])]

    def h_th_FE_reorder(self):
        """
        Generate sparse matrix to reorder h_th in FE DoFs to match the 
        index order of multi FFD
        """
        diag_vecs = []
        for s_ind in range(self.num_splines):
            diag_vecs += [np.eye((self.h_th_sizes[s_ind]))]
        block_mat_list = [[] for s_ind in range(self.num_splines)]
        for s_ind in range(self.num_splines):
            for ffd_ind in self.thopt_mixedffd_shell_inds:
                if s_ind == ffd_ind:
                    block_mat_list[s_ind] += [identity(self.h_th_sizes[s_ind],
                                                       format='coo')]
                else:
                    block_mat_list[s_ind] += [coo_matrix(
                                              (self.h_th_sizes[s_ind],
                                               self.h_th_sizes[ffd_ind]))]
        deriv_mat = bmat(block_mat_list, format='coo')
        return deriv_mat

    def set_thopt_multiFFD(self, thopt_knotsffd_list, thopt_cpffd_list):
        self.thopt_knotsffd_list = thopt_knotsffd_list
        self.thopt_cpffd_list = thopt_cpffd_list

        self.thopt_cpffd_flat_list = [cpffd[...,0:3].transpose(2,1,0,3)\
                                      .reshape(-1,3) for cpffd in 
                                      self.thopt_cpffd_list]
        self.thopt_cpffd_degree_list = [spline_degree(knotsffd[0], 
                                        knotsffd[0][0]) for knotsffd in 
                                        self.thopt_knotsffd_list]
        self.thopt_cpffd_shape_list = [cpffd.shape[0:3] for cpffd in 
                                       self.thopt_cpffd_list]
        self.thopt_cpffd_size_list = [np.prod(cpffd_shape) for cpffd_shape in 
                                      self.thopt_cpffd_shape_list]
        self.thopt_cpffd_design_size = np.sum(self.thopt_cpffd_size_list)
        self.thopt_dcpsurf_fedcpffd_list = []
        for ffd_ind in range(self.num_thopt_ffd):
            self.thopt_dcpsurf_fedcpffd_list += [CP_FFD_matrix(
                            self.thopt_cpsurf_fe_hom_list_multiffd[ffd_ind],
                            [self.thopt_cpffd_degree_list[ffd_ind]]*self.nsd,
                            self.thopt_knotsffd_list[ffd_ind])]
        self.thopt_dcpsurf_fedcpffd_mat = block_diag(
                                          self.thopt_dcpsurf_fedcpffd_list)

        # Get derivative of surface control points in FE function space
        # w.r.t. FFD blocks and constant thickness if any
        if self.num_thopt_nonffd_shells > 0:
            self.thopt_dcpsurf_fedcpconst_list = [np.ones(
                                    (self.h_th_sizes[ind], 1)) for ind in 
                                    self.thopt_nonffd_shell_inds]
            self.thopt_dcpsurf_fedcpconst_mat = block_diag(
                                    self.thopt_dcpsurf_fedcpconst_list)
            self.thopt_dcpsurf_fedcpmultiffd = block_diag(
                                        [self.thopt_dcpsurf_fedcpffd_mat,
                                         self.thopt_dcpsurf_fedcpconst_mat])
            self.thopt_cpffd_design_size += self.num_thopt_nonffd_shells
        else:
            self.thopt_dcpsurf_fedcpmultiffd = self.thopt_dcpsurf_fedcpffd_mat
        self.h_th_fe_reorder_mat = self.h_th_FE_reorder()
        self.thopt_dcpsurf_fedcpmultiffd = coo_matrix(
                                           self.h_th_fe_reorder_mat
                                           *self.thopt_dcpsurf_fedcpmultiffd)
        return self.thopt_dcpsurf_fedcpmultiffd

    def get_init_h_th_multiFFD(self):
        if self.init_h_th_ffd is None:
            init_h_th_ffd_list = []
            for i in range(len(self.thopt_dcpsurf_fedcpffd_list)):
                dfedffd_dense = self.thopt_dcpsurf_fedcpffd_list[i].todense()
                h_th_fe = [get_petsc_vec_array(self.h_th_fe_list[ind]) for
                           ind in self.thopt_multiffd_surf_ind_list[i]]
                h_th_fe = np.concatenate(h_th_fe).reshape(-1,1)
                init_h_th_ffd = solve_nonsquare(dfedffd_dense, h_th_fe)
                init_h_th_ffd = np.asarray(init_h_th_ffd).reshape(-1)
                init_h_th_ffd_list += [init_h_th_ffd]
            self.init_h_th_ffd = np.concatenate(init_h_th_ffd_list)
            if self.num_thopt_nonffd_shells > 0:
                self.init_h_th_const = [self.h_th_fe_list[ind][0] for ind in 
                                        self.thopt_nonffd_shell_inds]
                self.init_h_th_multiffd = np.concatenate([self.init_h_th_ffd,
                                          self.init_h_th_const])
            else:
                self.init_h_th_multiffd = self.init_h_th_ffd
        return self.init_h_th_multiffd

    ##########################################
    # Shape optimization related constraints #
    ##########################################

    def set_shopt_regu_CPFFD(self, shopt_regu_dir, shopt_regu_side, 
                             shopt_regu_align=None):
        """
        ``shopt_regu_dir`` is a list that has the same length with 
        ``opt_field``. If the entry is None, that means regularize all 
        layers minus one of control points along the direction in 
        ``opt_field``. If it's not None, we need the correspond value in 
        ``shopt_regu_side`` to determine which side to regularize along the 
        direction ``shopt_regu_dir``, the value of ``shopt_regu_dir`` 
        should be dirrection from ``opt_field``.
        Note: For i-th entry in ``shopt_regu_dir``, ``shopt_regu_dir[i]`` 
        cannot be equal to ``opt_field[i]``, otherwise, this constraint 
        would be trivial and this function returns a zero Jacobian matrix.
        For example, when optimizing vectical coordinates of control points,
        ``opt_field=[2]``, if ``shopt_regu_dir=[None]``, that means all 
        vertical coordinates are regularized to prevent self-penetration. 
        If ``shopt_regu_dir=[1]`` and ``shopt_regu_side=[0]``, that means 
        only the control points on the surface along direction 1 side 0 is 
        regularized.
        This function is used as a linear inequality constraint.

        Parameters
        ----------
        shopt_regu_dir : list of ints
        shopt_regu_side : list of ints
        """
        self.shopt_regu_dir = shopt_regu_dir
        self.shopt_regu_side = shopt_regu_side
        self.shopt_regu_align = shopt_regu_align

        self.shopt_cpregu_sizes = []
        for i, field in enumerate(self.opt_field):
            cpregu_size_temp = 1
            for j in range(self.nsd):
                if j == field:
                    cpregu_size_temp *= self.shopt_cpffd_shape[j]-1
                else:
                    cpregu_size_temp *= self.shopt_cpffd_shape[j]
            if self.shopt_regu_dir[i] is not None:
                cpregu_size_temp /= self.shopt_cpffd_shape\
                                    [self.shopt_regu_dir[i]]
            if self.shopt_regu_align is not None:
                # print("*"*50, i)
                cpregu_size_temp /= self.shopt_cpffd_shape\
                                    [self.shopt_regu_align[i]]

            self.shopt_cpregu_sizes += [int(cpregu_size_temp),]

        self.shopt_dcpregudcpffd_list = self.dCPregudCPFFD(
            self.shopt_cpregu_sizes, self.shopt_cpffd_size, 
            self.shopt_cpffd_shape, self.shopt_regu_dir, 
            self.shopt_regu_side, self.shopt_regu_align)
        return self.shopt_dcpregudcpffd_list


    def set_shopt_regu_CPFFD_old(self, shopt_regu_dir, shopt_regu_side):
        """
        ``shopt_regu_dir`` is a list that has the same length with 
        ``opt_field``. If the entry is None, that means regularize all 
        layers minus one of control points along the direction in 
        ``opt_field``. If it's not None, we need the correspond value in 
        ``shopt_regu_side`` to determine which side to regularize along the 
        direction ``shopt_regu_dir``, the value of ``shopt_regu_dir`` 
        should be dirrection from ``opt_field``.
        Note: For i-th entry in ``shopt_regu_dir``, ``shopt_regu_dir[i]`` 
        cannot be equal to ``opt_field[i]``, otherwise, this constraint 
        would be trivial and this function returns a zero Jacobian matrix.
        For example, when optimizing vectical coordinates of control points,
        ``opt_field=[2]``, if ``shopt_regu_dir=[None]``, that means all 
        vertical coordinates are regularized to prevent self-penetration. 
        If ``shopt_regu_dir=[1]`` and ``shopt_regu_side=[0]``, that means 
        only the control points on the surface along direction 1 side 0 is 
        regularized.
        This function is used as a linear inequality constraint.

        Parameters
        ----------
        shopt_regu_dir : list of ints
        shopt_regu_side : list of ints
        """
        self.shopt_regu_dir = shopt_regu_dir
        self.shopt_regu_side = shopt_regu_side

        self.shopt_cpregu_sizes = []
        for i, field in enumerate(self.opt_field):
            cpregu_size_temp = 1
            for j in range(self.nsd):
                if j == field:
                    cpregu_size_temp *= self.shopt_cpffd_shape[j]-1
                else:
                    cpregu_size_temp *= self.shopt_cpffd_shape[j]
            if self.shopt_regu_dir[i] is not None:
                cpregu_size_temp /= self.shopt_cpffd_shape\
                                    [self.shopt_regu_dir[i]]
            self.shopt_cpregu_sizes += [int(cpregu_size_temp),]

        self.shopt_dcpregudcpffd_list = self.dCPregudCPFFD_old(
            self.shopt_cpregu_sizes, self.shopt_cpffd_size, 
            self.shopt_cpffd_shape, self.shopt_regu_dir, 
            self.shopt_regu_side)
        return self.shopt_dcpregudcpffd_list

    def set_shopt_pin_CPFFD(self, pin_dir0, pin_side0=[0,1], 
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

        self.shopt_pin_dof = np.concatenate([self.shopt_pin_dof, 
                                       self.CPpinDoFs(self.shopt_cpffd_shape,
                                       self.pin_dir0, self.pin_side0,
                                       self.pin_dir1, self.pin_side1)], 
                                      dtype='int32')
        self.shopt_pin_dof = np.unique(self.shopt_pin_dof)
        self.shopt_cppin_size = self.shopt_pin_dof.size
        self.shopt_dcppindcpffd = self.dCPpindCPFFD(self.shopt_cpffd_size,
                                                    self.shopt_pin_dof)
        return self.shopt_dcppindcpffd

    def set_shopt_align_CPFFD(self, shopt_align_dir):
        """
        Set the direction of the FFD block to be alignment so that the 
        control points have the same coordinates along ``shopt_align_dir``.
        This function is used as a linear equality constraint.

        Parameters
        ----------
        shopt_align_dir : int
        """
        if not isinstance(shopt_align_dir, list):
            self.shopt_align_dir = [shopt_align_dir]
        else:
            self.shopt_align_dir = shopt_align_dir
        self.shopt_cp_align_size = 0
        for direction in self.shopt_align_dir:
            shopt_cp_align_size_sub = 1
            for i in range(self.nsd):
                if direction == i:
                    shopt_cp_align_size_sub *= self.shopt_cpffd_shape[i]-1
                else:
                    shopt_cp_align_size_sub *= self.shopt_cpffd_shape[i]
            self.shopt_cp_align_size += shopt_cp_align_size_sub
        self.shopt_dcpaligndcpffd = self.dCPaligndCPFFD(self.shopt_align_dir, 
                              self.shopt_cp_align_size, self.shopt_cpffd_size, 
                              self.shopt_cpffd_shape)
        return self.shopt_dcpaligndcpffd

    """
    def set_shopt_align_CP_multiFFD(self, shopt_align_dir_list):
        # Place holder for multi FFD shape optimization
        pass
    """

    ##############################################
    # Thickness optimization related constraints #
    ##############################################

    def set_thopt_align_CPFFD(self, thopt_align_dir):
        """
        Set the direction of the FFD block to be alignment so that the 
        control points have the same coordinates along ``thopt_align_dir``.
        This function is used as a linear equality constraint.

        Parameters
        ----------
        thopt_align_dir : int
        """
        if not isinstance(thopt_align_dir, list):
            self.thopt_align_dir = [thopt_align_dir]
        else:
            self.thopt_align_dir = thopt_align_dir
        self.thopt_cp_align_size = 0
        for direction in self.thopt_align_dir:
            thopt_cp_align_size_sub = 1
            for i in range(self.nsd):
                if direction == i:
                    thopt_cp_align_size_sub *= self.thopt_cpffd_shape[i]-1
                else:
                    thopt_cp_align_size_sub *= self.thopt_cpffd_shape[i]
            self.thopt_cp_align_size += thopt_cp_align_size_sub
        self.thopt_dcpaligndcpffd = self.dCPaligndCPFFD(self.thopt_align_dir, 
                              self.thopt_cp_align_size, self.thopt_cpffd_size, 
                              self.thopt_cpffd_shape)
        return self.thopt_dcpaligndcpffd

    def set_thopt_align_CP_multiFFD(self, thopt_align_dir_list):
        assert len(thopt_align_dir_list) == self.num_thopt_ffd
        self.thopt_align_dir_list = thopt_align_dir_list
        self.thopt_dcpaligndcpffd_list = []
        for ffd_ind in range(self.num_thopt_ffd):
            align_dir = thopt_align_dir_list[ffd_ind]
            if not isinstance(align_dir, list):
                align_dir = [align_dir]
            cp_align_size = 0
            for direction in align_dir:
                cp_align_size_sub = 1
                for i in range(self.nsd):
                    if direction == i:
                        cp_align_size_sub *= self.thopt_cpffd_shape_list[ffd_ind][i]-1
                    else:
                        cp_align_size_sub *= self.thopt_cpffd_shape_list[ffd_ind][i]
                cp_align_size += cp_align_size_sub
            self.thopt_dcpaligndcpffd_list += [self.dCPaligndCPFFD(align_dir, 
                                  cp_align_size, self.thopt_cpffd_size_list[ffd_ind], 
                                  self.thopt_cpffd_shape_list[ffd_ind])]
        self.thopt_dcpaligndcpffd = block_diag(self.thopt_dcpaligndcpffd_list)

        if self.num_thopt_nonffd_shells > 0:
            temp_block = coo_matrix((self.thopt_dcpaligndcpffd.shape[0],
                                    self.num_thopt_nonffd_shells))
            self.thopt_dcpaligndcpmultiffd = bmat([[self.thopt_dcpaligndcpffd, 
                                                    temp_block]])
        else:
            self.thopt_dcpaligndcpmultiffd = self.thopt_dcpaligndcpffd
        return self.thopt_dcpaligndcpmultiffd

    def set_thopt_regu_CPFFD(self, thopt_regu_dir, thopt_regu_side, 
                             thopt_regu_align=None):
        """
        ``shopt_regu_dir`` is a list that has the same length with 
        ``opt_field``. If the entry is None, that means regularize all 
        layers minus one of control points along the direction in 
        ``opt_field``. If it's not None, we need the correspond value in 
        ``shopt_regu_side`` to determine which side to regularize along the 
        direction ``shopt_regu_dir``, the value of ``shopt_regu_dir`` 
        should be dirrection from ``opt_field``.
        Note: For i-th entry in ``shopt_regu_dir``, ``shopt_regu_dir[i]`` 
        cannot be equal to ``opt_field[i]``, otherwise, this constraint 
        would be trivial and this function returns a zero Jacobian matrix.
        For example, when optimizing vectical coordinates of control points,
        ``opt_field=[2]``, if ``shopt_regu_dir=[None]``, that means all 
        vertical coordinates are regularized to prevent self-penetration. 
        If ``shopt_regu_dir=[1]`` and ``shopt_regu_side=[0]``, that means 
        only the control points on the surface along direction 1 side 0 is 
        regularized.
        This function is used as a linear inequality constraint.

        Parameters
        ----------
        shopt_regu_dir : list of ints
        shopt_regu_side : list of ints
        """
        self.thopt_regu_dir = thopt_regu_dir
        self.thopt_regu_side = thopt_regu_side
        self.thopt_regu_align = thopt_regu_align

        self.thopt_cpregu_sizes = []
        opt_field = [2]
        for i, field in enumerate(opt_field):
            cpregu_size_temp = 1
            for j in range(self.nsd):
                if j == field:
                    cpregu_size_temp *= self.thopt_cpffd_shape[j]-1
                else:
                    cpregu_size_temp *= self.thopt_cpffd_shape[j]
            if self.thopt_regu_dir[i] is not None:
                cpregu_size_temp /= self.thopt_cpffd_shape\
                                    [self.thopt_regu_dir[i]]
            if self.thopt_regu_align is not None:
                # print("*"*50, i)
                cpregu_size_temp /= self.thopt_cpffd_shape\
                                    [self.thopt_regu_align[i]]

            self.thopt_cpregu_sizes += [int(cpregu_size_temp),]

        self.thopt_dcpregudcpffd_list = self.dCPregudCPFFD(
            self.thopt_cpregu_sizes, self.thopt_cpffd_size, 
            self.thopt_cpffd_shape, self.thopt_regu_dir, 
            self.thopt_regu_side, self.thopt_regu_align, opt_field=opt_field)
        return self.thopt_dcpregudcpffd_list

    # def set_thopt_regu_CPFFD_old(self, thopt_regu_dir, thopt_regu_side):
    #     """
    #     ``thopt_regu_dir`` is a list that has the same length with 
    #     ``opt_field``. If the entry is None, that means regularize all 
    #     layers minus one of control points along the direction in 
    #     ``opt_field``. If it's not None, we need the correspond value in 
    #     ``thopt_regu_side`` to determine which side to regularize along the 
    #     direction ``thopt_regu_dir``, the value of ``thopt_regu_dir`` 
    #     should be dirrection from ``opt_field``.
    #     Note: For i-th entry in ``thopt_regu_dir``, ``thopt_regu_dir[i]`` 
    #     cannot be equal to ``opt_field[i]``, otherwise, this constraint 
    #     would be trivial and this function returns a zero Jacobian matrix.
    #     For example, when optimizing vectical coordinates of control points,
    #     ``opt_field=[2]``, if ``thopt_regu_dir=[None]``, that means all 
    #     vertical coordinates are regularized to prevent self-penetration. 
    #     If ``thopt_regu_dir=[1]`` and ``thopt_regu_side=[0]``, that means 
    #     only the control points on the surface along direction 1 side 0 is 
    #     regularized.
    #     This function is used as a linear inequality constraint.

    #     Parameters
    #     ----------
    #     thopt_regu_dir : list of ints
    #     thopt_regu_side : list of ints
    #     """
    #     self.thopt_regu_dir = thopt_regu_dir
    #     self.thopt_regu_side = thopt_regu_side

    #     self.thopt_cpregu_sizes = []
    #     for i, field in enumerate([2]):
    #         cpregu_size_temp = 1
    #         for j in range(self.nsd):
    #             if j == field:
    #                 cpregu_size_temp *= self.thopt_cpffd_shape[j]-1
    #             else:
    #                 cpregu_size_temp *= self.thopt_cpffd_shape[j]
    #         if self.thopt_regu_dir[i] is not None:
    #             cpregu_size_temp /= self.thopt_cpffd_shape\
    #                                 [self.thopt_regu_dir[i]]
    #         self.thopt_cpregu_sizes += [int(cpregu_size_temp),]

    #     self.thopt_dcpregudcpffd_list = self.dCPregudCPFFD_old(
    #         self.thopt_cpregu_sizes, self.thopt_cpffd_size, 
    #         self.thopt_cpffd_shape, self.thopt_regu_dir, 
    #         self.thopt_regu_side)
    #     return self.thopt_dcpregudcpffd_list

    ######################################
    # Methods for derivative computation #
    ######################################

    def dCPregudCPFFD(self, cpregu_sizes, cpffd_size, 
                      cpffd_shape, regu_dir, regu_side, 
                      align_dir=None, opt_field=None):
        if opt_field is None:
            opt_field = self.opt_field

        derivs = [np.zeros((cpregu_sizes[i], cpffd_size)) 
                  for i in range(len(opt_field))]
        l, m = cpffd_shape[0], cpffd_shape[1]
        for field_ind, field in enumerate(opt_field):
            row_ind = 0
            if field == 0:
                if regu_dir[field_ind] is None:
                    if align_dir is None:
                        for i in range(cpffd_shape[0]-1):
                            for j in range(cpffd_shape[1]):
                                for k in range(cpffd_shape[2]):
                                    col_ind0 = ijk2dof(i, j, k, l, m) 
                                    col_ind1 = ijk2dof(i+1, j, k, l, m)
                                    derivs[field_ind][row_ind, col_ind0] = -1.
                                    derivs[field_ind][row_ind, col_ind1] = 1.
                                    row_ind += 1
                    elif align_dir[field_ind] == 0:
                        raise ValueError("Optimization filed cannot equal to align direction")
                    elif align_dir[field_ind] == 1:
                        for i in range(cpffd_shape[0]-1):
                            for k in range(cpffd_shape[2]):
                                j = 0
                                col_ind0 = ijk2dof(i, j, k, l, m) 
                                col_ind1 = ijk2dof(i+1, j, k, l, m)
                                derivs[field_ind][row_ind, col_ind0] = -1.
                                derivs[field_ind][row_ind, col_ind1] = 1.
                                row_ind += 1
                    elif align_dir[field_ind] == 2:
                        for i in range(cpffd_shape[0]-1):
                            for j in range(cpffd_shape[1]):
                                k = 0
                                col_ind0 = ijk2dof(i, j, k, l, m) 
                                col_ind1 = ijk2dof(i+1, j, k, l, m)
                                derivs[field_ind][row_ind, col_ind0] = -1.
                                derivs[field_ind][row_ind, col_ind1] = 1.
                                row_ind += 1
                elif regu_dir[field_ind] == 1:
                    for i in range(cpffd_shape[0]-1):
                        for k in range(cpffd_shape[2]):
                            if regu_side[field_ind] == 0:
                                j = 0
                            elif regu_side[field_ind] == 1:
                                j = cpffd_shape[1]-1
                            col_ind0 = ijk2dof(i, j, k, l, m) 
                            col_ind1 = ijk2dof(i+1, j, k, l, m)
                            derivs[field_ind][row_ind, col_ind0] = -1.
                            derivs[field_ind][row_ind, col_ind1] = 1.
                            row_ind += 1
                elif regu_dir[field_ind] == 2:
                    for i in range(cpffd_shape[0]-1):
                        for j in range(cpffd_shape[1]):
                            if regu_side[field_ind] == 0:
                                k = 0
                            elif regu_side[field_ind] == 1:
                                k = cpffd_shape[2]-1
                            col_ind0 = ijk2dof(i, j, k, l, m) 
                            col_ind1 = ijk2dof(i+1, j, k, l, m)
                            derivs[field_ind][row_ind, col_ind0] = -1.
                            derivs[field_ind][row_ind, col_ind1] = 1.
                            row_ind += 1
            elif field == 1:
                if regu_dir[field_ind] is None:
                    if align_dir is None:
                        for j in range(cpffd_shape[1]-1):
                            for i in range(cpffd_shape[0]):
                                for k in range(cpffd_shape[2]):
                                    col_ind0 = ijk2dof(i, j, k, l, m) 
                                    col_ind1 = ijk2dof(i, j+1, k, l, m)
                                    derivs[field_ind][row_ind, col_ind0] = -1.
                                    derivs[field_ind][row_ind, col_ind1] = 1.
                                    row_ind += 1
                    elif align_dir[field_ind] == 0:
                        for j in range(cpffd_shape[1]-1):
                            for k in range(cpffd_shape[2]):
                                i = 0
                                col_ind0 = ijk2dof(i, j, k, l, m) 
                                col_ind1 = ijk2dof(i+1, j, k, l, m)
                                derivs[field_ind][row_ind, col_ind0] = -1.
                                derivs[field_ind][row_ind, col_ind1] = 1.
                                row_ind += 1
                    elif align_dir[field_ind] == 1:
                        raise ValueError("Optimization filed cannot equal to align direction")
                    elif align_dir[field_ind] == 2:
                        for j in range(cpffd_shape[1]-1):
                            for i in range(cpffd_shape[0]):
                                k = 0
                                col_ind0 = ijk2dof(i, j, k, l, m) 
                                col_ind1 = ijk2dof(i+1, j, k, l, m)
                                derivs[field_ind][row_ind, col_ind0] = -1.
                                derivs[field_ind][row_ind, col_ind1] = 1.
                                row_ind += 1
                elif regu_dir[field_ind] == 0:
                    for j in range(cpffd_shape[1]-1):
                        for k in range(cpffd_shape[2]):
                            if regu_side[field_ind] == 0:
                                i = 0
                            elif regu_side[field_ind] == 1:
                                i = cpffd_shape[0]-1
                            col_ind0 = ijk2dof(i, j, k, l, m) 
                            col_ind1 = ijk2dof(i, j+1, k, l, m)
                            derivs[field_ind][row_ind, col_ind0] = -1.
                            derivs[field_ind][row_ind, col_ind1] = 1.
                            row_ind += 1
                elif regu_dir[field_ind] == 2:
                    for j in range(cpffd_shape[1]-1):
                        for i in range(cpffd_shape[0]):
                            if regu_side[field_ind] == 0:
                                k = 0
                            elif regu_side[field_ind] == 1:
                                k = cpffd_shape[2]-1
                            col_ind0 = ijk2dof(i, j, k, l, m) 
                            col_ind1 = ijk2dof(i, j+1, k, l, m)
                            derivs[field_ind][row_ind, col_ind0] = -1.
                            derivs[field_ind][row_ind, col_ind1] = 1.
                            row_ind += 1
            elif field == 2:
                if regu_dir[field_ind] is None:
                    if align_dir is None:
                        for k in range(cpffd_shape[2]-1):
                            for i in range(cpffd_shape[0]):
                                for j in range(cpffd_shape[1]):
                                    col_ind0 = ijk2dof(i, j, k, l, m) 
                                    col_ind1 = ijk2dof(i, j, k+1, l, m)
                                    derivs[field_ind][row_ind, col_ind0] = -1.
                                    derivs[field_ind][row_ind, col_ind1] = 1.
                                    row_ind += 1
                    elif align_dir[field_ind] == 0:
                        for k in range(cpffd_shape[2]-1):
                            for j in range(cpffd_shape[1]):
                                i = 0
                                col_ind0 = ijk2dof(i, j, k, l, m) 
                                col_ind1 = ijk2dof(i, j, k+1, l, m)
                                derivs[field_ind][row_ind, col_ind0] = -1.
                                derivs[field_ind][row_ind, col_ind1] = 1.
                                row_ind += 1
                    elif align_dir[field_ind] == 1:
                        for k in range(cpffd_shape[2]-1):
                            for i in range(cpffd_shape[0]):
                                j = 0
                                col_ind0 = ijk2dof(i, j, k, l, m) 
                                col_ind1 = ijk2dof(i, j, k+1, l, m)
                                derivs[field_ind][row_ind, col_ind0] = -1.
                                derivs[field_ind][row_ind, col_ind1] = 1.
                                row_ind += 1
                    elif align_dir[field_ind] == 2:
                        raise ValueError("Optimization filed cannot equal to align direction")
                elif regu_dir[field_ind] == 0:
                    for k in range(cpffd_shape[2]-1):
                        for j in range(cpffd_shape[1]):
                            if regu_side[field_ind] == 0:
                                i = 0
                            elif regu_side[field_ind] == 1:
                                i = cpffd_shape[0] - 1
                            col_ind0 = ijk2dof(i, j, k, l, m) 
                            col_ind1 = ijk2dof(i, j, k+1, l, m)
                            derivs[field_ind][row_ind, col_ind0] = -1.
                            derivs[field_ind][row_ind, col_ind1] = 1.
                            row_ind += 1
                elif regu_dir[field_ind] == 1:
                    for k in range(cpffd_shape[2]-1):
                        for i in range(cpffd_shape[0]):
                            if regu_side[field_ind] == 0:
                                j = 0
                            elif regu_side[field_ind] == 1:
                                j = cpffd_shape[1] - 1
                            col_ind0 = ijk2dof(i, j, k, l, m) 
                            col_ind1 = ijk2dof(i, j, k+1, l, m)
                            derivs[field_ind][row_ind, col_ind0] = -1.
                            derivs[field_ind][row_ind, col_ind1] = 1.
                            row_ind += 1
        derivs_coo = [coo_matrix(A) for A in derivs]
        return derivs_coo


    def dCPregudCPFFD_old(self, cpregu_sizes, cpffd_size, 
                      cpffd_shape, regu_dir, regu_side):
        derivs = [np.zeros((cpregu_sizes[i], cpffd_size)) 
                  for i in range(len(self.opt_field))]
        l, m = cpffd_shape[0], cpffd_shape[1]
        for field_ind, field in enumerate(self.opt_field):
            row_ind = 0
            if field == 0:
                if regu_dir[field_ind] is None:
                    for i in range(cpffd_shape[0]-1):
                        for j in range(cpffd_shape[1]):
                            for k in range(cpffd_shape[2]):
                                col_ind0 = ijk2dof(i, j, k, l, m) 
                                col_ind1 = ijk2dof(i+1, j, k, l, m)
                                derivs[field_ind][row_ind, col_ind0] = -1.
                                derivs[field_ind][row_ind, col_ind1] = 1.
                                row_ind += 1
                elif regu_dir[field_ind] == 1:
                    for i in range(cpffd_shape[0]-1):
                        for k in range(cpffd_shape[2]):
                            if regu_side[field_ind] == 0:
                                j = 0
                            elif regu_side[field_ind] == 1:
                                j = cpffd_shape[1]-1
                            col_ind0 = ijk2dof(i, j, k, l, m) 
                            col_ind1 = ijk2dof(i+1, j, k, l, m)
                            derivs[field_ind][row_ind, col_ind0] = -1.
                            derivs[field_ind][row_ind, col_ind1] = 1.
                            row_ind += 1
                elif regu_dir[field_ind] == 2:
                    for i in range(cpffd_shape[0]-1):
                        for j in range(cpffd_shape[1]):
                            if regu_side[field_ind] == 0:
                                k = 0
                            elif regu_side[field_ind] == 1:
                                k = cpffd_shape[2]-1
                            col_ind0 = ijk2dof(i, j, k, l, m) 
                            col_ind1 = ijk2dof(i+1, j, k, l, m)
                            derivs[field_ind][row_ind, col_ind0] = -1.
                            derivs[field_ind][row_ind, col_ind1] = 1.
                            row_ind += 1
            elif field == 1:
                if regu_dir[field_ind] is None:
                    for j in range(cpffd_shape[1]-1):
                        for i in range(cpffd_shape[0]):
                            for k in range(cpffd_shape[2]):
                                col_ind0 = ijk2dof(i, j, k, l, m) 
                                col_ind1 = ijk2dof(i, j+1, k, l, m)
                                derivs[field_ind][row_ind, col_ind0] = -1.
                                derivs[field_ind][row_ind, col_ind1] = 1.
                                row_ind += 1
                elif regu_dir[field_ind] == 0:
                    for j in range(cpffd_shape[1]-1):
                        for k in range(cpffd_shape[2]):
                            if regu_side[field_ind] == 0:
                                i = 0
                            elif regu_side[field_ind] == 1:
                                i = cpffd_shape[0]-1
                            col_ind0 = ijk2dof(i, j, k, l, m) 
                            col_ind1 = ijk2dof(i, j+1, k, l, m)
                            derivs[field_ind][row_ind, col_ind0] = -1.
                            derivs[field_ind][row_ind, col_ind1] = 1.
                            row_ind += 1
                elif regu_dir[field_ind] == 2:
                    for j in range(cpffd_shape[1]-1):
                        for i in range(cpffd_shape[0]):
                            if regu_side[field_ind] == 0:
                                k = 0
                            elif regu_side[field_ind] == 1:
                                k = cpffd_shape[2]-1
                            col_ind0 = ijk2dof(i, j, k, l, m) 
                            col_ind1 = ijk2dof(i, j+1, k, l, m)
                            derivs[field_ind][row_ind, col_ind0] = -1.
                            derivs[field_ind][row_ind, col_ind1] = 1.
                            row_ind += 1
            elif field == 2:
                if regu_dir[field_ind] is None:
                    for k in range(cpffd_shape[2]-1):
                        for i in range(cpffd_shape[0]):
                            for j in range(cpffd_shape[1]):
                                col_ind0 = ijk2dof(i, j, k, l, m) 
                                col_ind1 = ijk2dof(i, j, k+1, l, m)
                                derivs[field_ind][row_ind, col_ind0] = -1.
                                derivs[field_ind][row_ind, col_ind1] = 1.
                                row_ind += 1
                elif regu_dir[field_ind] == 0:
                    for k in range(cpffd_shape[2]-1):
                        for j in range(cpffd_shape[1]):
                            if regu_side[field_ind] == 0:
                                i = 0
                            elif regu_side[field_ind] == 1:
                                i = cpffd_shape[0] - 1
                            col_ind0 = ijk2dof(i, j, k, l, m) 
                            col_ind1 = ijk2dof(i, j, k+1, l, m)
                            derivs[field_ind][row_ind, col_ind0] = -1.
                            derivs[field_ind][row_ind, col_ind1] = 1.
                            row_ind += 1
                elif regu_dir[field_ind] == 1:
                    for k in range(cpffd_shape[2]-1):
                        for i in range(cpffd_shape[0]):
                            if regu_side[field_ind] == 0:
                                j = 0
                            elif regu_side[field_ind] == 1:
                                j = cpffd_shape[1] - 1
                            col_ind0 = ijk2dof(i, j, k, l, m) 
                            col_ind1 = ijk2dof(i, j, k+1, l, m)
                            derivs[field_ind][row_ind, col_ind0] = -1.
                            derivs[field_ind][row_ind, col_ind1] = 1.
                            row_ind += 1
        derivs_coo = [coo_matrix(A) for A in derivs]
        return derivs_coo

    """
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
    """

    def CPpinDoFs(self, cpffd_shape, pin_dir0, pin_side0, pin_dir1, pin_side1):
        pin_dof = []
        l, m = cpffd_shape[0], cpffd_shape[1]
        for side0 in pin_side0:
            if pin_dir0 == 0:
                if pin_dir1 is not None:
                    for side1 in pin_side1:
                        if pin_dir1 == 1:
                            for k in range(cpffd_shape[2]):
                                pin_ijk = [int(side0*(cpffd_shape[0]-1)), 
                                           int(side1*(cpffd_shape[1]-1)), 
                                           k]
                                pin_dof += [ijk2dof(*pin_ijk, l, m)]
                        elif pin_dir1 == 2:
                            for j in range(cpffd_shape[1]):
                                pin_ijk = [int(side0*(cpffd_shape[0]-1)), 
                                           j, int(side1*(cpffd_shape[2]-1))]
                                pin_dof += [ijk2dof(*pin_ijk, l, m)]
                        else:
                            raise ValueError("Unsupported pin_dir1 {}".
                                             format(pin_dir1))
                else:
                    for k in range(cpffd_shape[2]):
                        for j in range(cpffd_shape[1]):
                            pin_ijk = [int(side0*(cpffd_shape[0]-1)), 
                                       j, k]
                            pin_dof += [ijk2dof(*pin_ijk, l, m)]
            elif pin_dir0 == 1:
                if pin_dir1 is not None:
                    for side1 in pin_side1:
                        if pin_dir1 == 0:
                            for k in range(cpffd_shape[2]):
                                pin_ijk = [int(side1*(cpffd_shape[0]-1)), 
                                           int(side0*(cpffd_shape[1]-1)), 
                                           k]
                                pin_dof += [ijk2dof(*pin_ijk, l, m)]
                        elif pin_dir1 == 2:
                            for i in range(cpffd_shape[0]):
                                pin_ijk = [i, int(side0*(cpffd_shape[1]-1)), 
                                           int(side1*(cpffd_shape[2]-1))]
                                pin_dof += [ijk2dof(*pin_ijk, l, m)]
                        else:
                            raise ValueError("Unsupported pin_dir1 {}".
                                             format(pin_dir1))
                else:
                    for k in range(cpffd_shape[2]):
                        for i in range(cpffd_shape[0]):
                            pin_ijk = [i, int(side0*(cpffd_shape[1]-1)), 
                                       k]
                            pin_dof += [ijk2dof(*pin_ijk, l, m)]
            elif pin_dir0 == 2:
                if pin_dir1 is not None:
                    for side1 in pin_side1:
                        if pin_dir1 == 0:
                            for j in range(cpffd_shape[1]):
                                pin_ijk = [int(side1*(cpffd_shape[0]-1)), j, 
                                           int(side0*(cpffd_shape[2]-1))]
                                pin_dof += [ijk2dof(*pin_ijk, l, m)]
                        elif pin_dir1 == 1:
                            for i in range(cpffd_shape[0]):
                                pin_ijk = [i, int(side1*(cpffd_shape[1]-1)), 
                                           int(side0*(cpffd_shape[2]-1))]
                                pin_dof += [ijk2dof(*pin_ijk, l, m)]
                        else:
                            raise ValueError("Unsupported pin_dir1 {}".
                                             format(pin_dir1))
                else:
                    for j in range(cpffd_shape[1]):
                        for i in range(cpffd_shape[0]):
                            pin_ijk = [i, j, 
                                       int(side0*(cpffd_shape[2]-1))]
                            pin_dof += [ijk2dof(*pin_ijk, l, m)]
            else:
                raise ValueError("Unsupported pin_dir0 {}".
                                 format(pin_dir0))
        pin_dof = np.array(pin_dof)
        return pin_dof

    def dCPpindCPFFD(self, cpffd_size, pin_dof):
        deriv = np.zeros((pin_dof.size, cpffd_size))
        for i in range(pin_dof.size):
            deriv[i, pin_dof[i]] = 1
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

    # def save_files(self, thickness=False):
    #     """
    #     Save splines' displacements and control points to pvd files.
    #     """
    #     for i in range(self.num_splines):
    #         u_split = self.spline_funcs[i].split()
    #         for j in range(3):
    #             u_split[j].rename('u'+str(i)+'_'+str(j),
    #                               'u'+str(i)+'_'+str(j))
    #             self.u_files[i][j] << u_split[j]
    #             self.splines[i].cpFuncs[j].rename('F'+str(i)+'_'+str(j),
    #                                               'F'+str(i)+'_'+str(j))
    #             self.F_files[i][j] << self.splines[i].cpFuncs[j]
    #             if j==2:
    #                 self.splines[i].cpFuncs[j+1].rename(
    #                     'F'+str(i)+'_'+str(j+1), 'F'+str(i)+'_'+str(j+1))
    #                 self.F_files[i][j+1] << self.splines[i].cpFuncs[j+1]
    #         if thickness:
    #             self.h_th[i].rename('t'+str(i), 't'+str(i))
    #             self.t_files[i] << self.h_th[i]

    # def save_FFD_block(self, ind):
    #     # Save FFD block for visualization
    #     CP2 = inputs[self.input_CP_name_list[0]]
    #     FFD_shape = self.FFD_block.shape
    #     CP_FFD = self.FFD_block.control[...,0:3].transpose(2,1,0,3).\
    #                 reshape(-1,3)
    #     CP_FFD[:,2] = CP2
    #     CP_FFD_temp = CP_FFD.reshape(self.FFD_block.control[:,:,:,0:3]\
    #                     .transpose(2,1,0,3).shape).transpose(2,1,0,3)
    #     FFD_temp = NURBS(self.FFD_block.knots, CP_FFD_temp)

    #     FFD_path = './geometry/'
    #     FFD_vtk_name_pre = 'FFD_block_opt'
    #     FFD_names = glob.glob(FFD_path+FFD_vtk_name_pre+'*.vtu')
    #     name_ind = len(FFD_names)
    #     name_ind_str = '0'*(6-len(str(name_ind)))+str(name_ind)
    #     FFD_vtk_name = FFD_vtk_name_pre+name_ind_str+'.vtk'
    #     FFD_vtu_name = FFD_vtk_name_pre+name_ind_str+'.vtu'

    #     print("FFD_vtk_name:", FFD_vtk_name)
    #     print("FFD_vtu_name:", FFD_vtu_name)

    #     VTK().write(FFD_path+FFD_vtk_name, FFD_temp)
    #     vtk_file = meshio.read(FFD_path+FFD_vtk_name)
    #     meshio.write(FFD_path+FFD_vtu_name, vtk_file)


if __name__ == '__main__':
    pass