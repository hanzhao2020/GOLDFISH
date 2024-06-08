from GOLDFISH.nonmatching_opt import *
from GOLDFISH.utils.ffd_utils import *
from scipy.sparse import block_diag, bmat, identity, hstack, vstack
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
                 comm=None):
        """
        Parameters
        ----------
        splines : list of ExtractedSplines
        E : ufl Constant or list, Young's modulus
        h_th : ufl Constant or list, thickness of the splines
        nu : ufl Constant or list, Poisson's ratio
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
                         contact, comm)

        # Bspline surfaces' control points in FE DoFs
        self.cpsurf_fe_hom_list_decate = [[] for i in range(self.nsd+1)]
        self.cpsurf_fe_hom_list = [None for i in range(self.nsd+1)]
        for field in range(self.nsd+1):
            for s_ind in range(self.num_splines):
                self.cpsurf_fe_hom_list_decate[field] += [get_petsc_vec_array(
                                         v2p(self.splines[s_ind].
                                         cpFuncs[field].vector()))]
            self.cpsurf_fe_hom_list[field] = np.concatenate(
                                             self.cpsurf_fe_hom_list_decate[field])
        self.cpsurf_fe_hom_list = np.array(self.cpsurf_fe_hom_list).transpose(1,0)
 
        # if opt_thickness and var_thickness:
        #     self.init_h_th_ffd = None

    ##################################
    # Methods for shape optimization #
    ##################################

    def set_shopt_surf_inds_FFD(self, opt_field, shopt_surf_inds):
        """
        Set surface indicies of the single FFD block.

        Parameters
        ----------
        shopt_surf_inds : list of ints
        opt_field : list of ints
        """
        self.opt_shape = True
        self.shopt_multiffd = False

        self.opt_field = opt_field
        self.shopt_surf_inds = [shopt_surf_inds]*len(self.opt_field)

        self.shopt_cpsurf_fe_hom_list_decate = [[] for i in range(self.nsd+1)]
        self.shopt_cpsurf_fe_hom_list = [None for i in range(self.nsd+1)]
        for field in range(self.nsd+1):
            for s_ind in shopt_surf_inds:
                self.shopt_cpsurf_fe_hom_list_decate[field] += \
                    [self.cpsurf_fe_hom_list_decate[field][s_ind]]
            self.shopt_cpsurf_fe_hom_list[field] = \
                np.concatenate(self.shopt_cpsurf_fe_hom_list_decate[field])

        self.shopt_cpsurf_fe_hom_list = np.array(self.shopt_cpsurf_fe_hom_list)\
                                      .transpose(1,0)

        # CP surf fe lims, used for creation of FFD block
        self.cpsurf_des_fe_list = [None for i in range(self.nsd)]
        self.cpsurf_des_lims = [None for i in range(self.nsd)]
        for field in range(self.nsd):
                self.cpsurf_des_fe_list[field] = \
                    self.shopt_cpsurf_fe_hom_list[:,field]\
                    /self.shopt_cpsurf_fe_hom_list[:,-1]
                self.cpsurf_des_lims[field] = \
                    [np.min(self.cpsurf_des_fe_list[field]),
                     np.max(self.cpsurf_des_fe_list[field])]
        self.cpsurf_des_fe_list = np.array(self.cpsurf_des_fe_list)\
                                  .transpose(1,0)

        # self.dcps_iga_list = [[] for field in self.opt_field]
        # self.dcps_iga = [None for field in self.opt_field]
        # self.dcps_fe_list = [[] for field in self.opt_field]
        # self.dcps_fe = [None for field in self.opt_field]
        # for field_ind, field in enumerate(self.opt_field):
        #     for i, s_ind in enumerate(shopt_surf_inds):
        #         self.dcps_iga_list[field_ind] += [self.vec_scalar_iga_list[s_ind]]
        #         self.dcps_fe_list[field_ind] += [self.vec_scalar_fe_list[s_ind]]
        #     self.dcps_iga[field_ind] = create_nest_PETScVec(self.dcps_iga_list[field_ind], comm=self.comm)
        #     self.dcps_fe[field_ind] = create_nest_PETScVec(self.dcps_fe_list[field_ind], comm=self.comm)

        # # Create nested control points in IGA DoFs
        self.cpdes_iga_list = [[] for i in range(len(self.opt_field))]
        self.cpdes_iga_nest = [None for i in range(len(self.opt_field))]
        # Create nested cpFuncs vectors (in FE DoFs)
        self.cpdes_fe_list = [[] for i in range(len(self.opt_field))]
        self.cpdes_fe_nest = [None for i in range(len(self.opt_field))]

        self.cpdes_iga_dofs_full_list = [[] for i in 
                                         range(len(self.opt_field))]
        for field_ind, field in enumerate(self.opt_field):
            ind_off = 0
            for s_ind in self.shopt_surf_inds[field_ind]:
                self.cpdes_iga_list[field_ind] += [self.vec_scalar_iga_list[s_ind]]
                self.cpdes_fe_list[field_ind] += [v2p(self.splines[s_ind].
                                              cpFuncs[field].vector()),]
                self.cpdes_iga_dofs_full_list[field_ind] += [list(range(ind_off, 
                    ind_off+self.vec_scalar_iga_dof_list[s_ind]))]
                ind_off += self.vec_scalar_iga_dof_list[s_ind]
            self.cpdes_iga_nest[field_ind] = create_nest_PETScVec(
                self.cpdes_iga_list[field_ind], comm=self.comm)
            self.cpdes_fe_nest[field_ind] = create_nest_PETScVec(
                self.cpdes_fe_list[field_ind], comm=self.comm)

        self.cpdes_iga_dofs = [[[dof for dof in subdof] 
                              for subdof in subdof_subfield] 
                              for subdof_subfield in 
                              self.cpdes_iga_dofs_full_list]
        self.cpdes_iga_dofs_full = [np.concatenate(dof_list) for dof_list 
                                    in self.cpdes_iga_dofs_full_list]

        self.init_cp_iga = self.get_init_CPIGA()

    def set_shopt_FFD(self, shopt_knotsffd, shopt_cpffd):
        """
        ``shopt_cpffd`` is in the igakit order convention
        Assume FFD block has identity geometric mapping.

        Parameters
        ----------
        shopt_knotsffd : list of ndarray, ndarray, knots of FFD block
        shopt_cpffd : ndarray, control points of FFD block
        """
        self.shopt_knotsffd = shopt_knotsffd
        self.shopt_cpffd = shopt_cpffd

        self.shopt_cpffd_flat = self.shopt_cpffd[...,0:3].transpose(2,1,0,3)\
                                .reshape(-1,3)
        self.shopt_ffd_degree = [spline_degree(knots, knots[0]) 
                                 for knots in self.shopt_knotsffd]

        self.shopt_cpffd_shape = self.shopt_cpffd.shape[0:3]
        self.shopt_cpffd_size = np.prod(self.shopt_cpffd_shape)

        self.shopt_num_desvars = [self.shopt_cpffd_size 
                                  for field in self.opt_field]
        self.shopt_cpffd_design_dof = [list(range(self.shopt_cpffd_size)) 
                                       for field in self.opt_field]
        self.shopt_cpffd_design_dof_full = [[dof for dof in dof_subfield]
                                            for dof_subfield in 
                                            self.shopt_cpffd_design_dof]

        self.shopt_dcpsurf_fedcpffd = CP_FFD_matrix(
                                      self.shopt_cpsurf_fe_hom_list,
                                      self.shopt_ffd_degree, 
                                      self.shopt_knotsffd)

        self.shopt_init_cpffd_full = [self.shopt_cpffd_flat[:,field]
                                      for field in self.opt_field]
        self.shopt_cpffd_pin_dof = [[] for field in self.opt_field]

        return self.shopt_dcpsurf_fedcpffd

    def set_shopt_surf_inds_multiFFD(self, opt_field_mffd, shopt_surf_ind_list_mffd):
        """
        Method for shape optimization with multiple FFD blocks,
        this involves change of parametric location of surface--surface
        intersections.

        Parameters
        ----------
        shopt_surf_ind_list_mffd : list of lists
        """
        assert len(opt_field_mffd) == len(shopt_surf_ind_list_mffd)
        self.opt_shape = True
        self.shopt_multiffd = True

        self.opt_field_mffd = opt_field_mffd
        self.shopt_surf_ind_list_mffd = shopt_surf_ind_list_mffd
        self.shopt_num_ffd = len(self.shopt_surf_ind_list_mffd)

        # Create ``opt_field`` using ``opt_field_mffd``
        self.opt_field = []
        for ffd_ind in range(self.shopt_num_ffd):
            for field in self.opt_field_mffd[ffd_ind]:
                if field not in self.opt_field:
                    self.opt_field += [field]
        self.opt_field = list(np.sort(self.opt_field))
        # Create ``shopt_surf_inds`` using ``shopt_surf_inds_mffd``
        self.shopt_surf_inds = [[] for field in self.opt_field]
        for ffd_ind in range(self.shopt_num_ffd):
            for i, field in enumerate(self.opt_field_mffd[ffd_ind]):
                field_ind = self.opt_field.index(field)
                self.shopt_surf_inds[field_ind] += \
                    self.shopt_surf_ind_list_mffd[ffd_ind]
        for field_ind, field in enumerate(self.opt_field):
            self.shopt_surf_inds[field_ind] = \
                list(np.sort(self.shopt_surf_inds[field_ind]))

        # For filed i, indicate blocks inds in ``opt_field_ffdinds[i]``
        self.opt_field_ffdinds = [[] for field in self.opt_field]
        for ffd_ind in range(self.shopt_num_ffd):
            for i, field in enumerate(self.opt_field_mffd[ffd_ind]):
                field_ind = self.opt_field.index(field)
                self.opt_field_ffdinds[field_ind] += [ffd_ind]
        

        # Get control point functions and lims for each FFD blocks
        shopt_cpsurf_fe_mffd_temp = [[[] for i in range(self.nsd+1)]
                                     for j in range(self.shopt_num_ffd)]
        self.shopt_cpsurf_fe_hom_list_mffd_decate = \
                                        [[None for i in range(self.nsd+1)]
                                         for j in range(self.shopt_num_ffd)]
        for ffd_ind in range(self.shopt_num_ffd):
            for field in range(self.nsd+1):
                for s_ind in self.shopt_surf_ind_list_mffd[ffd_ind]:
                    shopt_cpsurf_fe_mffd_temp[ffd_ind][field] += \
                        [get_petsc_vec_array(v2p(self.splines[s_ind].
                         cpFuncs[field].vector()))]
                self.shopt_cpsurf_fe_hom_list_mffd_decate[ffd_ind][field] = \
                    np.concatenate(shopt_cpsurf_fe_mffd_temp\
                                   [ffd_ind][field])
            self.shopt_cpsurf_fe_hom_list_mffd_decate[ffd_ind] = \
                np.array(self.shopt_cpsurf_fe_hom_list_mffd_decate[ffd_ind])\
                .transpose(1,0)
        self.shopt_cpsurf_fe_hom_list_mffd = np.concatenate(
            self.shopt_cpsurf_fe_hom_list_mffd_decate, axis=0)

        self.shopt_cpsurf_fe_list_mffd_decate = [[None for i in range(self.nsd)]
                                          for j in range(self.shopt_num_ffd)]
        self.shopt_cpsurf_lims_mffd = [[None for i in range(self.nsd)]
                                       for j in range(self.shopt_num_ffd)]
        for ffd_ind in range(self.shopt_num_ffd):
            for field in range(self.nsd):
                self.shopt_cpsurf_fe_list_mffd_decate[ffd_ind][field] = \
                    self.shopt_cpsurf_fe_hom_list_mffd_decate[ffd_ind][:,field]\
                    /self.shopt_cpsurf_fe_hom_list_mffd_decate[ffd_ind][:,-1]
                self.shopt_cpsurf_lims_mffd[ffd_ind][field] = \
                    [np.min(self.shopt_cpsurf_fe_list_mffd_decate\
                            [ffd_ind][field]),
                     np.max(self.shopt_cpsurf_fe_list_mffd_decate\
                            [ffd_ind][field])]
            self.shopt_cpsurf_fe_list_mffd_decate[ffd_ind] = \
                np.array(self.shopt_cpsurf_fe_list_mffd_decate[ffd_ind])\
                .transpose(1,0)

        self.shopt_cpsurf_fe_list_mffd = np.concatenate(
            self.shopt_cpsurf_fe_list_mffd_decate, axis=0)

        self.shopt_init_cpsurf_fe_list_mffd = []
        for field_ind, field in enumerate(self.opt_field):
            init_cpsurf_mffd_list = []
            for ffd_ind, ffd in enumerate(self.opt_field_ffdinds[field_ind]):
                init_cpsurf_mffd_list += [self.shopt_cpsurf_fe_list_mffd_decate[ffd][:,field]]
            init_cp_multi_ffd_list = np.concatenate(init_cpsurf_mffd_list)
            self.shopt_init_cpsurf_fe_list_mffd += [init_cp_multi_ffd_list]

        # self.cpdes_iga_list = [[] for i in range(len(self.opt_field))]

        # # Create nested control points in IGA DoFs
        self.cpdes_iga_list = [[] for i in range(len(self.opt_field))]
        self.cpdes_iga_nest = [None for i in range(len(self.opt_field))]
        # Create nested cpFuncs vectors (in FE DoFs)
        self.cpdes_fe_list = [[] for i in range(len(self.opt_field))]
        self.cpdes_fe_nest = [None for i in range(len(self.opt_field))]

        self.cpdes_iga_dofs_full_list = [[] for i in 
                                         range(len(self.opt_field))]
        for field_ind, field in enumerate(self.opt_field):
            ind_off = 0
            for s_ind in self.shopt_surf_inds[field_ind]:
                self.cpdes_iga_list[field_ind] += [self.vec_scalar_iga_list[s_ind]]
                self.cpdes_fe_list[field_ind] += [v2p(self.splines[s_ind].
                                              cpFuncs[field].vector()),]
                self.cpdes_iga_dofs_full_list[field_ind] += [list(range(ind_off, 
                    ind_off+self.vec_scalar_iga_dof_list[s_ind]))]
                ind_off += self.vec_scalar_iga_dof_list[s_ind]
            self.cpdes_iga_nest[field_ind] = create_nest_PETScVec(
                self.cpdes_iga_list[field_ind], comm=self.comm)
            self.cpdes_fe_nest[field_ind] = create_nest_PETScVec(
                self.cpdes_fe_list[field_ind], comm=self.comm)

        self.cpdes_iga_dofs = [[[dof for dof in subdof] 
                                    for subdof in subdof_subfield] 
                                    for subdof_subfield in 
                                    self.cpdes_iga_dofs_full_list]
        self.cpdes_iga_dofs_full = [np.concatenate(dof_list) for dof_list 
                                    in self.cpdes_iga_dofs_full_list]

        self.init_cp_iga = self.get_init_CPIGA()

    def set_shopt_multiFFD(self, shopt_knots_mffd, shopt_cp_mffd):
        """
        Place holder for shape optimization with multiple FFD blocks,
        this involves change of parametric location of surface--surface
        intersections.
        """
        assert len(shopt_knots_mffd) == len(self.opt_field_mffd)
        assert len(shopt_cp_mffd) == len(self.opt_field_mffd)

        self.shopt_knots_mffd = shopt_knots_mffd
        self.shopt_cp_mffd = shopt_cp_mffd

        self.shopt_cp_mffd_flat_decate = [cpffd[...,0:3].transpose(2,1,0,3)\
                                         .reshape(-1,3) for cpffd in 
                                         self.shopt_cp_mffd]
        self.shopt_cp_mffd_flat = np.concatenate(self.shopt_cp_mffd_flat_decate, 
                                                 axis=0)
        self.shopt_cp_mffd_degree = [[spline_degree(knots, knots[0]) 
                                      for knots in ffdknots] 
                                      for ffdknots in self.shopt_knots_mffd]
        self.shopt_cp_mffd_shape = [cpffd.shape[0:3] for cpffd in 
                                    self.shopt_cp_mffd]
        self.shopt_cp_mffd_size = [np.prod(cpffd_shape) for cpffd_shape in 
                                   self.shopt_cp_mffd_shape]
        self.shopt_cp_mffd_design_size = np.sum(self.shopt_cp_mffd_size)
        self.shopt_dcpsurf_fedcp_mffd_list = []
        for ffd_ind in range(self.shopt_num_ffd):
            self.shopt_dcpsurf_fedcp_mffd_list += [CP_FFD_matrix(
                            self.shopt_cpsurf_fe_hom_list_mffd_decate[ffd_ind],
                            self.shopt_cp_mffd_degree[ffd_ind],
                            self.shopt_knots_mffd[ffd_ind])]

        self.shopt_dcpsurf_fedcp_mffd = [None for field in self.opt_field]
        for field_ind, field in enumerate(self.opt_field):
            ffd_inds = self.opt_field_ffdinds[field_ind]
            deriv_mat_list = [self.shopt_dcpsurf_fedcp_mffd_list[ind]
                              for ind in ffd_inds]
            deriv_mat = block_diag(deriv_mat_list)
            reorder_mat = self.CPFE_reorder(field)
            if reorder_mat is not None:
                deriv_mat = coo_matrix(reorder_mat*deriv_mat)
            else:
                deriv_mat = coo_matrix(deriv_mat)
            self.shopt_dcpsurf_fedcp_mffd[field_ind] = deriv_mat

        ######## Design optimization related attributes

        self.shopt_num_desvars = [0 for field in self.opt_field]
        self.shopt_cp_mffd_design_dof = [[] for field in self.opt_field]
        self.shopt_cp_mffd_design_dof_full = [[] for field in self.opt_field]
        self.shopt_cp_mffd_design_dof_full_decate = [[] for field in self.opt_field]
        for field_ind, field in enumerate(self.opt_field):
            ind_off = 0
            for ffd_ind, ffd in enumerate(self.opt_field_ffdinds[field_ind]):
                self.shopt_num_desvars[field_ind] += self.shopt_cp_mffd_size[ffd]
                self.shopt_cp_mffd_design_dof_full_decate[field_ind] += \
                    [list(range(ind_off, ind_off+self.shopt_cp_mffd_size[ffd]))]
                self.shopt_cp_mffd_design_dof[field_ind] += \
                    [list(range(ind_off, ind_off+self.shopt_cp_mffd_size[ffd]))]
                self.shopt_cp_mffd_design_dof_full[field_ind] += \
                    list(range(ind_off, ind_off+self.shopt_cp_mffd_size[ffd]))
                ind_off += self.shopt_cp_mffd_size[ffd]


        self.shopt_init_cp_mffd_full = [self.get_init_CP_multiFFD(field)
                                        for field in self.opt_field]

        # Constraint related attributes
        # CP alignment
        self.shopt_align_dir_mffd = [None for ffd_ind in range(self.shopt_num_ffd)]
        self.shopt_dcpaligndcp_mffd_list = [[coo_matrix(np.eye(self.shopt_cp_mffd_size[ffd])) 
                                        for ffd in self.opt_field_ffdinds[field_ind]]
                                        for field_ind in range(len(self.opt_field))]
        # CP pin
        self.shopt_cp_mffd_pin_dof = [[] for field in self.opt_field]
        self.shopt_pin_vals = [None for field in self.opt_field]
        self.shopt_dcppindcp_mffd = [None for field in self.opt_field]
        
        return self.shopt_dcpsurf_fedcp_mffd

    def CPFE_reorder(self, field):
        """
        Generate sparse matrix to reorder CP in FE DoFs to match the 
        index order of multi FFD, i.e. [[1,3],[2,4]] -> [1,2,3,4]
        """
        # diag_vecs = []
        # for s_ind in range(self.num_splines):
        #     diag_vecs += [np.eye((self.vec_scalar_fe_dof_list[s_ind]))]
        field_ind = self.opt_field.index(field)
        ffd_inds = self.opt_field_ffdinds[field_ind]

        if len(ffd_inds) == 1:
            reorder_deriv_mat = None
        else:
            block_mat_list = [[] for s_ind in self.shopt_surf_inds[field_ind]]
            ffd_shell_inds = np.concatenate([self.shopt_surf_ind_list_mffd[i] 
                              for i in ffd_inds])
            for i, s_ind in enumerate(self.shopt_surf_inds[field_ind]):
                for ffd_s_ind in ffd_shell_inds:
                    if s_ind == ffd_s_ind:
                        block_mat_list[i] += [
                            identity(self.vec_scalar_fe_dof_list[s_ind],
                            format='coo')]
                    else:
                        block_mat_list[i] += [coo_matrix(
                            (self.vec_scalar_fe_dof_list[s_ind],
                             self.vec_scalar_fe_dof_list[ffd_s_ind]))]
            reorder_deriv_mat = bmat(block_mat_list, format='coo')
        return reorder_deriv_mat

    def get_init_CP_multiFFD(self, field):
        init_cp_multiffd_list = []
        field_ind = self.opt_field.index(field)
        for ffd_ind, ffd in enumerate(self.opt_field_ffdinds[field_ind]):
            init_cp_multiffd_list += [self.shopt_cp_mffd_flat_decate[ffd][:,field]]
        init_cp_multi_ffd_list = np.concatenate(init_cp_multiffd_list)
        return init_cp_multi_ffd_list

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
        reorder_deriv_mat = bmat(block_mat_list, format='coo')
        return reorder_deriv_mat

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

    def set_shopt_align_CPFFD(self, align_dir):
        """
        Set the direction of the FFD block to be alignment so that the 
        control points have the same coordinates along ``align_dir``.
        This function is used as a linear equality constraint.

        Parameters
        ----------
        shopt_align_dir : list of list of ints
            len(shopt_align_dir) == len(opt_field)
            for opt_field 0: shopt_align_dir[0] can be [1], [2], or [1,2]
        """
        assert len(align_dir) == len(self.opt_field)
        self.shopt_align_dir = align_dir
        self.shopt_dcpaligndcpffd = [coo_matrix(np.eye(self.shopt_cpffd_size)) 
                                     for field in self.opt_field]
        for field_ind, field in enumerate(self.opt_field):
            align_dir_sub = align_dir[field_ind]
            if align_dir_sub is not None:
                free_dof, deriv_mat = self.dCPaligndCPFFD(field, align_dir_sub, 
                                      self.shopt_cpffd_shape)
                self.shopt_cpffd_design_dof[field_ind] = free_dof
                self.shopt_dcpaligndcpffd[field_ind] = deriv_mat

        self.shopt_init_cpffd_design = [self.shopt_init_cpffd_full[field_ind]
                                        [self.shopt_cpffd_design_dof[field_ind]]
                                        for field_ind in range(len(self.opt_field))]
        return self.shopt_dcpaligndcpffd

    def set_shopt_align_CP_multiFFD(self, ffd_ind, align_dir):
        """
        Linear constraint information for FFD block CP alignment
        """
        assert len(align_dir) == len(self.opt_field_mffd[ffd_ind])
        self.shopt_align_dir_mffd[ffd_ind] = align_dir

        for field_ind, field in enumerate(self.opt_field_mffd[ffd_ind]):
            field_ind_gloabl = self.opt_field.index(field)
            ffd_ind_global = self.opt_field_ffdinds[field_ind_gloabl].index(ffd_ind)
            align_dir_sub = align_dir[field_ind]
            if align_dir_sub is not None:
                free_dof, deriv_mat = self.dCPaligndCPFFD(field, align_dir_sub, 
                                      self.shopt_cp_mffd_shape[ffd_ind])
                self.shopt_dcpaligndcp_mffd_list[field_ind_gloabl][ffd_ind_global] = deriv_mat
                ind_off = self.shopt_cp_mffd_design_dof[field_ind_gloabl][ffd_ind_global][0]
                free_dof_global = [dof+ind_off for dof in free_dof]
                self.shopt_cp_mffd_design_dof[field_ind_gloabl][ffd_ind_global] = free_dof_global

        self.shopt_init_cp_mffd_design = [None for field in self.opt_field]
        for field_ind, field in enumerate(self.opt_field):
            design_dof = [dof for dof_subfield in 
                          self.shopt_cp_mffd_design_dof[field_ind] 
                          for dof in dof_subfield]
            self.shopt_init_cp_mffd_design[field_ind] = \
                self.shopt_init_cp_mffd_full[field_ind][design_dof]

        self.shopt_dcpaligndcp_mffd = [coo_matrix(block_diag(
                                       self.shopt_dcpaligndcp_mffd_list[field_ind])) 
                                       for field_ind in range(len(self.opt_field))]
        return self.shopt_dcpaligndcp_mffd

    def set_shopt_pin_CPFFD(self, pin_dir0, pin_side0, pin_dir1=None, pin_side1=None):
        """
        Pin the control points with specific values.
        If pin_dir1 is None, a surface of DoFs will be pinned, if not, 
        a line of DoFs will be pinned.
        This function is used as a linear equality constraint.

        Parameters
        ----------
        pin_dir0 : list of ints, len(pin_dir0) == len(opt_field)
        pin_side0 : list of list of ints, default is [0, 1], pin both sides
        pin_dir1 : list of ints or None, {0,1,2}, default is None
        pin_side1 : list of list of ints, default is [0, 1] 
        """
        assert len(pin_dir0) == len(self.opt_field)
        assert len(pin_side0) == len(self.opt_field)

        for field_ind, field in enumerate(self.opt_field):
            pin_dir0_sub = pin_dir0[field_ind]
            pin_side0_sub = pin_side0[field_ind]
            if pin_dir0_sub is not None:
                if pin_dir1 is None:
                    pin_dir1_sub = None
                    pin_side1_sub = None
                else:
                    pin_dir1_sub = pin_dir1[field_ind]
                    pin_side1_sub = pin_side1[field_ind]
                pin_dof_temp = self.CPpinDoFs(pin_dir0_sub, pin_side0_sub, 
                                              pin_dir1_sub, pin_side1_sub, 
                                              self.shopt_cpffd_shape)
                pin_dof_des = []
                for dof in pin_dof_temp:
                    if dof in self.shopt_cpffd_design_dof[field_ind]:
                        pin_dof_des += [dof]
                self.shopt_cpffd_pin_dof[field_ind] += pin_dof_des

        self.shopt_pin_vals = [None for field in self.opt_field]
        self.shopt_dcppindcpffd = [None for field in self.opt_field]
        for field_ind, field in enumerate(self.opt_field):
            if len(self.shopt_cpffd_pin_dof[field_ind]) > 0:
                deriv_mat = self.dCPpindCPFFD(
                            self.shopt_cpffd_design_dof[field_ind],
                            self.shopt_cpffd_pin_dof[field_ind])
                self.shopt_dcppindcpffd[field_ind] = deriv_mat
                self.shopt_pin_vals[field_ind] = \
                    self.shopt_cpffd_flat[:,field]\
                    [self.shopt_cpffd_pin_dof[field_ind]]

        self.pin_field = []
        for field_ind, field in enumerate(self.opt_field):
            if self.shopt_dcppindcpffd[field_ind] is not None:
                self.pin_field += [field]

        return self.shopt_dcppindcpffd

    def set_shopt_pin_CP_multiFFD(self, ffd_ind, pin_dir0, pin_side0, 
                                  pin_dir1=None, pin_side1=None):

        assert len(pin_dir0) == len(self.opt_field_mffd[ffd_ind])
        assert len(pin_side0) == len(self.opt_field_mffd[ffd_ind])

        for field_ind, field in enumerate(self.opt_field_mffd[ffd_ind]):
            field_ind_gloabl = self.opt_field.index(field)
            ffd_ind_global = self.opt_field_ffdinds[field_ind_gloabl].index(ffd_ind)

            pin_dir0_sub = pin_dir0[field_ind]
            pin_side0_sub = pin_side0[field_ind]
            if pin_dir0_sub is not None:
                if pin_dir1 is None:
                    pin_dir1_sub = None
                    pin_side1_sub = None
                else:
                    pin_dir1_sub = pin_dir1[field_ind]
                    pin_side1_sub = pin_side1[field_ind]
                pin_dof_temp = self.CPpinDoFs(pin_dir0_sub, pin_side0_sub, 
                                              pin_dir1_sub, pin_side1_sub, 
                                              self.shopt_cp_mffd_shape[ffd_ind])

                ind_off = self.shopt_cp_mffd_design_dof[field_ind_gloabl][ffd_ind_global][0]
                pin_dof_temp_global = [dof+ind_off for dof in pin_dof_temp]

                pin_dof_des = []
                for dof in pin_dof_temp_global:
                    if dof in self.shopt_cp_mffd_design_dof[field_ind_gloabl][ffd_ind_global]:
                        pin_dof_des += [dof]
                self.shopt_cp_mffd_pin_dof[field_ind_gloabl] += pin_dof_des

        for field_ind, field in enumerate(self.opt_field_mffd[ffd_ind]):
            field_ind_gloabl = self.opt_field.index(field)
            if len(self.shopt_cp_mffd_pin_dof[field_ind_gloabl]) > 0:
                design_dof = [dof for dof_subfield in 
                              self.shopt_cp_mffd_design_dof[field_ind_gloabl] 
                              for dof in dof_subfield]
                deriv_mat = self.dCPpindCPFFD(design_dof,
                            self.shopt_cp_mffd_pin_dof[field_ind_gloabl])
                self.shopt_dcppindcp_mffd[field_ind_gloabl] = deriv_mat

                self.shopt_pin_vals[field_ind_gloabl] = \
                    self.shopt_cp_mffd_flat[:,field]\
                    [self.shopt_cp_mffd_pin_dof[field_ind_gloabl]]

        self.pin_field = []
        for field_ind, field in enumerate(self.opt_field):
            if self.shopt_dcppindcp_mffd[field_ind] is not None:
                self.pin_field += [field]
                
        return self.shopt_dcppindcp_mffd

    def set_shopt_regu_CPFFD(self):
        self.shopt_dcpregudcpffd = [None for field in self.opt_field]

        for field_ind, field in enumerate(self.opt_field):
            l,m,n = self.shopt_cpffd_shape
            align_dir = self.shopt_align_dir[field_ind]
            if align_dir is not None:
                if 0 in align_dir: l=1
                if 1 in align_dir: m=1
                if 2 in align_dir: n=1
            cpffd_des_dof = self.shopt_cpffd_design_dof[field_ind]
            deriv_mat = self.dCPregudCPFFD(field,l,m,n,cpffd_des_dof)
            self.shopt_dcpregudcpffd[field_ind] = deriv_mat
        return self.shopt_dcpregudcpffd

    def set_shopt_regu_CP_multiFFD(self):
        self.shopt_dcpregudcp_mffd_list = [[None for ffd in 
                                     self.opt_field_ffdinds[field_ind]] 
                                     for field_ind in range(len(self.opt_field))]
        for ffd_ind in range(self.shopt_num_ffd):
            for field_ind, field in enumerate(self.opt_field_mffd[ffd_ind]):
                field_ind_gloabl = self.opt_field.index(field)
                ffd_ind_gloabl = self.opt_field_ffdinds[field_ind_gloabl].index(ffd_ind)
                l,m,n = self.shopt_cp_mffd_shape[ffd_ind]
                align_dir = self.shopt_align_dir_mffd[ffd_ind][field_ind]
                if align_dir is not None:
                    if 0 in align_dir: l=1
                    if 1 in align_dir: m=1
                    if 2 in align_dir: n=1
                cpffd_des_dof = self.shopt_cp_mffd_design_dof\
                                [field_ind_gloabl][ffd_ind_gloabl]
                deriv_mat = self.dCPregudCPFFD(field,l,m,n,cpffd_des_dof)
                self.shopt_dcpregudcp_mffd_list[field_ind_gloabl][ffd_ind_gloabl] = deriv_mat

        self.shopt_dcpregudcp_mffd = [None for field in self.opt_field]
        for field_ind, field in enumerate(self.opt_field):
            self.shopt_dcpregudcp_mffd[field_ind] = \
                coo_matrix(block_diag(self.shopt_dcpregudcp_mffd_list[field_ind]))
        return self.shopt_dcpregudcp_mffd


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

    ######################################
    # Methods for derivative computation #
    ######################################

    def dCPaligndCPFFD(self, field, align_dir, cpffd_shape, side=0):
        cpffd_size = np.prod(cpffd_shape)
        l, m, n = cpffd_shape
        free_dof = []
        if field in align_dir:
            raise ValueError(f"Illegal CPFFD align direction {align_dir}"
                             f" for opt field {field}")

        if align_dir == [0]:
            for k in range(n):
                for j in range(m):
                    free_dof += [ijk2dof(side*(l-1),j,k,l,m)]
            sub_deriv_mat = np.ones((l,1))
            deriv_mat_list = [sub_deriv_mat]*(m*n)
            deriv_mat = coo_matrix(block_diag(deriv_mat_list))
        elif align_dir == [1]:
            for k in range(n):
                for i in range(l):
                    free_dof += [ijk2dof(i,side*(m-1),k,l,m)]
            sub_deriv_mat = vstack([coo_matrix(np.eye(l))]*m)
            deriv_mat_list = [sub_deriv_mat]*n
            deriv_mat = coo_matrix(block_diag(deriv_mat_list))
        elif align_dir == [2]:
            for j in range(m):
                for i in range(l):
                    free_dof += [ijk2dof(i,j,side*(n-1),l,m)]
            sub_deriv_mat = coo_matrix(np.eye(l*m))
            deriv_mat_list = [sub_deriv_mat]*n
            deriv_mat = coo_matrix(vstack(deriv_mat_list))
        elif align_dir == [1,2]:
            for i in range(l):
                free_dof += [ijk2dof(i,side*(m-1),side*(n-1),l,m)]
            sub_deriv_mat = coo_matrix(np.eye(l))
            deriv_mat_list = [sub_deriv_mat]*(m*n)
            deriv_mat = coo_matrix(vstack(deriv_mat_list))
        elif align_dir == [0,2]:
            for j in range(m):
                free_dof += [ijk2dof(side*(l-1),j,side*(n-1),l,m)]
            sub_deriv_mat = coo_matrix(block_diag([np.ones((l,1))]*m))
            deriv_mat_list = [sub_deriv_mat]*n
            deriv_mat = coo_matrix(vstack(deriv_mat_list))
        elif align_dir == [0,1]:
            for k in range(n):
                free_dof += [ijk2dof(side*(l-1),side*(m-1),k,l,m)]
            sub_deriv_mat = np.ones((l*m,1))
            deriv_mat_list = [sub_deriv_mat]*n
            deriv_mat = coo_matrix(block_diag(deriv_mat_list))
        else:
            raise ValueError(f"Undefined CPFFD ailgn direction {align_dir}")
        return free_dof, deriv_mat

    def CPpinDoFs(self, pin_dir0, pin_side0, pin_dir1, pin_side1, cpffd_shape):
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

    def dCPpindCPFFD(self, cpffd_des_dof, cpffd_pin_dof):
        deriv = np.zeros((len(cpffd_pin_dof), len(cpffd_des_dof)))
        for i in range(len(cpffd_pin_dof)):
            col_ind = cpffd_des_dof.index(cpffd_pin_dof[i])
            deriv[i, col_ind] = 1.
        deriv_coo = coo_matrix(deriv)
        return deriv_coo

    def dCPregudCPFFD(self, field, l, m, n, cpffd_design_dof):
        num_cols = len(cpffd_design_dof)
        if field == 0:
            num_rows = (l-1)*m*n
        elif field == 1:
            num_rows = l*(m-1)*n
        elif field == 2:
            num_rows = l*m*(n-1)

        deriv_mat = np.zeros((num_rows, num_cols))
        row_ind = 0
        if field == 0:
            for i in range(l-1):
                for j in range(m):
                    for k in range(n):
                        col_ind0 = ijk2dof(i,j,k,l,m)
                        col_ind1 = ijk2dof(i+1,j,k,l,m)
                        deriv_mat[row_ind, col_ind0] = -1.
                        deriv_mat[row_ind, col_ind1] = 1.
                        row_ind += 1
        elif field == 1:
            for i in range(l):
                for j in range(m-1):
                    for k in range(n):
                        col_ind0 = ijk2dof(i,j,k,l,m)
                        col_ind1 = ijk2dof(i,j+1,k,l,m)
                        deriv_mat[row_ind, col_ind0] = -1.
                        deriv_mat[row_ind, col_ind1] = 1.
                        row_ind += 1
        elif field == 2:
            for i in range(l):
                for j in range(m):
                    for k in range(n-1):
                        col_ind0 = ijk2dof(i,j,k,l,m)
                        col_ind1 = ijk2dof(i,j,k+1,l,m)
                        deriv_mat[row_ind, col_ind0] = -1.
                        deriv_mat[row_ind, col_ind1] = 1.
                        row_ind += 1
        return coo_matrix(deriv_mat)

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