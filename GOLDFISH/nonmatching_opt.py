from PENGoLINS.nonmatching_coupling import *
from GOLDFISH.utils.opt_utils import *
from GOLDFISH.utils.bsp_utils import *

from GOLDFISH.cpiga2xi import *

class NonMatchingOpt(NonMatchingCoupling):
    """
    Subclass of NonmatchingCoupling which serves as the base class
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
                         int_V_family, int_V_degree,
                         int_dx_metadata, contact, comm)

        self.opt_shape = False
        self.opt_field = []
        self.opt_thickness = False
        self.var_thickness = False
        self.use_aero_pressure = False

        # Create nested vectors in IGA DoFs
        self.vec_iga_list              = []
        self.vec_scalar_iga_list       = []
        self.vec_iga_dof_list          = []
        self.vec_scalar_iga_dof_list   = []
        for s_ind in range(self.num_splines):
            self.vec_iga_list += [zero_petsc_vec(self.splines[s_ind].M.size(1),
                                                 comm=self.comm)]
            self.vec_scalar_iga_list += [zero_petsc_vec(
                                         self.splines[s_ind].M_control.size(1),
                                         comm=self.comm)]
            self.vec_iga_dof_list += [self.splines[s_ind].M.size(1),]
            self.vec_scalar_iga_dof_list += \
                                    [self.splines[s_ind].M_control.size(1),]

        self.vec_iga_nest        = create_nest_PETScVec(
                                   self.vec_iga_list, comm=self.comm)
        self.vec_scalar_iga_nest = create_nest_PETScVec(
                                   self.vec_scalar_iga_list, comm=self.comm)
        self.vec_iga_dof         = self.vec_iga_nest.getSizes()[1]
        self.vec_scalar_iga_dof  = self.vec_scalar_iga_nest.getSizes()[1]

        # Create nested vectors in FE DoFs
        self.vec_fe_list               = []
        self.vec_scalar_fe_list        = []
        self.vec_fe_dof_list           = []
        self.vec_scalar_fe_dof_list    = []
        for s_ind in range(self.num_splines):
            self.vec_fe_list += [zero_petsc_vec(self.splines[s_ind].M.size(0),
                                 comm=self.comm)]
            self.vec_scalar_fe_list += [zero_petsc_vec(
                                        self.splines[s_ind].M_control.size(0),
                                        comm=self.comm)]
            self.vec_fe_dof_list += [self.splines[s_ind].M.size(0),]
            self.vec_scalar_fe_dof_list += \
                                    [self.splines[s_ind].M_control.size(0),]

        self.vec_fe_nest        = create_nest_PETScVec(self.vec_fe_list, 
                                                       comm=self.comm)
        self.vec_scalar_fe_nest = create_nest_PETScVec(self.vec_scalar_fe_list,
                                  comm=self.comm)
        self.vec_fe_dof         = self.vec_fe_nest.getSizes()[1]
        self.vec_scalar_fe_dof  = self.vec_scalar_fe_nest.getSizes()[1]

        # Create nested displacements in IGA DoFs
        self.u_iga_nest = self.vec_iga_nest.copy()

        # # Create nested control points in IGA DoFs
        # self.cp_iga_nest = self.vec_scalar_iga_nest.copy()

        # # Create nested cpFuncs vectors (in FE DoFs)
        # self.cp_funcs_list      = [[] for i in range(self.nsd)]
        # self.cp_funcs_nest      = [None for i in range(self.nsd)]
        # for field in range(self.nsd):
        #     for s_ind in range(self.num_splines):
        #         self.cp_funcs_list[field] += [v2p(self.splines[s_ind].
        #                                       cpFuncs[field].vector()),]
        #     self.cp_funcs_nest[field] = create_nest_PETScVec(
        #                                 self.cp_funcs_list[field],
        #                                 comm=self.comm)

        # Set initial control points in IGA DoFs as None
        self.init_cp_iga = None

        # Initial attributes for element sizes
        self.init_cpfuncs_list  = [[] for s_ind in range(self.num_splines)]
        self.hl_phy             = [] # Physical element length
        self.ha_phy             = [] # Physical element area
        self.hl_phy_linear      = [] # Physical element length in linear space
        self.ha_phy_linear      = [] # Physical element area in linear space

        for s_ind in range(self.num_splines):
            for field in range(self.nsd+1):
                self.init_cpfuncs_list[s_ind] += \
                                     [Function(self.splines[s_ind].V_control)]
                self.init_cpfuncs_list[s_ind][field].assign(
                                      self.splines[s_ind].cpFuncs[field])
            self.hl_phy += [spline_mesh_size(self.splines[s_ind])]
            self.ha_phy += [spline_mesh_area(self.splines[s_ind])]
            self.hl_phy_linear += [self.splines[s_ind].\
                                projectScalarOntoLinears(self.hl_phy[s_ind])] 
            self.ha_phy_linear += [self.splines[s_ind].\
                                projectScalarOntoLinears(self.ha_phy[s_ind])]

    def set_geom_preprocessor(self, preprocessor):
        """
        Set geometric processor
        """
        self.preprocessor = preprocessor
        self.cp_shapes = [surf_data.control.shape[0:2] for surf_data 
                         in self.preprocessor.BSpline_surfs_data]
        if self.preprocessor.reparametrize:
            self.cp_shapes = [surf_data.control.shape[0:2] for surf_data 
                         in self.preprocessor.BSpline_surfs_repara_data]
        if self.preprocessor.refine:
            self.cp_shapes = [surf_data.control.shape[0:2] for surf_data 
                         in self.preprocessor.BSpline_surfs_refine_data]


    #######################################################
    ######## Shape optimization setup methods #############
    #######################################################

    def set_shopt_surf_inds(self, opt_field, shopt_surf_inds):
        self.opt_shape = True
        self.opt_field = opt_field
        self.shopt_surf_inds = shopt_surf_inds

        assert len(opt_field) == len(shopt_surf_inds)
        
        self.shopt_num_desvars = [0 for field in self.opt_field]
        
        # # Create nested control points in IGA DoFs
        self.cpdes_iga_list    = [[] for i in range(len(self.opt_field))]
        self.cpdes_iga_nest    = [None for i in range(len(self.opt_field))]
        # Create nested cpFuncs vectors (in FE DoFs)
        self.cpdes_fe_list     = [[] for i in range(len(self.opt_field))]
        self.cpdes_fe_nest     = [None for i in range(len(self.opt_field))]

        self.cpdes_iga_dofs_full_list    = [[] for i in range(len(self.opt_field))]

        for i, field in enumerate(self.opt_field):
            ind_off = 0
            for s_ind in self.shopt_surf_inds[i]:
                self.shopt_num_desvars[i] += self.vec_scalar_iga_dof_list[s_ind]
                self.cpdes_iga_list[i] += [self.vec_scalar_iga_list[s_ind]]
                self.cpdes_fe_list[i] += [v2p(self.splines[s_ind].
                                              cpFuncs[field].vector()),]
                self.cpdes_iga_dofs_full_list[i] += [list(range(ind_off, 
                    ind_off+self.vec_scalar_iga_dof_list[s_ind]))]
                ind_off += self.vec_scalar_iga_dof_list[s_ind]
            self.cpdes_iga_nest[i] = create_nest_PETScVec(
                                     self.cpdes_iga_list[i], comm=self.comm)
            self.cpdes_fe_nest[i] = create_nest_PETScVec(
                                    self.cpdes_fe_list[i], comm=self.comm)

        self.cpdes_iga_dofs = [[[dof for dof in subdof] 
                              for subdof in subdof_subfield] 
                              for subdof_subfield in self.cpdes_iga_dofs_full_list]
        self.cpdes_iga_dofs_full = [np.concatenate(dof_list) for dof_list 
                                    in self.cpdes_iga_dofs_full_list]

        # Initialize pin dofs
        self.shopt_pin_dofs = [[] for i in range(len(self.opt_field))]

        self.dcps_iga_list = [[] for field in self.opt_field]
        self.dcps_iga = [None for field in self.opt_field]
        for field_ind, field in enumerate(self.opt_field):
            for i, s_ind in enumerate(self.shopt_surf_inds[field_ind]):
                self.dcps_iga_list[field_ind] += [self.vec_scalar_iga_list[s_ind]]
            self.dcps_iga[field_ind] = create_nest_PETScVec(
                self.dcps_iga_list[field_ind], comm=self.comm)

    def set_init_CPIGA(self, cp_iga):
        """
        Flattend control points for shell patches with
        shape of ``(self.vec_scalar_iga_dof, 3)``
        """
        self.init_cp_iga = cp_iga

    def get_init_CPIGA(self):
        """
        Return the flattend control points for all spline patches
        in IGA DoFs, if it hasn't be set yet, solve the pseudo-
        inverse problem to get the control points in IGA DoFs 
        form FE DoFs.
        """
        if self.init_cp_iga is None:
            self.init_cp_iga = self.solve_init_CPIGA()
        return self.init_cp_iga

    def solve_init_CPIGA(self):
        init_cp_iga = [None for i in range(len(self.opt_field))]
        init_cp_iga_list = [[] for i in range(len(self.opt_field))]
        for i, field in enumerate(self.opt_field):
            for j, s_ind in enumerate(self.shopt_surf_inds[i]):
                cp_fe = self.cpdes_fe_list[i][j]
                Mc = m2p(self.splines[s_ind].M_control)
                McTMc = Mc.transposeMatMult(Mc)
                McTcp_fe = AT_x(Mc, cp_fe)
                cp_iga = solve_Ax_b(McTMc, McTcp_fe)
                init_cp_iga_list[i] += [get_petsc_vec_array(cp_iga, 
                                            comm=self.comm)]
            init_cp_iga[i] = np.concatenate(init_cp_iga_list[i])
        return init_cp_iga

    ## CP constraints
    def set_shopt_align_CP(self, align_surf_inds=[], align_dir=[]):
        """
        Set surface control points alignment with surfaces indices
        ``align_surf_inds`` in direction ``align_dir``.

        Parameters
        ----------
        align_surf_inds : list of list of inds
            len(align_surf_inds) == len(opt_field)
        align_dir : list of list of inds, align_dir[i][j] in {0,1}
            len(align_dir) == len(opt_field)
            len(align_surf_inds[i]) == len(align_dir[i])
        """
        assert len(align_surf_inds) == len(self.opt_field)
        assert len(align_dir) == len(self.opt_field)
        # cp_shapes = self.cpiga2xi.cp_shapes
        cp_sizes = self.vec_scalar_iga_dof_list
        self.align_surf_inds = align_surf_inds
        self.align_dir = align_dir

        self.align_cp_deriv_list = [[] for field in self.opt_field]
        for field_ind, field in enumerate(self.opt_field):
            for s_ind in self.shopt_surf_inds[field_ind]:
                self.align_cp_deriv_list[field_ind] += [np.eye(cp_sizes[s_ind])]

        for field_ind, field in enumerate(self.opt_field):
            ind_off = 0
            if self.align_surf_inds[field_ind] is not None:
                for align_ind, s_ind in enumerate(self.align_surf_inds[field_ind]):
                    if s_ind not in self.shopt_surf_inds[field_ind]:
                        raise ValueError(f"Aligned surface {s_ind} is not optimized.")
                    num_col, num_row = self.cp_shapes[s_ind]
                    opt_surf_ind = self.shopt_surf_inds[field_ind].index(s_ind)
                    self.shopt_num_desvars[field_ind] \
                        -= self.vec_scalar_iga_dof_list[s_ind]

                    if self.align_dir[field_ind][align_ind] == 0:
                        self.shopt_num_desvars[field_ind] += num_row
                        deriv_mat = np.zeros((cp_sizes[s_ind], num_row))
                        for row_ind in range(num_row):
                            deriv_mat[num_col*row_ind:num_col*(row_ind+1), 
                                      row_ind] = 1.
                        self.align_cp_deriv_list[field_ind][align_ind] = deriv_mat
                        dofs_full = self.cpdes_iga_dofs_full_list[field_ind][opt_surf_ind]
                        self.cpdes_iga_dofs[field_ind][opt_surf_ind] =  \
                            dofs_full[0:cp_sizes[s_ind]:num_col]
                    elif self.align_dir[field_ind][align_ind] == 1:
                        self.shopt_num_desvars[field_ind] += num_col
                        deriv_mat = np.zeros((cp_sizes[s_ind], num_col))
                        for row_ind in range(num_row):
                            deriv_mat[num_col*row_ind:num_col*(row_ind+1), 
                                      :] = np.eye(num_col)
                        self.align_cp_deriv_list[field_ind][align_ind] = deriv_mat
                        dofs_full = self.cpdes_iga_dofs_full_list[field_ind][opt_surf_ind]
                        self.cpdes_iga_dofs[field_ind][opt_surf_ind] =  \
                            dofs_full[0:num_col]
                    else:
                        raise ValueError("Undefined direction: {}"
                                         .format(self.align_dir[align_ind]))
            self.cpdes_iga_dofs[field_ind] = np.concatenate(self.cpdes_iga_dofs[field_ind])

        self.init_cp_iga_design = [None for field in self.opt_field]
        self.shopt_dcpaligndcpsurf = [None for field in self.opt_field]
        for field_ind, field in enumerate(self.opt_field):
            self.shopt_dcpaligndcpsurf[field_ind] = coo_matrix(
                block_diag(*self.align_cp_deriv_list[field_ind]))
            self.init_cp_iga_design[field_ind] = \
                self.init_cp_iga[field_ind][self.cpdes_iga_dofs[field_ind]]
        return self.shopt_dcpaligndcpsurf

    def set_shopt_pin_CP(self, pin_surf_inds=[], pin_dir=[], pin_side=[], 
                         pin_dofs=None, pin_vals=None):
        """
        len(pin_surf_inds) == len(self.opt_field)
        """
        if pin_dofs is None:
            assert len(pin_surf_inds) == len(self.opt_field)
            assert len(pin_dir) == len(self.opt_field)
            assert len(pin_side) == len(self.opt_field)
            self.pin_surf_inds = pin_surf_inds
            self.pin_dir = pin_dir
            self.pin_side = pin_side

            # cp_shapes = self.cpiga2xi.cp_shapes
            cp_sizes = self.vec_scalar_iga_dof_list

            for field_ind, field in enumerate(self.opt_field):
                for pin_ind, s_ind in enumerate(self.pin_surf_inds[field_ind]):
                    if s_ind not in self.shopt_surf_inds[field_ind]:
                        raise ValueError(f"Pinned surface {s_ind} is not optimized.")
                    opt_surf_ind = self.shopt_surf_inds[field_ind].index(s_ind)
                    num_col, num_row = self.cp_shapes[s_ind]
                    dofs_full = self.cpdes_iga_dofs_full_list[field_ind][opt_surf_ind]
                    if self.pin_dir[field_ind][pin_ind] == 0:
                        if self.pin_side[field_ind][pin_ind] == 0:
                            local_pin_dof = list(range(0, cp_sizes[s_ind], num_col))
                        elif self.pin_side[field_ind][pin_ind] == 1:
                            local_pin_dof = list(range(num_col-1, cp_sizes[s_ind], num_col))
                    elif self.pin_dir[field_ind][pin_ind] == 1:
                        if self.pin_side[field_ind][pin_ind] == 0:
                            local_pin_dof = list(range(0, num_col))
                        elif self.pin_side[field_ind][pin_ind] == 1:
                            local_pin_dof = list(range(cp_sizes[s_ind]-num_col, cp_sizes[s_ind]))
                    else:
                        raise ValueError("Undefined direction: {}"
                                         .format(self.align_dir[align_ind]))
                    opt_pin_dofs = [dofs_full[local_dof_ind] for local_dof_ind in local_pin_dof]

                    self.shopt_pin_dofs[field_ind] += [pin_dof for pin_dof in opt_pin_dofs 
                                                    if pin_dof in self.cpdes_iga_dofs[field_ind]]
                    # print("field ind:", field_ind)                   
                    # print(opt_pin_dofs)
                    # print(self.shopt_pin_dofs[field_ind])
        else:
            self.shopt_pin_dofs = pin_dofs

        if shopt_pin_vals is None:
            self.shopt_pin_vals = []
            for field_ind, field in enumerate(self.opt_field):
                self.shopt_pin_vals += [self.init_cp_iga[field_ind]
                                  [self.shopt_pin_dofs[field_ind]]]
        else:
            self.shopt_pin_vals = shopt_pin_vals

        self.shopt_dcppindcpsurf = [None for field in self.opt_field]
        for field_ind, field in enumerate(self.opt_field):
            partial_mat = np.zeros((len(self.shopt_pin_dofs[field_ind]), 
                                    len(self.cpdes_iga_dofs[field_ind])))
            for pin_ind in range(len(self.shopt_pin_dofs[field_ind])):
                col_ind = np.where(self.cpdes_iga_dofs[field_ind]
                        ==self.shopt_pin_dofs[field_ind][pin_ind])[0][0]
                partial_mat[pin_ind, col_ind] = 1.
            self.shopt_dcppindcpsurf[field_ind] = coo_matrix(partial_mat)
        return self.shopt_dcppindcpsurf


    #######################################################
    ######## Thickness optimization setup methods #########
    #######################################################

    def set_thickness_opt(self, var_thickness=False):
        self.opt_thickness = True
        self.var_thickness = var_thickness

        if var_thickness:
            self.h_th_fe_list = [v2p(h_th.vector()) for h_th in self.h_th]
            self.h_th_fe_nest = create_nest_PETScVec(self.h_th_fe_list,
                                                     comm=self.comm)
            self.h_th_iga_nest = self.vec_scalar_iga_nest.copy()
            self.init_h_th_fe = get_petsc_vec_array(self.h_th_fe_nest,
                                                    comm=self.comm)
            self.h_th_sizes = [h_th_fe_sub.getSizes()[1] for h_th_fe_sub in
                               self.h_th_fe_list]
            self.init_h_th_iga = None
        else:
            # Create nested thickness vector
            self.h_th_vec_list = [v2p(h_th.vector()) for h_th in self.h_th]
            self.h_th_nest = create_nest_PETScVec(self.h_th_vec_list,
                                                  comm=self.comm)
            self.h_th_dof = self.h_th_nest.getSizes()[1]
            self.h_th_sizes = [h_th_sub.getSizes()[1] for h_th_sub in 
                               self.h_th_vec_list]
            self.init_h_th_list = [get_petsc_vec_array(h_th_sub, 
                                   comm=self.comm) for h_th_sub in
                                   self.h_th_vec_list]
            self.init_h_th = get_petsc_vec_array(self.h_th_nest, 
                                                 comm=self.comm)

    #######################################################
    ######## Derivatives computatoin ######################
    #######################################################

    def mortar_dRmdCPm_symexp(self, field):
        """
        Create dolfin forms of ``dRmdCPm`` for all intersections.
        """
        dRm_dcpm_single_field = [None for i in range(self.num_intersections)]
        for i in range(self.num_intersections):
            dRm_dcpm_temp = dRmdcpm_sub(self.Rm_symexp_list[i],
                            self.mortar_cpfuncs[i], field)
            dRm_dcpm_temp_to_assemble = [[[[Form(dRm_dcpm_ijkl) 
                                            if dRm_dcpm_ijkl is not None 
                                            else None 
                                            for dRm_dcpm_ijkl in dRm_dcpm_ijk]
                                            for dRm_dcpm_ijk in dRm_dcpm_ij]
                                            for dRm_dcpm_ij in dRm_dcpm_i]
                                            for dRm_dcpm_i in dRm_dcpm_temp]
            dRm_dcpm_single_field[i] = dRm_dcpm_temp_to_assemble
        return dRm_dcpm_single_field

    def mortar_meshes_setup(self, mapping_list, mortar_parametric_coords, 
                            penalty_coefficient=1000, transfer_mat_deriv=1,
                            penalty_method="minimum"):
        NonMatchingCoupling.mortar_meshes_setup(self, 
                            mapping_list, mortar_parametric_coords, 
                            penalty_coefficient, transfer_mat_deriv, 
                            penalty_method)
        if self.opt_shape:
            self.dRm_dcpm_list = [self.mortar_dRmdCPm_symexp(field) 
                                  for field in self.opt_field]

    def set_residuals(self, residuals, residuals_deriv=None):
        NonMatchingCoupling.set_residuals(self, residuals, residuals_deriv)
        if self.opt_shape:
            print("set residuals....")
            dR_dcp_ufl_symexp = [[] for field in self.opt_field]
            for i, field in enumerate(self.opt_field):
                for s_ind in range(self.num_splines):
                    dR_dcp_ufl_symexp[i] += [derivative(residuals[s_ind], 
                                       self.splines[s_ind].cpFuncs[field]),]
            self.dR_dcp_symexp = [[Form(dRdcp) for dRdcp in 
                                  dR_dcp_single_field]
                                  for dR_dcp_single_field in 
                                  dR_dcp_ufl_symexp]
        if self.opt_thickness:
            dR_dh_th_ufl_symexp = []
            for s_ind in range(self.num_splines):
                dR_dh_th_ufl_symexp += [derivative(residuals[s_ind],
                                        self.h_th[s_ind])]
            self.dR_dh_th_symexp = [Form(dR_dh_th) for dR_dh_th 
                                    in dR_dh_th_ufl_symexp]

    def vec_IGA2FE(self, v_iga, v_fe, s_ind):
        """
        Compute vector in FE DoFs from values in IGA DoFs for 
        spline ``s_ind``.
        """
        M_petsc = m2p(self.splines[s_ind].M)
        M_petsc.mult(v_iga, v_fe)
        v_fe.ghostUpdate()
        v_fe.assemble()

    def vec_scalar_IGA2FE(self, v_scalar_iga, v_scalar_fe, s_ind):
        """
        Compute scalar field vector in FE DoFs from values in
        IGA DoFs for spline ``s_ind``.
        """
        Mc_petsc = m2p(self.splines[s_ind].M_control)
        Mc_petsc.mult(v_scalar_iga, v_scalar_fe)
        v_scalar_fe.ghostUpdate()
        v_scalar_fe.assemble()

    def update_uIGA(self, u_array_iga):
        """
        Update splines' displacement functions with input array
        in IGA DoFs.
        """
        update_nest_vec(u_array_iga, self.u_iga_nest, comm=self.comm)
        u_iga_sub = self.u_iga_nest.getNestSubVecs()
        for s_ind in range(self.num_splines):
            self.vec_IGA2FE(u_iga_sub[s_ind], 
                            v2p(self.spline_funcs[s_ind].vector()), s_ind)
        self.update_mortar_funcs()

    def update_CPFE(self, cp_array_fe, field):
        """
        Update splines' control point functions with input array
        in FE DoFs
        """
        update_nest_vec(cp_array_fe, self.cpdes_fe_nest[
                        self.opt_field.index(field)], 
                        comm=self.comm)

    def update_CPIGA(self, cp_array_iga, field):
        """
        Update splines' control point functions with input array
        in IGA DoFs
        """
        field_ind = self.opt_field.index(field)
        update_nest_vec(cp_array_iga, self.cpdes_iga_nest[field_ind], 
                        comm=self.comm)
        cp_iga_sub = self.cpdes_iga_nest[field_ind].getNestSubVecs()
        for i, s_ind in enumerate(self.shopt_surf_inds[field_ind]):
            self.vec_scalar_IGA2FE(cp_iga_sub[i],
                 v2p(self.splines[s_ind].cpFuncs[field].vector()), s_ind)

    def update_h_th_FE(self, h_th_fe_array):
        """
        Update splines' thickness functions with input array
        in FE DoFs
        """
        update_nest_vec(h_th_fe_array, self.h_th_fe_nest, 
                        comm=self.comm)

    def update_h_th_IGA(self, h_th_iga_array):
        """
        Update splines' thickness functions with input array
        in IGA DoFs
        """
        update_nest_vec(h_th_iga_array, self.h_th_iga_nest, comm=self.comm)
        h_th_iga_sub = self.h_th_iga_nest.getNestSubVecs()
        for s_ind in range(self.num_splines):
            self.vec_scalar_IGA2FE(h_th_iga_sub[s_ind],
                 self.h_th_fe_list[s_ind], s_ind)           

    def update_h_th(self, h_th_array):
        """
        Update splines' thickness with input array
        """
        update_nest_vec(h_th_array, self.h_th_nest, comm=self.comm)

    def create_diff_intersections(self, num_edge_pts=None):
        """
        This function is noly needed when differentiating intersections'
        parametric coordiantes, and requires the class ``CPIGA2Xi``
        """
        assert self.transfer_mat_deriv == 2
        self.cpiga2xi = CPIGA2Xi(self.preprocessor, self.shopt_surf_inds, 
                                 self.opt_field, num_edge_pts)
        self.diff_int_inds = self.preprocessor.diff_int_inds
        self.Vms_2dim = [VectorFunctionSpace(
                         self.mortar_meshes[diff_int_ind], 'CG', 1)
                         for diff_int_ind in self.diff_int_inds]
        self.xi_funcs = []
        self.xi_vecs = []
        for i, diff_int_ind in enumerate(self.diff_int_inds):
            self.xi_funcs += [Function(self.Vms_2dim[i]),
                              Function(self.Vms_2dim[i])]
            self.xi_vecs += [v2p(self.xi_funcs[-2].vector()),
                             v2p(self.xi_funcs[-1].vector())]
        self.xi_nest = create_nest_PETScVec(self.xi_vecs, comm=self.comm)
        self.update_xi(self.cpiga2xi.xi_flat_global)
        self.xi_size = self.cpiga2xi.xi_size_global

    # def update_xi_old(self, xi_flat):
    #     """
    #     Update intersections' parametric coordinates
    #     """
    #     sub_vecs = self.xi_nest.getNestSubVecs()
    #     num_sub_vecs = len(sub_vecs)

    #     sub_vecs_range = []
    #     sub_vecs_size = []
    #     for i in range(num_sub_vecs):
    #         sub_vecs_range += [sub_vecs[i].getOwnershipRange(),]
    #         sub_vecs_size += [sub_vecs[i].size,]

    #     sub_array_list = []
    #     array_ind_off = 0
    #     for i in range(num_sub_vecs):
    #         sub_array = xi_flat[array_ind_off+sub_vecs_range[i][0]: 
    #                               array_ind_off+sub_vecs_range[i][1]]
    #         sub_array = sub_array.reshape(-1, self.npd)
    #         sub_array = sub_array[::-1].reshape(-1)
    #         sub_array_list += [sub_array,]
    #         array_ind_off += sub_vecs_size[i]
    #     nest_array = np.concatenate(sub_array_list)
    #     self.xi_nest.setArray(nest_array)
    #     self.xi_nest.assemble()

    def update_xi(self, xi_flat):
        """
        Update intersections' parametric coordinates
        """
        self.xi_nest.setArray(xi_flat)
        self.xi_nest.assemble()

    def update_transfer_matrices_sub(self, moartar_coords, index, side):
        """
        Update transfer matrices for single intersection on one side
        """
        move_mortar_mesh(self.mortar_meshes[index], moartar_coords)
        self.transfer_matrices_list[index][side] = \
            create_transfer_matrix_list(self.splines[
            self.mapping_list[index][side]].V, 
            self.Vms[index], self.transfer_mat_deriv)
        self.transfer_matrices_control_list[index][side] = \
            create_transfer_matrix_list(self.splines[
            self.mapping_list[index][side]].V_control, 
            self.Vms_control[index], self.transfer_mat_deriv)
        # Update mortar mesh functions and gemetric mappling
        for i in range(len(self.mortar_funcs[index][side])):
            A_x_b(self.transfer_matrices_list[index][side][i], 
                self.spline_funcs[self.mapping_list[index][side]].vector(), 
                self.mortar_funcs[index][side][i].vector())
        for i in range(len(self.mortar_funcs[index][side])):
            for j in range(self.nsd+1):
                A_x_b(self.transfer_matrices_control_list[index][side][i], 
                    self.splines[self.mapping_list[index][side]]
                    .cpFuncs[j].vector(), 
                    self.mortar_cpfuncs[index][side][i][j].vector())

    def update_transfer_matrices(self):
        """
        Update transfer matrices for all intersections
        """
        for i, diff_int_ind in enumerate(self.diff_int_inds):
            for side in range(self.npd):
            # for side in [1,0]:
                mortar_coords = v2p(self.xi_funcs[int(i*self.npd+side)].vector())\
                                .array.reshape(-1,2)
                self.update_transfer_matrices_sub(mortar_coords, diff_int_ind, side)

    # def update_transfer_matrices(self):
    #     for int_ind, int_ind_global in enumerate(self.diff_int_inds):
    #         xi_flat_sub = xi_flat[self.cpiga2xi.xi_flat_inds[int_ind]:
    #                               self.cpiga2xi.xi_flat_inds[int_ind+1]]
    #         xi_coord_sub = xi_flat_sub.reshape(-1,2)
    #         num_pts = int(xi_coord_sub.shape[0])
    #         for side in range(2):
    #             if side == 0:
    #                 mesh_coord = xi_coord_sub[0:num_pts, :]
    #             elif side == 1:
    #                 mesh_coord = xi_coord_sub[num_pts:, :]
    #             self.update_transfer_matrices_sub(mesh_coord,
    #                 int_ind_global, side)

    def extract_nonmatching_vec(self, vec_list, ind_list=None,
                                scalar=False, apply_bcs=False):
        """
        Extract non-matching vector from FE to IGA DoFs
        """
        if ind_list is None:
            ind_list = list(range(self.num_splines))
        vec_iga_list = []
        for i, s_ind in enumerate(ind_list):
            if scalar:
                M = m2p(self.splines[s_ind].M_control)
            else:
                M = m2p(self.splines[s_ind].M)
            vec_iga_sub = AT_x(M, vec_list[i])
            # Only apply bcs to non-scalar field vectors
            if apply_bcs and not scalar:
                apply_bcs_vec(self.splines[s_ind], vec_iga_sub)
            vec_iga_list += [vec_iga_sub,]
        vec_iga = create_nest_PETScVec(vec_iga_list)
        return vec_iga

    def extract_nonmatching_mat(self, mat_list, 
                                left_ind_list=None, right_ind_list=None, 
                                ext_right=True,
                                left_scalar=False, right_scalar=False, 
                                apply_row_bcs=False, apply_col_bcs=False,
                                diag=1):
        """
        Extract non-matching matrix from FE to IGA DoFs.
        """
        if left_ind_list is None:
            left_ind_list = list(range(self.num_splines))
        if right_ind_list is None:
            right_ind_list = list(range(self.num_splines))
        mat_iga_list = []
        for i, s_ind0 in enumerate(left_ind_list):
            mat_iga_list += [[],]
            for j, s_ind1 in enumerate(right_ind_list):
                if mat_list[i][j] is not None:
                    # Extract matrix
                    if left_scalar:
                        M_left = m2p(self.splines[s_ind0].M_control)
                    else:
                        M_left = m2p(self.splines[s_ind0].M)
                    if ext_right:
                        if right_scalar:
                            M_right = m2p(self.splines[s_ind1].M_control)
                        else:
                            M_right = m2p(self.splines[s_ind1].M)
                        mat_iga_temp = AT_R_B(M_left, mat_list[i][j], M_right)
                    else:
                        mat_iga_temp = M_left.transposeMatMult(mat_list[i][j])
                    # Apply boundary conditions
                    # Only these two conditions are considered
                    if apply_row_bcs and apply_col_bcs:
                        if i == j:
                            mat_iga_temp = apply_bcs_mat(self.splines[s_ind0],
                                           mat_iga_temp, diag=diag)
                        else:
                            mat_iga_temp = apply_bcs_mat(self.splines[s_ind0],
                                           mat_iga_temp, self.splines[s_ind1], 
                                           diag=0)
                    elif apply_row_bcs and not apply_col_bcs:
                        if i == j:
                            diag = diag
                        else:
                            diag=0
                        mat_iga_temp.zeroRows(self.splines[s_ind0].zeroDofs,
                                              diag=diag)
                else:
                    mat_iga_temp = None

                mat_iga_list[i] += [mat_iga_temp,]

        # print('='*50, mat_iga_list)

        self.mat_iga_list = mat_iga_list

        mat_iga = create_nest_PETScMat(mat_iga_list, comm=self.comm)

        if MPI.size(self.comm) == 1:
            mat_iga.convert('seqaij')
        else:
            mat_iga = create_aijmat_from_nestmat(mat_iga, 
                      mat_iga_list, comm=self.comm)
        return mat_iga

    def assemble_RFE(self):
        """
        Non-matching residual in FE DoFs
        """
        ## Step 1: assemble residuals of ExtractedSplines 
        # Compute contributions from shell residuals
        R_FE = []
        for i in range(self.num_splines):
            R_assemble = assemble(self.residuals[i])
            if self.point_sources is not None:
                for j, ps_ind in enumerate(self.point_source_inds):
                    if ps_ind == i:
                        self.point_sources[j].apply(R_assemble)
            R_FE += [v2p(R_assemble),]

        ## Step 2: assemble non-matching contributions
        # Create empty lists for non-matching contributions
        Rm_FE = [None for i1 in range(self.num_splines)]
        # Compute non-matching contributions ``Rm_FE``
        for i in range(self.num_intersections):
            Rm = transfer_penalty_residual(self.Rm_list[i], 
                      self.transfer_matrices_list[i])
            for j in range(len(Rm)):
                if Rm_FE[self.mapping_list[i][j]] is not None:
                    Rm_FE[self.mapping_list[i][j]] += Rm[j]
                else:
                    Rm_FE[self.mapping_list[i][j]] = Rm[j]

        ## Step 3: add spline residuals and non-matching 
        # contribution together
        Rt_FE = [None for i1 in range(self.num_splines)]
        for i in range(self.num_splines):
            if Rm_FE[i] is not None:
                Rt_FE[i] = R_FE[i] + Rm_FE[i]
            else:
                Rt_FE[i] = R_FE[i]

        ## Step 4: add contact contributions if contact is given
        if self.contact is not None:
            Kcs, Fcs = self.contact.assembleContact(self.spline_funcs, 
                                                    output_PETSc=True)
            for i in range(self.num_splines):
                if Fcs[i] is not None:
                    Rt_FE[i] += Fcs[i]
        return Rt_FE

    def assemble_dRFEduFE(self):
        """
        Derivatives of non-matching residual w.r.t. displacements in
        FE DoFs.
        """
        # Compute contributions from shell derivatives
        dR_du_FE = []
        for i in range(self.num_splines):
            dR_du_assemble = assemble(self.residuals_deriv[i])
            dR_du_FE += [m2p(dR_du_assemble),]

        ## Step 2: assemble non-matching contributions
        # Create empty lists for non-matching contributions
        dRm_dum_FE = [[None for i1 in range(self.num_splines)] 
                            for i2 in range(self.num_splines)]

        # Compute non-matching contributions ``dRm_dum_FE``.
        for i in range(self.num_intersections):
            dRm_dum = transfer_penalty_residual_deriv(
                           self.dRm_dum_list[i],  
                           self.transfer_matrices_list[i])
            for j in range(len(dRm_dum)):
                for k in range(j, len(dRm_dum[j])):
                    if dRm_dum_FE[self.mapping_list[i][j]]\
                       [self.mapping_list[i][k]] is not None:
                        dRm_dum_FE[self.mapping_list[i][j]]\
                            [self.mapping_list[i][k]] += dRm_dum[j][k]
                    else:
                        dRm_dum_FE[self.mapping_list[i][j]]\
                            [self.mapping_list[i][k]] = dRm_dum[j][k]

        # Filling lower triangle blocks of non-matching derivatives
        for i in range(self.num_splines-1):
            for j in range(i+1, self.num_splines):
                if dRm_dum_FE[i][j] is not None:
                    dRm_dum_temp = dRm_dum_FE[i][j].copy()
                    dRm_dum_temp.transpose()
                    dRm_dum_FE[j][i] = dRm_dum_temp

        ## Step 3: add spline residuals and non-matching 
        # contribution together
        dRt_dut_FE = [[None for i1 in range(self.num_splines)] 
                           for i2 in range(self.num_splines)]
        for i in range(self.num_splines):
            for j in range(self.num_splines):
                if i == j:
                    if dRm_dum_FE[i][i] is not None:
                        dRt_dut_FE[i][i] = dR_du_FE[i] + dRm_dum_FE[i][i]
                    else:
                        dRt_dut_FE[i][i] = dR_du_FE[i]
                else:
                    dRt_dut_FE[i][j] = dRm_dum_FE[i][j]

        ## Step 4: add contact contributions if contact is given
        if self.contact is not None:
            Kcs, Fcs = self.contact.assembleContact(self.spline_funcs, 
                                                    output_PETSc=True)
            for i in range(self.num_splines):
                for j in range(self.num_splines):
                    if i == j:
                        if Fcs[i] is not None:
                            dRt_dut_FE[i][i] += Kcs[i][i]
                    else:
                        if dRt_dut_FE[i][j] is not None:
                            if Kcs[i][j] is not None:
                                dRt_dut_FE[i][j] += Kcs[i][j]
                        else:
                            if Kcs[i][j] is not None:
                                dRt_dut_FE[i][j] = Kcs[i][j]
        return dRt_dut_FE

    def assemble_dRFEdCPFE(self, ind_list, field):
        """
        Derivatives of non-matching residual w.r.t. displacements in
        FE DoFs.
        """
        # Compute contributions from shell residuals and derivatives
        field_ind = self.opt_field.index(field)
        dR_dcp_FE = []
        for i, s_ind in enumerate(ind_list):
            dR_dcp_assemble = assemble(self.dR_dcp_symexp[field_ind][s_ind])
            dR_dcp_FE += [m2p(dR_dcp_assemble),]

        ## Step 2: assemble non-matching contributions
        # Create empty lists for non-matching contributions
        dRm_dcpm_FE = [[None for i1 in range(len(ind_list))] 
                             for i2 in range(self.num_splines)]

        # Compute non-matching contributions ``dRm_dcpm_FE``.
        for i in range(self.num_intersections):
            s_ind0, s_ind1 = self.mapping_list[i]
            if s_ind0 in ind_list or s_ind1 in ind_list:
                dRm_dcpm = transfer_dRmdcpm_sub(
                           self.dRm_dcpm_list[field_ind][i],  
                           self.transfer_matrices_list[i],
                           self.transfer_matrices_control_list[i])
                if s_ind0 in ind_list:
                    s_opt_ind0 = ind_list.index(s_ind0)
                    if dRm_dcpm_FE[s_ind0][s_opt_ind0] is not None:
                        dRm_dcpm_FE[s_ind0][s_opt_ind0] += dRm_dcpm[0][0]
                    else:
                        dRm_dcpm_FE[s_ind0][s_opt_ind0] = dRm_dcpm[0][0]
                    if dRm_dcpm_FE[s_ind1][s_opt_ind0] is not None:
                        dRm_dcpm_FE[s_ind1][s_opt_ind0] += dRm_dcpm[1][0]
                    else:
                        dRm_dcpm_FE[s_ind1][s_opt_ind0] = dRm_dcpm[1][0]
                if s_ind1 in ind_list:
                    s_opt_ind1 = ind_list.index(s_ind1)
                    if dRm_dcpm_FE[s_ind0][s_opt_ind1] is not None:
                        dRm_dcpm_FE[s_ind0][s_opt_ind1] += dRm_dcpm[0][1]
                    else:
                        dRm_dcpm_FE[s_ind0][s_opt_ind1] = dRm_dcpm[0][1]
                    if dRm_dcpm_FE[s_ind1][s_opt_ind1] is not None:
                        dRm_dcpm_FE[s_ind1][s_opt_ind1] += dRm_dcpm[1][1]
                    else:
                        dRm_dcpm_FE[s_ind1][s_opt_ind1] = dRm_dcpm[1][1]

        ## Step 3: add derivatives from splines and mortar meshes together
        dRt_dcp_FE = [[None for i1 in range(len(ind_list))] 
                            for i2 in range(self.num_splines)]
        for i, s_ind0 in enumerate(list(range(self.num_splines))):
            for j, s_ind1 in enumerate(ind_list):                
                if s_ind0 == s_ind1:
                    if dRm_dcpm_FE[i][j] is not None:
                        dRt_dcp_FE[i][j] = dR_dcp_FE[j] + dRm_dcpm_FE[i][j]
                    else:
                        dRt_dcp_FE[i][j] = dR_dcp_FE[j]
                else:
                    dRt_dcp_FE[i][j] = dRm_dcpm_FE[i][j]

        for i, s_ind0 in enumerate(list(range(self.num_splines))):
            if dRt_dcp_FE[i].count(None) == len(dRt_dcp_FE[i]):
                temp_mat = m2p(assemble(self.dR_dcp_symexp[field_ind][s_ind0]))
                row_sizes = temp_mat.getSizes()[0]
                if i == 0:
                    row_ind = 1
                else:
                    row_ind = 0
                col_sizes = dRt_dcp_FE[row_ind][0].getSizes()[1]
                zero_mat = zero_petsc_mat(row_sizes, col_sizes, 
                           PREALLOC=100, comm=self.comm)
                dRt_dcp_FE[s_ind0][0] = zero_mat


        if len(ind_list) == 1:
            for s_ind in range(self.num_splines):
                if dRt_dcp_FE[s_ind][0] is None:
                    temp_mat = m2p(assemble(self.dR_dcp_symexp[field_ind][s_ind]))
                    row_sizes = temp_mat.getSizes()[0]
                    col_sizes = dRt_dcp_FE[0][0].getSizes()[1]
                    zero_mat = zero_petsc_mat(row_sizes, col_sizes, 
                               PREALLOC=100, comm=self.comm)
                    dRt_dcp_FE[s_ind][0] = zero_mat

        return dRt_dcp_FE

    def assemble_dRFEdh_th(self):
        """
        Derivatives of non-matching residual w.r.t. shell thickness
        in FE DoFs.
        """
        dRFE_dh_th = [[None for i1 in range(self.num_splines)] 
                            for i2 in range(self.num_splines)]
        for i in range(self.num_splines):
            dRFE_dh_th_assemble = assemble(self.dR_dh_th_symexp[i])
            dRFE_dh_th[i][i] = m2p(dRFE_dh_th_assemble)
        return dRFE_dh_th


    def RIGA(self):
        """
        Return the non-matching residual in IGA DoFs.
        """
        Rt_FE = self.assemble_RFE()
        Rt_IGA = self.extract_nonmatching_vec(Rt_FE, scalar=False, 
                                              apply_bcs=True)
        return Rt_IGA

    def dRIGAduIGA(self):
        """
        Return the derivative of non-matching residual in IGA DoFs
        w.r.t. displacements in IGA DoFs.
        """
        dRdu_FE = self.assemble_dRFEduFE()
        dRdu_IGA = self.extract_nonmatching_mat(dRdu_FE, ext_right=True,
                   left_scalar=False, right_scalar=False,
                   apply_col_bcs=True, apply_row_bcs=True)
        return dRdu_IGA

    def dRIGAdCPFE(self, field):
        """
        Return the derivative of non-matching residual in IGA DoFs
        w.r.t. control points in FE DoFs.
        """
        ind_list = self.shopt_surf_inds[self.opt_field.index(field)]
        dRdcp_FE = self.assemble_dRFEdCPFE(ind_list, field)
        dRIGAdcp_funcs = self.extract_nonmatching_mat(dRdcp_FE, 
                         right_ind_list=ind_list,
                         ext_right=False, 
                         apply_col_bcs=False, apply_row_bcs=True, diag=0)
        return dRIGAdcp_funcs


    def dRIGAdCPIGA_FD(self, CP, field, h=1e-8):
        self.update_CPIGA(CP, field)
        R_init = self.RIGA().array

        self.dRigadcpiga_FD = np.zeros((R_init.size, CP.size))
        cp_init = CP.copy()

        for cp_ind, cpi in enumerate(CP):
            perturb = np.zeros(CP.size)
            perturb[cp_ind] = h
            cp_perturb = cp_init + perturb
            self.update_CPIGA(cp_perturb, field)
            R_perturb = self.RIGA().array
            R_diff = R_perturb - R_init
            self.dRigadcpiga_FD[:,cp_ind] = R_diff/h
        return self.dRigadcpiga_FD

    def dRIGAdCPIGA(self, field):
        """
        Return the derivative of non-matching residual in IGA DoFs
        w.r.t. control points in IGA DoFs.
        """
        ind_list = self.shopt_surf_inds[self.opt_field.index(field)]
        dRdcp_FE = self.assemble_dRFEdCPFE(ind_list, field)
        dRIGAdcp_IGA = self.extract_nonmatching_mat(dRdcp_FE, 
                         right_ind_list=ind_list,
                         ext_right=True, right_scalar=True, 
                         apply_col_bcs=False, apply_row_bcs=True,
                         diag=0)
        return dRIGAdcp_IGA

    def dRIGAdh_th(self):
        """
        Return the derivative of non-matching residual in IGA DoFs
        w.r.t. shell thickness in IGA DoFs.
        """
        dRFEdh_th = self.assemble_dRFEdh_th()
        dRIGAdh_th_mat = self.extract_nonmatching_mat(dRFEdh_th,
                         ext_right=self.var_thickness, right_scalar=True, 
                         apply_col_bcs=False)
        return dRIGAdh_th_mat


    def dRIGAdxi_FD(self, xi_flat, h=1e-8):

        xi_init = xi_flat.copy()
        self.update_xi(xi_init)
        self.update_transfer_matrices()
        R_init = self.RIGA().array

        self.dRigadxi_FD = np.zeros((R_init.size, xi_flat.size))

        for xi_ind, xi in enumerate(xi_flat):
            print("Computing FD dRIGAdxi, column: {} out of {}".format(xi_ind, xi_flat.size))
            perturb = np.zeros(xi_flat.size)
            perturb[xi_ind] = h
            xi_perturb = xi_init + perturb
            self.update_xi(xi_perturb)
            self.update_transfer_matrices()
            R_perturb = self.RIGA().array
            R_diff = R_perturb - R_init

            self.dRigadxi_FD[:,xi_ind] = R_diff/h

        return self.dRigadxi_FD

    # def dRIGAdxi(self, diff_int_inds=None):
    def dRIGAdxi(self):
        """
        Reserved for shape optimization with moving intersections
        """
        num_sides = 2
        dRIGAdxi_sub_list = []
        for i, diff_int_ind in enumerate(self.diff_int_inds):
            dRIGAdxi_sub_list += [[[None, None],[None, None]]]
            for side in range(num_sides):
                dRIGAdxi_sub_temp = self.dRIGAdxi_sub(diff_int_ind, side)
                ##### Apply BCs to dRIGAdxi###########
                s_ind0, s_ind1 = self.mapping_list[diff_int_ind]
                zero_dof0 = self.splines[s_ind0].zeroDofs
                zero_dof1 = self.splines[s_ind1].zeroDofs
                dRIGAdxi_sub_temp[0].zeroRows(zero_dof0, diag=0)
                dRIGAdxi_sub_temp[0].assemble()
                dRIGAdxi_sub_temp[1].zeroRows(zero_dof1, diag=0)
                dRIGAdxi_sub_temp[1].assemble()
                ######################################

                dRIGAdxi_sub_list[i][0][side] = dRIGAdxi_sub_temp[0]
                dRIGAdxi_sub_list[i][1][side] = dRIGAdxi_sub_temp[1]

        self.dRIGAdxi_list = [[None for i1 in range(int(len(self.diff_int_inds)*num_sides))] 
                               for i2 in range(self.num_splines)]
        for i, diff_int_ind in enumerate(self.diff_int_inds):
            s_ind0, s_ind1 = self.mapping_list[diff_int_ind]
            self.dRIGAdxi_list[s_ind0][i*num_sides] = dRIGAdxi_sub_list[i][0][0]
            self.dRIGAdxi_list[s_ind0][i*num_sides+1] = dRIGAdxi_sub_list[i][0][1]
            self.dRIGAdxi_list[s_ind1][i*num_sides] = dRIGAdxi_sub_list[i][1][0]
            self.dRIGAdxi_list[s_ind1][i*num_sides+1] = dRIGAdxi_sub_list[i][1][1]

        # Fill out empty rows before creating nest matrix
        for s_ind in range(self.num_splines):
            none_row = True
            for j in range(int(len(self.diff_int_inds)*num_sides)):
                if self.dRIGAdxi_list[s_ind][j] is not None:
                    none_row = False
            if none_row:
                num_pts = int(self.mortar_nels[self.diff_int_inds[0]]+1)
                row_sizes = m2p(self.splines[s_ind].M).sizes[1]
                col_sizes = 2*num_pts # TODO: This size doesn't work in parallel
                self.dRIGAdxi_list[s_ind][0] = zero_petsc_mat(row_sizes, col_sizes,
                                               comm=self.comm)

        dRIGAdxi_mat = create_nest_PETScMat(self.dRIGAdxi_list, comm=self.comm)
        return dRIGAdxi_mat

    def dRIGAdxi_sub(self, index=0, side=0):
        """
        Return a list of PETSc matrix with two elements
        if side is 0: [dRAIGA/dxiA, dRBIGA/dxiA]
        if side is 1: [dRAIGA/dxiB, dRBIGA/dixB]

        index : intersection index
        side : int, {0,1}, side for mortar mesh coordinates
        """
        # TODO: this number of points don't work in parallel
        num_pts = int(self.mortar_nels[index]+1)
        mapping_list = self.mapping_list
        s_ind0, s_ind1 = self.mapping_list[index]
        other_side = int(1 - side)
        if side == 0:
            proj_tan = False
        else:
            proj_tan = True

        PE = penalty_energy(self.splines[s_ind0], self.splines[s_ind1], 
            self.spline_funcs[s_ind0], self.spline_funcs[s_ind1], 
            self.mortar_meshes[index], 
            self.mortar_funcs[index], self.mortar_cpfuncs[index], 
            self.transfer_matrices_list[index],
            self.transfer_matrices_control_list[index],
            self.alpha_d_list[index], self.alpha_r_list[index], 
            proj_tan=proj_tan)

        ####################################################
        # Get extraction, transfer matrices, displacements, 
        # control point functions
        ####################################################
        # Off-diagonal
        Mo = m2p(self.splines[mapping_list[index][other_side]].M)
        A0o = m2p(self.transfer_matrices_list[index][other_side][0])
        A1o = m2p(self.transfer_matrices_list[index][other_side][1])

        u0Mo = self.mortar_funcs[index][other_side][0]
        u1Mo = self.mortar_funcs[index][other_side][1]

        # Diagonal
        M = m2p(self.splines[mapping_list[index][side]].M)
        A0 = m2p(self.transfer_matrices_list[index][side][0])
        A1 = m2p(self.transfer_matrices_list[index][side][1])
        A2 = m2p(self.transfer_matrices_list[index][side][2])
        A1c = m2p(self.transfer_matrices_control_list[index][side][1])
        A2c = m2p(self.transfer_matrices_control_list[index][side][2])

        u0M = self.mortar_funcs[index][side][0]
        u1M = self.mortar_funcs[index][side][1]
        P0M = self.mortar_cpfuncs[index][side][0]
        P1M = self.mortar_cpfuncs[index][side][1]
        uFE = self.spline_funcs[mapping_list[index][side]]
        PFE = self.splines[mapping_list[index][side]].cpFuncs

        ###############################
        # Derivatives of penalty energy 
        ###############################
        # Off-diagonal
        R_pen0Mo = derivative(PE, u0Mo)
        R_pen1Mo = derivative(PE, u1Mo)
        R_pen0Mo_vec = v2p(assemble(R_pen0Mo))
        R_pen1Mo_vec = v2p(assemble(R_pen1Mo))
        
        # Diagonal
        R_pen0M = derivative(PE, u0M)
        R_pen1M = derivative(PE, u1M)
        R_pen0M_vec = v2p(assemble(R_pen0M))
        R_pen1M_vec = v2p(assemble(R_pen1M))
        R_pen0M_mat = self.R_vec2mat(R_pen0M_vec, num_pts, self.nsd, self.npd, 0)
        R_pen1M_mat = self.R_vec2mat(R_pen1M_vec, num_pts, self.nsd, self.npd, 1)

        der_mat_FE_diag = A1.transposeMatMult(R_pen0M_mat.transpose())
        der_mat_FE_diag += A2.transposeMatMult(R_pen1M_mat.transpose())

        ######################################
        # Derivatives for mortar displacements 
        ######################################
        # Off-diagonal
        dR0Modu0M = derivative(R_pen0Mo, u0M)
        dR0Modu1M = derivative(R_pen0Mo, u1M)
        dR1Modu0M = derivative(R_pen1Mo, u0M)
        dR1Modu1M = derivative(R_pen1Mo, u1M)
        dR0Modu0M_mat = m2p(assemble(dR0Modu0M))
        dR0Modu1M_mat = m2p(assemble(dR0Modu1M))
        dR1Modu0M_mat = m2p(assemble(dR1Modu0M))
        dR1Modu1M_mat = m2p(assemble(dR1Modu1M))
        A1u_vec = A_x(A1, uFE)
        A2u_vec = A_x(A2, uFE)
        A1u_mat = self.Au_vec2mat(A1u_vec, num_pts, self.nsd, self.npd, order=1)
        A2u_mat = self.Au_vec2mat(A2u_vec, num_pts, self.nsd, self.npd, order=2)

        temp_mat = A0o.transposeMatMult(dR0Modu0M_mat) + \
                   A1o.transposeMatMult(dR1Modu0M_mat)
        der_mat_FE_offdiag = temp_mat.matMult(A1u_mat)

        temp_mat = A0o.transposeMatMult(dR0Modu1M_mat) + \
                   A1o.transposeMatMult(dR1Modu1M_mat)
        der_mat_FE_offdiag += temp_mat.matMult(A2u_mat)

        # Diagonal
        dR0Mdu0M = derivative(R_pen0M, u0M)
        dR0Mdu1M = derivative(R_pen0M, u1M)
        dR1Mdu0M = derivative(R_pen1M, u0M)
        dR1Mdu1M = derivative(R_pen1M, u1M)
        dR0Mdu0M_mat = m2p(assemble(dR0Mdu0M))
        dR0Mdu1M_mat = m2p(assemble(dR0Mdu1M))
        dR1Mdu0M_mat = m2p(assemble(dR1Mdu0M))
        dR1Mdu1M_mat = m2p(assemble(dR1Mdu1M))

        temp_mat = A0.transposeMatMult(dR0Mdu0M_mat) + \
                   A1.transposeMatMult(dR1Mdu0M_mat)
        der_mat_FE_diag += temp_mat.matMult(A1u_mat)

        temp_mat = A0.transposeMatMult(dR0Mdu1M_mat) + \
                   A1.transposeMatMult(dR1Mdu1M_mat)
        der_mat_FE_diag += temp_mat.matMult(A2u_mat)


        ##############################################
        # For derivatives w.r.t. mortar control points 
        ##############################################
        # Off-diagonal
        dR0ModP0M_list = []
        dR0ModP1M_list = []
        dR1ModP0M_list = []
        dR1ModP1M_list = []
        dR0ModP0M_mat_list = []
        dR0ModP1M_mat_list = []
        dR1ModP0M_mat_list = []
        dR1ModP1M_mat_list = []
        A1cP_vec_list = []
        A2cP_vec_list = []
        A1cP_mat_list = []
        A2cP_mat_list = []
        temp_mat1_list_offdiag = []
        temp_mat2_list_offdiag = []
        # Diagonal
        dR0MdP0M_list = []
        dR0MdP1M_list = []
        dR1MdP0M_list = []
        dR1MdP1M_list = []
        dR0MdP0M_mat_list = []
        dR0MdP1M_mat_list = []
        dR1MdP0M_mat_list = []
        dR1MdP1M_mat_list = []
        A1cP_vec_list = []
        A2cP_vec_list = []
        A1cP_mat_list = []
        A2cP_mat_list = []
        temp_mat1_list_diag = []
        temp_mat2_list_diag = []

        for i in range(len(PFE)):
            # Off-diagonal
            dR0ModP0M_list += [derivative(R_pen0Mo, P0M[i])]
            dR0ModP1M_list += [derivative(R_pen0Mo, P1M[i])]
            dR1ModP0M_list += [derivative(R_pen1Mo, P0M[i])]
            dR1ModP1M_list += [derivative(R_pen1Mo, P1M[i])]
            dR0ModP0M_mat_list += [m2p(assemble(dR0ModP0M_list[i]))]
            dR0ModP1M_mat_list += [m2p(assemble(dR0ModP1M_list[i]))]
            dR1ModP0M_mat_list += [m2p(assemble(dR1ModP0M_list[i]))]
            dR1ModP1M_mat_list += [m2p(assemble(dR1ModP1M_list[i]))]
            temp_mat1_list_offdiag += [A0o.transposeMatMult(dR0ModP0M_mat_list[i]) 
                               + A1o.transposeMatMult(dR1ModP0M_mat_list[i])]
            temp_mat2_list_offdiag += [A0o.transposeMatMult(dR0ModP1M_mat_list[i]) 
                               + A1o.transposeMatMult(dR1ModP1M_mat_list[i])]
            A1cP_vec_list += [A_x(A1c, PFE[i])]
            A1cP_mat_list += [self.Au_vec2mat(A1cP_vec_list[i], num_pts, 
                                          1, self.npd, 1)]
            A2cP_vec_list += [A_x(A2c, PFE[i])]
            A2cP_mat_list += [self.Au_vec2mat(A2cP_vec_list[i], num_pts, 
                                          1, self.npd, 2)]
            der_mat_FE_offdiag += temp_mat1_list_offdiag[i].matMult(A1cP_mat_list[i])
            der_mat_FE_offdiag += temp_mat2_list_offdiag[i].matMult(A2cP_mat_list[i])

            # Diagonal
            dR0MdP0M_list += [derivative(R_pen0M, P0M[i])]
            dR0MdP1M_list += [derivative(R_pen0M, P1M[i])]
            dR1MdP0M_list += [derivative(R_pen1M, P0M[i])]
            dR1MdP1M_list += [derivative(R_pen1M, P1M[i])]
            dR0MdP0M_mat_list += [m2p(assemble(dR0MdP0M_list[i]))]
            dR0MdP1M_mat_list += [m2p(assemble(dR0MdP1M_list[i]))]
            dR1MdP0M_mat_list += [m2p(assemble(dR1MdP0M_list[i]))]
            dR1MdP1M_mat_list += [m2p(assemble(dR1MdP1M_list[i]))]
            temp_mat1_list_diag += [A0.transposeMatMult(dR0MdP0M_mat_list[i]) 
                               + A1.transposeMatMult(dR1MdP0M_mat_list[i])]
            temp_mat2_list_diag += [A0.transposeMatMult(dR0MdP1M_mat_list[i]) 
                               + A1.transposeMatMult(dR1MdP1M_mat_list[i])]
            der_mat_FE_diag += temp_mat1_list_diag[i].matMult(A1cP_mat_list[i])
            der_mat_FE_diag += temp_mat2_list_diag[i].matMult(A2cP_mat_list[i])

        ###############################################
        # For derivatives w.r.t. parametric coordinates 
        ###############################################
        # Off-diagonal
        xi_m = SpatialCoordinate(self.mortar_meshes[index])
        dR0Modxi_m = derivative(R_pen0Mo, xi_m)
        dR1Modxi_m = derivative(R_pen1Mo, xi_m)
        dR0Modxi_m_mat = m2p(assemble(dR0Modxi_m))
        dR1Modxi_m_mat = m2p(assemble(dR1Modxi_m))

        der_mat_FE_offdiag += A0o.transposeMatMult(dR0Modxi_m_mat)
        der_mat_FE_offdiag += A1o.transposeMatMult(dR1Modxi_m_mat)

        # Diagonal
        xi_m = SpatialCoordinate(self.mortar_meshes[index])
        dR0Mdxi_m = derivative(R_pen0M, xi_m)
        dR1Mdxi_m = derivative(R_pen1M, xi_m)
        dR0Mdxi_m_mat = m2p(assemble(dR0Mdxi_m))
        dR1Mdxi_m_mat = m2p(assemble(dR1Mdxi_m))

        der_mat_FE_diag += A0.transposeMatMult(dR0Mdxi_m_mat)
        der_mat_FE_diag += A1.transposeMatMult(dR1Mdxi_m_mat)

        #######################
        # Off-diagonal
        der_mat_FE_offdiag.assemble()
        # Diagonal
        der_mat_FE_diag.assemble()

        #######################
        # Off-diagonal
        der_mat_IGA_rev_offdiag = Mo.transposeMatMult(der_mat_FE_offdiag)
        der_mat_IGA_rev_offdiag.assemble()
        # Diagonal
        der_mat_IGA_rev_diag = M.transposeMatMult(der_mat_FE_diag)
        der_mat_IGA_rev_diag.assemble()

        #######################
        # Off-diagonal
        switch_col_mat = zero_petsc_mat(num_pts*self.npd, num_pts*self.npd, 
                                             PREALLOC=num_pts*self.npd)
        for i in range(num_pts):
            switch_col_mat.setValue(i*self.npd, num_pts*self.npd-i*self.npd-2, 1.)
            switch_col_mat.setValue(i*self.npd+1, num_pts*self.npd-i*self.npd-1, 1.)
            # switch_col_mat.setValue(i*self.npd, num_pts*self.npd-i*self.npd-2, -1.)
            # switch_col_mat.setValue(i*self.npd+1, num_pts*self.npd-i*self.npd-1, -1.)
        switch_col_mat.assemble()
        der_mat_IGA_offdiag = der_mat_IGA_rev_offdiag.matMult(switch_col_mat)

        # Diagonal
        der_mat_IGA_diag = der_mat_IGA_rev_diag.matMult(switch_col_mat)

        # # No coloum switch
        # der_mat_IGA_offdiag = der_mat_IGA_rev_offdiag
        # der_mat_IGA_diag = der_mat_IGA_rev_diag
        if side == 0:
            dRIGAdxi_sub = [der_mat_IGA_diag, der_mat_IGA_offdiag]
        elif side == 1:
            dRIGAdxi_sub = [der_mat_IGA_offdiag, der_mat_IGA_diag]
        return dRIGAdxi_sub


    def R_vec2mat(self, R, num_pts, phy_dim, para_dim, order=0):
        """
        Arrange input vector ``R`` into a diagonal matrix to 
        avoid tensor-vector multiplication. 
        Note: corresponds to "lambda_tilde" in the formulation 

        Parameters
        ---------- 
        R : PETSc vec
        num_pts : int
        phy_dim : int
        para_dim : int
        order : int, 0 or 1

        Returns
        -------
        R_mat : PETSc mat
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

    def Au_vec2mat(self, Au, num_pts, phy_dim, para_dim, order=1):
        """
        Arrange input vector ``Au`` into a matrix to represent the 
        result of tensor-vector multiplication so that we can 
        avoid the construction of the 3D matrix
        Note: corresponds to "lambda" in the formulation 

        Parameters
        ----------
        Au : PETSc vector
        num_pts : int
        phy_dim : int
        para_dim : int
        order : int, 1 or 2

        Returns
        -------
        Au_mat : PETSc mat
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

    #######################################################
    ####### Create and save pvd files #####################
    #######################################################

    def create_files(self, save_path="./", folder_name="results/", 
                     refine_mesh=False, ref_nel=32):
        """
        Create pvd files for all spline patches' displacements 
        and control points (and thickness is needed).
        """
        self.refine_mesh = refine_mesh
        self.save_path = save_path
        self.folder_name = folder_name
        self.u_file_names = []
        self.u_files = []
        self.F_file_names = []
        self.F_files = []
        if self.opt_thickness:
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
                    self.F_files[i] += [File(self.comm, 
                                             self.F_file_names[i][j+1]),]
            if self.opt_thickness:
                self.t_file_names += [save_path+folder_name+'t'+str(i)
                                      +'_file.pvd',]
                self.t_files += [File(self.comm, self.t_file_names[i]),]

        # Save results on a refine mesh for better visualization
        if self.refine_mesh:
            self.mesh_ref_list = []
            self.V_ref_list = []
            self.Vv_ref_list = []
            self.Au_ref_list = []
            self.AF_ref_list = []
            self.u_func_ref_list = [None for s_ind in range(self.num_splines)]
            self.F_func_ref_list = [[] for s_ind in range(self.num_splines)]
            if self.opt_thickness:
                self.Ah_ref_list = []
                self.h_th_func_ref_list = [None for s_ind in 
                                           range(self.num_splines)]
            for s_ind in range(self.num_splines):
                self.mesh_ref_list += [UnitSquareMesh(ref_nel, ref_nel),]
                self.V_ref_list += [FunctionSpace(self.mesh_ref_list[s_ind], 
                                                  'CG', 1),]
                self.Vv_ref_list += [VectorFunctionSpace(
                                     self.mesh_ref_list[s_ind], 
                                     'CG', 1, dim=3),]
                self.Au_ref_list += [create_transfer_matrix(
                                     self.splines[s_ind].V, 
                                     self.Vv_ref_list[s_ind])]
                self.AF_ref_list += [create_transfer_matrix(
                                     self.splines[s_ind].V_control, 
                                     self.V_ref_list[s_ind])]
                self.u_func_ref_list[s_ind] = \
                        Function(self.Vv_ref_list[s_ind])
                for field in range(3):
                    self.F_func_ref_list[s_ind] += \
                        [Function(self.V_ref_list[s_ind])]
                    if field == 2:
                        self.F_func_ref_list[s_ind] += \
                            [Function(self.V_ref_list[s_ind])]
                if self.opt_thickness:
                    self.Ah_ref_list += [create_transfer_matrix(
                                     self.h_th[s_ind].function_space(),
                                     self.V_ref_list[s_ind])]
                    self.h_th_func_ref_list[s_ind] = \
                        Function(self.V_ref_list[s_ind])
                
    def save_files(self):
        """
        Save splines' displacements and control points to pvd files.
        """
        for i in range(self.num_splines):
            if self.refine_mesh:
                A_x_b(self.Au_ref_list[i], 
                      v2p(self.spline_funcs[i].vector()),
                      v2p(self.u_func_ref_list[i].vector()))
                u_split = self.u_func_ref_list[i].split()
            else:
                u_split = self.spline_funcs[i].split()

            for j in range(3):
                u_func = u_split[j]
                u_func.rename('u'+str(i)+'_'+str(j), 'u'+str(i)+'_'+str(j))
                self.u_files[i][j] << u_func

                if self.refine_mesh:
                    A_x_b(self.AF_ref_list[i], 
                          v2p(self.splines[i].cpFuncs[j].vector()),
                          v2p(self.F_func_ref_list[i][j].vector()))
                    f_func = self.F_func_ref_list[i][j]
                else:
                    f_func = self.splines[i].cpFuncs[j]

                f_func.rename('F'+str(i)+'_'+str(j), 'F'+str(i)+'_'+str(j))
                self.F_files[i][j] << f_func

                if j==2:
                    if self.refine_mesh:
                        A_x_b(self.AF_ref_list[i], 
                              v2p(self.splines[i].cpFuncs[j+1].vector()),
                              v2p(self.F_func_ref_list[i][j+1].vector()))
                        w_func = self.F_func_ref_list[i][j+1]
                    else:
                        w_func = self.splines[i].cpFuncs[j+1]

                    w_func.rename('F'+str(i)+'_'+str(j+1), 
                                  'F'+str(i)+'_'+str(j+1))
                    self.F_files[i][j+1] << w_func
            if self.opt_thickness:
                if self.refine_mesh:
                    A_x_b(self.Ah_ref_list[i], v2p(self.h_th[i].vector()),
                          v2p(self.h_th_func_ref_list[i].vector()))
                    h_th_func = self.h_th_func_ref_list[i]
                else:
                    h_th_func = self.h_th[i]
                h_th_func.rename('t'+str(i), 't'+str(i))
                self.t_files[i] << h_th_func

    #######################################################
    ######## Aero pressure related methods ################
    #######################################################

    def set_aero_linear_splines(self, linear_splines, Paero):
        self.use_aero_pressure = True
        self.linear_splines = linear_splines
        self.Paero = Paero

        # Create nested vectors in IGA DoFs
        self.linear_spline_vec_iga_list = []
        self.linear_spline_vec_iga_dof_list = []
        for s_ind in range(self.num_splines):
            self.linear_spline_vec_iga_list += [zero_petsc_vec(
                self.linear_splines[s_ind].M.size(1), comm=self.comm)]
            self.linear_spline_vec_iga_dof_list += [
                self.linear_splines[s_ind].M.size(1),]

        self.linear_spline_vec_iga_nest = create_nest_PETScVec(
            self.linear_spline_vec_iga_list, comm=self.comm)
        self.linear_spline_vec_iga_dof = \
            self.linear_spline_vec_iga_nest.getSizes()[1]

        dR_dPaero_ufl_symexp = []
        for s_ind in range(self.num_splines):
            dR_dPaero_ufl_symexp += [derivative(self.residuals_form[s_ind],
                                    self.Paero[s_ind])]
        self.dR_dPaero_symexp = [Form(dR_dh_th) for dR_dh_th 
                                in dR_dPaero_ufl_symexp]

        self.Paero_fe_list = [v2p(Paero_sub.vector()) 
                              for Paero_sub in self.Paero]
        self.Paero_fe_nest = create_nest_PETScVec(self.Paero_fe_list,
                                                  comm=self.comm)
        self.Paero_iga_nest = self.linear_spline_vec_iga_nest.copy()

        self.Paero_fe_sizes = [Paero_fe_sub.getSizes()[1] for Paero_fe_sub in
                           self.Paero_fe_list]
        self.init_Paero = None

    def assemble_dRFEdPaero(self):
        """
        Derivatives of non-matching residual w.r.t. aero pressures
        in FE DoFs.
        """
        dRFEdPaero = [[None for i1 in range(self.num_splines)] 
                            for i2 in range(self.num_splines)]
        for i in range(self.num_splines):
            dRFEdPaero_assemble = assemble(self.dR_dPaero_symexp[i])
            dRFEdPaero[i][i] = m2p(dRFEdPaero_assemble)
        return dRFEdPaero

    def dRIGAdPaero(self):
        dRFEdPaero = self.assemble_dRFEdPaero()

        dRIGAdPaero_list = []
        for i in range(self.num_splines):
            dRIGAdPaero_list += [[],]
            for j in range(self.num_splines):
                if dRFEdPaero[i][j] is not None:
                    # Extract matrix
                    M_left = m2p(self.splines[i].M)
                    M_right = m2p(self.linear_splines[j].M)
                    mat_iga_temp = AT_R_B(M_left, dRFEdPaero[i][j], M_right)
                else:
                    mat_iga_temp = None
                dRIGAdPaero_list[i] += [mat_iga_temp,]
        mat_iga = create_nest_PETScMat(dRIGAdPaero_list, comm=self.comm)

        if MPI.size(self.comm) == 1:
            mat_iga.convert('seqaij')
        else:
            mat_iga = create_aijmat_from_nestmat(mat_iga, 
                      dRIGAdPaero_list, comm=self.comm)
        return mat_iga

    def update_Paero(self, Paero_array):
        """
        Update splines' thickness functions with input array
        in IGA DoFs
        """
        update_nest_vec(Paero_array, self.Paero_iga_nest, comm=self.comm)
        Paero_iga_sub = self.Paero_iga_nest.getNestSubVecs()
        for s_ind in range(self.num_splines):
            M_petsc = m2p(self.linear_splines[s_ind].M)
            M_petsc.mult(Paero_iga_sub[s_ind], self.Paero_fe_list[s_ind])
            self.Paero_fe_list[s_ind].ghostUpdate()
            self.Paero_fe_list[s_ind].assemble()

    #######################################################
    #######################################################  

if __name__ == '__main__':
    pass