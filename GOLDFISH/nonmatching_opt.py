from PENGoLINS.nonmatching_coupling import *
from GOLDFISH.utils.opt_utils import *
from GOLDFISH.utils.bsp_utils import *

# from cpiga2xi import *

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

class NonMatchingOpt(NonMatchingCoupling):
    """
    Subclass of NonmatchingCoupling which serves as the base class
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
                         int_V_family, int_V_degree,
                         int_dx_metadata, contact, comm)
        self.opt_field = opt_field
        self.opt_shape = opt_shape
        self.opt_thickness = opt_thickness
        self.var_thickness = var_thickness
        self.nsd = self.splines[0].nsd

        # Create nested vectors in IGA DoFs
        self.vec_iga_list = []
        self.vec_scalar_iga_list = []
        for s_ind in range(self.num_splines):
            self.vec_iga_list += [zero_petsc_vec(
                                  self.splines[s_ind].M.size(1),
                                  comm=self.comm)]
            self.vec_scalar_iga_list += [zero_petsc_vec(
                                     self.splines[s_ind].M_control.size(1),
                                     comm=self.comm)]
        self.vec_iga_nest = create_nest_PETScVec(self.vec_iga_list,
                                                 comm=self.comm)
        self.vec_scalar_iga_nest = create_nest_PETScVec(
                                   self.vec_scalar_iga_list,
                                   comm=self.comm)
        self.vec_iga_dof = self.vec_iga_nest.getSizes()[1]
        self.vec_scalar_iga_dof = self.vec_scalar_iga_nest.getSizes()[1]

        # Create nested vectors in FE DoFs
        self.vec_fe_list = []
        self.vec_scalar_fe_list = []
        for s_ind in range(self.num_splines):
            self.vec_fe_list += [zero_petsc_vec(
                                 self.splines[s_ind].M.size(0),
                                 comm=self.comm)]
            self.vec_scalar_fe_list += [zero_petsc_vec(
                                    self.splines[s_ind].M_control.size(0),
                                    comm=self.comm)]
        self.vec_fe_nest = create_nest_PETScVec(self.vec_fe_list, 
                                                comm=self.comm)
        self.vec_scalar_fe_nest = create_nest_PETScVec(
                                  self.vec_scalar_fe_list,
                                  comm=self.comm)
        self.vec_fe_dof = self.vec_fe_nest.getSizes()[1]
        self.vec_scalar_fe_dof = self.vec_scalar_fe_nest.getSizes()[1]

        # Create nested displacements in IGA DoFs
        self.u_iga_nest = self.vec_iga_nest.copy()

        # Create nested cpFuncs vectors (in FE DoFs)
        self.cp_funcs_list = [[] for i in range(self.nsd)]
        self.cp_funcs_nest = [None for i in range(self.nsd)]
        for field in range(self.nsd):
            for s_ind in range(self.num_splines):
                self.cp_funcs_list[field] += [v2p(self.splines[s_ind].
                                              cpFuncs[field].vector()),]
            self.cp_funcs_nest[field] = create_nest_PETScVec(
                                        self.cp_funcs_list[field],
                                        comm=self.comm)

        # Create nested control points in IGA DoFs
        self.cp_iga_nest = self.vec_scalar_iga_nest.copy()
        # Set initial control points in IGA DoFs as None
        self.init_cp_iga = None

        if self.opt_thickness:
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

        # Initial attributes
        self.init_cpfuncs_list = [[] for s_ind in range(self.num_splines)]
        self.hl_phy = []
        # self.h_phy_linear = []
        # self.h_phy_avg = []
        self.ha_phy = []
        self.ha_phy_linear = []
        self.hl_phy_linear = []
        for s_ind in range(self.num_splines):
            for field in range(self.nsd+1):
                self.init_cpfuncs_list[s_ind] += [Function(self.splines[s_ind].V_control)]
                self.init_cpfuncs_list[s_ind][field].assign(self.splines[s_ind].cpFuncs[field])
            self.hl_phy += [spline_mesh_size(self.splines[s_ind])]
            self.ha_phy += [spline_mesh_area(self.splines[s_ind])]
            self.hl_phy_linear += [self.splines[s_ind].\
                                   projectScalarOntoLinears(self.hl_phy[s_ind])] 
            self.ha_phy_linear += [self.splines[s_ind].\
                                   projectScalarOntoLinears(self.ha_phy[s_ind])]

            # self.h_phy_linear += [self.splines[s_ind].\
            #                       projectScalarOntoLinears(self.h_phy[s_ind])]
            # self.h_phy_avg += [np.average(self.h_phy_linear[s_ind].vector().get_local())]

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
        init_cp_iga = [None for i in range(self.nsd)]
        init_cp_iga_list = [[] for i in range(self.nsd)]
        for field in range(self.nsd):
            for s_ind in range(self.num_splines):
                cp_fe = self.cp_funcs_list[field][s_ind]
                Mc = m2p(self.splines[s_ind].M_control)
                McTMc = Mc.transposeMatMult(Mc)
                McTcp_fe = AT_x(Mc, cp_fe)
                cp_iga = solve_Ax_b(McTMc, McTcp_fe)
                init_cp_iga_list[field] += [get_petsc_vec_array(cp_iga, 
                                            comm=self.comm)]
            init_cp_iga[field] = np.concatenate(init_cp_iga_list[field])
        init_cp_iga = np.array(init_cp_iga).T
        return init_cp_iga

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
        update_nest_vec(cp_array_fe, self.cp_funcs_nest[field], 
                        comm=self.comm)

    def update_CPIGA(self, cp_array_iga, field):
        """
        Update splines' control point functions with input array
        in IGA DoFs
        """
        update_nest_vec(cp_array_iga, self.cp_iga_nest, comm=self.comm)
        cp_iga_sub = self.cp_iga_nest.getNestSubVecs()
        for s_ind in range(self.num_splines):
            self.vec_scalar_IGA2FE(cp_iga_sub[s_ind],
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

    def set_xi_diff_info(self, preprocessor, int_indices_diff=None):
        """
        This function is noly need when differentiating intersections'
        parametric coordiantes, and require the class ``CPIGA2Xi``
        """
        assert self.transfer_mat_deriv == 2
        self.cpiga2xi = CPIGA2Xi(preprocessor, int_indices_diff, self.opt_field)
        self.int_indices_diff = self.cpiga2xi.int_indices_diff
        self.Vms_2dim = [VectorFunctionSpace(
                         self.mortar_meshes[int_ind_global], 'CG', 1)
                         for int_ind_global in self.int_indices_diff]
        self.xi_funcs = []
        self.xi_vecs = []
        for int_ind, int_ind_global in enumerate(self.int_indices_diff):
            self.xi_funcs += [Function(self.Vms_2dim[int_ind]),
                              Function(self.Vms_2dim[int_ind])]
            self.xi_vecs += [v2p(self.xi_funcs[-2].vector()),
                             v2p(self.xi_funcs[-1].vector())]
        self.xi_nest = create_nest_PETScVec(self.xi_vecs, comm=self.comm)
        self.xi_size = self.cpiga2xi.xi_size_global

    def update_xi(self, xi_flat):
        """
        Update intersections' parametric coordinates
        """
        sub_vecs = self.xi_nest.getNestSubVecs()
        num_sub_vecs = len(sub_vecs)

        sub_vecs_range = []
        sub_vecs_size = []
        for i in range(num_sub_vecs):
            sub_vecs_range += [sub_vecs[i].getOwnershipRange(),]
            sub_vecs_size += [sub_vecs[i].size,]

        sub_array_list = []
        array_ind_off = 0
        for i in range(num_sub_vecs):
            sub_array = xi_flat[array_ind_off+sub_vecs_range[i][0]: 
                                  array_ind_off+sub_vecs_range[i][1]]
            sub_array = sub_array.reshape(-1, self.para_dim)
            sub_array = sub_array[::-1].reshape(-1)
            sub_array_list += [sub_array,]
            array_ind_off += sub_vecs_size[i]
        nest_array = np.concatenate(sub_array_list)
        self.xi_nest.setArray(nest_array)
        self.xi_nest.assemble()

    def update_transfer_matrices_sub(self, xi_func, index, side):
        """
        Update transfer matrices for single intersection on one side
        """
        move_mortar_mesh(self.mortar_meshes[index], xi_func)
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
            for j in range(self.num_field+1):
                A_x_b(self.transfer_matrices_control_list[index][side][i], 
                    self.splines[self.mapping_list[index][side]]
                    .cpFuncs[j].vector(), 
                    self.mortar_cpfuncs[index][side][i][j].vector())

    def update_transfer_matrices(self):
        """
        Update transfer matrices for all intersections
        """
        for int_ind, int_ind_global in enumerate(self.int_indices_diff):
            for side in range(self.para_dim):
                self.update_transfer_matrices_sub(
                    self.xi_funcs[int(int_ind*self.para_dim+side)],
                    int_ind_global, side)

    # def update_transfer_matrices(self):
    #     for int_ind, int_ind_global in enumerate(self.int_indices_diff):
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

    def extract_nonmatching_vec(self, vec_list, scalar=False, 
                                apply_bcs=False):
        """
        Extract non-matching vector from FE to IGA DoFs
        """
        vec_iga_list = []
        for i in range(self.num_splines):
            if scalar:
                M = m2p(self.splines[i].M_control)
            else:
                M = m2p(self.splines[i].M)
            vec_iga_sub = AT_x(M, vec_list[i])
            # Only apply bcs to non-scalar field vectors
            if apply_bcs and not scalar:
                apply_bcs_vec(self.splines[i], vec_iga_sub)
            vec_iga_list += [vec_iga_sub,]
        vec_iga = create_nest_PETScVec(vec_iga_list)
        return vec_iga

    def extract_nonmatching_mat(self, mat_list, ext_right=True,
                                left_scalar=False, right_scalar=False, 
                                apply_row_bcs=False, apply_col_bcs=False):
        """
        Extract non-matching matrix from FE to IGA DoFs.
        """
        mat_iga_list = []
        for i in range(self.num_splines):
            mat_iga_list += [[],]
            for j in range(self.num_splines):
                if mat_list[i][j] is not None:
                    # Extract matrix
                    if left_scalar:
                        M_left = m2p(self.splines[i].M_control)
                    else:
                        M_left = m2p(self.splines[i].M)
                    if ext_right:
                        if right_scalar:
                            M_right = m2p(self.splines[j].M_control)
                        else:
                            M_right = m2p(self.splines[j].M)
                        mat_iga_temp = AT_R_B(M_left, mat_list[i][j], M_right)
                    else:
                        mat_iga_temp = M_left.transposeMatMult(mat_list[i][j])
                    # Apply boundary conditions
                    # Only these two conditions are considered
                    if apply_row_bcs and apply_col_bcs:
                        if i == j:
                            mat_iga_temp = apply_bcs_mat(self.splines[i],
                                           mat_iga_temp, diag=1)
                        else:
                            mat_iga_temp = apply_bcs_mat(self.splines[i],
                                           mat_iga_temp, self.splines[j], 
                                           diag=0)
                    elif apply_row_bcs and not apply_col_bcs:
                        if i == j:
                            diag = 1
                        else:
                            diag=0
                        mat_iga_temp.zeroRows(self.splines[i].zeroDofs,
                                              diag=diag)
                else:
                    mat_iga_temp = None

                mat_iga_list[i] += [mat_iga_temp,]

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

    def assemble_dRFEdCPFE(self, field):
        """
        Derivatives of non-matching residual w.r.t. displacements in
        FE DoFs.
        """
        # Compute contributions from shell residuals and derivatives
        field_ind = self.opt_field.index(field)
        dR_dcp_FE = []
        for i in range(self.num_splines):
            dR_dcp_assemble = assemble(self.dR_dcp_symexp[field_ind][i])
            dR_dcp_FE += [m2p(dR_dcp_assemble),]

        ## Step 2: assemble non-matching contributions
        # Create empty lists for non-matching contributions
        dRm_dcpm_FE = [[None for i1 in range(self.num_splines)] 
                             for i2 in range(self.num_splines)]

        # Compute non-matching contributions ``dRm_dcpm_FE``.
        for i in range(self.num_intersections):
            dRm_dcpm = transfer_dRmdcpm_sub(self.dRm_dcpm_list[field_ind][i],  
                       self.transfer_matrices_list[i],
                       self.transfer_matrices_control_list[i])
            for j in range(len(dRm_dcpm)):
                for k in range(len(dRm_dcpm[j])):
                    if dRm_dcpm_FE[self.mapping_list[i][j]]\
                       [self.mapping_list[i][k]] is not None:
                        dRm_dcpm_FE[self.mapping_list[i][j]]\
                            [self.mapping_list[i][k]] += dRm_dcpm[j][k]
                    else:
                        dRm_dcpm_FE[self.mapping_list[i][j]]\
                            [self.mapping_list[i][k]] = dRm_dcpm[j][k]

        ## Step 3: add derivatives from splines and mortar meshes together
        dRt_dcp_FE = [[None for i1 in range(self.num_splines)] 
                            for i2 in range(self.num_splines)]
        for i in range(self.num_splines):
            for j in range(self.num_splines):
                if i == j:
                    if dRm_dcpm_FE[i][i] is not None:
                        dRt_dcp_FE[i][i] = dR_dcp_FE[i] + dRm_dcpm_FE[i][i]
                    else:
                        dRt_dcp_FE[i][i] = dR_dcp_FE[i]
                else:
                    dRt_dcp_FE[i][j] = dRm_dcpm_FE[i][j]
        return dRt_dcp_FE

    def assemble_RFEdh_th(self):
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
        dRdcp_FE = self.assemble_dRFEdCPFE(field)
        dRIGAdcp_funcs = self.extract_nonmatching_mat(dRdcp_FE, 
                         ext_right=False, apply_col_bcs=False)
        return dRIGAdcp_funcs

    def dRIGAdCPIGA(self, field):
        """
        Return the derivative of non-matching residual in IGA DoFs
        w.r.t. control points in IGA DoFs.
        """
        dRdcp_FE = self.assemble_dRFEdCPFE(field)
        dRIGAdcp_IGA = self.extract_nonmatching_mat(dRdcp_FE, 
                         ext_right=True, right_scalar=True, 
                         apply_col_bcs=False)
        return dRIGAdcp_IGA

    def dRIGAdh_th(self):
        """
        Return the derivative of non-matching residual in IGA DoFs
        w.r.t. shell thickness in IGA DoFs.
        """
        dRFEdh_th = self.assemble_RFEdh_th()
        dRIGAdh_th_mat = self.extract_nonmatching_mat(dRFEdh_th,
                         ext_right=self.var_thickness, right_scalar=True, 
                         apply_col_bcs=False)
        return dRIGAdh_th_mat

    # def dRIGAdxi(self, int_indices_diff=None):
    def dRIGAdxi(self):
        """
        Reserved for shape optimization with moving intersections
        """
        num_sides = 2
        dRIGAdxi_sub_list = []
        for i, index in enumerate(self.int_indices_diff):
            dRIGAdxi_sub_list += [[[None, None],[None, None]]]
            for side in range(num_sides):
                dRIGAdxi_sub_temp = self.dRIGAdxi_sub(index, side)
                dRIGAdxi_sub_list[i][0][side] = dRIGAdxi_sub_temp[0]
                dRIGAdxi_sub_list[i][1][side] = dRIGAdxi_sub_temp[1]

        self.dRIGAdxi_list = [[None for i1 in range(int(len(self.int_indices_diff)*num_sides))] 
                               for i2 in range(self.num_splines)]
        for i, index in enumerate(self.int_indices_diff):
            s_ind0, s_ind1 = self.mapping_list[index]
            self.dRIGAdxi_list[s_ind0][i*num_sides] = dRIGAdxi_sub_list[i][0][0]
            self.dRIGAdxi_list[s_ind0][i*num_sides+1] = dRIGAdxi_sub_list[i][0][1]
            self.dRIGAdxi_list[s_ind1][i*num_sides] = dRIGAdxi_sub_list[i][1][0]
            self.dRIGAdxi_list[s_ind1][i*num_sides+1] = dRIGAdxi_sub_list[i][1][1]

        # Fill out empty rows before creating nest matrix
        for s_ind in range(self.num_splines):
            none_row = True
            for j in range(int(len(self.int_indices_diff)*num_sides)):
                if self.dRIGAdxi_list[s_ind][j] is not None:
                    none_row = False
            if none_row:
                num_pts = int(self.mortar_nels[self.int_indices_diff[0]]+1)
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

        # print("Step 1 ", "*"*30)
        # Get extraction, transfer matrices, displacements, control point functions
        # Off-diagonal
        self.Mo = m2p(self.splines[mapping_list[index][other_side]].M)
        self.A0o = m2p(self.transfer_matrices_list[index][other_side][0])
        self.A1o = m2p(self.transfer_matrices_list[index][other_side][1])

        self.u0Mo = self.mortar_funcs[index][other_side][0]
        self.u1Mo = self.mortar_funcs[index][other_side][1]

        # Diagonal
        self.M = m2p(self.splines[mapping_list[index][side]].M)
        self.A0 = m2p(self.transfer_matrices_list[index][side][0])
        self.A1 = m2p(self.transfer_matrices_list[index][side][1])
        self.A2 = m2p(self.transfer_matrices_list[index][side][2])
        self.A1c = m2p(self.transfer_matrices_control_list[index][side][1])
        self.A2c = m2p(self.transfer_matrices_control_list[index][side][2])

        self.u0M = self.mortar_funcs[index][side][0]
        self.u1M = self.mortar_funcs[index][side][1]
        self.P0M = self.mortar_cpfuncs[index][side][0]
        self.P1M = self.mortar_cpfuncs[index][side][1]
        self.uFE = self.spline_funcs[mapping_list[index][side]]
        self.PFE = self.splines[mapping_list[index][side]].cpFuncs

        ###################################
        # print("Step 2 ", "*"*30)
        # Derivatives of penalty energy
        # Off-diagonal
        self.R_pen0Mo = derivative(PE, self.u0Mo)
        self.R_pen1Mo = derivative(PE, self.u1Mo)
        self.R_pen0Mo_vec = v2p(assemble(self.R_pen0Mo))
        self.R_pen1Mo_vec = v2p(assemble(self.R_pen1Mo))
        
        # Diagonal
        self.R_pen0M = derivative(PE, self.u0M)
        self.R_pen1M = derivative(PE, self.u1M)
        self.R_pen0M_vec = v2p(assemble(self.R_pen0M))
        self.R_pen1M_vec = v2p(assemble(self.R_pen1M))
        self.R_pen0M_mat = Lambda_tilde(self.R_pen0M_vec, num_pts, self.num_field, self.para_dim, 0)
        self.R_pen1M_mat = Lambda_tilde(self.R_pen1M_vec, num_pts, self.num_field, self.para_dim, 1)

        self.der_mat_FE_diag = self.A1.transposeMatMult(self.R_pen0M_mat.transpose())
        self.der_mat_FE_diag += self.A2.transposeMatMult(self.R_pen1M_mat.transpose())

        #############################
        # print("Step 3 ", "*"*30)
        # Derivatives for mortar displacements
        # Off-diagonal
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
        self.A1u_mat = Lambda(self.A1u_vec, num_pts, self.num_field, self.para_dim, order=1)
        self.A2u_mat = Lambda(self.A2u_vec, num_pts, self.num_field, self.para_dim, order=2)

        temp_mat = self.A0o.transposeMatMult(self.dR0Modu0M_mat) + \
                   self.A1o.transposeMatMult(self.dR1Modu0M_mat)
        self.der_mat_FE_offdiag = temp_mat.matMult(self.A1u_mat)

        temp_mat = self.A0o.transposeMatMult(self.dR0Modu1M_mat) + \
                   self.A1o.transposeMatMult(self.dR1Modu1M_mat)
        self.der_mat_FE_offdiag += temp_mat.matMult(self.A2u_mat)

        # Diagonal
        self.dR0Mdu0M = derivative(self.R_pen0M, self.u0M)
        self.dR0Mdu1M = derivative(self.R_pen0M, self.u1M)
        self.dR1Mdu0M = derivative(self.R_pen1M, self.u0M)
        self.dR1Mdu1M = derivative(self.R_pen1M, self.u1M)
        self.dR0Mdu0M_mat = m2p(assemble(self.dR0Mdu0M))
        self.dR0Mdu1M_mat = m2p(assemble(self.dR0Mdu1M))
        self.dR1Mdu0M_mat = m2p(assemble(self.dR1Mdu0M))
        self.dR1Mdu1M_mat = m2p(assemble(self.dR1Mdu1M))
        # self.A1u_vec = A_x(self.A1, self.uFE)
        # self.A2u_vec = A_x(self.A2, self.uFE)
        # self.A1u_mat = Lambda(self.A1u_vec, num_pts, self.num_field, self.para_dim, order=1)
        # self.A2u_mat = Lambda(self.A2u_vec, num_pts, self.num_field, self.para_dim, order=2)

        temp_mat = self.A0.transposeMatMult(self.dR0Mdu0M_mat) + \
                   self.A1.transposeMatMult(self.dR1Mdu0M_mat)
        self.der_mat_FE_diag += temp_mat.matMult(self.A1u_mat)

        temp_mat = self.A0.transposeMatMult(self.dR0Mdu1M_mat) + \
                   self.A1.transposeMatMult(self.dR1Mdu1M_mat)
        self.der_mat_FE_diag += temp_mat.matMult(self.A2u_mat)


        ###############################
        # For derivatives w.r.t. mortar control points
        # print("Step 4 ", "*"*30)
        # Off-diagonal
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
        self.temp_mat1_list_offdiag = []
        self.temp_mat2_list_offdiag = []
        # Diagonal
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
        self.temp_mat1_list_diag = []
        self.temp_mat2_list_diag = []

        for i in range(len(self.PFE)):
            # Off-diagonal
            self.dR0ModP0M_list += [derivative(self.R_pen0Mo, self.P0M[i])]
            self.dR0ModP1M_list += [derivative(self.R_pen0Mo, self.P1M[i])]
            self.dR1ModP0M_list += [derivative(self.R_pen1Mo, self.P0M[i])]
            self.dR1ModP1M_list += [derivative(self.R_pen1Mo, self.P1M[i])]
            self.dR0ModP0M_mat_list += [m2p(assemble(self.dR0ModP0M_list[i]))]
            self.dR0ModP1M_mat_list += [m2p(assemble(self.dR0ModP1M_list[i]))]
            self.dR1ModP0M_mat_list += [m2p(assemble(self.dR1ModP0M_list[i]))]
            self.dR1ModP1M_mat_list += [m2p(assemble(self.dR1ModP1M_list[i]))]
            self.temp_mat1_list_offdiag += [self.A0o.transposeMatMult(self.dR0ModP0M_mat_list[i]) 
                               + self.A1o.transposeMatMult(self.dR1ModP0M_mat_list[i])]
            self.temp_mat2_list_offdiag += [self.A0o.transposeMatMult(self.dR0ModP1M_mat_list[i]) 
                               + self.A1o.transposeMatMult(self.dR1ModP1M_mat_list[i])]
            self.A1cP_vec_list += [A_x(self.A1c, self.PFE[i])]
            self.A1cP_mat_list += [Lambda(self.A1cP_vec_list[i], num_pts, 
                                          1, self.para_dim, 1)]
            self.A2cP_vec_list += [A_x(self.A2c, self.PFE[i])]
            self.A2cP_mat_list += [Lambda(self.A2cP_vec_list[i], num_pts, 
                                          1, self.para_dim, 2)]
            self.der_mat_FE_offdiag += self.temp_mat1_list_offdiag[i].matMult(self.A1cP_mat_list[i])
            self.der_mat_FE_offdiag += self.temp_mat2_list_offdiag[i].matMult(self.A2cP_mat_list[i])

            # Diagonal
            self.dR0MdP0M_list += [derivative(self.R_pen0M, self.P0M[i])]
            self.dR0MdP1M_list += [derivative(self.R_pen0M, self.P1M[i])]
            self.dR1MdP0M_list += [derivative(self.R_pen1M, self.P0M[i])]
            self.dR1MdP1M_list += [derivative(self.R_pen1M, self.P1M[i])]
            self.dR0MdP0M_mat_list += [m2p(assemble(self.dR0MdP0M_list[i]))]
            self.dR0MdP1M_mat_list += [m2p(assemble(self.dR0MdP1M_list[i]))]
            self.dR1MdP0M_mat_list += [m2p(assemble(self.dR1MdP0M_list[i]))]
            self.dR1MdP1M_mat_list += [m2p(assemble(self.dR1MdP1M_list[i]))]
            self.temp_mat1_list_diag += [self.A0.transposeMatMult(self.dR0MdP0M_mat_list[i]) 
                               + self.A1.transposeMatMult(self.dR1MdP0M_mat_list[i])]
            self.temp_mat2_list_diag += [self.A0.transposeMatMult(self.dR0MdP1M_mat_list[i]) 
                               + self.A1.transposeMatMult(self.dR1MdP1M_mat_list[i])]
            # self.A1cP_vec_list += [A_x(self.A1c, self.PFE[i])]
            # self.A1cP_mat_list += [Lambda(self.A1cP_vec_list[i], num_pts, 
            #                               1, self.para_dim, 1)]
            # self.A2cP_vec_list += [A_x(self.A2c, self.PFE[i])]
            # self.A2cP_mat_list += [Lambda(self.A2cP_vec_list[i], num_pts, 
            #                               1, self.para_dim, 2)]
            self.der_mat_FE_diag += self.temp_mat1_list_diag[i].matMult(self.A1cP_mat_list[i])
            self.der_mat_FE_diag += self.temp_mat2_list_diag[i].matMult(self.A2cP_mat_list[i])


        #####################
        # print("Step 5 ", "*"*30)
        # For derivatives w.r.t. parametric coordinates
        # Off-diagonal
        self.xi_m = SpatialCoordinate(self.mortar_meshes[index])
        self.dR0Modxi_m = derivative(self.R_pen0Mo, self.xi_m)
        self.dR1Modxi_m = derivative(self.R_pen1Mo, self.xi_m)
        self.dR0Modxi_m_mat = m2p(assemble(self.dR0Modxi_m))
        self.dR1Modxi_m_mat = m2p(assemble(self.dR1Modxi_m))

        self.der_mat_FE_offdiag += self.A0o.transposeMatMult(self.dR0Modxi_m_mat)
        self.der_mat_FE_offdiag += self.A1o.transposeMatMult(self.dR1Modxi_m_mat)

        # Diagonal
        self.xi_m = SpatialCoordinate(self.mortar_meshes[index])
        self.dR0Mdxi_m = derivative(self.R_pen0M, self.xi_m)
        self.dR1Mdxi_m = derivative(self.R_pen1M, self.xi_m)
        self.dR0Mdxi_m_mat = m2p(assemble(self.dR0Mdxi_m))
        self.dR1Mdxi_m_mat = m2p(assemble(self.dR1Mdxi_m))

        self.der_mat_FE_diag += self.A0.transposeMatMult(self.dR0Mdxi_m_mat)
        self.der_mat_FE_diag += self.A1.transposeMatMult(self.dR1Mdxi_m_mat)

        #######################
        # print("Step 6 ", "*"*30)
        # off-diagonal
        self.der_mat_FE_offdiag.assemble()
        # Diagonal
        self.der_mat_FE_diag.assemble()

        #######################
        # print("Step 7 ", "*"*30)
        # Off-diagonal
        self.der_mat_IGA_rev_offdiag = self.Mo.transposeMatMult(self.der_mat_FE_offdiag)
        self.der_mat_IGA_rev_offdiag.assemble()
        # Diagonal
        self.der_mat_IGA_rev_diag = self.M.transposeMatMult(self.der_mat_FE_diag)
        self.der_mat_IGA_rev_diag.assemble()

        #######################
        # print("Step 8 ", "*"*30)
        # Off-diagonal
        self.switch_col_mat = zero_petsc_mat(num_pts*self.para_dim, num_pts*self.para_dim, 
                                             PREALLOC=num_pts*self.para_dim)
        for i in range(num_pts):
            self.switch_col_mat.setValue(i*self.para_dim, num_pts*self.para_dim-i*self.para_dim-2, 1.)
            self.switch_col_mat.setValue(i*self.para_dim+1, num_pts*self.para_dim-i*self.para_dim-1, 1.)
            # self.switch_col_mat.setValue(i*self.para_dim, num_pts*self.para_dim-i*self.para_dim-2, -1.)
            # self.switch_col_mat.setValue(i*self.para_dim+1, num_pts*self.para_dim-i*self.para_dim-1, -1.)
        self.switch_col_mat.assemble()
        self.der_mat_IGA_offdiag = self.der_mat_IGA_rev_offdiag.matMult(self.switch_col_mat)

        # Diagonal
        self.der_mat_IGA_diag = self.der_mat_IGA_rev_diag.matMult(self.switch_col_mat)

        if side == 0:
            dRIGAdxi_sub = [self.der_mat_IGA_diag, self.der_mat_IGA_offdiag]
        elif side == 1:
            dRIGAdxi_sub = [self.der_mat_IGA_offdiag, self.der_mat_IGA_diag]
        return dRIGAdxi_sub

    def create_files(self, save_path="./", folder_name="results/", 
                     thickness=False, refine_mesh=False, ref_nel=32):
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
                    self.F_files[i] += [File(self.comm, 
                                             self.F_file_names[i][j+1]),]
            if thickness:
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
            if thickness:
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
                if thickness:
                    self.Ah_ref_list += [create_transfer_matrix(
                                     self.h_th[s_ind].function_space(),
                                     self.V_ref_list[s_ind])]
                    self.h_th_func_ref_list[s_ind] = \
                        Function(self.V_ref_list[s_ind])
                

    def save_files(self, thickness=False):
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
            if thickness:
                if self.refine_mesh:
                    A_x_b(self.Ah_ref_list[i], v2p(self.h_th[i].vector()),
                          v2p(self.h_th_func_ref_list[i].vector()))
                    h_th_func = self.h_th_func_ref_list[i]
                else:
                    h_th_func = self.h_th[i]
                h_th_func.rename('t'+str(i), 't'+str(i))
                self.t_files[i] << h_th_func

if __name__ == '__main__':
    pass