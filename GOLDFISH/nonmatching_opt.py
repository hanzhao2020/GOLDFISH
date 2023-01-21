from PENGoLINS.nonmatching_coupling import *
from GOLDFISH.utils.opt_utils import *
from GOLDFISH.utils.bsp_utils import *

class NonMatchingOpt(NonMatchingCoupling):
    """
    Subclass of NonmatchingCoupling which serves as the base class
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
                         int_dx_metadata, contact, comm)
        self.opt_field = opt_field
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
        self.cp_iga_nest = self.vec_scalar_iga_nest
        # Set initial control points in IGA DoFs as None
        self.init_cp_iga = None

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
            dRm_dcpm_temp = dRmdcpm_sub(self.Rm_form_list[i],
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
                            penalty_coefficient=1000, 
                            penalty_method="minimum"):
        NonMatchingCoupling.mortar_meshes_setup(self, 
                            mapping_list, mortar_parametric_coords, 
                            penalty_coefficient, penalty_method)
        self.dRm_dcpm_list = [self.mortar_dRmdCPm_symexp(field) 
                              for field in self.opt_field]

    def set_residuals(self, residuals, residuals_deriv=None):
        NonMatchingCoupling.set_residuals(self, residuals, residuals_deriv)
        dR_dcp_ufl_symexp = [[] for field in self.opt_field]
        for i, field in enumerate(self.opt_field):
            for s_ind in range(self.num_splines):
                dR_dcp_ufl_symexp[i] += [derivative(residuals[s_ind], 
                                   self.splines[s_ind].cpFuncs[field]),]
        self.dR_dcp_symexp = [[Form(dRdcp) for dRdcp in dR_dcp_single_field]
                             for dR_dcp_single_field in dR_dcp_ufl_symexp]

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

    def extract_nonmatching_vec(self, vec_list, scalar=False, 
                                apply_bcs=False):
        """
        extract non-matching vector from FE to IGA DoFs
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

    def dRIGAdxi(self):
        """
        Reserved for shape optimization with moving intersections
        """
        return

    def dRIGAdh_th(self):
        """
        Reserved for thickness optimization
        """
        return 

    def create_files(self, save_path="./", folder_name="results/", 
                     thickness=False):
        """
        Create pvd files for all spline patches' displacements 
        and control points (and thickness is needed).
        """
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

    def save_files(self, thickness=False):
        """
        Save splines' displacements and control points to pvd files.
        """
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
                    self.splines[i].cpFuncs[j+1].rename(
                        'F'+str(i)+'_'+str(j+1), 'F'+str(i)+'_'+str(j+1))
                    self.F_files[i][j+1] << self.splines[i].cpFuncs[j+1]
            if thickness:
                self.h_ths[i].rename('t'+str(i), 't'+str(i))
                self.t_files[i] << self.h_ths[i]

if __name__ == '__main__':
    pass