from GOLDFISH.nonmatching_opt_ffd import *

class MaxvMStressExOperation(object):

    def __init__(self, nonmatching_opt, rho=1., alpha=None, m=None, 
                 surf="top", method="pnorm", linearize_stress=False):
        self.nonmatching_opt = nonmatching_opt
        self.num_splines = self.nonmatching_opt.num_splines
        self.splines = self.nonmatching_opt.splines
        self.opt_field = self.nonmatching_opt.opt_field
        self.opt_shape = self.nonmatching_opt.opt_shape
        self.opt_thickness = self.nonmatching_opt.opt_thickness
        self.surf = surf

        if self.surf == "top":
            self.xi2 = [h_th/2 for h_th in self.nonmatching_opt.h_th]
        elif self.surf == "bottom":
            self.xi2 = [-h_th/2 for h_th in self.nonmatching_opt.h_th]
        elif self.surf == "middle":
            self.xi2 = [Constant(0.) for i in range(self.nonmatching_opt.num_splines)]
        else:
            raise ValueError("Unknown surface type:", self.surf)

        self.rho = rho
        self.method = method
        self.linearize_stress = linearize_stress

        self.shellSVKstress = []
        self.vMstress = []
        for s_ind in range(self.num_splines):
            self.shellSVKstress += [ShellStressSVK(
                                    self.nonmatching_opt.splines[s_ind], 
                                    self.nonmatching_opt.spline_funcs[s_ind],
                                    self.nonmatching_opt.E[s_ind], 
                                    self.nonmatching_opt.nu[s_ind], 
                                    self.nonmatching_opt.h_th[s_ind], 
                                    linearize=self.linearize_stress)]
            self.vMstress += [self.shellSVKstress[s_ind].
                              vonMisesStress(self.xi2[s_ind])]

        if alpha is not None:
            self.alpha = alpha
        else:
            self.compute_alpha()
        if m is not None:
            self.given_m = True
            self.m = m
            self.m_list = [m for i in range(self.num_splines)]
        else:
            self.given_m = False
            self.compute_m()

        self.max_vM_symexps = []
        self.max_vM_forms = []
        for s_ind in range(self.num_splines):
            self.max_vM_symexps += [self.max_vM_symexp(
                                   self.vMstress[s_ind], s_ind)]
            self.max_vM_forms += [Form(self.max_vM_symexps[s_ind])]

        # Derivatives w.r.t. displacements
        self.dmax_vMdu_symexp = []
        self.dmax_vMdu_forms = []
        for s_ind in range(self.num_splines):
            self.dmax_vMdu_symexp += [derivative(self.max_vM_symexps[s_ind],
                                self.nonmatching_opt.spline_funcs[s_ind])]
            self.dmax_vMdu_forms += [Form(self.dmax_vMdu_symexp[s_ind])]

        # Derivatives w.r.t. control points
        if self.opt_shape:
            self.dmax_vMdcp_symexp = [[] for i in range(len(self.opt_field))]
            self.dmax_vMdcp_forms = [[] for i in range(len(self.opt_field))]
            for i, field in enumerate(self.opt_field):
                for s_ind in range(self.num_splines):
                    self.dmax_vMdcp_symexp[i] += [derivative(
                                    self.max_vM_symexps[s_ind], 
                                    self.splines[s_ind].cpFuncs[field])]
                    self.dmax_vMdcp_forms[i] += [Form(
                                    self.dmax_vMdcp_symexp[i][s_ind])]

        # Derivatives w.r.t. thickness (vM stress on middle surface is 
        # independent of thickness)
        if self.opt_thickness:
            self.dmax_vMdh_th_symexp = []
            self.dmax_vMdh_th_forms = []
            for s_ind in range(self.num_splines):
                self.dmax_vMdh_th_symexp += [derivative(
                            self.max_vM_symexps[s_ind],
                            self.nonmatching_opt.h_th[s_ind])]
                self.dmax_vMdh_th_forms += [Form(
                            self.dmax_vMdh_th_symexp[s_ind])]

    def compute_alpha(self):
        cell_vol_list = []
        for s_ind in range(self.num_splines):
            cell_vol = CellVolume(self.splines[s_ind].mesh)
            cell_vol_proj = self.splines[s_ind].\
                            projectScalarOntoLinears(cell_vol)
            cell_vol_list += [np.average(get_petsc_vec_array(
                              v2p(cell_vol_proj.vector())))]
        self.alpha = np.min(cell_vol_list)
        return self.alpha

    def compute_m(self):
        self.m_list = []
        for s_ind in range(self.num_splines):
            vm_proj = self.splines[s_ind].projectScalarOntoLinears(
                      self.vMstress[s_ind], lumpMass=False)
            self.m_list += [v2p(vm_proj.vector()).max()[1]]
        self.m = np.max(self.m_list)
        # print("Computed m:", self.m)
        # print("Computed m_list:", self.m_list)
        return self.m_list, self.m

    def compute_max_vM(self):
        self.max_vM_sub_proj = []
        for s_ind in range(self.num_splines):
            vm_proj = self.splines[s_ind].projectScalarOntoLinears(
                      self.vMstress[s_ind], lumpMass=False)
            self.max_vM_sub_proj += [v2p(vm_proj.vector()).max()[1]]
        self.max_global = np.max(self.max_vM_sub_proj)
        print("Projected max vM global:", self.max_global)
        # print("Projected max vM sub:", self.max_vM_sub_proj)
        return self.max_vM_sub_proj, self.max_global

    def KS_symexp(self, stress, ind):
        return exp(self.rho*(stress-self.m_list[ind]))*self.splines[ind].dx

    def pnorm_symexp(self, stress, ind):
        return ((stress/self.m_list[ind])**self.rho)*self.splines[ind].dx

    def max_vM_symexp(self, stress, ind):
        if self.method == 'KS':
            max_vM_symexp = self.KS_symexp(stress, ind)
        elif self.method == 'pnorm':
            max_vM_symexp = self.pnorm_symexp(stress, ind)
        else:
            raise ValueError("Unsupported max stress method "+self.method)
        return max_vM_symexp

    def continuous_KS_function(self, KS_form, ind):
        KS_val = assemble(KS_form)
        if KS_val == 0:
            KS_val = 1e-9
        elif KS_val == inf:
            KS_val = 1e14
        return self.m_list[ind] + 1/self.rho*np.log(1/self.alpha*KS_val)

    def continuous_pnorm_function(self, pnorm_form, ind):
        pnorm_val = assemble(pnorm_form)
        if pnorm_val == 0:
            pnorm_val = 1e-12
        elif pnorm_val == inf:
            pnorm_val = inf
        return self.m_list[ind]*(1/self.alpha*pnorm_val)**(1/self.rho)

    def continuous_max_vM_stress(self, max_vM_from, ind):
        if self.method == 'KS':
            max_vM_stress = self.continuous_KS_function(max_vM_from, ind)
        elif self.method == 'pnorm':
            max_vM_stress = self.continuous_pnorm_function(max_vM_from, ind)
        else:
            raise ValueError("Unsupported max stress method "+self.method)
        return max_vM_stress

    def discrete_KS_function(self, stress_list):
        max_dks = 0
        for i in range(len(stress_list)):
            max_dks += np.exp(self.rho*(stress_list[i]-self.m))
        max_dks = self.m + 1/self.rho*np.log(1/self.alpha*max_dks)
        return max_dks

    def discrete_pnorm_function(self, stress_list):
        max_dp = 0
        for i in range(len(stress_list)):
            max_dp += (stress_list[i]/self.m)**self.rho
        max_dp = self.m*(1/self.alpha*max_dp)**(1/self.rho)
        return max_dp

    def discrete_max_vM_stress(self, stress_list):
        if self.method == 'KS':
            max_vM_stress = self.discrete_KS_function(stress_list)
        elif self.method == 'pnorm':
            max_vM_stress = self.discrete_pnorm_function(stress_list)
        else:
            raise ValueError("Unsupported max stress method "+self.method)
        return max_vM_stress

    def max_vM_stress_global(self):
        if not self.given_m:
            self.compute_m()
        self.compute_max_vM()
        max_vM_sub = []
        for s_ind in range(self.num_splines):
            max_vM_sub += [self.continuous_max_vM_stress(
                           self.max_vM_forms[s_ind], s_ind)]
        max_vM_global = self.discrete_max_vM_stress(max_vM_sub)
        print("max_vM_global:", max_vM_global)
        # print("max_vM_sub:", max_vM_sub)
        print("relative error of max stress:", 
              np.abs(max_vM_global-self.max_global)/self.max_global)
        return max_vM_global

    def dglobal_KSdlocal_KS(self, stress_list, ind):
        temp_val0 = 0
        for s_ind in range(self.num_splines):
            temp_val0 += np.exp(self.rho*(stress_list[s_ind]-self.m))
        temp_val1 = np.exp(self.rho*(stress_list[ind]-self.m))/temp_val0
        return temp_val1

    def dlocal_KSdKS_form(self, max_vM_forms, ind):
        KS_val = assemble(max_vM_forms[ind])
        if KS_val == 0:
            KS_val = 1e-9
        elif KS_val == inf:
            KS_val = 1e14
        return 1./(self.rho*KS_val)

    def dglobal_pnormdlocal_prnom(self, stress_list, ind):
        temp_val0 = 0
        for s_ind in range(self.num_splines):
            temp_val0 += (stress_list[s_ind]/self.m)**self.rho
        temp_val1 = 1./self.alpha*(1/self.alpha*temp_val0)**(1/self.rho-1)\
                    *(stress_list[ind]/self.m)**(self.rho-1)
        return temp_val1

    def dlocal_pnormdpnorm_form(self, max_vM_forms, ind):
        pnorm_val = assemble(max_vM_forms[ind])
        if pnorm_val == 0:
            pnorm_val = 1e-12
        elif pnorm_val == inf:
            pnorm_val = inf
        temp_val0 = self.m_list[ind]*(1/self.alpha)**(1/self.rho)\
                  *(1/self.rho)*pnorm_val**(1/self.rho-1)
        return temp_val0

    def dmax_vMduIGA_global(self, array=True, apply_bcs=True):
        max_vM_sub = []
        for s_ind in range(self.num_splines):
            max_vM_sub += [self.continuous_max_vM_stress(
                           self.max_vM_forms[s_ind], s_ind)]

        dglobal_max_vMdu_list = []
        for s_ind in range(self.num_splines):
            if self.method == 'KS':
                a0 = self.dglobal_KSdlocal_KS(max_vM_sub, s_ind)
                a1 = self.dlocal_KSdKS_form(self.max_vM_forms, s_ind)
            elif self.method == 'pnorm':
                a0 = self.dglobal_pnormdlocal_prnom(max_vM_sub, s_ind)
                a1 = self.dlocal_pnormdpnorm_form(self.max_vM_forms, s_ind)
            dfdu = assemble(self.dmax_vMdu_forms[s_ind])
            sub_deriv = v2p(a0*a1*dfdu)
            dglobal_max_vMdu_list += [sub_deriv]

        dglobal_max_vMduIGA_nest = self.nonmatching_opt.\
                                   extract_nonmatching_vec(
                                   dglobal_max_vMdu_list, scalar=False,
                                   apply_bcs=apply_bcs)
        if array:
            return get_petsc_vec_array(dglobal_max_vMduIGA_nest,
                                       comm=self.nonmatching_opt.comm)
        else:
            return dglobal_max_vMduIGA_nest

    def dmax_vMdCPIGA_global(self, field, array=True):
        max_vM_sub = []
        for s_ind in range(self.num_splines):
            max_vM_sub += [self.continuous_max_vM_stress(
                           self.max_vM_forms[s_ind], s_ind)]
        field_ind = self.opt_field.index(field)
        dglobal_max_vMdcp_fe_list = []
        for s_ind in range(self.num_splines):
            if self.method == 'KS':
                a0 = self.dglobal_KSdlocal_KS(max_vM_sub, s_ind)
                a1 = self.dlocal_KSdKS_form(self.max_vM_forms, s_ind)
            elif self.method == 'pnorm':
                a0 = self.dglobal_pnormdlocal_prnom(max_vM_sub, s_ind)
                a1 = self.dlocal_pnormdpnorm_form(self.max_vM_forms, s_ind)
            dfdcp_fe = assemble(self.dmax_vMdcp_forms[field_ind][s_ind])
            sub_deriv = v2p(a0*a1*dfdcp_fe)
            dglobal_max_vMdcp_fe_list += [sub_deriv]

        dglobal_max_vMdcpIGA_nest = self.nonmatching_opt.\
                                   extract_nonmatching_vec(
                                   dglobal_max_vMdcp_fe_list, scalar=True)
        if array:
            return get_petsc_vec_array(dglobal_max_vMdcpIGA_nest,
                                       comm=self.nonmatching_opt.comm)
        else:
            return dglobal_max_vMdcpIGA_nest

    def dmax_vMdh_th_global(self, array=True):
        max_vM_sub = []
        for s_ind in range(self.num_splines):
            max_vM_sub += [self.continuous_max_vM_stress(
                           self.max_vM_forms[s_ind], s_ind)]

        dglobal_max_vMdh_th_list = []
        for s_ind in range(self.num_splines):
            if self.method == 'KS':
                a0 = self.dglobal_KSdlocal_KS(max_vM_sub, s_ind)
                a1 = self.dlocal_KSdKS_form(self.max_vM_forms, s_ind)
            elif self.method == 'pnorm':
                a0 = self.dglobal_pnormdlocal_prnom(max_vM_sub, s_ind)
                a1 = self.dlocal_pnormdpnorm_form(self.max_vM_forms, s_ind)
            dfdh_th = assemble(self.dmax_vMdh_th_forms[s_ind])
            sub_deriv = v2p(a0*a1*dfdh_th)
            dglobal_max_vMdh_th_list += [sub_deriv]

        if self.nonmatching_opt.var_thickness:
            dglobal_max_vMdh_th_nest = self.nonmatching_opt.\
                                       extract_nonmatching_vec(
                                       dglobal_max_vMdh_th_list, scalar=True)
        else:
            dglobal_max_vMdh_th_nest = create_nest_PETScVec(
                                   dglobal_max_vMdh_th_list,
                                   comm=self.nonmatching_opt.comm)
        if array:
            return get_petsc_vec_array(dglobal_max_vMdh_th_nest,
                                       comm=self.nonmatching_opt.comm)
        else:
            return dglobal_max_vMdh_th_nest


if __name__ == "__main__":
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

    def OCCBSpline2tIGArSpline(surface, num_field=3, quad_deg_const=3, 
                               spline_bc=None, index=0):
        """
        Generate ExtractedBSpline from OCC B-spline surface.
        """
        quad_deg = surface.UDegree()*quad_deg_const
        # DIR = SAVE_PATH+"spline_data/extraction_"+str(index)+"_init"
        # spline = ExtractedSpline(DIR, quad_deg)
        spline_mesh = NURBSControlMesh4OCC(surface, useRect=False)
        spline_generator = EqualOrderSpline(worldcomm, num_field, spline_mesh)
        if spline_bc is not None:
            spline_bc.set_bc(spline_generator)
        # spline_generator.writeExtraction(DIR)
        spline = ExtractedSpline(spline_generator, quad_deg)
        return spline

    filename_igs = "./geometry/init_Tbeam_geom.igs"
    igs_shapes = read_igs_file(filename_igs, as_compound=False)
    occ_surf_list = [topoface2surface(face, BSpline=True) 
                     for face in igs_shapes]
    occ_surf_data_list = [BSplineSurfaceData(surf) for surf in occ_surf_list]
    num_surfs = len(occ_surf_list)
    p = occ_surf_data_list[0].degree[0]

    # Define material and geometric parameters
    E = Constant(1.0e12)
    nu = Constant(0.)
    # h_th = Constant(0.1)
    penalty_coefficient = 1.0e3
    pressure = Constant(1.)

    fields0 = [None, [[0,1,2]],]
    spline_bc0 = SplineBC(directions=[1], sides=[None, [0]],
                         fields=fields0, n_layers=[None, [1]])
    spline_bcs = [spline_bc0,]*2

    # Geometry preprocessing and surface-surface intersections computation
    preprocessor = OCCPreprocessing(occ_surf_list, reparametrize=False, 
                                    refine=False)
    print("Computing intersections...")
    int_data_filename = "int_data.npz"
    if os.path.isfile(int_data_filename):
        preprocessor.load_intersections_data(int_data_filename)
    else:
        preprocessor.compute_intersections(mortar_refine=2)
        preprocessor.save_intersections_data(int_data_filename)

    if mpirank == 0:
        print("Total DoFs:", preprocessor.total_DoFs)
        print("Number of intersections:", preprocessor.num_intersections_all)

    # Display B-spline surfaces and intersections using 
    # PythonOCC build-in 3D viewer.
    # display, start_display, add_menu, add_function_to_menu = init_display()
    # preprocessor.display_surfaces(display, save_fig=False)
    # preprocessor.display_intersections(display, save_fig=False)

    if mpirank == 0:
        print("Creating splines...")
    # Create tIGAr extracted spline instances
    splines = []
    for i in range(num_surfs):
            spline = OCCBSpline2tIGArSpline(preprocessor.BSpline_surfs[i], 
                                            spline_bc=spline_bcs[i], index=i)
            splines += [spline,]

    h_th = []
    h_val = [0.01, 0.0547]
    # h_val = [0.1, 0.1]
    # expression = Expression("0.1*sin(pi*x[0])*sin(pi*x[1])+0.05", degree=2)
    for i in range(num_surfs):
        h_th += [Function(splines[i].V_linear)]
        h_th[i].interpolate(Constant(h_val[i]))
        # temp_vec = project(expression, splines[i].V_linear)
        # h_th[i].assign(temp_vec)

    # Create non-matching problem
    nonmatching_opt = NonMatchingOptFFD(splines, E, h_th, nu, opt_shape=True, 
                                        opt_thickness=True, comm=worldcomm)
    nonmatching_opt.create_mortar_meshes(preprocessor.mortar_nels)

    if mpirank == 0:
        print("Setting up mortar meshes...")
    nonmatching_opt.mortar_meshes_setup(preprocessor.mapping_list, 
                    preprocessor.intersections_para_coords, 
                    penalty_coefficient)
    pressure = -Constant(10.)
    f = as_vector([Constant(0.), Constant(0.), pressure])
    source_terms = []
    residuals = []
    for s_ind in range(nonmatching_opt.num_splines):
        z = nonmatching_opt.splines[s_ind].rationalize(
            nonmatching_opt.spline_test_funcs[s_ind])
        source_terms += [inner(f, z)*nonmatching_opt.splines[s_ind].dx]
        residuals += [SVK_residual(nonmatching_opt.splines[s_ind], 
                      nonmatching_opt.spline_funcs[s_ind], 
                      nonmatching_opt.spline_test_funcs[s_ind], 
                      E, nu, h_th[s_ind], source_terms[s_ind])]
    nonmatching_opt.set_residuals(residuals)

    if mpirank == 0:
        print("Solving linear non-matching problem...")
    nonmatching_opt.solve_nonlinear_nonmatching_problem(solver="direct")

    # Compute von Mises stress
    print("Computing von Mises stresses...")
    von_Mises_tops = []
    von_Mises_tops_proj = []
    # von_Mises_bots = []
    for i in range(nonmatching_opt.num_splines):
        spline_stress = ShellStressSVK(nonmatching_opt.splines[i], 
                                       nonmatching_opt.spline_funcs[i],
                                       E, nu, h_th, linearize=False)
        # von Mises stresses on top surfaces
        von_Mises_top = spline_stress.vonMisesStress(0)#h_val[i]/2)
        von_Mises_top_proj = nonmatching_opt.splines[i].\
                             projectScalarOntoLinears(
                             von_Mises_top, lumpMass=False)
        v2p(von_Mises_top_proj.vector()).ghostUpdate()
        von_Mises_tops += [von_Mises_top]
        von_Mises_tops_proj += [von_Mises_top_proj]

    SAVE_PATH = "/home/han/Documents/test_results/"
    for i in range(nonmatching_opt.num_splines):
        save_results(nonmatching_opt.splines[i], nonmatching_opt.spline_funcs[i], i, 
                    save_cpfuncs=True, save_path=SAVE_PATH, comm=nonmatching_opt.comm)
        von_Mises_tops_proj[i].rename("von_Mises_top_"+str(i), 
                                 "von_Mises_top_"+str(i))
        File(SAVE_PATH+"results/von_Mises_top_"+str(i)+".pvd") \
            << von_Mises_tops_proj[i]
        # von_Mises_bots[i].rename("von_Mises_bot_"+str(i), 
        #                          "von_Mises_bot_"+str(i))
        # File(SAVE_PATH+"results/von_Mises_bot_"+str(i)+".pvd") \
        #     << von_Mises_bots[i]

    # xi2 = 0.05
    # rho = 1e-2
    # # alpha = 0.00367
    # # m = 30000
    # max_vm = MaxvMStressExOperation(
    #          nonmatching_opt, xi2, rho=rho, alpha=None, 
    #          m=None, method="KS", linearize_stress=False)
    
    # val0 = max_vm.max_vM_stress_global()
    # max_val = np.max(von_Mises_tops_proj[1].vector().get_local())

    # deriv = max_vm.dmax_vMduIGA_global()
    # deriv1 = max_vm.dmax_vMdCPIGA_global(1)