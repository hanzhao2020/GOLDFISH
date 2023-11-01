# # FEMO related dependencies
# from femo.fea.fea_dolfinx import FEA
# from femo.csdl_opt.fea_model import FEAModel
# from femo.csdl_opt.state_model import StateModel
# from femo.csdl_opt.output_model import OutputModel, OutputFieldModel
# # RM shell related properties
# from shell_analysis_fenicsx.read_properties import readCLT, sortIndex

import psutil

# GOLDFISH related models
from GOLDFISH.nonmatching_opt_om import *
from GOLDFISH.nonmatching_opt import NonMatchingOpt
from GOLDFISH.csdl_models.hth_map_model import HthMapModel
from GOLDFISH.csdl_models.disp_states_model import DispStatesModel
from GOLDFISH.csdl_models.volume_model import VolumeModel
from GOLDFISH.csdl_models.max_vmstress_model import MaxvMStressModel 
# from GOLDFISH.csdl_models.tip_disp_model import TipDispModel
# # TODO: write MaxVmstressModel, update DispStatesModel

from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
import scipy.sparse as sp
import csdl
import numpy as np

SAVE_PATH = '/home/han/Documents/test_results/'

def memory_usage_psutil():
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0]/float(1024**2)
    return mem


# print("Inspection: Memory usage: {:8.2f} MB.\n"\
#           .format(memory_usage_psutil()))

class KLShellModule(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('nonmatching_opt', default=None)
        self.parameters.declare('klshell_pde', default=None)
        self.parameters.declare('shells', default={})
    
    def init_parameters(self):
    # def define(self):
        self.nonmatching_opt = self.parameters['nonmatching_opt']
        self.klshell_pde = self.parameters['klshell_pde']
        self.shell_properties = self.parameters['shells']
        self.shell_name = list(self.shell_properties.keys())[0]
        self.E = self.shell_properties[self.shell_name]['E']
        self.nu = self.shell_properties[self.shell_name]['nu']
        self.rho = self.shell_properties[self.shell_name]['rho']
        # self.g = self.shell_properties[self.shell_name]['g']
        # self.record = self.shell_properties[self.shell_name]['record']

        # Define variable names
        self.h_th_name_design = self.shell_name+'_hth_design'
        self.h_th_name_full = self.shell_name+'_hth_full'
        self.volume_name = 'volume'
        self.max_vM_name = 'max_vM_stress'

        self.disp_solid_name = 'disp_solid'
        self.disp_exract_name = self.shell_name+'_displacement'

        self.fsolid_name = 'force_solid'
        self.force_name = self.shell_name+'_forces'

        # Define model names
        self.h_th_map_model_name = 'h_th_map_model'
        self.disp_states_model_name = 'disp_states_model'
        self.volume_model_name = 'volume_model'
        self.max_vM_model_name = 'max_vM_model'
        self.force_reshaping_model_name = 'force_reshaping_model'
        self.disp_extraction_model_name = 'disp_extraction_model'

        # Create force functions and set residuals
        self.spline_forces = []
        self.source_terms = []
        self.residuals = []
        for s_ind in range(self.nonmatching_opt.num_splines):
            a0,a1,a2,_,a,_ = surfaceGeometry(
                             self.nonmatching_opt.splines[s_ind], 
                             self.nonmatching_opt.splines[s_ind].F\
                             +self.nonmatching_opt.spline_funcs[s_ind])
            A0,A1,A2,_,A,_ = surfaceGeometry(
                             self.nonmatching_opt.splines[s_ind], 
                             self.nonmatching_opt.splines[s_ind].F)
            if s_ind in self.klshell_pde.lin_spline_inds:
                lin_spline = self.klshell_pde.lin_splines.surfaces[
                                       self.klshell_pde.lin_spline_inds.
                                       index(s_ind)]
                self.spline_forces += [Function(lin_spline.V)]
                self.source_terms += [Constant(1e-1)*lin_spline.rationalize(
                    self.spline_forces[s_ind])*sqrt(det(a)/det(A))
                    *inner(a2, self.nonmatching_opt.spline_test_funcs[s_ind])*
                    self.nonmatching_opt.splines[s_ind].dx]
                # self.source_terms += [
                #     *inner(self.spline_forces[s_ind], 
                #     self.nonmatching_opt.spline_test_funcs[s_ind])*
                #     self.nonmatching_opt.splines[s_ind].dx]
            else:
                # self.spline_forces += [Constant((0.))]
                # self.source_terms += [
                #     self.spline_forces[s_ind]*sqrt(det(a)/det(A))
                #     *inner(a2, self.nonmatching_opt.spline_test_funcs[s_ind])*
                #     self.nonmatching_opt.splines[s_ind].dx]    
                self.spline_forces += [Constant((0.,0.,0.))]        
                self.source_terms += [
                    inner(self.spline_forces[s_ind], 
                    self.nonmatching_opt.spline_test_funcs[s_ind])*
                    self.nonmatching_opt.splines[s_ind].dx]
            self.residuals += [SVK_residual(
                self.nonmatching_opt.splines[s_ind], 
                self.nonmatching_opt.spline_funcs[s_ind], 
                self.nonmatching_opt.spline_test_funcs[s_ind], 
                self.nonmatching_opt.E[s_ind], 
                self.nonmatching_opt.nu[s_ind], 
                self.nonmatching_opt.h_th[s_ind], 
                self.source_terms[s_ind])]
        self.nonmatching_opt.set_residuals(self.residuals)

        self.nonmatching_opt.set_aero_linear_splines(
            self.klshell_pde.lin_splines.surfaces,
            self.spline_forces)


    def define(self):
        # Force reshaping model
        self.force_reshaping_model = ForceReshapingModel(
                                     nonmatching_opt=self.nonmatching_opt,
                                     klshell_pde=self.klshell_pde,
                                     input_force_name=self.force_name,
                                     output_force_name=self.fsolid_name)
        # self.force_reshaping_model.init_parameters()
        self.add(self.force_reshaping_model, self.force_reshaping_model_name)

        # Displacement extraction model
        self.disp_extraction_model = DisplacementExtractionModel(
                                     nonmatching_opt=self.nonmatching_opt,
                                     klshell_pde=self.klshell_pde,
                                     input_disp_name=self.disp_solid_name,
                                     output_disp_name=self.disp_exract_name)
        # self.disp_extraction_model.init_parameters()
        self.add(self.disp_extraction_model, self.disp_extraction_model_name)

        # Patch to CG1 thickness mapping model
        self.h_th_map_model = HthMapModel(
                              nonmatching_opt=self.nonmatching_opt,
                              input_h_th_name_design=self.h_th_name_design,
                              output_h_th_name_full=self.h_th_name_full)
        print("Initializing model: ", self.h_th_map_model_name)
        self.h_th_map_model.init_parameters()
        # self.add(self.h_th_map_model_name, self.h_th_map_model)
        self.add(self.h_th_map_model, name=self.h_th_map_model_name)

        # Displacement model
        self.disp_states_model = DispStatesModel(
                                 nonmatching_opt=self.nonmatching_opt,
                                 input_h_th_name=self.h_th_name_full,
                                 input_Paero_name=self.fsolid_name,
                                 output_u_name=self.disp_solid_name)
        print("Initializing model: ", self.disp_states_model_name)
        self.disp_states_model.init_parameters(save_files=True)
        self.add(self.disp_states_model, self.disp_states_model_name)

        # Volume model
        self.volume_model = VolumeModel(
                            nonmatching_opt=self.nonmatching_opt,
                            input_h_th_name=self.h_th_name_full,
                            output_vol_name=self.volume_name)
        print("Initializing model: ", self.volume_model_name)
        self.volume_model.init_parameters()
        self.add(self.volume_model, name=self.volume_model_name)

        # Max vM stress model
        vm_surf = 'top'
        rho = 1e2
        upper_vM = 324E6/1.5
        self.max_vM_model = MaxvMStressModel(
                            nonmatching_opt=self.nonmatching_opt,
                            rho=rho, alpha=None, 
                            m=upper_vM, surf=vm_surf,
                            input_u_name=self.disp_solid_name,
                            input_h_th_name=self.h_th_name_full,
                            output_max_vM_name=self.max_vM_name)
        print("Initializing model: ", self.max_vM_model_name)
        self.max_vM_model.init_parameters()
        self.add(self.max_vM_model, self.max_vM_model_name)

        # self.connect(self.h_th_name_design,
        #              self.h_th_map_model_name+'.'+self.h_th_name_full)
        # self.connect(self.h_th_map_model_name+'.'+self.h_th_name_full,
        #              self.disp_states_model_name+'.'+self.h_th_name_full)
        # self.connect(self.force_reshaping_model_name+'.'+self.fsolid_name,
        #              self.disp_states_model_name+'.'+self.fsolid_name)
        # self.connect(self.disp_states_model_name+'.'+self.disp_solid_name,
        #              self.max_vM_model_name+'.'+self.disp_solid_name)
        # self.connect(self.h_th_map_model_name+'.'+self.h_th_name_full,
        #              self.max_vM_model_name+'.'+self.h_th_name_full)
        # self.connect(self.h_th_map_model_name+'.'+self.h_th_name_full,
        #              self.volume_model_name+'.'+self.h_th_name_full)


class ForceReshapingModel(csdl.Model):
    def initialize(self):
        self.parameters.declare('nonmatching_opt')
        self.parameters.declare('klshell_pde')
        self.parameters.declare('input_force_name', default='input_solid_forces')
        self.parameters.declare('output_force_name', default='output_solid_force')

    # def init_parameters(self):
    #     self.nonmatching_opt = self.parameters['nonmatching_opt']
    #     self.klshell_pde = self.parameters['klshell_pde']
    #     self.input_force_name = self.parameters['input_force_name']
    #     self.output_force_name = self.parameters['output_force_name']

    def define(self):
        self.nonmatching_opt = self.parameters['nonmatching_opt']
        self.klshell_pde = self.parameters['klshell_pde']
        self.input_force_name = self.parameters['input_force_name']
        self.output_force_name = self.parameters['output_force_name']

        # phy_dim = 3
        # cg1_size_list = []
        # for s_ind in range(self.nonmatching_opt.num_splines):
        #     cg1_size_list += [self.nonmatching_opt.splines[s_ind]
        #                        .V_linear.dim()]
        # force_shape = (np.sum(cg1_size_list), phy_dim)
        # force_size = force_shape[0]*force_shape[1]

        force_shape = self.klshell_pde.solid_mesh_phys.shape
        force_size = force_shape[0]*force_shape[1]

        # print("Test 1 ................")
        # print("force_size:", force_size)
        # print("force_shape:", force_shape)

        vector = np.arange(force_size)
        tensor = vector.reshape(force_shape)
        
        nodal_force_mat = self.declare_variable(self.input_force_name,
                           val=tensor)
        # self.register_output(self.output_force_name,
        #      csdl.reshape(nodal_force_mat, new_shape=(force_size,)))
        self.register_output(self.output_force_name,
            csdl.pnorm(nodal_force_mat, 2, axis=1))
        

class DisplacementExtractionModel(csdl.Model):
    """
    Input: displacement in IGA DoFs
    Output: nodal displacement for linear splines
    """
    def initialize(self):
        self.parameters.declare('nonmatching_opt')
        self.parameters.declare('klshell_pde')
        self.parameters.declare('input_disp_name', default='input_solid_disp')
        self.parameters.declare('output_disp_name', default='output_solid_disp')

    # def init_parameters(self):
    #     self.nonmatching_opt = self.parameters['nonmatching_opt']
    #     self.klshell_pde = self.parameters['klshell_pde']
    #     self.input_disp_name = self.parameters['input_disp_name']
    #     self.output_disp_name = self.parameters['output_disp_name']

    def define(self):
        self.nonmatching_opt = self.parameters['nonmatching_opt']
        self.klshell_pde = self.parameters['klshell_pde']
        self.input_disp_name = self.parameters['input_disp_name']
        self.output_disp_name = self.parameters['output_disp_name']
        self.phy_dim = 3

        solid_mesh_shape = self.klshell_pde.solid_mesh_phys.shape

        # cp_size_list = []
        # for s_ind in range(self.nonmatching_opt.num_splines):
        #     cp_size_list += [self.nonmatching_opt.splines[s_ind]
        #                      .M.size(1)]

        # lin_disp_mat_list = [[csr_matrix(np.zeros((cp_size_list[j], cp_size_list[i]))) 
        #                       for i in range(self.nonmatching_opt.num_splines)]
        #                       for j in self.klshell_pde.lin_spline_inds]
        # for i, s_ind in enumerate(self.klshell_pde.lin_spline_inds):
        #     lin_disp_mat_list[i][s_ind] = csr_matrix(np.eye(cp_size_list[s_ind]))

        print("Inspection disp extraction 0: Memory usage: {:8.2f} MB.\n"\
              .format(memory_usage_psutil()))

        disp_ext_mat = self.klshell_pde.construct_nodal_disp_map()

        print("Inspection disp extraction 1: Memory usage: {:8.2f} MB.\n"\
              .format(memory_usage_psutil()))


        input_size = np.sum(self.klshell_pde.cp_size_list)

        vector = np.arange(input_size)
        disp_vec = self.declare_variable(self.input_disp_name, val=vector)
        nodal_disp_vec = csdl.matvec(disp_ext_mat, disp_vec)

        # print("Test 2 .......")
        # print("mesh shape:", solid_mesh_shape)
        # print("disp_ext_mat shape:", disp_ext_mat.shape)


        nodal_disp_mat = csdl.reshape(nodal_disp_vec,
                         new_shape=(solid_mesh_shape[1], solid_mesh_shape[0]))
        self.register_output(self.output_disp_name,
                             csdl.transpose(nodal_disp_mat))


class KLShellPDE(object):
    def __init__(self, nonmatching_opt, lin_splines, lin_spline_inds):
        self.nonmatching_opt = nonmatching_opt
        self.num_splines = self.nonmatching_opt.num_splines
        self.splines = self.nonmatching_opt.splines
        self.lin_spline_inds = lin_spline_inds
        self.phy_dim = self.splines[0].nFields

        # self.bf_sup_sizes_list = []
        # for s_ind in range(self.num_splines):
        #     v_temp = TestFunction(self.nonmatching_opt.splines[s_ind].V)
        #     self.bf_sup_sizes_list += [assemble(
        #         self.splines[s_ind].rationalize(v_temp)
        #         *self.splines[s_ind].dx).get_local()]
        # self.bf_sup_sizes = np.concatenate(self.bf_sup_sizes_list)

        self.lin_splines = lin_splines 
        self.surf_sol_obj = self.splines 
        # self.solid_mesh_phys, self.solid_mesh_par = \

        print("Inspection klshell pde 0: Memory usage: {:8.2f} MB.\n"\
                .format(memory_usage_psutil()))

        self.compute_solid_mesh_phys_coords()
        self.compute_basisfunc_support_size()

        print("Inspection klshell pde 1: Memory usage: {:8.2f} MB.\n"\
                .format(memory_usage_psutil()))

    def compute_basisfunc_support_size(self):
        # # vec = as_vector([Constant(1.0), Constant(0.0)])#, Constant(0.0)])
        # vec = Constant((1.)*)
        # self.bf_sup_sizes = \
        #     [AT_x(spline.M, 
        #         assemble(inner(spline.rationalize(TestFunction(spline.V)), vec)*spline.dx))\
        #         .getArray()[:int(spline.M.size(1)/spline.nFields)] 
        #             for spline in self.lin_splines.surfaces]
        # self.bf_sup_sizes = np.concatenate(self.bf_sup_sizes)

        self.bf_sup_sizes = []
        for spline in self.lin_splines.surfaces:
            # vec = Constant((1.,)*spline.nFields)
            vec = Constant(1.)
            sup_FE = assemble(inner(spline.rationalize(
                     TestFunction(spline.V)), vec)*spline.dx)
            # sup_FE_array = assemble(sup_FE).getArray()
            self.bf_sup_sizes += [AT_x(spline.M, sup_FE).getArray()]
                                  #[:int(spline.M.size(1)/spline.nFields)]]
        self.bf_sup_sizes = np.concatenate(self.bf_sup_sizes)

    def compute_solid_mesh_phys_coords(self):
        self.solid_mesh_par = []
        for spline_surf in self.lin_splines.surfaces:
            self.solid_mesh_par += [spline_surf.mesh.coordinates()]
        solid_mesh_phys_coords_list = self.lin_splines.compute_phys_coords(
                                 par_coords_list=self.solid_mesh_par)
        self.solid_mesh_phys = np.concatenate(solid_mesh_phys_coords_list, axis=0)
        return self.solid_mesh_phys, self.solid_mesh_par

    def compute_alpha(self):
        self.alpha_list = []
        for s_ind in range(self.num_splines):
            alpha_vec = self.nonmatching_opt.ha_phy_linear[s_ind].\
                        vector().get_local()
            self.alpha_list = [np.average(alpha_vec),]
        return self.alpha_list

    def construct_nodal_disp_map(self):
        self.phy_dim = 3
        solid_mesh_shape = self.solid_mesh_phys.shape

        self.cp_size_list = []
        for s_ind in range(self.nonmatching_opt.num_splines):
            self.cp_size_list += [self.nonmatching_opt.splines[s_ind].M.size(1)]

        lin_disp_mat_list = [[csr_matrix(np.zeros((self.cp_size_list[j], self.cp_size_list[i]))) 
                              for i in range(self.nonmatching_opt.num_splines)]
                              for j in self.lin_spline_inds]
        for i, s_ind in enumerate(self.lin_spline_inds):
            lin_disp_mat_list[i][s_ind] = csr_matrix(np.eye(self.cp_size_list[s_ind]))

        self.lin_disp_mat = bmat(lin_disp_mat_list).tocsr()
        self.construct_disp_interp_map()
        self.disp_ext_mat = csr_matrix(sp.vstack([self.Q_map]*self.phy_dim)*self.lin_disp_mat)

        return self.disp_ext_mat

    def construct_disp_interp_map(self):
        # Q_map shape: par_coord[0]*iga_dof
        surf_sol_obj_temp = [self.surf_sol_obj[i] for i in self.lin_spline_inds]
        self.Q_map = sp.block_diag(
                     self.compute_IGA_bases_in_points(
                     surf_sol_obj_temp, 
                     self.solid_mesh_par, 
                     num_spline_fields=3)).tocsr()
        self.Q_map.eliminate_zeros()
        # self.Q_map = self.Q_map[:, 
        #              :int(self.Q_map.shape[1]/self.phy_dim)]
        return self.Q_map

    def compute_IGA_bases_in_points(self, surf_list, par_coords_list,
                                    num_spline_fields=1):
        # num_spline_fields = surf_list[0].nFields
        samples_FE = self.compute_FE_bases_in_points(
                     surf_list, par_coords_list, 
                     num_spline_fields=num_spline_fields)
        samples_IGA = []
        for i in range(len(surf_list)):
            M_petsc = m2p(surf_list[i].M).getValuesCSR()
            M = sp.csr_matrix((M_petsc[2], M_petsc[1], M_petsc[0]))
            samples_IGA += [samples_FE[i]@M]
        return samples_IGA

    def compute_FE_bases_in_points(self, surf_list, par_coords_list,
                                   num_spline_fields=1):
        # num_spline_fields = surf_list[0].nFields
        sample_list = []
        for l in range(len(surf_list)):
            coord_vec = par_coords_list[l]
            V = surf_list[l].V
            mesh_cur = surf_list[l].mesh
            sample_mat = np.zeros((coord_vec.shape[0], V.dim()))
            for i in range(coord_vec.shape[0]):
                x_point = np.array(coord_vec[i, :])
                x_point_eval = eval_fe_basis_all(x_point, V, mesh_cur, 
                               num_fields=num_spline_fields)
                sample_mat[i, :] = x_point_eval       
            sample_list += [sample_mat]
        return sample_list

    def compute_sparse_mass_matrix(self, normal_vec_list=None):
        self.normal_vec_list = normal_vec_list
        mat_assemble = []
        mat_assemble_fe = []
        mat_assemble_list = []
        vec_assemble_list = []
        vec_reg_assemble_list = []
        spline_funcs = [TrialFunction(spline.V) for spline in self.lin_splines.surfaces]
        spline_test_funcs = [TestFunction(spline.V) 
                                    for spline in self.lin_splines.surfaces]
        for i in range(len(spline_funcs)):
            spline_surf = self.lin_splines.surfaces[i]
            # if self.normal_vec_list is None:
            mat = assemble(inner(spline_surf.rationalize(spline_funcs[i]), 
                  spline_surf.rationalize(spline_test_funcs[i]))*spline_surf.dx)
            iga_mat = AT_R_B(m2p(spline_surf.M), m2p(mat), m2p(spline_surf.M))
            # convert iga_mat to sparse python matrix
            iga_mat_csr = iga_mat.getValuesCSR()
            mat_assemble += [sp.csr_array((iga_mat_csr[2], iga_mat_csr[1], 
                                                 iga_mat_csr[0]))]
            
            mat_fe_csr = m2p(mat).getValuesCSR()
            mat_assemble_fe += [sp.csr_array((mat_fe_csr[2], mat_fe_csr[1], 
                                                    mat_fe_csr[0]))]
            if self.normal_vec_list is not None:
                # create list of component matrices per surface
                mat_list = []
                vec_list = []
                vec_list_regular = []
                for j in range(self.normal_vec_list[i].ufl_shape[0]):
                    mat = assemble(inner(self.normal_vec_list[i][j]\
                          *spline_surf.rationalize(spline_funcs[i]), 
                          spline_surf.rationalize(spline_test_funcs[i]))*spline_surf.dx)
                    iga_mat = AT_R_B(m2p(spline_surf.M), m2p(mat), m2p(spline_surf.M))
                    # convert iga_mat to sparse python matrix
                    iga_mat_csr = iga_mat.getValuesCSR()
                    mat_list += [sp.csr_array((iga_mat_csr[2], iga_mat_csr[1], 
                                 iga_mat_csr[0]))]

                    vec = assemble(self.normal_vec_list[i][j]\
                          *spline_surf.rationalize(spline_test_funcs[i])*spline_surf.dx)
                    vec_regular = assemble(spline_surf.rationalize(
                                    spline_test_funcs[i])*spline_surf.dx)
                    iga_vec = AT_x(m2p(spline_surf.M), v2p(vec))
                    iga_vec_regular = AT_x(m2p(spline_surf.M), v2p(vec_regular))
                    vec_list += [iga_vec.getArray()]
                    vec_list_regular += [iga_vec_regular.getArray()]

                # append list of component matrices per surface to list of matrices of all surfaces
                mat_assemble_list += [mat_list]
                vec_assemble_list += [vec_list]
                vec_reg_assemble_list += [vec_list_regular]
        
        # concatenate matrices
        self.mass_mat_list = mat_assemble
        self.mass_mat = sp.block_diag(mat_assemble).tocsr()
        self.mass_mat.eliminate_zeros()
        self.mass_mat_fe_list = mat_assemble_fe
        self.mass_mat_fe = sp.block_diag(mat_assemble_fe).tocsr()
        self.mass_mat_fe.eliminate_zeros()
        if self.normal_vec_list is not None:
            # create concatenated matrices per (x,y,z)-component
            self.mass_mat_list = []
            self.mass_vec_list = []
            self.mass_vec_reg_list = []
            for j in range(self.normal_vec_list[0].ufl_shape[0]):
                elements_j_of_surf_lists = [mat_assemble_el[j] for mat_assemble_el in mat_assemble_list]
                mass_mat_el = sp.block_diag(elements_j_of_surf_lists).tocsr()
                mass_mat_el.eliminate_zeros()
                self.mass_mat_list += [mass_mat_el]

                elements_j_of_vec_lists = [vec_assemble_el[j] for vec_assemble_el in vec_assemble_list]
                self.mass_vec_list += [np.concatenate(elements_j_of_vec_lists)]
                elements_j_of_vec_reg_lists = [vec_assemble_el[j] for vec_assemble_el in vec_reg_assemble_list]
                self.mass_vec_reg_list += [np.concatenate(elements_j_of_vec_reg_lists)]

        return self.mass_mat

    # def compute_G_mat(self, oml):
    #     self.G_mat = NodalMap(self.solid_mesh_phys, 
    #                  reshape_3D_array_to_2D(oml), 
    #                  RBF_width_par=1., RBF_func=RadialBasisFunctions.Gaussian, 
    #                  column_scaling_vec=self.bf_sup_sizes)

def eval_fe_basis_all(x, V, mesh_cur, num_fields=1, x_alt=None, V_alt=None, mesh_cur_alt=None):
    bbt = BoundingBoxTree()
    bbt.build(mesh_cur)
    cell_id = bbt.compute_first_entity_collision(Point(*x))
    cell = Cell(mesh_cur, cell_id)
    dof_coords = cell.get_vertex_coordinates()
    basis_idxs = V.dofmap().cell_dofs(cell_id)
    basis_evals = V.element().evaluate_basis_all(x, dof_coords, cell_id)
    # print(basis_evals)

    basis_vec = np.zeros((V.dim(),))
    basis_vec[basis_idxs] = basis_evals[::num_fields]

    if x_alt is not None and V_alt is not None and mesh_cur_alt is not None:
        bbt_alt = BoundingBoxTree()
        bbt_alt.build(mesh_cur_alt)
        cell_id_alt = bbt_alt.compute_first_entity_collision(Point(*x_alt))
        cell_alt = Cell(mesh_cur_alt, cell_id_alt)
        dof_coords_alt = cell_alt.get_vertex_coordinates()
        basis_idxs_alt = V_alt.dofmap().cell_dofs(cell_id_alt)
        basis_evals_alt = V_alt.element().evaluate_basis_all(x_alt, dof_coords_alt, cell_id_alt)
        # TODO: Figure out why using `x` instead of `x_alt` gives odd results
        # NOTE: Adding 1e-16 to each coordinate of x=[0,0] gives correct results
        # NOTE:  V_alt.element().evaluate_basis(3, x, cell_alt.get_vertex_coordinates(), cell_alt.orientation()) also gives odd results
        # print(basis_evals)

        basis_vec_alt = np.zeros((V_alt.dim(),))
        basis_vec_alt[basis_idxs_alt] = basis_evals_alt
    return basis_vec


def reshape_3D_array_to_2D(mesh_file):
    # convert mesh to an array of (n_y*n_x, 3)
    return np.reshape(mesh_file, (mesh_file.shape[0]*mesh_file.shape[1], 3), order='F')




class Forces_on_surfaces:
    def __init__(self):#, outer_surface_vec):
        # initialize this class just to initialize `self`
        pass

class PENGoLINS_dist_loads(Forces_on_surfaces):
    def __init__(self, surf_list, displacements=None, 
                 interpolation_type="trivial", num_interp_pts=None):
        super(PENGoLINS_dist_loads, self).__init__()
        # check whether inputs of correct type are present for the given interpolation type
        if interpolation_type=="least-squares": 
            if num_interp_pts is None:
                raise ValueError("Number of interpolation points must be specified when using least-squares interpolation")
            elif type(num_interp_pts) is tuple:
                self.num_interp_pts = [num_interp_pts]*len(surf_list)
            elif type(num_interp_pts) is list:
                if not len(num_interp_pts)==len(surf_list):
                    raise ValueError("Number of interpolation points must be specified either as a tuple (n_1, n_2) \
                                      or as list of tuples with length equal to the number of surfaces")
                self.num_interp_pts = num_interp_pts

        self.surfaces = surf_list
        self.interpolation_type = interpolation_type
        # self.splines, self.spline_testfuncs = self.generate_funcs_on_surfaces()
        # # TODO: Allow user to specify the number of interpolation points in both directions below, rather than fixing it to the number of DoFs
        # self.interp_par_coords = self.determine_parametric_surf_coords()
        # self.interp_phys_coords = self.compute_phys_coords(displacements=displacements)

        # # construct the interpolation matrix for each shell surface
        # self.interpolation_matrices = compute_IGA_bases_in_points(self.surfaces, self.interp_par_coords, num_spline_fields=self.surfaces[0].nFields)
        # self.DoFs_per_surf = [matrix.shape[1] for matrix in self.interpolation_matrices]

    def compute_phys_coords(self, par_coords_list=None, displacements=None):
        if par_coords_list is None:
            par_coords_list = self.interp_par_coords
        dof_phys_coords = []
        for i, dof_coord_arr in enumerate(par_coords_list):
            if displacements is not None:
                disp_inp = displacements[i]
            else:
                disp_inp = None
            phys_coord_arr = np.zeros((dof_coord_arr.shape[0], 3))
            for j in range(dof_coord_arr.shape[0]):
                phys_coord_arr[j, :] = compute_phys_coord_tuple(self.surfaces[i], dof_coord_arr[j, :], displacements=disp_inp)
                # phys_coord_arr[j, :] = undeformed_position(self.surfaces[i], dof_coord_arr[j, :])
            dof_phys_coords += [phys_coord_arr]

        return dof_phys_coords


def compute_phys_coord_tuple(surf, xi, displacements=None, disp_plot_factor=1.):#, occ_surf=None):
    if displacements is not None:
        disp_surf = displacements[0]
        disp_vec = displacements[1]
    
    coords = []
    w = eval_func(surf.mesh, surf.cpFuncs[3], xi)
    for i in range(3):
        phys_coord_i = surf.F[i](xi)
        if displacements is not None:
            # An additional displacement is present, so needs to be added to the parametric->physical domain mapping.
            # This is used to impose displacements on the VLM mesh
            phys_coord_i += disp_plot_factor*eval_func(disp_surf.mesh, disp_vec[i], xi)

        coords += [phys_coord_i/w]
    # if occ_surf is not None:
    #     p_temp = gp_Pnt()
    #     occ_surf.D0(xi[0], xi[1], p_temp)
    #     coords_alt = np.multiply(1, p_temp.Coord())
        # print("Diff OCC and tIGAr physical coordinates: {}".format(np.linalg.norm(np.subtract(coords, coords_alt))))

    return coords



# class KLShellNodalMap(object):
#     def __init__(self):
#         pass

#     def temp(self):
#         pass


class RadialBasisFunctions:
    def Gaussian(x_dist, eps=1.):
        return np.exp(-(eps*x_dist)**2)

    def BumpFunction(x_dist, eps=1.):
        # filter x_dist to get rid of x_dist >= 1/eps, this prevents overflow warnings
        x_dist_filt = np.where(x_dist < 1/eps, x_dist, 0.)
        f_mat = np.where(x_dist < 1/eps, np.exp(-1./(1.-(eps*x_dist_filt)**2)), 0.)
        return f_mat/np.exp(-1)  # normalize output so x_dist = 0 corresponds to f = 1

    def ThinPlateSpline(x_dist):
        return np.multiply(np.power(x_dist, 2), np.log(x_dist))

class NodalMap:
    def __init__(self, solid_nodal_mesh, fluid_nodal_mesh, 
                 RBF_width_par=np.inf, RBF_func=RadialBasisFunctions.Gaussian, 
                 column_scaling_vec=None):
        self.solid_nodal_mesh = solid_nodal_mesh
        self.fluid_nodal_mesh = fluid_nodal_mesh
        self.RBF_width_par = RBF_width_par
        self.RBF_func = RBF_func
        if column_scaling_vec is not None:
            self.column_scaling_vec = column_scaling_vec
        else:
            self.column_scaling_vec = np.ones((solid_nodal_mesh.shape[0],))

        self.map_shape = [self.fluid_nodal_mesh.shape[0], self.solid_nodal_mesh.shape[0]]
        self.distance_matrix = self.compute_distance_matrix()
        self.map = self.construct_map()
    
    def compute_distance_matrix(self):
        coord_dist_mat = np.zeros((self.map_shape + [3]))
        for i in range(3):
            coord_dist_mat[:, :, i] = NodalMap.coord_diff(self.fluid_nodal_mesh[:, i], 
                                                          self.solid_nodal_mesh[:, i])

        coord_dist_mat = NodalMap.compute_pairwise_Euclidean_distance(coord_dist_mat)
        return coord_dist_mat
    
    def construct_map(self):
        influence_coefficients = self.RBF_func(self.distance_matrix, eps=self.RBF_width_par)

        # influence_coefficients = np.multiply(influence_coefficients, influence_dist_below_max_mask)

        # include influence of column scaling
        influence_coefficients = np.einsum('ij,j->ij', influence_coefficients, self.column_scaling_vec)
        # influence_coefficients = np.dot(influence_coefficients, self.column_scaling_vec.reshape((-1,3))).reshape(-1,1)

        influence_coefficients[influence_coefficients < 1e-16] = 0.
        # TODO: Make influence_coefficients matrix sparse before summation below?
        #       -> seems like the matrix sparsity depends heavily on the value of self.RBF_par_width, 
        #           probably not worthwhile to make the matrix sparse

        # sum influence coefficients in each row and normalize the coefficients with those row sums
        inf_coeffs_per_row = np.sum(influence_coefficients, axis=1)
        normalized_inf_coeff_map = np.divide(influence_coefficients, inf_coeffs_per_row[:, None])
        return normalized_inf_coeff_map

    @staticmethod
    def coord_diff(arr_1, arr_2):
        return np.subtract(arr_1[:, None], arr_2[None, :])

    @staticmethod
    def compute_pairwise_Euclidean_distance(coord_dist_mat):
        coord_dist_mat_sqrd = np.power(coord_dist_mat, 2)
        coord_dist_mat_summed = np.sum(coord_dist_mat_sqrd, axis=2)
        return np.sqrt(coord_dist_mat_summed)


def clampedBC(spline_generator, side=0, direction=0):
    """
    Apply clamped boundary condition to spline.
    """
    for field in [0,1,2]:
        if field in [1]:
            n_layers = 1
        else:
            n_layers = 2
        scalar_spline = spline_generator.getScalarSpline(field)
        side_dofs = scalar_spline.getSideDofs(direction, side, 
                                              nLayers=n_layers)
        spline_generator.addZeroDofs(field, side_dofs)

def OCCBSpline2tIGArSpline(surface, num_field=3, quad_deg_const=4, 
                           setBCs=None, side=0, direction=0, index=0):
    """
    Generate ExtractedBSpline from OCC B-spline surface.
    """
    quad_deg = surface.UDegree()*quad_deg_const
    DIR = SAVE_PATH+"spline_data/extraction_"+str(index)+"_init"
    if os.path.isdir(DIR):
        spline = ExtractedSpline(DIR, quad_deg)
    else:
        spline_mesh = NURBSControlMesh4OCC(surface, useRect=False)
        spline_generator = EqualOrderSpline(worldcomm, num_field, spline_mesh)
        if setBCs is not None:
            setBCs(spline_generator, side, direction)
        spline_generator.writeExtraction(DIR)
        spline = ExtractedSpline(spline_generator, quad_deg)
    return spline

def OCCBSpline2LoadSpline(surface, sol_spline, n_fields=1, index=0):
    quad_deg = 4
    DIR = SAVE_PATH+"load_spline_data/extraction_"+str(index)+"_init"
    if os.path.isdir(DIR):
        spline = ExtractedSpline(DIR, quad_deg, mesh=sol_spline.mesh)
    else:
        spline_mesh = NURBSControlMesh4OCC(surface, useRect=False)
        bs_data = BSplineSurfaceData(surface)
        
        # TODO: Generalize this construction approach to allow for higher-degree force fields?
        spline_deg_diff = np.subtract(bs_data.degree, (1,1))
        
        # cut off some of the repeated first and last knots to create open knot vectors for the linear spline
        lin_knots = (bs_data.knots[0][spline_deg_diff[0]:-spline_deg_diff[0]], bs_data.knots[1][spline_deg_diff[1]:-spline_deg_diff[1]])
        
        # update spline_mesh with new data (the control mesh remains unchanged)
        # spline_mesh.scalarSpline = BSpline((1,1), lin_knots, False, 0)
        # spline_generator = EqualOrderSpline(worldcomm, 1, spline_mesh)

        # construct the spline generator as a FieldListSpline, since the control map and the solution fields exist in different spaces
        lin_spline = BSpline((1,1), lin_knots, useRect=False, overRefine=0)
        spline_generator = FieldListSpline(worldcomm, spline_mesh, [lin_spline]*n_fields)
        spline_generator.writeExtraction(DIR)
        # spline = ExtractedSpline(DIR, quad_deg, mesh=sol_spline.mesh)
        spline = ExtractedSpline(spline_generator, quad_deg, mesh=sol_spline.mesh)
    return spline


if __name__ == '__main__':
    from GOLDFISH.tests.test_tbeam import (nonmatching_opt, 
        srf0, srf1, spline0, spline1)
    # from GOLDFISH.tests.test_slr import nonmatching_opt
    from GOLDFISH.nonmatching_opt_ffd import *
    from PENGoLINS.igakit_utils import *

    ffd_block_num_el = [4,4,1]
    p = 3
    # Create FFD block in igakit format
    cp_ffd_lims = nonmatching_opt.cpsurf_lims
    for field in [2]:
        cp_range = cp_ffd_lims[field][1] - cp_ffd_lims[field][0]
        cp_ffd_lims[field][0] = cp_ffd_lims[field][0] - 0.2*cp_range
        cp_ffd_lims[field][1] = cp_ffd_lims[field][1] + 0.2*cp_range
    FFD_block = create_3D_block(ffd_block_num_el, p, cp_ffd_lims)
    nonmatching_opt.set_shopt_FFD(FFD_block.knots, FFD_block.control)
    nonmatching_opt.set_shopt_pin_CPFFD(pin_dir0=1, pin_side0=[0],
                                        pin_dir1=2, pin_side1=[0])

    occ_srf0 = ikNURBS2BSpline_surface(srf0)
    occ_srf1 = ikNURBS2BSpline_surface(srf1)

    occ_srfs = [occ_srf0, occ_srf1]
    splines = [spline0, spline1]

    lin_splines = []
    for s_ind in range(nonmatching_opt.num_splines):
        lin_splines += [OCCBSpline2LoadSpline(occ_srfs[s_ind], splines[s_ind])]

    lin_loads = PENGoLINS_dist_loads(lin_splines)

    klshell_pde = KLShellPDE(nonmatching_opt, lin_loads)
    
    klshell_pde.construct_nodal_disp_map()
    klshell_pde.compute_sparse_mass_matrix()
    # klshell_pde.compute_G_mat()