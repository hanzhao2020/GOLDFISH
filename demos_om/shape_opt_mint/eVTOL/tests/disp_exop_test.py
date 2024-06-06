from GOLDFISH.nonmatching_opt_ffd import *
import openmdao.api as om
from openmdao.api import Problem

class DispMintExOpeartion(object):
    """
    Implicit operation to solve non-matching shell structures 
    displacements using nonlinear solver.
    """
    def __init__(self, nonmatching_opt, save_files=False):
        # print("Running init ...")
        self.nonmatching_opt = nonmatching_opt
        self.save_files = save_files
        self.comm = self.nonmatching_opt.comm
        self.opt_field = self.nonmatching_opt.opt_field

        self.dres_iga = self.nonmatching_opt.vec_iga_nest.copy()
        self.du_iga = self.nonmatching_opt.vec_iga_nest.copy()
        # self.dcp_iga = self.nonmatching_opt.vec_scalar_iga_nest.copy()
        self.dcps_iga = [vec.copy() for vec in self.nonmatching_opt.cpdes_iga_nest]
        self.dxi_vec = self.nonmatching_opt.xi_nest.copy()
        # print("Finish runing init ...")

    def compute(self):
        # print("Running apply_nonlinear...")
        res_iga = self.nonmatching_opt.RIGA()
        res_iga_array = get_petsc_vec_array(res_iga, self.comm)
        # print("Finish Running apply_nonlinear...")
        return res_iga_array

    def compute_partials(self):
        # print("Running linearize ...")
        self.dRdu_iga = self.nonmatching_opt.dRIGAduIGA()
        self.dRigadcpiga_list = []
        for i, field in enumerate(self.opt_field):
            # print('*'*50, field)
            self.dRigadcpiga_list += [self.nonmatching_opt.
                                      dRIGAdCPIGA(field),]
        self.dRigadxi = self.nonmatching_opt.dRIGAdxi()
        return self.dRdu_iga, self.dRigadcpiga_list, self.dRigadxi
        # print("Finish Running linearize ...")



class DispMintExComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('nonmatching_opt')
        self.options.declare('input_cp_iga_name_pre', default='CP_IGA')
        self.options.declare('input_xi_name', default='int_para')
        self.options.declare('input_u_name', default='displacements')
        self.options.declare('output_res_name', default='res')

    def init_parameters(self, save_files=False, nonlinear_solver_rtol=1e-3,
                       nonlinear_solver_max_it=10):
        self.nonmatching_opt = self.options['nonmatching_opt']
        self.input_cp_iga_name_pre = self.options['input_cp_iga_name_pre']
        self.input_xi_name = self.options['input_xi_name']
        self.input_u_name = self.options['input_u_name']
        self.output_res_name = self.options['output_res_name']


        self.save_files = save_files
        self.nonlinear_solver_max_it = nonlinear_solver_max_it
        self.nonlinear_solver_rtol = nonlinear_solver_rtol

        self.disp_mint_exop = DispMintExOpeartion(self.nonmatching_opt)
        self.opt_field = self.nonmatching_opt.opt_field
        self.input_cp_shapes = []
        for field_ind, field in enumerate(self.opt_field):        
            self.input_cp_shapes += [len(self.nonmatching_opt.cpdes_iga_dofs_full[field_ind])]

        self.input_xi_shape = self.nonmatching_opt.xi_size
        self.input_u_shape = self.nonmatching_opt.vec_iga_dof
        self.output_res_shape = self.nonmatching_opt.vec_iga_dof

        self.init_cp_iga = self.nonmatching_opt.get_init_CPIGA()
        self.init_xi = self.nonmatching_opt.cpiga2xi.xi_flat_global
        _, a0 = self.nonmatching_opt.solve_linear_nonmatching_problem(iga_dofs=True)
        self.init_disp_array = a0.array
        # self.current_xi = self.init_xi.copy()

        self.input_cp_iga_name_list = []
        for i, field in enumerate(self.opt_field):
            self.input_cp_iga_name_list += \
                [self.input_cp_iga_name_pre+str(field)]

    def setup(self):
        for i, field in enumerate(self.opt_field):
            self.add_input(self.input_cp_iga_name_list[i],
                           shape=self.input_cp_shapes[i],
                           val=self.init_cp_iga[i])
        self.add_input(self.input_xi_name, 
                       shape=self.input_xi_shape,
                       val=self.init_xi)
        self.add_input(self.input_u_name, shape=self.input_u_shape,
                       val=self.input_xi_shape)

        self.add_output(self.output_res_name, shape=self.output_res_shape)

        for i, field in enumerate(self.opt_field):
            self.declare_partials(self.output_res_name,
                                  self.input_cp_iga_name_list[i])
        self.declare_partials(self.output_res_name, self.input_xi_name)
        self.declare_partials(self.output_res_name, self.input_u_name)

    def update_inputs(self, inputs):
        # print("Updating inputs outputs ...")
        for i, field in enumerate(self.opt_field):
            self.nonmatching_opt.update_CPIGA(
                inputs[self.input_cp_iga_name_list[i]], field)
        
        # xi_diff = np.linalg.norm(inputs[self.input_xi_name]-self.current_xi)\
        #           /np.linalg.norm(self.current_xi)
        # print("xi diff:", xi_diff)
        # if xi_diff > 1e-14:
        # print("Updating xi and update_transfer_matrices...")
        self.nonmatching_opt.update_xi(inputs[self.input_xi_name])
        self.nonmatching_opt.update_transfer_matrices()
        self.nonmatching_opt.update_uIGA(inputs[self.input_u_name])

    def compute(self, inputs, outputs):
        self.update_inputs(inputs)
        outputs[self.output_res_name] = \
            self.disp_mint_exop.compute()

    def compute_partials(self, inputs, partials):
        self.update_inputs(inputs)
        self.dRdu_iga, self.dRigadcpiga_list, self.dRigadxi = \
            self.disp_mint_exop.compute_partials()

        for i, field in enumerate(self.opt_field):
            partials[self.output_res_name, self.input_cp_iga_name_list[i]] = csr_matrix(self.dRigadcpiga_list[i].getValuesCSR()[::-1]).todense()

        self.dRigadxi.convert('seqaij')
        partials[self.output_res_name, self.input_xi_name] = csr_matrix(self.dRigadxi.getValuesCSR()[::-1]).todense()
        partials[self.output_res_name, self.input_u_name] = csr_matrix(self.dRdu_iga.getValuesCSR()[::-1]).todense()
        


if __name__ == '__main__':
    # from GOLDFISH.tests.test_tbeam import nonmatching_opt
    from GOLDFISH.tests.test_slr import nonmatching_opt

    disp_op = DispMintExOpeartion(nonmatching_opt)

    disp_op.apply_nonlinear()
    disp_op.solve_nonlinear()
    disp_op.linearize()

    # t0 = nonmatching_opt.dRIGAdxi_sub(0, 0)
    indices = [1,3,5,7]
    t = nonmatching_opt.dRIGAdxi(indices)