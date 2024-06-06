from GOLDFISH.nonmatching_opt_ffd import *
import openmdao.api as om
from openmdao.api import Problem

class CPIGA2XiExop(object):
    """
    Explicit operation to compute internal energy of non-matching 
    structure, derivatives of compliacne w.r.t. displacements 
    and control points both in IGA DoFs.
    """
    def __init__(self, nonmatching_opt):
        self.nonmatching_opt = nonmatching_opt
        self.cpiga2xi = self.nonmatching_opt.cpiga2xi
        self.num_splines = self.nonmatching_opt.num_splines
        self.splines = self.nonmatching_opt.splines
        self.opt_field = self.nonmatching_opt.opt_field
        
    def compute(self, xi_flat):
        self.cpiga2xi.update_occ_surfs()
        res = self.cpiga2xi.residual(xi_flat)
        return res


    def compute_partials(self, xi_flat, coo=False):
        self.cpiga2xi.update_occ_surfs()
        self.dRdxi_mat = self.cpiga2xi.dRdxi(xi_flat, coo=coo)
        # print("xi_flat:", xi_flat)
        # print("xi_flat norm:", np.linalg.norm(xi_flat))
        # print("det dRdxi:", np.linalg.det(self.dRdxi_mat.todense()))

        self.dRdCP_mat_list = []
        for i, field in enumerate(self.opt_field):
            self.dRdCP_mat_list += [self.cpiga2xi.dRdCP(xi_flat, 
                                    field, coo=coo)]
        return self.dRdxi_mat, self.dRdCP_mat_list



class CPIGA2XiExComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('nonmatching_opt')
        self.options.declare('input_cp_iga_name_pre', default='CP_IGA')
        self.options.declare('input_xi_name', default='int_para_coord')
        self.options.declare('output_res_name', default='res')

    def init_parameters(self):
        self.nonmatching_opt = self.options['nonmatching_opt']
        self.opt_field = self.nonmatching_opt.opt_field
        self.input_cp_iga_name_pre = self.options['input_cp_iga_name_pre']
        self.input_xi_name = self.options['input_xi_name']
        self.output_res_name = self.options['output_res_name']

        self.cpiga2xi_exop = CPIGA2XiExop(self.nonmatching_opt)

        self.input_cp_shapes = []
        for field_ind, field in enumerate(self.opt_field):        
            self.input_cp_shapes += [len(self.nonmatching_opt.cpdes_iga_dofs_full[field_ind])]
        self.init_cp_iga = self.nonmatching_opt.init_cp_iga
        self.input_xi_shape = self.cpiga2xi_exop.cpiga2xi.xi_size_global

        self.output_res_shape = self.input_xi_shape

        self.input_cp_iga_name_list = []
        for i, field in enumerate(self.opt_field):
            self.input_cp_iga_name_list += \
                [self.input_cp_iga_name_pre+str(field)]

    def setup(self):
        for i, field in enumerate(self.opt_field):
            self.add_input(self.input_cp_iga_name_list[i],
                           shape=self.input_cp_shapes[i],
                           val=self.init_cp_iga[i])
        self.add_input(self.input_xi_name, shape=self.input_xi_shape)
        self.add_output(self.output_res_name, shape=self.output_res_shape)

        for i, field in enumerate(self.opt_field):
            self.declare_partials(self.output_res_name,
                                  self.input_cp_iga_name_list[i])
        self.declare_partials(self.output_res_name, self.input_xi_name)

    def update_inputs(self, inputs):
        for i, field in enumerate(self.opt_field):
            self.cpiga2xi_exop.cpiga2xi.update_CPs(
                inputs[self.input_cp_iga_name_list[i]], field)

    def compute(self, inputs, outputs):
        self.update_inputs(inputs)
        xi_flat = inputs[self.input_xi_name]
        outputs[self.output_res_name] = self.cpiga2xi_exop.compute(xi_flat)

    def compute_partials(self, inputs, partials):
        self.update_inputs(inputs)
        xi_flat = inputs[self.input_xi_name]
        dRdxi, dRdcp_list = self.cpiga2xi_exop.compute_partials(xi_flat)


        partials[self.output_res_name, self.input_xi_name] = dRdxi
        for i, field in enumerate(self.opt_field):
            partials[self.output_res_name, 
                     self.input_cp_iga_name_list[i]] = dRdcp_list[i]

if __name__ == '__main__':
    # from GOLDFISH.tests.test_tbeam import nonmatching_opt
    # from GOLDFISH.tests.test_slr import nonmatching_opt
    from GOLDFISH.tests.test_dRdt import nonmatching_opt

    wint_op = CPIGA2XiExop(nonmatching_opt)

    vec0 = np.ones(wint_op.nonmatching_opt.vec_iga_dof)
    vec1 = np.ones(wint_op.nonmatching_opt.vec_iga_dof)
    vec_disp = np.concatenate([vec0, vec1])
    wint_op.nonmatching_opt.update_uIGA(vec_disp)

    wint = wint_op.Wint()
    dwint_duiga = wint_op.dWintduIGA()
    dwint_dcpiga = wint_op.dWintdCPIGA(1)
    dwintdh_th = wint_op.dWintdh_th()