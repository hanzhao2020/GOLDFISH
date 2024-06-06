import openmdao.api as om
from openmdao.api import Problem
from scipy.sparse import coo_matrix, bmat
from PENGoLINS.occ_preprocessing import *
from GOLDFISH.utils.opt_utils import *
from GOLDFISH.utils.bsp_utils import *

class CPIGA2CPIntComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('nonmatching_opt')
        self.options.declare('input_cp_iga_name_pre', default='CPS_IGA')
        self.options.declare('output_cp_iga_int_name_pre', default='CPS_IGA_int')

    def init_parameters(self):
        self.nonmatching_opt = self.options['nonmatching_opt']
        self.input_cp_iga_name_pre = self.options['input_cp_iga_name_pre']
        self.output_cp_iga_int_name_pre = self.options['output_cp_iga_int_name_pre']

        self.opt_field = self.nonmatching_opt.opt_field
        self.cp_iga_dof_list = self.nonmatching_opt.vec_scalar_iga_dof_list

        # self.input_cpiga_shape = self.nonmatching_opt.vec_scalar_iga_dof
        # self.init_cp_iga = self.nonmatching_opt.get_init_CPIGA()


        self.input_cp_shapes = []
        for field_ind, field in enumerate(self.opt_field):        
            self.input_cp_shapes += [len(self.nonmatching_opt.cpdes_iga_dofs_full[field_ind])]
        self.init_cp_iga = self.nonmatching_opt.init_cp_iga
        # self.output_shape = self.cpiga2xi_imop.cpiga2xi.xi_size_global
        

        self.int_surf_inds = self.nonmatching_opt.cpiga2xi.int_surf_inds
        self.output_cpiga_shape = self.nonmatching_opt.cpiga2xi.cp_size_global

        self.input_cp_iga_name_list = []
        self.output_cp_iga_name_list = []
        for i, field in enumerate(self.opt_field):
            self.input_cp_iga_name_list += \
                [self.input_cp_iga_name_pre+str(field)]
            self.output_cp_iga_name_list += \
                [self.output_cp_iga_int_name_pre+str(field)]

        self.partial_mat = self.get_derivative()

    def setup(self):
        for i, field in enumerate(self.opt_field):
            self.add_input(self.input_cp_iga_name_list[i], 
                           shape=self.input_cpiga_shape,
                           val=self.init_cp_iga[:,field])
            self.add_output(self.output_cp_iga_name_list[i], 
                            shape=self.output_cpiga_shape)
            self.declare_partials(self.output_cp_iga_name_list[i],
                                  self.input_cp_iga_name_list[i],
                                  val=self.partial_mat.data,
                                  rows=self.partial_mat.row,
                                  cols=self.partial_mat.col)

    def get_derivative(self, coo=True):
        self.partial_mat_list = [[coo_matrix((self.cp_iga_dof_list[j], 
                                         self.cp_iga_dof_list[i]))
                             for i in range(self.nonmatching_opt.num_splines)]
                             for j in self.int_surf_inds]
        for i, ind in enumerate(self.int_surf_inds):
            self.partial_mat_list[i][ind] = coo_matrix(np.eye(self.cp_iga_dof_list[ind]))
        partial_mat = bmat(self.partial_mat_list)
        if coo:
            partial_mat = coo_matrix(partial_mat)
        return partial_mat

    def compute(self, inputs, outputs):
        for i, field in enumerate(self.opt_field):
            outputs[self.output_cp_iga_name_list[i]] = \
                self.partial_mat*inputs[self.input_cp_iga_name_list[i]]

if __name__ == '__main__':
    from GOLDFISH.tests.test_tbeam_mint import preprocessor, nonmatching_opt

    #################################
    preprocessor.check_intersections_type()
    preprocessor.get_diff_intersections()
    nonmatching_opt.set_diff_intersections(preprocessor)
    #################################

    prob = Problem()
    comp = CPIGA2CPIntComp(nonmatching_opt=nonmatching_opt)
    comp.init_parameters()
    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print('check_partials:')
    prob.check_partials(compact_print=True)