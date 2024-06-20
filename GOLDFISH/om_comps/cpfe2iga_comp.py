from GOLDFISH.nonmatching_opt_ffd import *
from GOLDFISH.operations.cpfe2iga_imop import *
import openmdao.api as om
from openmdao.api import Problem

class CPFE2IGAComp(om.ImplicitComponent):

    def initialize(self):
        self.options.declare('nonmatching_opt')
        self.options.declare('input_cp_fe_name_pre', default='CP_FE')
        self.options.declare('output_cp_iga_name_pre', default='CP_IGA')

    def init_parameters(self):
        self.nonmatching_opt = self.options['nonmatching_opt']
        self.input_cp_fe_name_pre = self.options['input_cp_fe_name_pre']
        self.output_cp_iga_name_pre = self.options['output_cp_iga_name_pre']

        self.cpfe2iga_imop = CPFE2IGAImOperation(self.nonmatching_opt)
        self.opt_field = self.nonmatching_opt.opt_field
        self.input_shapes = [cp_fe.size for cp_fe in 
                             self.nonmatching_opt.cpdes_fe_nest]
        self.output_shapes = [cp_iga.size for cp_iga in 
                              self.nonmatching_opt.cpdes_iga_nest]
        self.cp_fe_vecs = self.cpfe2iga_imop.cp_fe_vecs
        self.cp_iga_vecs = self.cpfe2iga_imop.cp_iga_vecs

        if self.nonmatching_opt.shopt_multiffd:
            self.init_cp_fe_array_list = self.nonmatching_opt.shopt_init_cpsurf_fe_list_mffd
        else:
            self.init_cp_fe_array_list = self.nonmatching_opt.shopt_cpsurf_fe_hom_list
        self.input_cp_fe_name_list = []
        self.output_cp_iga_name_list = []
        for i, field in enumerate(self.opt_field):
            self.input_cp_fe_name_list += \
                [self.input_cp_fe_name_pre+str(field)]
            self.output_cp_iga_name_list += \
                [self.output_cp_iga_name_pre+str(field)]

    def setup(self):
        for i, field in enumerate(self.opt_field):
            # init_cp_fe_array = get_petsc_vec_array(
            #                    self.nonmatching_opt.cp_funcs_nest[field], 
            #                    comm=self.nonmatching_opt.comm)
            self.add_input(self.input_cp_fe_name_list[i],
                           shape=self.input_shapes[i],)
                           # val=self.init_cp_fe_array_list[i])
            self.add_output(self.output_cp_iga_name_list[i],
                            shape=self.output_shapes[i])
            self.declare_partials(self.output_cp_iga_name_list[i],
                                  self.input_cp_fe_name_list[i],
                                  val=self.cpfe2iga_imop.dRdcp_fe_coo[i].data,
                                  rows=self.cpfe2iga_imop.dRdcp_fe_coo[i].row,
                                  cols=self.cpfe2iga_imop.dRdcp_fe_coo[i].col)
            self.declare_partials(self.output_cp_iga_name_list[i],
                                  self.output_cp_iga_name_list[i],
                                  val=self.cpfe2iga_imop.dRdcp_iga_coo[i].data,
                                  rows=self.cpfe2iga_imop.dRdcp_iga_coo[i].row,
                                  cols=self.cpfe2iga_imop.dRdcp_iga_coo[i].col)

    def update_cp_vecs(self, inputs, outputs):
        for i, field in enumerate(self.opt_field):
            update_nest_vec(inputs[self.input_cp_fe_name_list[i]],
                            self.cp_fe_vecs[i])
            update_nest_vec(outputs[self.output_cp_iga_name_list[i]],
                            self.cp_iga_vecs[i])

    def apply_nonlinear(self, inputs, outputs, residuals):
        self.update_cp_vecs(inputs, outputs)
        res_array_list = self.cpfe2iga_imop.apply_nonlinear()
        for i, field in enumerate(self.opt_field):
            residuals[self.output_cp_iga_name_list[i]] = res_array_list[i]

    def solve_nonlinear(self, inputs, outputs):
        self.update_cp_vecs(inputs, outputs)
        cp_iga_arary_list = self.cpfe2iga_imop.solve_nonlinear()
        for i, field in enumerate(self.opt_field):
            outputs[self.output_cp_iga_name_list[i]] = cp_iga_arary_list[i]

    def solve_linear(self, d_outputs, d_residuals, mode):
        d_outputs_array_list = []
        d_residuals_array_list = []
        for i, field in enumerate(self.opt_field):
            d_outputs_array_list += [d_outputs[
                                     self.output_cp_iga_name_list[i]]]
            d_residuals_array_list += [d_residuals[
                                       self.output_cp_iga_name_list[i]]]
        if mode == 'fwd':
            self.cpfe2iga_imop.solve_linear_fwd(d_outputs_array_list,
                                                d_residuals_array_list)
        if mode == 'rev':
            self.cpfe2iga_imop.solve_linear_rev(d_outputs_array_list,
                                                d_residuals_array_list)

if __name__ == "__main__":
    from GOLDFISH.tests.test_tbeam import nonmatching_opt
    # from GOLDFISH.tests.test_slr import nonmatching_opt

    nonmatching_opt.set_shopt_FFD_surf_inds(opt_field=[0,1,2], opt_surf_inds=[0,1])
    
    ffd_block_num_el = [4,4,1]
    p = 3
    # Create FFD block in igakit format
    cp_ffd_lims = nonmatching_opt.cpsurf_des_lims
    for field in [2]:
        cp_range = cp_ffd_lims[field][1] - cp_ffd_lims[field][0]
        cp_ffd_lims[field][0] = cp_ffd_lims[field][0] - 0.2*cp_range
        cp_ffd_lims[field][1] = cp_ffd_lims[field][1] + 0.2*cp_range
    FFD_block = create_3D_block(ffd_block_num_el, p, cp_ffd_lims)
    nonmatching_opt.set_shopt_FFD(FFD_block.knots, FFD_block.control)
    nonmatching_opt.set_shopt_align_CPFFD(align_dir=[[1],[2],[0]])
    nonmatching_opt.set_shopt_regu_CPFFD()

    prob = Problem()
    comp = CPFE2IGAComp(nonmatching_opt=nonmatching_opt)
    comp.init_parameters()
    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print('check_partials:')
    prob.check_partials(compact_print=True)