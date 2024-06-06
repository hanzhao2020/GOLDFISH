import sys
sys.path.append("../opers/")
sys.path.append("../")

from GOLDFISH.nonmatching_opt_ffd import *
from cpiga2xi_imop_v0 import *
import openmdao.api as om
from openmdao.api import Problem

class CPIGA2XiComp(om.ImplicitComponent):

    def initialize(self):
        self.options.declare('preprocessor')
        self.options.declare('int_indices_diff', default=None)
        self.options.declare('opt_field', default=[0,1,2])
        self.options.declare('input_cp_iga_name_pre', default='CP_IGA')
        self.options.declare('output_xi_name', default='int_para_coord')

    def init_parameters(self):
        self.preprocessor = self.options['preprocessor']
        self.int_indices_diff = self.options['int_indices_diff']
        self.opt_field = self.options['opt_field']
        self.input_cp_iga_name_pre = self.options['input_cp_iga_name_pre']
        self.output_xi_name = self.options['output_xi_name']

        self.cpiga2xi_imop = CPIGA2XiImOperation(self.preprocessor,
                               self.int_indices_diff,
                               self.opt_field)

        self.input_shape = self.cpiga2xi_imop.cp_size_global
        self.output_shape = self.cpiga2xi_imop.xi_size_global

        self.input_cp_iga_name_list = []
        for i, field in enumerate(self.opt_field):
            self.input_cp_iga_name_list += \
                [self.input_cp_iga_name_pre+str(field)]

    def setup(self):
        for i, field in enumerate(self.opt_field):
            self.add_input(self.input_cp_iga_name_list[i],
                           shape=self.input_shape,
                           val=self.cpiga2xi_imop.cp_flat_global[:,field])
        self.add_output(self.output_xi_name, shape=self.output_shape)
        for i, field in enumerate(self.opt_field):
            self.declare_partials(self.output_xi_name,
                                  self.input_cp_iga_name_list[i])
        self.declare_partials(self.output_xi_name, self.output_xi_name)

    def update_inputs(self, inputs):
        for i, field in enumerate(self.opt_field):
            self.cpiga2xi_imop.update_CPs(
                inputs[self.input_cp_iga_name_list[i]], field)

    def apply_nonlinear(self, inputs, outputs, residuals):
        self.update_inputs(inputs)
        xi_flat = outputs[self.output_xi_name]
        residuals[self.output_xi_name] = self.cpiga2xi_imop.apply_nonlinear(xi_flat)

    def solve_nonlinear(self, inputs, outputs):
        self.update_inputs(inputs)
        xi_flat_init = self.cpiga2xi_imop.xi_flat_global
        outputs[self.output_xi_name] = self.cpiga2xi_imop.solve_nonlinear(xi_flat_init)

    def linearize(self, inputs, outputs, partials):
        self.update_inputs(inputs)
        xi_flat = outputs[self.output_xi_name]
        self.cpiga2xi_imop.linearize(xi_flat)

    def apply_linear(self, inputs, outputs, d_inputs, 
                     d_outputs, d_residuals, mode):
        self.update_inputs(inputs)
        d_inputs_array_list = []
        for i, field in enumerate(self.opt_field):
            if self.input_cp_iga_name_list[i] in d_inputs:
                d_inputs_array_list += \
                    [d_inputs[self.input_cp_iga_name_list[i]]]
        if len(d_inputs_array_list) == 0:
            d_inputs_array_list = None

        d_outputs_array = None
        if self.output_xi_name in d_outputs:
            d_outputs_array = d_outputs[self.output_xi_name]
        d_residuals_array = None
        if self.output_xi_name in d_residuals:
            d_residuals_array = d_residuals[self.output_xi_name]
            
        if mode == 'fwd':
            self.cpiga2xi_imop.apply_linear_fwd(d_inputs_array_list, 
                d_outputs_array, d_residuals_array)
        elif mode == 'rev':
            self.cpiga2xi_imop.apply_linear_rev(d_inputs_array_list, 
                d_outputs_array, d_residuals_array)

    def solve_linear(self, d_outputs, d_residuals, mode):
        d_outputs_array = d_outputs[self.output_xi_name]
        d_residuals_array = d_residuals[self.output_xi_name]
        if mode == 'fwd':
            self.cpiga2xi_imop.solve_linear_fwd(d_outputs_array,
                                                  d_residuals_array)
        if mode == 'rev': 
            self.cpiga2xi_imop.solve_linear_rev(d_outputs_array,
                                                  d_residuals_array)

if __name__ == "__main__":
    from PENGoLINS.occ_preprocessing import *

    filename_igs = "../geometry/init_Tbeam_geom_moved.igs"
    igs_shapes = read_igs_file(filename_igs, as_compound=False)
    occ_surf_list = [topoface2surface(face, BSpline=True) 
                     for face in igs_shapes]
    occ_surf_data_list = [BSplineSurfaceData(surf) for surf in occ_surf_list]
    num_surfs = len(occ_surf_list)
    p = occ_surf_data_list[0].degree[0]

    # Geometry preprocessing and surface-surface intersections computation
    preprocessor = OCCPreprocessing(occ_surf_list, reparametrize=False, 
                                    refine=False)
    print("Computing intersections...")
    # int_data_filename = "int_data.npz"
    # if os.path.isfile(int_data_filename):
    #     preprocessor.load_intersections_data(int_data_filename)
    # else:
    #     preprocessor.compute_intersections(mortar_refine=2)
    #     preprocessor.save_intersections_data(int_data_filename)

    preprocessor.compute_intersections(mortar_refine=2)
    if mpirank == 0:
        print("Total DoFs:", preprocessor.total_DoFs)
        print("Number of intersections:", preprocessor.num_intersections_all)

    prob = Problem()
    comp = CPIGA2XiComp(preprocessor=preprocessor, opt_field=[0,1,2])
    comp.init_parameters()
    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print('check_partials:')
    prob.check_partials(compact_print=True)