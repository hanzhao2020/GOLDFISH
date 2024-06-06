from GOLDFISH.nonmatching_opt_ffd import *
import openmdao.api as om
from openmdao.api import Problem


class ShellDistComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('dist_dir', default=0)
        self.options.declare('cp_size', default=3)
        self.options.declare('num_surfs', default=2)
        self.options.declare('rev_dir', default=False)
        self.options.declare('input_cp_design_name_pre', default='CP_design')
        self.options.declare('output_cp_dist_name_pre', default='CP_dist')

    def init_parameters(self):
        self.dist_dir = self.options['dist_dir']
        self.cp_size = self.options['cp_size']
        self.num_surfs = self.options['num_surfs']
        self.rev_dir = self.options['rev_dir']
        self.input_cp_design_name_pre = self.options['input_cp_design_name_pre']
        self.output_cp_dist_name_pre = self.options['output_cp_dist_name_pre']

        # self.deriv = create_surf_regu_operator(cp_size=3, num_surfs=2, rev_dir=self.rev_dir, coo=True)
        self.deriv = create_surf_regu_operator(cp_size=3, num_surfs=3, rev_dir=self.rev_dir, coo=True)
        self.input_shape = self.deriv.shape[1]
        self.output_shape = self.deriv.shape[0]

        self.input_cp_design_name = self.input_cp_design_name_pre+str(self.dist_dir)
        self.output_cp_dist_name = self.output_cp_dist_name_pre+str(self.dist_dir)

    def setup(self):
        self.add_input(self.input_cp_design_name,
                       shape=self.input_shape)
                       # val=self.init_cp_design[i])
        self.add_output(self.output_cp_dist_name,
                        shape=self.output_shape)
        self.declare_partials(self.output_cp_dist_name,
                              self.input_cp_design_name,
                              val=self.deriv.data,
                              rows=self.deriv.row,
                              cols=self.deriv.col)

    def compute(self, inputs, outputs):
        outputs[self.output_cp_dist_name] = \
            self.deriv*inputs[self.input_cp_design_name]


if __name__ == "__main__":

    prob = Problem()
    comp = ShellDistComp(dist_dir=0)
    comp.init_parameters()
    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print('check_partials:')
    prob.check_partials(compact_print=True)