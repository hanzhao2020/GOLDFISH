import numpy as np
from scipy.sparse import coo_matrix
import openmdao.api as om
from openmdao.api import Problem

class XiConsComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('input_xi_name', default='int_xi')
        self.options.declare('output_xi_cons_name', default='w_int')

    def init_parameters(self, input_shape):
        self.input_xi_name = self.options['input_xi_name']
        self.output_xi_cons_name = self.options['output_xi_cons_name']

        self.input_shape = input_shape
        self.num_pts = int(self.input_shape/4)
        self.output_shape = 4
        self.deriv = np.zeros((self.output_shape, self.input_shape))
        self.deriv[0,0] = 1
        self.deriv[1,1] = 1
        self.deriv[2,self.num_pts*2] = 1
        self.deriv[3,self.num_pts*2+1] = 1
        self.deriv = coo_matrix(self.deriv)


    def setup(self):
        self.add_input(self.input_xi_name,
                       shape=self.input_shape)
        self.add_output(self.output_xi_cons_name,
                        shape=self.output_shape)
        self.declare_partials(self.output_xi_cons_name,
                              self.input_xi_name,
                              val=self.deriv.data,
                              rows=self.deriv.row,
                              cols=self.deriv.col)

    def compute(self, inputs, outputs):
        outputs[self.output_xi_cons_name] = np.dot(self.deriv.todense(),
                                                   inputs[self.input_xi_name])