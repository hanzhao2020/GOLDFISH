from csdl import Model, CustomExplicitOperation
import csdl
import numpy as np
from csdl_om import Simulator
from int_ss import *

class IntSSModel(Model):

    def initialize(self):
        self.parameters.declare('int_ss')
        self.parameters.declare('int_ind')

    def define(self):
        self.int_ss = self.parameters['int_ss']
        self.int_ind = self.parameters['int_ind']

        xi = self.declare_variable('xi', 
                                   shape=(self.int_ss.num_interior_pts*4,))

        operation = IntSSCompute(int_ss=self.int_ss, int_ind=self.int_ind)
        res = csdl.custom(xi, op=operation)

        self.register_output('res', res)


class IntSSCompute(CustomExplicitOperation):

    def initialize(self):
        self.parameters.declare('int_ss')
        self.parameters.declare('int_ind')
        
    def define(self):
        self.int_ss = self.parameters['int_ss']
        self.int_ind = self.parameters['int_ind']

        xi_init = np.concatenate(
            [int_ss.ints_para_coord_equidist_para[self.int_ind][0][1:-1], 
             int_ss.ints_para_coord_equidist_para[self.int_ind][1][1:-1]], 
            axis=1)
        xi_init_flat = xi_init.reshape(-1,1)[:,0]

        self.add_input('xi', shape=(self.int_ss.num_interior_pts*4,),
                       val=xi_init_flat)
        self.add_output('res', shape=(self.int_ss.num_interior_pts*4,))
        self.declare_derivatives('*', '*')

    def compute(self, inputs, outputs):
        outputs['res'] = self.int_ss.coupled_residual(inputs['xi'], 
                                                      self.int_ind)

    def compute_derivatives(self, inputs, derivatives):
        derivatives['res', 'xi'] = self.int_ss.dRdxi(inputs['xi'], 
                                                     self.int_ind)

if __name__ == "__main__":

    int_ind = 0
    sim = Simulator(IntSSModel(int_ss=int_ss, int_ind=int_ind))

    sim['xi'] = xi_root
    sim.run()

    print("Checking partials ...")
    sim.check_partials()