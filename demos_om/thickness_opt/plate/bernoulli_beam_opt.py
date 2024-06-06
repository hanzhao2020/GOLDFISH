import numpy as np
import matplotlib.pyplot as plt

# th = np.array([0.14915749, 0.14764328, 0.14611321, 0.1445672 , 0.14300422, 0.1414241,
#  0.13982611, 0.13820973, 0.13657401, 0.13491866, 0.13324259, 0.1315453,
#  0.12982575, 0.1280832 , 0.12631652, 0.12452483, 0.12270692, 0.12086181,
#  0.11898801, 0.1170842 , 0.11514905, 0.11318073, 0.11117756, 0.10913768,
#  0.10705896, 0.104939  , 0.10277542, 0.10056528, 0.09830549, 0.09599251,
#  0.09362249, 0.09119083, 0.08869258, 0.08612199, 0.08347229, 0.08073574,
#  0.07790325, 0.07496383, 0.07190449, 0.06870933, 0.06535831, 0.06182632,
#  0.05808048, 0.05407662, 0.04975297, 0.04501847, 0.03972914, 0.03363155,
#  0.02620191, 0.01610859])

# th_norm = th/np.max(th)
# x_coord = np.linspace(0.,1.,len(th))

# from openmdao.test_suite.test_examples.beam_optimization.beam_group import BeamGroup
# import openmdao.api as om

import numpy as np
import openmdao.api as om

from openmdao.test_suite.test_examples.beam_optimization.components.moment_comp import MomentOfInertiaComp
from openmdao.test_suite.test_examples.beam_optimization.components.local_stiffness_matrix_comp import LocalStiffnessMatrixComp
from openmdao.test_suite.test_examples.beam_optimization.components.states_comp import StatesComp
from openmdao.test_suite.test_examples.beam_optimization.components.compliance_comp import ComplianceComp
from openmdao.test_suite.test_examples.beam_optimization.components.volume_comp import VolumeComp


class BeamGroup(om.Group):

    def initialize(self):
        self.options.declare('E')
        self.options.declare('L')
        self.options.declare('b')
        self.options.declare('volume')
        self.options.declare('num_elements', int)

    def setup(self):
        E = self.options['E']
        L = self.options['L']
        b = self.options['b']
        volume = self.options['volume']
        num_elements = self.options['num_elements']
        num_nodes = num_elements + 1

        force_vector = np.zeros(2 * num_nodes)
        force_vector[-2] = -1.

        I_comp = MomentOfInertiaComp(num_elements=num_elements, b=b)
        self.add_subsystem('I_comp', I_comp, promotes_inputs=['h'])

        comp = LocalStiffnessMatrixComp(num_elements=num_elements, E=E, L=L)
        self.add_subsystem('local_stiffness_matrix_comp', comp)

        comp = StatesComp(num_elements=num_elements, force_vector=force_vector)
        self.add_subsystem('states_comp', comp)

        comp = ComplianceComp(num_elements=num_elements, force_vector=force_vector)
        self.add_subsystem('compliance_comp', comp)

        comp = VolumeComp(num_elements=num_elements, b=b, L=L)
        self.add_subsystem('volume_comp', comp, promotes_inputs=['h'])

        self.connect('I_comp.I', 'local_stiffness_matrix_comp.I')
        self.connect('local_stiffness_matrix_comp.K_local', 'states_comp.K_local')
        self.connect('states_comp.d', 'compliance_comp.displacements',
                     src_indices=np.arange(2 *num_nodes))

        self.add_design_var('h', lower=5e-3, upper=10.)
        self.add_objective('compliance_comp.compliance')
        self.add_constraint('volume_comp.volume', equals=volume)

E = 1.
L = 1.
b = 0.5
volume = 0.05

num_elements = 108

prob = om.Problem(model=BeamGroup(E=E, L=L, b=b, volume=volume, num_elements=num_elements))

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['tol'] = 1e-15
prob.driver.options['disp'] = True
prob.driver.options['maxiter'] = 50000

prob.setup()

prob.run_driver()

th = prob['h']
x_coord = np.linspace(0.,1.,len(th))
th_norm = th/np.max(th)

# print(prob['h'])

plt.figure()
plt.plot(x_coord, th_norm)
plt.show()