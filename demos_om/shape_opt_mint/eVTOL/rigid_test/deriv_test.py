import numpy as np
import openmdao.api as om
from openmdao.test_suite.components.sellar import SellarDerivatives

prob = om.Problem(model=SellarDerivatives())
model = prob.model
model.nonlinear_solver = om.NonlinearBlockGS()

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['tol'] = 1e-9

model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
model.add_design_var('x', lower=0.0, upper=10.0, scaler=100)
model.add_objective('obj')
model.add_constraint('con1', upper=0.0)
model.add_constraint('con2', upper=0.0)

prob.setup()
prob.set_solver_print(level=0)
prob.run_driver()
print(f"con1 = {prob.get_val('con1')}")