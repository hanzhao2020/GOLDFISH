from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

m = 5.4779
# m = 3.6870
a = -4*m/100
b = 4*m/10
c = 0

w = 10.
L = 3.

# x = np.linspace(0,w,100)
# y = a*x**2 + b*x + c
# plt.figure()
# plt.plot(x, y, '-*')
# plt.show()

mesh = RectangleMesh(Point(0.,0.), Point(w,L), 201, 4)
V = VectorFunctionSpace(mesh, 'CG', 1, dim=3)
u_disp = Expression(('0.', '0.', 'a*pow(x[0],2)+b*x[0]+c'), 
                    a=a, b=b, c=c, degree=2)
u = project(u_disp, V)
u.rename("CP_ex", "CP_ex")
# File("./geometry/exact_shape.pvd") << u
File("./geometry/parabola_shape.pvd") << u