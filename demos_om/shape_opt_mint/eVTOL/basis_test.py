from PENGoLINS.occ_preprocessing import *
from PENGoLINS.igakit_utils import *
from igakit import igalib as _igalib
from igakit.cad import NURBS
_bsp = _igalib.bsp

from tIGAr import BSplines


knots0 = [0,0,0,0,1,1,1,1]
knots1 = [0,0,1,1]
knots = [knots0, knots1]
p = [3,1]

cp = array([[[0.        , 0.        , 2.        , 1.        ],
            [0.        , 1.        , 2.        , 1.        ]],
           [[0.33333333, 0.        , 4.        , 1.        ],
            [0.33333333, 1.        , 4.        , 1.        ]],
           [[0.66666667, 0.        , 3.        , 1.        ],
            [0.66666667, 1.        , 3.        , 1.        ]],
           [[1.        , 0.        , 6.        , 1.        ],
            [1.        , 1.        , 6.        , 1.        ]]])

cp_flat = cp.transpose(1,0,2).reshape(-1,4)


xi = [0.4,0.9]

ikbs = NURBS(knots, cp)
tigar_bs = BSplines.BSpline(p, knots, xi)




# ik eval basis
ikbasis0 = _bsp.EvalBasisFunsDers(p[0],knots[0],xi[0])
ikbasis1 = _bsp.EvalBasisFunsDers(p[1],knots[1],xi[1])
ikbasis = []
for i in range(len(ikbasis0[0])):
    for j in range(len(ikbasis1[0])):
        ikbasis += [ikbasis0[0][i]*ikbasis1[0][j]]

ikbasis_deriv0 = []
for i in range(len(ikbasis0[1])):
    for j in range(len(ikbasis1[0])):
        ikbasis_deriv0 += [ikbasis0[1][i]*ikbasis1[0][j]]


ikbasis_dderiv0 = []
for i in range(len(ikbasis0[2])):
    for j in range(len(ikbasis1[0])):
        ikbasis_dderiv0 += [ikbasis0[2][i]*ikbasis1[0][j]]

ikbasis_deriv1 = []
for i in range(len(ikbasis0[0])):
    for j in range(len(ikbasis1[1])):
        ikbasis_deriv1 += [ikbasis0[0][i]*ikbasis1[1][j]]

span0 = _bsp.FindSpan(p[0], knots[0], xi[0])
span1 = _bsp.FindSpan(p[1], knots[1], xi[1])
nodes0 = list(range(span0-p[0], span0+1))
nodes1 = list(range(span1-p[1], span1+1))
cp_shape0 = cp.shape[0]

def ij2dof(i,j,l):
    return i+j*l

nodes = []
for node0_temp in nodes0:
    for node1_temp in nodes1:
        nodes += [ij2dof(node0_temp, node1_temp, cp_shape0)]


# tIGAr eval basis
tigar_basis_all = tigar_bs.getNodesAndEvals(xi)
tigar_basis = [i[1] for i in tigar_basis_all]
tigar_nodes = [i[0] for i in tigar_basis_all]


F0 = np.dot(ikbasis, cp_flat[:,0][nodes])
F2 = np.dot(ikbasis, cp_flat[:,2][nodes])
dFdxi00 = np.dot(ikbasis_deriv0, cp_flat[:,0][nodes])
dFdxi02 = np.dot(ikbasis_deriv0, cp_flat[:,2][nodes])

dFddxi00 = np.dot(ikbasis_dderiv0, cp_flat[:,0][nodes])
dFddxi01 = np.dot(ikbasis_dderiv0, cp_flat[:,1][nodes])
dFddxi02 = np.dot(ikbasis_dderiv0, cp_flat[:,2][nodes])

dFdxi10 = np.dot(ikbasis_deriv1, cp_flat[:,0][nodes])


# For OCC basis

occbs = ikNURBS2BSpline_surface(ikbs)
phy_pt = gp_Pnt()
dFdxi1_vec = gp_Vec()
dFdxi2_vec = gp_Vec()

occbs.D1(xi[0], xi[1], phy_pt, dFdxi1_vec, dFdxi2_vec)


p = gp_Pnt()
d1u = gp_Vec()
d1v = gp_Vec()
d2u = gp_Vec()
d2v = gp_Vec()
d2uv = gp_Vec()
occbs.D2(xi[0], xi[1], p, d1u, d1v, d2u, d2v, d2uv)