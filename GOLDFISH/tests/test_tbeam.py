from tIGAr.NURBS import *
from GOLDFISH.nonmatching_opt_ffd import *
from igakit.cad import *

def create_surf(pts, num_el0, num_el1, p):
    knots0 = np.linspace(0,1,num_el0+1)[1:-1]
    knots1 = np.linspace(0,1,num_el1+1)[1:-1]
    L1 = line(pts[0],pts[1])
    L2 = line(pts[2],pts[3])
    srf = ruled(L1,L2)
    deg0, deg1 = srf.degree 
    srf.elevate(0,p-deg0)
    srf.elevate(1,p-deg1)
    srf.refine(0,knots0)
    srf.refine(1,knots1)
    return srf

def create_spline(srf, num_field, BCs=[0,1]):
    spline_mesh = NURBSControlMesh(srf, useRect=False)
    spline_generator = EqualOrderSpline(COMM, num_field, spline_mesh)

    for field in range(num_field):
        scalar_spline = spline_generator.getScalarSpline(field)
        for para_direction in range(2):
            if BCs[para_direction] == 1:
                side = 0  # Only consider fixing the 0 side
                side_dofs = scalar_spline.getSideDofs(para_direction, 
                                                      side, nLayers=1)
                spline_generator.addZeroDofs(field, side_dofs)

    quad_deg = 3*srf.degree[0]
    spline = ExtractedSpline(spline_generator, quad_deg)
    return spline


class EqualOrderSplineCustom(EqualOrderSpline):
    def getDegree(self, field):
        return self.controlMesh.scalarSpline.splines[0].p

COMM = worldcomm
E = Constant(1.0e7)
nu = Constant(0.)
h_th = Constant(0.1)

L = 20.
w = 2.
h = 2.
num_field = 3

pts0 = [[-w/2., 0., 0.], [w/2., 0., 0.],\
        [-w/2., L, 0.], [w/2., L, 0.]]
pts1 = [[0., 0., 0.], [0.,0.,-h],\
        [0., L, 0.], [0., L, -h]]


num_el = 10
penalty_coefficient = 1e3
if mpirank == 0:
    print("Number of elements:", num_el)
    print("Penalty coefficient:", penalty_coefficient)
p = 3
num_el0 = num_el
num_el1 = num_el + 1
p0 = p
p1 = p
if mpirank == 0:
    print("Creating geometry...")
srf0 = create_surf(pts0, int(num_el0/2), num_el0, p0)
srf1 = create_surf(pts1, int(num_el1/2), num_el1, p1)
spline0 = create_spline(srf0, num_field, BCs=[0,1])
spline1 = create_spline(srf1, num_field, BCs=[0,1])

splines = [spline0, spline1]

h_th_list = []
for i in range(len(splines)):
    h_th_list += [Function(splines[i].V_control)]
    h_th_list[i].interpolate(Constant(0.1))

nonmatching_opt = NonMatchingOptFFD(splines, E, h_th_list, nu)

nonmatching_opt.set_thickness_opt(var_thickness=False)

mortar_nels = [2*num_el1]
nonmatching_opt.create_mortar_meshes(mortar_nels)

mapping_list = [[0,1],]
physical_locations = [np.array([[0.,0.,0.],[0.,20.,0.]]),]
mortar_mesh_locations = [[]]*nonmatching_opt.num_intersections
for i in range(nonmatching_opt.num_intersections):
    for j in range(2):
        mortar_mesh_locations[i] += [interface_parametric_location(
                                     splines[mapping_list[i][j]], 
                                     nonmatching_opt.mortar_meshes[i], 
                                     physical_locations[i]),]

nonmatching_opt.mortar_meshes_setup(mapping_list, mortar_mesh_locations,
                                    penalty_coefficient, 2)

source_terms = []
residuals = []
f0 = as_vector([Constant(0.), Constant(0.), Constant(0.)])
for i in range(len(splines)):
    source_terms += [inner(f0, nonmatching_opt.splines[i].\
                     rationalize(nonmatching_opt.spline_test_funcs[i]))\
                     *nonmatching_opt.splines[i].dx,]
    residuals += [SVK_residual(nonmatching_opt.splines[i], 
                               nonmatching_opt.spline_funcs[i], 
                               nonmatching_opt.spline_test_funcs[i], 
                               E, nu, h_th, source_terms[i])]
nonmatching_opt.set_residuals(residuals)

# PointSource will be applied mpisize times in parallel
tip_load = -10./MPI.size(COMM)
ps0 = PointSource(spline0.V.sub(2), Point(1.,1.), -tip_load)
ps_list = [ps0,]
ps_ind = [0,]
nonmatching_opt.set_point_sources(point_sources=ps_list, 
                                  point_source_inds=ps_ind)