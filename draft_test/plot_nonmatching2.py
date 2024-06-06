from GOLDFISH.nonmatching_opt_om import *
from PENGoLINS.igakit_utils import *
import igakit.plot as ikplot
import matplotlib.pyplot as plt

def create_surf(pts, num_el0, num_el1, p):
    """
    Create igakit NURBS instances
    """
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

def ikSurf2tIGArSpline(surface, num_field=3, quad_deg_const=3, 
                        spline_bc=None, index=0):
    """
    Generate ExtractedBSpline from OCC B-spline surface.
    """
    quad_deg = surface.degree[0]*quad_deg_const
    # DIR = SAVE_PATH+"spline_data/extraction_"+str(index)+"_init"
    # spline = ExtractedSpline(DIR, quad_deg)
    spline_mesh = NURBSControlMesh(surface, useRect=True)
    spline_generator = EqualOrderSpline(worldcomm, num_field, spline_mesh)
    if spline_bc is not None:
        spline_bc.set_bc(spline_generator)
    # spline_generator.writeExtraction(DIR)
    spline = ExtractedSpline(spline_generator, quad_deg)
    return spline


vtk_writer = VTKWriter()

p = 2
p_ffd = 2

num_els = [2,2]
x_off0 = -0.06
pts0 = np.array([[0.+x_off0, 0., 0.], 
                 [1.+x_off0, 0., 0.],
                 [0.+x_off0, 1., 0.],
                 [1.+x_off0, 1., 0.]])

y_off1 = -0.2
pts1 = np.array([[0.5, 0.+y_off1, -0.3], 
                 [0.5, 1.+y_off1, -0.3],
                 [0.5, 0.+y_off1, 0.3],
                 [0.5, 1.+y_off1, 0.3]])
pts_list = [pts0, pts1]

surfs = []
for i in range(len(num_els)):
    surfs += [create_surf(pts_list[i], num_els[i], num_els[i], p)]
    # vtk_writer.write('./nm_geom/surf'+str(i)+'_init.vtk', surfs[i], ref_level=4)
    # vtk_writer.write_cp('./nm_geom/surf'+str(i)+'_cp_init.vtk', surfs[i])

surfs_new = []
npts = 4
s0_control = surfs[0].control
# s0_control[...,2][1] += np.ones(npts)*0.1
# s0_control[...,2][2] += np.ones(npts)*(-0.1)
# s0_control[...,2][:,0] += np.ones(npts)*(-0.1)
# s0_control[...,2][:,3] += np.ones(npts)*(-0.1)

s0_control[...,2][0] += np.ones(npts)*(-0.1)
s0_control[...,2][3] += np.ones(npts)*(0.1)

s0_control[...,2][:,0] += np.ones(npts)*(-0.1)
s0_control[...,2][:,3] += np.ones(npts)*(-0.15)

surfs_new += [NURBS(surfs[0].knots, s0_control)]

new_knots0 = np.array([0.125, 0.25, 0.375, 0.625, 0.75, 0.875])
surfs_new[0].refine(0, new_knots0)
surfs_new[0].refine(1, new_knots0)

s1_control = surfs[1].control
s1_control[...,0][1] += np.ones(npts)*(-0.3)
s1_control[...,0][2] += np.ones(npts)*(-0.2)
s1_control[...,0][3] += np.ones(npts)*(0.1)
s1_control[...,0][:,0] += np.ones(npts)*(-0.1)
s1_control[...,0][:,3] += np.ones(npts)*(0.1)
surfs_new += [NURBS(surfs[1].knots, s1_control)]

new_knots1 = np.array([0.16666667, 0.33333333, 0.66666667, 0.83333333])
surfs_new[1].refine(0, new_knots1)
surfs_new[1].refine(1, new_knots1)

for i in range(len(num_els)):
    vtk_writer.write('./nm_geom/surf'+str(i)+'_init.vtk', surfs_new[i], ref_level=5)
    vtk_writer.write_cp('./nm_geom/surf'+str(i)+'_cp_init.vtk', surfs_new[i])


occ_surfs = []
for i in range(len(num_els)):
    occ_surfs += [ikNURBS2BSpline_surface(surfs_new[i])]
write_geom_file(occ_surfs, './nm_geom/geom.igs')

# Geometry preprocessing and surface-surface intersections computation
preprocessor = OCCPreprocessing(occ_surfs, reparametrize=False, refine=False)
# preprocessor.compute_intersections(mortar_refine=2)
preprocessor.compute_intersections(mortar_nels=[8])

mesh_para0 = generate_mortar_mesh(data=preprocessor.intersections_para_coords[0][0])
mesh_para1 = generate_mortar_mesh(data=preprocessor.intersections_para_coords[0][1])
File('./nm_geom/mortar_mesh0.pvd') << mesh_para0
File('./nm_geom/mortar_mesh1.pvd') << mesh_para1
mesh_phy = generate_mortar_mesh(preprocessor.intersections_phy_coords[0], num_el=128)
File('./nm_geom/int_curve.pvd') << mesh_phy
mesh_phy = generate_mortar_mesh(data=preprocessor.intersections_phy_coords[0])
File('./nm_geom/int_curve_pts.pvd') << mesh_phy

para_mortar_mesh = generate_mortar_mesh(np.array([[0,0,0], [1,0,0]]), num_el=8)
File('./nm_geom/para_mortar_mesh.pvd') << para_mortar_mesh


splines = []
for i in range(len(surfs)):
        spline = ikSurf2tIGArSpline(surfs_new[i])
        splines += [spline,]
        File("./nm_geom/para_mesh"+str(i)+".pvd") << splines[i].mesh