import math
from igakit.cad import *
from igakit.io import VTK
import igakit.plot as ikplot
import matplotlib.pyplot as plt
from PENGoLINS.occ_utils import *
from PENGoLINS.igakit_utils import *

def create_arc_surf(num_el0, num_el1, p, R, angle_lim, y_lim):
    """
    Create igakit NURBS instances
    """
    knots0 = np.linspace(0,1,num_el0+1)[1:-1]
    knots1 = np.linspace(0,1,num_el1+1)[1:-1]

    angle = (math.radians(angle_lim[0]), math.radians(angle_lim[1]))
    C0 = circle(center=[0,0,0], radius=R, angle=angle)
    C0.rotate(np.pi/2, axis=0)
    L = line([0,y_lim[0],0], [0,y_lim[1],0])
    srf = sweep(C0, L)
    deg0, deg1 = srf.degree 
    srf.elevate(0,p-deg0)
    srf.elevate(1,p-deg1)
    srf.refine(0,knots0)
    srf.refine(1,knots1)
    return srf

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

if __name__ == '__main__':
    # E = Constant(1.0e7)
    # nu = Constant(0.)
    # h_th = Constant(0.1)

    import os
    from os import path
    # if path.exists("./int_data_moved.npz"):
    #     os.remove("./int_data_moved.npz")

    L = 5.
    w = 2.
    h = 2.
    num_field = 3

    pts0 = [[-w/2., 0., 0.], [w/2., 0., 0.],\
            [-w/2., L, 0.], [w/2., L, 0.]]
    # pts1 = [[0., 0., 0.], [0.,0.,-h],\
    #         [0., L, 0.], [0., L, -h]]
    # pts1 = [[w/8.*3., 0., 0.], [w/8.*3.,0.,-h],\
    #         [w/8.*3., L, 0.], [w/8.*3., L, -h]]
    pts1 = [[0.5, 0., 0.], [0.5,0.,-h],\
            [0.5, L, 0.], [0.5, L, -h]]

    p = 3
    num_el0x = 2
    num_el0y = 1
    num_el1x = 1
    num_el1y = 1
    p0 = p
    p1 = p

    print("Creating geometry...")
    srf0_temp = create_surf(pts0, num_el0x, num_el0y, p0)
    srf1_temp = create_surf(pts1, num_el1x, num_el1y, p1)

    srf0_shape = srf0_temp.shape
    srf1_shape = srf1_temp.shape

    srf0_control = srf0_temp.control
    srf1_control = srf1_temp.control

    # ###########################
    # # Sin-like top plate
    # cp0z_min = -0.5
    # cp0z_max = 0.5
    # srf0_control[1][:,2] = np.ones(srf0_shape[1])*cp0z_min
    # srf0_control[3][:,2] = np.ones(srf0_shape[1])*cp0z_max
    # ############################

    ###########################
    # Sin-like top plate
    cp0z_min = 0.2
    cp0z_max = 0.4
    srf0_control[1][:,2] = np.ones(srf0_shape[1])*cp0z_min
    srf0_control[2][:,2] = np.ones(srf0_shape[1])*cp0z_max
    srf0_control[3][:,2] = np.ones(srf0_shape[1])*cp0z_min
    ############################

    srf0 = NURBS(srf0_temp.knots, srf0_control)
    bs0 = ikNURBS2BSpline_surface(srf0)
    para_coord0 = 0.75
    para_coord1 = 0.
    p0 = gp_Pnt()
    bs0.D0(para_coord0, para_coord1, p0)
    p0z_coord = p0.Coord()[2]

    srf1_control[:,:,2] = srf1_control[:,:,2] + p0z_coord
    srf1 = NURBS(srf1_temp.knots, srf1_control)
    ############################

    num_el0x_ref = 5
    num_el0y_ref = 5
    srf0x_ref_knots = np.linspace(0, 1, num_el0x_ref+1)[1:-1]
    srf0y_ref_knots = np.linspace(0, 1, num_el0y_ref+1)[1:-1]

    srf0.refine(0, srf0x_ref_knots)
    srf0.refine(1, srf0y_ref_knots)

    num_el1x_ref = 4
    num_el1y_ref = 4
    srf1x_ref_knots = np.linspace(0, 1, num_el1x_ref+1)[1:-1]
    srf1y_ref_knots = np.linspace(0, 1, num_el1y_ref+1)[1:-1]

    srf1.refine(0, srf1x_ref_knots)
    srf1.refine(1, srf1y_ref_knots)

    ikNURBS_surf_list = [srf0, srf1]

    occ_surf_list = []
    for i in range(len(ikNURBS_surf_list)):
        occ_surf_list += [ikNURBS2BSpline_surface(ikNURBS_surf_list[i])]
        # VTK().write("./geometry/Tbeam_surf"+str(i)+".vtk", ikNURBS_surf_list[i])

    # write_geom_file(occ_surf_list, "./geometry/initial_geometry.igs")
    write_geom_file(occ_surf_list, "./geometry/init_Tbeam_geom_curved.igs")