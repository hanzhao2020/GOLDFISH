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

    w = 1.
    L = 4.
    p = 3
    h = 0.1
    pts0 = [[0., w, 0.], [w/2+h, w/2, 0.], 
            [0., w, L/2], [w/2+h, w/2, L/2]]
    pts1 = [[w/2, w/2+h, 0.], [w, 0., 0.], 
            [w/2, w/2+h, L/2], [w, 0., L/2]]

    num_el0x = 6
    num_el0y = 8
    num_el1x = 6
    num_el1y = 7

    p = 3
    p0 = p
    p1 = p
    p2 = p

    print("Creating geometry...")
    srf0 = create_surf(pts0, num_el0x, num_el0y, p0)
    srf1 = create_surf(pts1, num_el1x, num_el1y, p1)

    ikNURBS_surf_list = [srf0, srf1]

    occ_surf_list = []
    for i in range(len(ikNURBS_surf_list)):
        occ_surf_list += [ikNURBS2BSpline_surface(ikNURBS_surf_list[i])]
        # VTK().write("./geometry/Tbeam_surf"+str(i)+".vtk", ikNURBS_surf_list[i])

    write_geom_file(occ_surf_list, "./geometry/init_tube_geom_2patch.igs")