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
    w = 1.
    L = 2.
    p = 3
    num_surfs = 4
    pts0 = [[0., w, 0.], [w/2, w/2, 0.], [0., w, L/2], [w/2, w/2, L/2]]
    pts1 = [[w/2, w/2, 0.], [w, 0., 0.], [w/2, w/2, L/2], [w, 0., L/2]]
    pts2 = [[0., w, L/2], [w/2, w/2, L/2], [0., w, L], [w/2, w/2, L]]
    pts3 = [[w/2, w/2, L/2], [w, 0., L/2], [w/2, w/2, L], [w, 0., L]]
    pts_list = [pts0,  pts1, pts2, pts3]
    num_el_list = [6, 7, 7, 5]
    ikNURBS_surf_list = []
    occ_surf_list = []
    for i in range(num_surfs):
        ikNURBS_surf_list += [create_surf(pts_list[i], num_el_list[i], 
                                          num_el_list[i], p)]
        occ_surf_list += [ikNURBS2BSpline_surface(ikNURBS_surf_list[i])]
        VTK().write("./geometry/tube_surf"+str(i)+".vtk", ikNURBS_surf_list[i])

    write_geom_file(occ_surf_list, "./geometry/initial_geometry.igs")