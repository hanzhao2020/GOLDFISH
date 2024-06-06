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
    p = 3

    scale = 1e0
    ref_level = 1
    num_surfs = 6
    x_lim_list = np.linspace(0,1,num_surfs+1)*scale
    # x_lim_list = np.array([0.,1.])*scale
    y_lim = np.array([0.,1.])*scale
    z_coord = 0.*scale
    num_el_x = 4*ref_level
    num_el_y = np.array([8,9,10,9,8,7])*ref_level
    # num_el_y = np.array([8,9,10,9])*ref_level
    # # One patch geometry
    # x_lim_list = np.array([0.,1.])*scale
    # num_el_x = 16
    # num_el_y = [16]
    num_surfs = len(x_lim_list)-1

    pts_list = []
    ikNURBS_surf_list = []
    occ_surf_list = []
    for i in range(num_surfs):
        pts_list += [[[x_lim_list[i], y_lim[0], z_coord], 
                      [x_lim_list[i+1], y_lim[0], z_coord], 
                      [x_lim_list[i], y_lim[1], z_coord], 
                      [x_lim_list[i+1], y_lim[+1], z_coord]]]
        ikNURBS_surf_list += [create_surf(pts_list[i], num_el_x,
                                          num_el_y[i], p)]

    for i in range(len(ikNURBS_surf_list)):
        occ_surf_list += [ikNURBS2BSpline_surface(ikNURBS_surf_list[i])]
        VTK().write("./geometry/plate_surf"+str(i)+".vtk", ikNURBS_surf_list[i])

    if p == 2:
        write_geom_file(occ_surf_list, "./geometry/plate_geometry_quadratic.igs")
    elif p == 3:
        # write_geom_file(occ_surf_list, "./geometry/plate_geometry_cubic.igs")
        write_geom_file(occ_surf_list, "./geometry/plate_geometry.igs")
    elif p == 4:
        write_geom_file(occ_surf_list, "./geometry/plate_geometry_quartic.igs")
