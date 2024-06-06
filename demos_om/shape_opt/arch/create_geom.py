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

def create_arc_surf_new(num_el0, num_el1, p, R, angle_lim0, angle_lim1, y_lim):
    """
    Create igakit NURBS instances
    """
    knots0 = np.linspace(0,1,num_el0+1)[1:-1]
    knots1 = np.linspace(0,1,num_el1+1)[1:-1]

    angle0 = (math.radians(angle_lim0[0]), math.radians(angle_lim0[1]))
    angle1 = (math.radians(angle_lim1[0]), math.radians(angle_lim1[1]))
    C0 = circle(center=[0,0,0], radius=R, angle=angle0)
    C1 = circle(center=[0,0,0], radius=R, angle=angle1)
    C0.rotate(np.pi/2, axis=0)
    C1.rotate(np.pi/2, axis=0)
    C1.translate(y_lim[1], axis=1)
    # L = line([0,y_lim[0],0], [0,y_lim[1],0])
    # srf = sweep(C0, L)
    srf = ruled(C0, C1)
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
    R = 10.
    t = 0.01
    E = 1.0e12
    nu = 0.
    load = 1

    # # Create 9 non-matching patches
    # num_surfs = 9
    # y_lim_list = [0,3,6,9]
    # angle_lim_list = [120,100,80,60]
    # num_el_list = np.random.randint(6,12,num_surfs)

    # ikNURBS_surf_list = []
    # occ_surf_list = []

    # for i in range(len(y_lim_list)-1):
    #     for j in range(len(angle_lim_list)-1):
    #         surf = create_arc_surf(num_el_list[i*(len(y_lim_list)-1)+j], 
    #                                num_el_list[i*(len(y_lim_list)-1)+j], p, R,
    #                                angle_lim=[angle_lim_list[j],
    #                                           angle_lim_list[j+1]],
    #                                y_lim=[y_lim_list[i], 
    #                                       y_lim_list[i+1]])
    #         ikNURBS_surf_list += [surf]

    # Create 3 non-matching arc patches
    num_surfs = 3
    y_lim_list = [0,3]
    # angle_lim_list = [120,100,80,60]
    angle_lim_list0 = [120,100,90,80,60]
    angle_lim_list1 = [120,105,90,75,60]
    # angle_lim_list1 = [120,100,90,80,60]
    # num_el_list = np.random.randint(6,12,num_surfs)
    num_el_list = np.array([7, 6, 7, 6])*1
    # num_el_list = np.array([15, 16, 17, 15])*1

    ikNURBS_surf_list = []
    occ_surf_list = []

    for i in range(len(y_lim_list)-1):
        for j in range(len(angle_lim_list0)-1):
            # surf = create_arc_surf(num_el_list[i*(len(y_lim_list)-1)+j], 
            #                        num_el_list[i*(len(y_lim_list)-1)+j], p, R,
            #                        angle_lim=[angle_lim_list[j],
            #                                   angle_lim_list[j+1]],
            #                        y_lim=[y_lim_list[i],y_lim_list[i+1]])
            surf = create_arc_surf_new(num_el_list[i*(len(y_lim_list)-1)+j], 
                                   num_el_list[i*(len(y_lim_list)-1)+j], p, R,
                                   angle_lim0=[angle_lim_list0[j],
                                              angle_lim_list0[j+1]],
                                   angle_lim1=[angle_lim_list1[j],
                                              angle_lim_list1[j+1]],
                                   y_lim=[y_lim_list[i],y_lim_list[i+1]])
            CP_new = np.zeros(surf.control[...,0:3].shape)
            for field in range(3):
                CP_new[...,field] = surf.control[...,field]/surf.control[...,-1]
            ikNURBS_surf_list += [NURBS(surf.knots, CP_new)]

    z_disp = np.min(ikNURBS_surf_list[0].control[:,:,2])
    x_disp = np.min(ikNURBS_surf_list[0].control[:,:,0])
    for i in range(len(ikNURBS_surf_list)):
        ikNURBS_surf_list[i].move(-x_disp, 0)
        ikNURBS_surf_list[i].move(-z_disp, 2)

    # # Create initial flat non-matching patches
    # y_lim = 3
    # z_lim = 3
    # x_lim_list = [0,3,7,10]

    # pts0 = [[x_lim_list[0], 0, 0], [x_lim_list[1], 0, z_lim],
    #         [x_lim_list[0], y_lim, 0], [x_lim_list[1], y_lim, z_lim]]
    # pts1 = [[x_lim_list[1], 0, z_lim], [x_lim_list[2], 0, z_lim],
    #         [x_lim_list[1], y_lim, z_lim], [x_lim_list[2], y_lim, z_lim]]
    # pts2 = [[x_lim_list[2], 0, z_lim], [x_lim_list[3], 0, 0],
    #         [x_lim_list[2], y_lim, z_lim], [x_lim_list[3], y_lim, 0]]
    # pts_list = [pts0, pts1, pts2]
    # num_el_list = np.array([6, 8, 7])*1

    # ikNURBS_surf_list = []
    # occ_surf_list = []
    # for i in range(3):
    #     ikNURBS_surf_list += [create_surf(pts_list[i], num_el_list[i], 
    #                                       num_el_list[i], p)]

    for i in range(len(ikNURBS_surf_list)):
        occ_surf_list += [ikNURBS2BSpline_surface(ikNURBS_surf_list[i])]
        VTK().write("./geometry/arc_surf"+str(i)+".vtk", ikNURBS_surf_list[i])

    write_geom_file(occ_surf_list, "./geometry/init_arch_geom_ref1.igs")