import numpy as np
from scipy.optimize import fsolve, newton_krylov
from PENGoLINS.occ_utils import *
from PENGoLINS.igakit_utils import *

######## 1. Create surface 1 and surface 2 ########
num_pts_int = 100  # Number of points to evaluate for intersection curve

# Add random disturbance in (0,1) times parametric element length times
# `para_disturb` to the linear spaced parametric coordinates.
# If `para_disturb` is 0, curve parametric coordinates are linear spaced.
para_disturb = 0.8
p = 3  # Spline degree
L = 10  # Length of spline surface

#### Surface 1 ####
pts1 = [[-L,0,-L/2], [0,0,-L/2],
        [-L,0,L/2], [0,0,L/2]]
# Create distorted NURBS surface
distort_mag1 = 3
c1 = line(pts1[0], pts1[1])
c1.elevate(0,p)
c1.control[1][1] = -distort_mag1
c1.control[3][1] = distort_mag1
t1 = line(pts1[1], pts1[3])
t1.elevate(0,p)
t1.control[1][1] = -distort_mag1
t1.control[3][1] = distort_mag1
ik_surf1 = sweep(c1, t1)
ik_surf1.move(L/2, axis=2)

#### Surface 2 ####
pts2 = [[-L/2,-L/2,0], [L/2,-L/2,0],
        [-L/2,L/2,0], [L/2,L/2,0]]
distort_mag2 = 2
c2 = line(pts2[0], pts2[1])
c2.elevate(0,p)
c2.control[1][2] = -distort_mag2
c2.control[3][2] = distort_mag2
t2 = line(pts2[1], pts2[3])
t2.elevate(0,p)
t2.control[1][2] = -distort_mag2
t2.control[3][2] = distort_mag2
ik_surf2 = sweep(c2, t2)
ik_surf2.move(-8, axis=0)
ik_surf2.move(6, axis=1)

# Convert igakit NURBS instance to OCC BSplineSurface
surf1 = ikNURBS2BSpline_surface(ik_surf1)
surf2 = ikNURBS2BSpline_surface(ik_surf2)

####### 2. Compute intersections by PythonOCC ########
surf_int = GeomAPI_IntSS(surf1, surf2, 1e-6)
int_curve = surf_int.Line(1)

# Display surfaces and intersection
# display, start_display, add_menu, add_function_to_menu = init_display()
# display.DisplayShape(make_face(surf1, 1e-6))
# display.DisplayShape(make_face(surf2, 1e-6))
# display.DisplayShape(int_curve, color='RED')
# exit()

######## 3. Get points on two ends of intersection ########
# Parametric coordinate of left (first) end for intersection curve
s0 = int_curve.FirstParameter()
sn = int_curve.LastParameter()
# Create equally spaced parametric coordinates for intersection curve
int_curve_para = np.linspace(s0, sn, num_pts_int)
# Add small disturbance to the initial parametric coordinates
# on intersection curve
int_curve_para_disturb = int_curve_para.copy()
int_curve_para_disturb[1:-1] = int_curve_para_disturb[1:-1] \
    + np.random.random(num_pts_int-2)/(num_pts_int-1)*para_disturb
# Corresponding physical coordinates on intersection curve
int_curve_phy = np.zeros((num_pts_int, 3))
pt_temp = gp_Pnt()
for i in range(num_pts_int):
    int_curve.D0(int_curve_para_disturb[i], pt_temp)
    int_curve_phy[i] = pt_temp.Coord()
# Corresponding physical coordinates on two surfaces
int_para_loc_surf1 = parametric_coord(int_curve_phy, surf1)
int_para_loc_surf2 = parametric_coord(int_curve_phy, surf2)

######## 4. Solve coupled system for interior points ########
def int_uv_coords(x, surf1=surf1, surf2=surf2, 
                  uv0_surf1=int_para_loc_surf1[0], 
                  uvn_surf1=int_para_loc_surf1[-1]):
    """
    Returns residuals of coupled system when solving interior
    parametric coordinates for intersection curve

    Parameters
    ----------
    x : ndarray, size: (num_pts_int-2)*4
        x = [u_1^1, v_1^1, u_2^1, v_2^1, ..., u_{n-1}^1, v_{n-1}^1,
             u_1^2, v_1^2, u_2^2, v_2^2, ..., u_{n-1}^2, v_{n-1}^2]
        Subscript indicates interior index, {1, n-1}
        Superscript means surface index, {1, 2}
    surf1 : OCC BSplineSurface
    surf2 : OCC BSplineSurface
    uv0_surf1 : ndarray
        Parametric coordinate of point corresponds to the 
        first parameter of intersection curve on surface 1
    uvn_surf1 : ndarray
        Parametric coordinate of point corresponds to the 
        last parameter of intersection curve on surface 1

    Returns
    -------
    res : ndarray: size: (num_pts_int-2)*4
    """
    # print("Calling function evaluation ....")
    x_coords = x.reshape(-1,2)
    num_pts_interior = int(x.size/4)
    res = np.zeros(x.size)

    # Enforce each pair of parametric points from two surfaces
    # have the same physical location.
    pt_temp1 = gp_Pnt()
    pt_temp2 = gp_Pnt()
    ind_off1 = num_pts_interior

    for i in range(num_pts_interior):
        surf1.D0(x_coords[i,0], x_coords[i,1], pt_temp1)
        surf2.D0(x_coords[i+ind_off1,0], x_coords[i+ind_off1,1], pt_temp2)
        res[i*3:(i+1)*3] = np.array(pt_temp1.Coord()) \
                         - np.array(pt_temp2.Coord())

    # Enforce two adjacent elements has the same magnitude 
    # in physical space for surface 1.
    pt_temp1 = gp_Pnt()
    pt_temp2 = gp_Pnt()
    pt_temp3 = gp_Pnt()
    ind_off2 = num_pts_interior*3

    for i in range(num_pts_interior*3, num_pts_interior*4):
        # For the first two elements
        if i == num_pts_interior*3:
            surf1.D0(uv0_surf1[0], uv0_surf1[1], pt_temp1)
            surf1.D0(x_coords[i-ind_off2,0],
                     x_coords[i-ind_off2,1], pt_temp2)
            surf1.D0(x_coords[i+1-ind_off2,0], 
                     x_coords[i+1-ind_off2,1], pt_temp3)
        # For the last two elements
        elif i == num_pts_interior*4-1:
            surf1.D0(x_coords[i-1-ind_off2,0], 
                     x_coords[i-1-ind_off2,1], pt_temp1)
            surf1.D0(x_coords[i-ind_off2,0], 
                     x_coords[i-ind_off2,1], pt_temp2)
            surf1.D0(uvn_surf1[0], uvn_surf1[1], pt_temp3)
        # For interior elements
        else:
            surf1.D0(x_coords[i-1-ind_off2,0], 
                     x_coords[i-1-ind_off2,1], pt_temp1)
            surf1.D0(x_coords[i-ind_off2,0],
                     x_coords[i-ind_off2,1], pt_temp2)
            surf1.D0(x_coords[i+1-ind_off2,0], 
                     x_coords[i+1-ind_off2,1], pt_temp3)

        # res[i] = np.linalg.norm(np.array(pt_temp1.Coord())
        #                       - np.array(pt_temp2.Coord())) \
        #        - np.linalg.norm(np.array(pt_temp2.Coord())
        #                       - np.array(pt_temp3.Coord()))

        diff1 = np.array(pt_temp1.Coord()) - np.array(pt_temp2.Coord())
        diff2 = np.array(pt_temp2.Coord()) - np.array(pt_temp3.Coord())
        res[i] = np.dot(diff1, diff1) - np.dot(diff2, diff2)

    return res

# Create initial guess.
uv_interior = np.concatenate([int_para_loc_surf1[1:-1], 
                              int_para_loc_surf2[1:-1]], axis=0)
# Initial guess
x0 = uv_interior.reshape(-1,1)[:,0]
print("Solving parametric coordinates ...")
uv_root = fsolve(int_uv_coords, x0=x0)
# uv_root = newton_krylov(int_uv_coords, xin=x0_disturb, 
#                         maxiter=1000, f_tol=1e-12)

######## 5. Check solution ########
# Check function residual after solve
uv_res = int_uv_coords(uv_root)
print("Residual after solve:", np.linalg.norm(uv_res))

# Check element length after solve
# Element length on surface 1
int_el_length_list1 = np.zeros(num_pts_int-1)
pt_temp1 = gp_Pnt()
pt_temp2 = gp_Pnt()
uv_root_coor1 = uv_root.reshape(-1,2)[0:num_pts_int-2]
uv_root_coor1 = np.concatenate([int_para_loc_surf1[0].reshape(1,2), 
                uv_root_coor1, int_para_loc_surf1[-1].reshape(1,2)], axis=0)
for i in range(num_pts_int-1):
    surf1.D0(uv_root_coor1[i,0], uv_root_coor1[i,1], pt_temp1)
    surf1.D0(uv_root_coor1[i+1,0], uv_root_coor1[i+1,1], pt_temp2)
    int_el_length_list1[i] = np.linalg.norm(np.array(pt_temp1.Coord()) 
                          - np.array(pt_temp2.Coord()))
# Element length on surface 2
int_el_length_list2 = np.zeros(num_pts_int-1)
uv_root_coor2 = uv_root.reshape(-1,2)[num_pts_int-2:]
uv_root_coor2 = np.concatenate([int_para_loc_surf2[0].reshape(1,2), 
                uv_root_coor2, int_para_loc_surf2[-1].reshape(1,2)], axis=0)
for i in range(num_pts_int-1):
    surf2.D0(uv_root_coor2[i,0], uv_root_coor2[i,1], pt_temp1)
    surf2.D0(uv_root_coor2[i+1,0], uv_root_coor2[i+1,1], pt_temp2)
    int_el_length_list2[i] = np.linalg.norm(np.array(pt_temp1.Coord()) 
                          - np.array(pt_temp2.Coord()))

int_length = curve_length(int_curve)
# Exact element length for intersection curve
int_el_length = int_length/(num_pts_int-1)

print("Exact element length:", int_el_length)
print("Element length of intersection on surface 1 "
      "after solve:\n", int_el_length_list1)
print("Element length of intersection on surface 2 "
      "after solve:\n", int_el_length_list2)

int_pt_list1 = []
int_pt_list2 = []
for i in range(num_pts_int):
    pt1 = gp_Pnt()
    pt2 = gp_Pnt()
    surf1.D0(uv_root_coor1[i,0], uv_root_coor1[i,1], pt1)
    surf2.D0(uv_root_coor2[i,0], uv_root_coor2[i,1], pt2)
    int_pt_list1 += [pt1,]
    int_pt_list2 += [pt2,]

# Display exact intersection and computed intersection points
display, start_display, add_menu, add_function_to_menu = init_display()
display.DisplayShape(make_face(surf1, 1e-6))
display.DisplayShape(make_face(surf2, 1e-6))
display.DisplayShape(int_curve, color='RED')
for i in range(num_pts_int):
    display.DisplayShape(int_pt_list1[i], color='BLUE')
    display.DisplayShape(int_pt_list2[i], color='GREEN')