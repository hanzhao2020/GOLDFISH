from PENGoLINS.occ_preprocessing import *
from PENGoLINS.igakit_utils import *

# def reparametrize_BSpline_surfaces(occ_bs_surfs, u_num_eval=4, v_num_eval=1, 
#                                    bs_degree=3, bs_continuity=3, connect_dir=0,
#                                    tol3D=1e-2, geom_scale=1., 
#                                    remove_dense_knots=False, 
#                                    dist_ratio_remove=0.5, rtol=1e-2):
#     """
#     Return reparametrized B-spline surface by evaluating the positions
#     of the original surface.

#     Parameters
#     ----------
#     occ_bs_surf : OCC BSplineSurface
#     u_num_eval : int, optional
#         The number of points evaluated in the u-direction. Default is 30.
#     v_num_eval : int, optional
#         The number of points evaluated in the v-direction. Default is 30.
#     bs_degree : int, optional, default is 3.
#     bs_continuity : int, optional, default is 3.
#     tol3D : float, optional, default is 1e-3.
#     geom_scale : float, optional, default is 1.0.

#     Returns
#     -------
#     occ_bs_res : occ Geom BSplineSurface
#     """
#     u_num_eval_max = u_num_eval
#     v_num_eval_max = v_num_eval

#     num_surfs = len(occ_bs_surfs)

#     occ_bs_res_pts_list = []

#     for surf_ind in range(num_surfs):
#         # for num_iter in range(int(u_num_eval)):
#         u_knot_range = occ_bs_surfs[surf_ind].Bounds()[1] \
#                      - occ_bs_surfs[surf_ind].Bounds()[0]
#         v_knot_range = occ_bs_surfs[surf_ind].Bounds()[3] \
#                      - occ_bs_surfs[surf_ind].Bounds()[2]
#         if surf_ind == 0:
#             u_bound_off = 0
#             v_bound_off = 0
#         else:
#             if connect_dir == 0:
#                 u_bound_off = u_knot_range/u_num_eval
#                 v_bound_off = 0
#             elif connect_dir == 1: 
#                 u_bound_off = 0
#                 v_bound_off = u_knot_range/v_num_eval
#         # print('u_off:', u_bound_off)
#         # print('v_off:', v_bound_off)

#         occ_bs_res_pts = TColgp_Array2OfPnt(1, u_num_eval, 1, v_num_eval)
#         para_u = np.linspace(occ_bs_surfs[surf_ind].Bounds()[0]+u_bound_off, 
#                              occ_bs_surfs[surf_ind].Bounds()[1], 
#                              u_num_eval)
#         para_v = np.linspace(occ_bs_surfs[surf_ind].Bounds()[2]+v_bound_off, 
#                              occ_bs_surfs[surf_ind].Bounds()[3], 
#                              v_num_eval)
#         pt_temp = gp_Pnt()
#         for i in range(u_num_eval):
#             for j in range(v_num_eval):
#                 occ_bs_surfs[surf_ind].D0(para_u[i], para_v[j], pt_temp)
#                 pt_temp0 = gp_Pnt(pt_temp.Coord()[0]*geom_scale, 
#                                   pt_temp.Coord()[1]*geom_scale, 
#                                   pt_temp.Coord()[2]*geom_scale)
#                 occ_bs_res_pts.SetValue(i+1, j+1, pt_temp0)
#         occ_bs_res_pts_list += [occ_bs_res_pts]

#     if connect_dir == 0:
#         u_size = u_num_eval*num_surfs
#         v_size = v_num_eval
#     elif connect_dir == 1:
#         u_size = u_num_eval
#         v_size = v_num_eval*num_surfs

#     occ_bs_res_pts_global = TColgp_Array2OfPnt(1, u_size, 1, v_size)

#     ind_off = 0    
#     for surf_ind in range(num_surfs):
#         for i in range(u_num_eval):
#             for j in range(v_num_eval):
#                 pt_value = occ_bs_res_pts_list[surf_ind].Value(i+1,j+1)
#                 # print("coord value:", pt_value.Coord())
#                 if connect_dir == 0:
#                     # print('i:', i, '-j:', j, '-ind_off:', ind_off)
#                     occ_bs_res_pts_global.SetValue(i+1+ind_off, j+1, pt_value)
#                 elif connect_dir == 1:
#                     # print('i:', i, '-j:', j, '-ind_off:', ind_off)
#                     occ_bs_res_pts_global.SetValue(i+1, j+1+ind_off, pt_value)
#         if connect_dir == 0:
#             ind_off += u_num_eval
#         elif connect_dir == 1:
#             ind_off += v_num_eval

#     # return occ_bs_res_pts_global

#     occ_bs_res_temp = GeomAPI_PointsToBSplineSurface(occ_bs_res_pts_global, 
#                     Approx_ParametrizationType(0), bs_degree, bs_degree, 
#                     bs_continuity, tol3D)

#     occ_bs_res = occ_bs_res_temp.Surface()

#     # Check if surface has excessive interior u and v knots
#     decrease_knot_multiplicity(occ_bs_res, rtol)
#     # Remove densely distributed knots
#     if remove_dense_knots:
#         remove_surf_dense_knots(occ_bs_res, dist_ratio_remove, rtol)

#     # return occ_bs_res_pts_global, occ_bs_res
#     return occ_bs_res


def reparametrize_BSpline_surfaces_new(occ_bs_surfs, u_num_eval=3, v_num_eval=8, 
                                   bs_degree=3, bs_continuity=3, connect_dir=0,
                                   tol3D=1e-3, geom_scale=1., 
                                   remove_dense_knots=False, 
                                   dist_ratio_remove=0.5, rtol=1e-2):
    """
    Return reparametrized B-spline surface by evaluating the positions
    of the original surface.

    Parameters
    ----------
    occ_bs_surf : OCC BSplineSurface
    u_num_eval : int, optional
        The number of points evaluated in the u-direction. Default is 30.
    v_num_eval : int, optional
        The number of points evaluated in the v-direction. Default is 30.
    bs_degree : int, optional, default is 3.
    bs_continuity : int, optional, default is 3.
    tol3D : float, optional, default is 1e-3.
    geom_scale : float, optional, default is 1.0.

    Returns
    -------
    occ_bs_res : occ Geom BSplineSurface
    """
    u_num_eval_max = u_num_eval
    v_num_eval_max = v_num_eval

    num_surfs = len(occ_bs_surfs)

    occ_bs_res_pts_list = []

    for surf_ind in range(num_surfs):
        # for num_iter in range(int(u_num_eval)):
        u_knot_range = occ_bs_surfs[surf_ind].Bounds()[1] \
                     - occ_bs_surfs[surf_ind].Bounds()[0]
        v_knot_range = occ_bs_surfs[surf_ind].Bounds()[3] \
                     - occ_bs_surfs[surf_ind].Bounds()[2]
        if surf_ind == 0:
            u_start_ind = 0
            v_start_ind = 0
        else:
            if connect_dir == 0:
                u_start_ind = 1
                v_start_ind = 0
            elif connect_dir == 1: 
                u_start_ind = 0
                v_start_ind = 1
        # print('u_off:', u_bound_off)
        # print('v_off:', v_bound_off)

        para_u = np.linspace(occ_bs_surfs[surf_ind].Bounds()[0], 
                             occ_bs_surfs[surf_ind].Bounds()[1], 
                             u_num_eval)[u_start_ind:]
        para_v = np.linspace(occ_bs_surfs[surf_ind].Bounds()[2], 
                             occ_bs_surfs[surf_ind].Bounds()[3], 
                             v_num_eval)[v_start_ind:]

        occ_bs_res_pts = TColgp_Array2OfPnt(1, para_u.size, 1, para_v.size)

        pt_temp = gp_Pnt()
        for i in range(para_u.size):
            for j in range(para_v.size):
                # print('para_u:', para_u)
                # print('para_v:', para_v)
                occ_bs_surfs[surf_ind].D0(para_u[i], para_v[j], pt_temp)
                pt_temp0 = gp_Pnt(pt_temp.Coord()[0]*geom_scale, 
                                  pt_temp.Coord()[1]*geom_scale, 
                                  pt_temp.Coord()[2]*geom_scale)
                occ_bs_res_pts.SetValue(i+1, j+1, pt_temp0)
        occ_bs_res_pts_list += [occ_bs_res_pts]

    u_size_list = [bs_pts.NbRows() for bs_pts in occ_bs_res_pts_list]
    v_size_list = [bs_pts.NbColumns()  for bs_pts in occ_bs_res_pts_list]
    if connect_dir == 0:
        u_size = int(np.sum(u_size_list))
        v_size = v_num_eval
    elif connect_dir == 1:
        u_size = u_num_eval
        v_size = int(np.sum(v_size_list))
    # print(u_size)
    # print(v_size)
    occ_bs_res_pts_global = TColgp_Array2OfPnt(1, u_size, 1, v_size)

    ind_off = 0    
    for surf_ind in range(num_surfs):
        for i in range(occ_bs_res_pts_list[surf_ind].NbRows()):
            for j in range(occ_bs_res_pts_list[surf_ind].NbColumns()):
                pt_value = occ_bs_res_pts_list[surf_ind].Value(i+1,j+1)
                # print("coord value:", pt_value.Coord())
                if connect_dir == 0:
                    # print('i:', i, '-j:', j, '-ind_off:', ind_off)
                    occ_bs_res_pts_global.SetValue(i+1+ind_off, j+1, pt_value)
                elif connect_dir == 1:
                    # print('i:', i, '-j:', j, '-ind_off:', ind_off)
                    occ_bs_res_pts_global.SetValue(i+1, j+1+ind_off, pt_value)
        if connect_dir == 0:
            ind_off += occ_bs_res_pts_list[surf_ind].NbRows()
        elif connect_dir == 1:
            ind_off += occ_bs_res_pts_list[surf_ind].NbColumns()

    # return occ_bs_res_pts_global

    occ_bs_res_temp = GeomAPI_PointsToBSplineSurface(occ_bs_res_pts_global, 
                    Approx_ParametrizationType(0), bs_degree, bs_degree, 
                    bs_continuity, tol3D)

    occ_bs_res = occ_bs_res_temp.Surface()

    # Check if surface has excessive interior u and v knots
    decrease_knot_multiplicity(occ_bs_res, rtol)
    # Remove densely distributed knots
    if remove_dense_knots:
        remove_surf_dense_knots(occ_bs_res, dist_ratio_remove, rtol)

    # return occ_bs_res_pts_global, occ_bs_res
    return occ_bs_res

geom_scale = 2.54e-5 

print("Importing geometry...")
# filename_igs = "./geometry/pegasus_wing.iges"
filename_igs = "./geometry/eVTOL_wing_structure.igs"
igs_shapes = read_igs_file(filename_igs, as_compound=False)
evtol_surfaces = [topoface2surface(face, BSpline=True) 
                  for face in igs_shapes]

# Outer skin indices: list(range(12, 18))
# Spars indices: [78, 92, 79]
# Ribs indices: list(range(80, 92))
wing_indices = list(range(12, 18)) + [78, 92, 79]  + list(range(80, 92))
# wing_indices = list(range(12, 18)) + [78, 79]  + [80, 86]
wing_surfaces = [evtol_surfaces[i] for i in wing_indices]
num_surfs = len(wing_surfaces)
# if mpirank == 0:
#     print("Number of surfaces:", num_surfs)

# for i in range(len(wing_surfaces)):
#     ik_surf = BSpline_surface2ikNURBS(wing_surfaces[i])
#     VTK().write("./geometry/surf"+str(i)+".vtk", ik_surf)

# print(aaa)

wing_bs_data = [BSplineSurfaceData(wing_surf) for wing_surf in wing_surfaces]

lower_skin_cp0 = wing_bs_data[0].control
lower_skin_cp1 = wing_bs_data[2].control[1:,:,:]
lower_skin_cp = np.concatenate([lower_skin_cp0, lower_skin_cp1], axis=0)
lower_skin_knots = [np.array([0,0,0.5,1,1]), wing_bs_data[0].knots[1]]

upper_skin_cp0 = wing_bs_data[1].control
upper_skin_cp1 = wing_bs_data[3].control[1:,:,:]
upper_skin_cp = np.concatenate([upper_skin_cp0, upper_skin_cp1], axis=0)
upper_skin_knots = [np.array([0,0,0.5,1,1]), wing_bs_data[1].knots[1]]

wing_tip_cp0 = wing_bs_data[4].control
wing_tip_cp1 = wing_bs_data[5].control[::-1,::-1,:][1:,:,:]
wing_tip_cp = np.concatenate([wing_tip_cp0, wing_tip_cp1], axis=0)
wing_tip_knots = [np.array([0,0,0.5,1,1]), wing_bs_data[4].knots[1]]

lower_skin_surf_ik = NURBS(lower_skin_knots, lower_skin_cp)
upper_skin_surf_ik = NURBS(upper_skin_knots, upper_skin_cp)
wing_tip_surf_ik = NURBS(wing_tip_knots, wing_tip_cp)

lower_skin = ikNURBS2BSpline_surface(lower_skin_surf_ik)
upper_skin = ikNURBS2BSpline_surface(upper_skin_surf_ik)
wing_tip = ikNURBS2BSpline_surface(wing_tip_surf_ik)

# lower_skins = [wing_surfaces[i] for i in [0,2]]
# lower_skin = reparametrize_BSpline_surfaces_new(lower_skins, 
#                         connect_dir=0, tol3D=1e-2, 
#                         geom_scale=1)

# upper_skins = [wing_surfaces[i] for i in [1,3]]
# upper_skin = reparametrize_BSpline_surfaces_new(upper_skins, 
#                         connect_dir=0, tol3D=1e-2, 
#                         geom_scale=1)

# display, start_display, add_menu, add_function_to_menu = init_display()
# # display.DisplayShape(lower_skin)
# # display.DisplayShape(upper_skin)
# display.DisplayShape(wing_tip)
# print(aaa)

# wing_tips = [wing_surfaces[i] for i in [4,5]]
# wing_tip = reparametrize_BSpline_surfaces_new(wing_tips, 
#                         connect_dir=1, tol3D=1e-3, 
#                         geom_scale=geom_scale)


# evtol_wing_surfaces = [lower_skin, upper_skin, wing_tip] \
#                     + [wing_surfaces[6] ,wing_surfaces[8]]

# # rib_inds = [9, 12, 14, 16, 18, 20]
# # rib_inds = [9, 14, 18]
# rib_inds = []
# num_ribs = len(rib_inds)
# # evtol_wing_surfaces += [wing_surfaces[i] for i in rib_inds]

evtol_wing_surfaces = [lower_skin, upper_skin, wing_tip]

spar_inds = [6, 8] #[6,7,8]
num_spars = len(spar_inds)
evtol_wing_surfaces += [wing_surfaces[i] for i in spar_inds]

# rib_inds = [9, 12, 14, 16, 18, 20]
rib_inds = [9, 14, 18]
# rib_inds = []
num_ribs = len(rib_inds)
evtol_wing_surfaces += [wing_surfaces[i] for i in rib_inds]

# # PythonOCC build-in 3D viewer.
# display, start_display, add_menu, add_function_to_menu = init_display()
# display.DisplayShape(upper_skin)
# display.DisplayShape(lower_skin)
# display.DisplayShape(front_spar)
# display.DisplayShape(rear_spar)

# raise RuntimeError()

num_pts_eval = [12]*num_surfs
num_pts_eval[0] = 8
num_pts_eval[1] = 8
num_pts_eval[3] = 6
num_pts_eval[4] = 6
ref_level_list = [1]*num_surfs

# Meshes that are close to the results in the paper
u_insert_list = [8, 8, 1] \
              + [16, 18] + [4]*num_ribs
v_insert_list = [6, 6, 8] \
              + [1]*2 + [1]*num_ribs
# # Meshes with equal numbers of knot insertion, this scheme
# # has slightly smaller QoI due to skinny elements at wingtip
# u_insert_list = [8]*num_surfs
# v_insert_list = [8]*num_surfs

# For the two small NURBS patches at the wingtip, we control the
# refinement level less than 3 to prevent over refinement.
wing_tip_ind = 2
if ref_level_list[wing_tip_ind] > 4:
    ref_level_list[wing_tip_ind] = 2
elif ref_level_list[wing_tip_ind] <=4 and ref_level_list[wing_tip_ind] >= 1:
    ref_level_list[wing_tip_ind] = 1

u_num_insert = []
v_num_insert = []
for i in range(len(u_insert_list)):
    u_num_insert += [ref_level_list[i]*u_insert_list[i]]
    v_num_insert += [ref_level_list[i]*v_insert_list[i]]

p = 3

# Geometry preprocessing and surface-surface intersections computation
preprocessor = OCCPreprocessing(evtol_wing_surfaces, reparametrize=True, 
                                refine=True)
preprocessor.reparametrize_BSpline_surfaces(num_pts_eval, num_pts_eval,
                                            bs_degrees=p, bs_continuities=p,
                                            geom_scale=geom_scale,
                                            remove_dense_knots=True,
                                            rtol=1e-4)
preprocessor.refine_BSpline_surfaces(p, p, u_num_insert, v_num_insert, 
                                     correct_element_shape=True)

bs_data_repara = [BSplineSurfaceData(surf) for surf in preprocessor.BSpline_surfs_repara]
bs_data_refine = [BSplineSurfaceData(surf) for surf in preprocessor.BSpline_surfs_refine]
total_cp = 0
for i in range(3, len(bs_data_repara)):
    total_cp += bs_data_repara[i].control.shape[0]*bs_data_repara[i].control.shape[1]

# for i in range(preprocessor.num_surfs):
#     ik_surf = BSpline_surface2ikNURBS(preprocessor.BSpline_surfs_repara[i])
#     VTK().write("./geometry/wing_surf"+str(i)+".vtk", ik_surf)
# raise RuntimeError


# if mpirank == 0:
#     print("Computing intersections...")
# int_data_filename = "pegasus_wing_int_data.npz"
# if os.path.isfile(int_data_filename):
#     preprocessor.load_intersections_data(int_data_filename)
# else:
#     preprocessor.compute_intersections(rtol=1e-6, mortar_refine=2, 
#                                        edge_rel_ratio=1e-3)
#     preprocessor.save_intersections_data(int_data_filename)

preprocessor.compute_intersections(rtol=1e-6, mortar_refine=1, 
                                   edge_rel_ratio=1e-3)

# if mpirank == 0:
#     print("Total DoFs:", preprocessor.total_DoFs)
#     print("Number of intersections:", preprocessor.num_intersections_all)

# # Display B-spline surfaces and intersections using 
# # PythonOCC build-in 3D viewer.
# display, start_display, add_menu, add_function_to_menu = init_display()
# preprocessor.display_surfaces(display, transparency=0.5, show_bdry=False, save_fig=False)
# preprocessor.display_intersections(display, color='RED', save_fig=False)

int_types = preprocessor.check_intersections_type()
mapping_list = preprocessor.mapping_list
wing_surf_names = ['skin_l', 'skin_u', 'tip',
                   'spar_f', 'spar_r']
wing_surf_names += ['rib_'+str(i) for i in range(num_ribs)]

int_types_new = []
int_xi_type = ['xi0.0', 'xi0.1', 'xi1.0', 'xi1.1']

for int_ind in range(preprocessor.num_intersections_all):
    s_ind0, s_ind1 = mapping_list[int_ind]
    s_name0 = wing_surf_names[s_ind0]
    s_name1 = wing_surf_names[s_ind1]
    int_type = int_types[int_ind][0]
    int_xi = int_types[int_ind][1]
    int_para_coord0 = preprocessor.intersections_para_coords[int_ind][0]
    int_para_coord1 = preprocessor.intersections_para_coords[int_ind][1]
    if 'skin' in s_name0:
        if 'skin' in s_name1 or 'tip' in s_name1: # Requires to be 'edge-edge'
            if int_type == 'surf-surf':
                int_type_new = 'edge-edge'
                err_norm0 = [np.linalg.norm(int_para_coord0[:,0]),
                             np.linalg.norm(int_para_coord0[:,0]
                             -np.ones(int_para_coord0[:,0].size)),
                             np.linalg.norm(int_para_coord0[:,1]),
                             np.linalg.norm(int_para_coord0[:,1]
                             -np.ones(int_para_coord0[:,1].size))]
                xi0_type = int_xi_type[np.argmin(err_norm0)]
                err_norm1 = [np.linalg.norm(int_para_coord1[:,0]),
                             np.linalg.norm(int_para_coord1[:,0]
                             -np.ones(int_para_coord1[:,0].size)),
                             np.linalg.norm(int_para_coord1[:,1]),
                             np.linalg.norm(int_para_coord1[:,1]
                             -np.ones(int_para_coord1[:,1].size))]
                xi1_type = int_xi_type[np.argmin(err_norm1)]
                int_xi_new = xi0_type+'-'+xi1_type
            elif int_type == 'surf-edge':
                int_type_new = 'edge-edge'
                err_norm0 = [np.linalg.norm(int_para_coord0[:,0]),
                             np.linalg.norm(int_para_coord0[:,0]
                             -np.ones(int_para_coord0[:,0].size)),
                             np.linalg.norm(int_para_coord0[:,1]),
                             np.linalg.norm(int_para_coord0[:,1]
                             -np.ones(int_para_coord0[:,1].size))]
                xi0_type = int_xi_type[np.argmin(err_norm0)]
                int_xi_new = xi0_type+'-'+int_xi.split('-')[1]
            elif int_type == 'edge-surf':
                int_type_new = 'edge-edge'
                err_norm1 = [np.linalg.norm(int_para_coord1[:,0]),
                             np.linalg.norm(int_para_coord1[:,0]
                             -np.ones(int_para_coord1[:,0].size)),
                             np.linalg.norm(int_para_coord1[:,1]),
                             np.linalg.norm(int_para_coord1[:,1]
                             -np.ones(int_para_coord1[:,1].size))]
                xi1_type = int_xi_type[np.argmin(err_norm1)]
                int_xi_new = int_xi.split('-')[0]+'-'+xi1_type
            else:
                int_type_new = int_type
                int_xi_new = int_xi
        if 'spar' in s_name1 or 'rib' in s_name1: # Requires to be 'surf-edge'
            if int_type == 'surf-surf':
                int_type_new = 'surf-edge'
                err_norm1 = [np.linalg.norm(int_para_coord1[:,0]),
                             np.linalg.norm(int_para_coord1[:,0]
                             -np.ones(int_para_coord1[:,0].size)),
                             np.linalg.norm(int_para_coord1[:,1]),
                             np.linalg.norm(int_para_coord1[:,1]
                             -np.ones(int_para_coord1[:,1].size))]
                xi1_type = int_xi_type[np.argmin(err_norm1)]
                int_xi_new = int_xi.split('-')[0]+'-'+xi1_type
            # elif int_type == 'edge-surf':
            #     int_type_new = 'surf-edge'
            #     pass
            else:
                int_type_new = int_type
                int_xi_new = int_xi
    elif 'spar' in s_name0:
        # if 'rib' in s_name1: # Requires to be 'surf-edge'
        #     if int_type == 'surf-surf':
        #         int_type_new = 'surf-edge'
        #         err_norm1 = [np.linalg.norm(int_para_coord1[:,0]),
        #                      np.linalg.norm(int_para_coord1[:,0]
        #                      -np.ones(int_para_coord1[:,0].size)),
        #                      np.linalg.norm(int_para_coord1[:,1]),
        #                      np.linalg.norm(int_para_coord1[:,1]
        #                      -np.ones(int_para_coord1[:,1].size))]
        #         xi1_type = int_xi_type[np.argmin(err_norm1)]
        #         int_xi_new = int_xi.split('-')[0]+'-'+xi1_type
        #     # elif int_type == 'edge-surf':
        #     #     int_type_new = 'surf-edge'
        #     #     pass
        #     else:
        #         int_type_new = int_type
        #         int_xi_new = int_xi
        if 'rib' in s_name1: # Requires to be 'surf-surf'
            if not int_type == 'surf-surf':
                int_type_new = 'surf-surf'
                int_xi_new = 'na-na'
            else:
                int_type_new = int_type
                int_xi_new = int_xi

    int_types_new += [[int_type_new, int_xi_new]]

# for int_ind in range(preprocessor.num_intersections_all):
#     s_ind0, s_ind1 = mapping_list[int_ind]
#     s_name0 = wing_surf_names[s_ind0]
#     s_name1 = wing_surf_names[s_ind1]
#     int_type = int_types[int_ind][0]
#     int_xi = int_types[int_ind][1]
#     int_type_new = int_types_new[int_ind][0]
#     int_xi_new = int_types_new[int_ind][1]
#     print('-'*50)
#     print("intersecting surfaces: {} -- {}".format(s_name0, s_name1))
#     print("Int type new: {}, xi new: {}".format(int_type_new, int_xi_new))
#     print("Int type old: {}, xi old: {}".format(int_type, int_xi))
#     print(' '*50)

preprocessor.intersections_type = int_types_new

pair_pop = [2,3]
pop_ind = preprocessor.mapping_list.index(pair_pop)
preprocessor.intersections_type.pop(pop_ind)
preprocessor.mapping_list.pop(pop_ind)
preprocessor.intersections_para_coords.pop(pop_ind)
preprocessor.mortar_nels.pop(pop_ind)

pair_pop = [2,4]
pop_ind = preprocessor.mapping_list.index(pair_pop)
preprocessor.intersections_type.pop(pop_ind)
preprocessor.mapping_list.pop(pop_ind)
preprocessor.intersections_para_coords.pop(pop_ind)
preprocessor.mortar_nels.pop(pop_ind)


preprocessor.num_intersections_all = len(preprocessor.mapping_list)
preprocessor.get_diff_intersections()