"""
Initial geometry can be downloaded from the following link:
https://drive.google.com/file/d/1IVNmFAEEMyM0p4QuEITgGMuG43UF2q5U/view?usp=sharing
"""

from PENGoLINS.occ_preprocessing import *
from PENGoLINS.igakit_utils import *

geom_scale = 2.54e-5 

print("Importing geometry...")
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

# Adjusting outer surfaces' control points to clean up geometry
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

lower_skin_surf_ik_temp = NURBS(lower_skin_knots, lower_skin_cp)
upper_skin_surf_ik_temp = NURBS(upper_skin_knots, upper_skin_cp)
wing_tip_surf_ik_temp = NURBS(wing_tip_knots, wing_tip_cp)

lower_skin_surf_ik_control = lower_skin_surf_ik_temp.control.copy()
for i in range(len(lower_skin_surf_ik_control[:,:,1])):
    lower_skin_surf_ik_control[:,:,1][i] = lower_skin_surf_ik_control[:,:,1][i][0]

lower_skin_surf_ik = NURBS(lower_skin_knots, lower_skin_surf_ik_control)

upper_skin_surf_ik_control = upper_skin_surf_ik_temp.control.copy()
upper_skin_surf_ik_knots = upper_skin_surf_ik_temp.knots
for i in range(len(upper_skin_surf_ik_control[:,:,1])):
    upper_skin_surf_ik_control[:,:,1][i] = upper_skin_surf_ik_control[:,:,1][i][0]
upper_skin_surf_ik = NURBS(upper_skin_knots, upper_skin_surf_ik_control)

wing_tip_surf_ik_control = wing_tip_surf_ik_temp.control.copy()
wing_tip_surf_ik_knots = wing_tip_surf_ik_temp.knots
for i in range(len(wing_tip_surf_ik_control[:,:,1])):
    wing_tip_surf_ik_control[:,:,1][i] = wing_tip_surf_ik_control[:,:,1][i][0]
wing_tip_surf_ik = NURBS(wing_tip_knots, wing_tip_surf_ik_control)

lower_skin = ikNURBS2BSpline_surface(lower_skin_surf_ik)
upper_skin = ikNURBS2BSpline_surface(upper_skin_surf_ik)
wing_tip = ikNURBS2BSpline_surface(wing_tip_surf_ik)

evtol_wing_surfaces = [lower_skin, upper_skin, wing_tip]

spar_inds = [6, 8] # Use two spars in the optimization
num_spars = len(spar_inds)

spar_f_ik_temp = BSpline_surface2ikNURBS(wing_surfaces[spar_inds[0]])
spar_f_control = spar_f_ik_temp.control.copy()
spar_f_control[:,:,1][0] = lower_skin_surf_ik_control[:,:,1][0][0]*(1+1e-3)
spar_f_control[:,:,1][-1] = lower_skin_surf_ik_control[:,:,1][-1][0]*(1-1e-3)
spar_f_ik = NURBS(spar_f_ik_temp.knots, spar_f_control)
spar_f_surf = ikNURBS2BSpline_surface(spar_f_ik)

spar_r_ik_temp = BSpline_surface2ikNURBS(wing_surfaces[spar_inds[1]])
spar_r_control = spar_r_ik_temp.control.copy()
spar_r_control[:,:,1][0] = lower_skin_surf_ik_control[:,:,1][0][0]*(1+1e-3)
spar_r_control[:,:,1][-1] = lower_skin_surf_ik_control[:,:,1][-1][0]*(1-1e-3)
spar_r_ik = NURBS(spar_r_ik_temp.knots, spar_r_control)
spar_r_surf = ikNURBS2BSpline_surface(spar_r_ik)

evtol_wing_surfaces += [spar_f_surf, spar_r_surf]

rib_inds = [9, 12, 14, 16, 18, 20] # Use six ribs in the optimization
num_ribs = len(rib_inds)
evtol_wing_surfaces += [wing_surfaces[i] for i in rib_inds]

# Reparametrize and refine spline surfaces
num_pts_eval = [12]*num_surfs
num_pts_eval[0] = 12
num_pts_eval[1] = 12
num_pts_eval[3] = 6
num_pts_eval[4] = 6
ref_level_list = [1]*num_surfs

# Meshes that are close to the results in the paper
# Working discretization
u_insert_list = [12, 12, 1] \
              + [16, 18] + [4]*num_ribs
v_insert_list = [12, 12, 8] \
              + [1]*2 + [1]*num_ribs

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

preprocessor.compute_intersections(rtol=1e-6, mortar_refine=1, 
                                   edge_rel_ratio=1e-4)

# Clean up intersections' type for shape optimization
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
            else:
                int_type_new = int_type
                int_xi_new = int_xi
    elif 'spar' in s_name0:
        if 'rib' in s_name1: # Requires to be 'surf-edge'
            if int_type == 'surf-edge':
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
if pair_pop in preprocessor.mapping_list:
    pop_ind = preprocessor.mapping_list.index(pair_pop)
    preprocessor.intersections_type.pop(pop_ind)
    preprocessor.mapping_list.pop(pop_ind)
    preprocessor.intersections_para_coords.pop(pop_ind)
    preprocessor.mortar_nels.pop(pop_ind)

pair_pop = [2,4]
if pair_pop in preprocessor.mapping_list:
    pop_ind = preprocessor.mapping_list.index(pair_pop)
    preprocessor.intersections_type.pop(pop_ind)
    preprocessor.mapping_list.pop(pop_ind)
    preprocessor.intersections_para_coords.pop(pop_ind)
    preprocessor.mortar_nels.pop(pop_ind)


preprocessor.num_intersections_all = len(preprocessor.mapping_list)
preprocessor.get_diff_intersections()

# from GOLDFISH.nonmatching_opt_om import *
# from PENGoLINS.nonmatching_utils import *

# for i in range(preprocessor.num_surfs):
#     # ik_surf = BSpline_surface2ikNURBS(preprocessor.BSpline_surfs_repara[i])
#     ik_surf = BSpline_surface2ikNURBS(preprocessor.BSpline_surfs_refine[i])
#     # VTK().write("./geometry/wing_surf"+str(i)+".vtk", ik_surf)
#     # # Write refine surface for visualization
#     VTKWriter().write("./geometry/wing_surf"+str(i)+".vtk", ik_surf, ref_level=3)

# for i in range(preprocessor.num_intersections_all):
#     mesh_phy = generate_mortar_mesh(preprocessor.intersections_phy_coords[i], num_el=128)
#     File('./geometry/int_curve'+str(i)+'.pvd') << mesh_phy