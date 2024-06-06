from GOLDFISH.nonmatching_opt_om import *
import igakit.plot as ikplot
import matplotlib.pyplot as plt

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

def ikSurf2tIGArSpline(surface, num_field=3, quad_deg_const=3, 
                        spline_bc=None, index=0):
    """
    Generate ExtractedBSpline from OCC B-spline surface.
    """
    quad_deg = surface.degree[0]*quad_deg_const
    # DIR = SAVE_PATH+"spline_data/extraction_"+str(index)+"_init"
    # spline = ExtractedSpline(DIR, quad_deg)
    spline_mesh = NURBSControlMesh(surface, useRect=False)
    spline_generator = EqualOrderSpline(worldcomm, num_field, spline_mesh)
    if spline_bc is not None:
        spline_bc.set_bc(spline_generator)
    # spline_generator.writeExtraction(DIR)
    spline = ExtractedSpline(spline_generator, quad_deg)
    return spline

def surf_cpflat2tIGAr():
    pass

def surf_cpflat2igakit():
    pass

vtk_writer = VTKWriter()
p = 3
p_ffd = 2
R = 10
angle_lim0 = [120, 90]
angle_lim1 = [90, 60]
y_lim0 = [0,5]
y_lim1 = [5,10]
# Number of elements for each surface
# num_el0 = [8,4]
# num_el1 = [7,5]
# num_el2 = [6,5]
# num_el3 = [8,4]
num_el0 = [5,5]
num_el1 = [6,6]
num_el2 = [6,6]
num_el3 = [7,7]
num_els = [num_el0, num_el1, num_el2, num_el3]
angle_lims = [angle_lim0, angle_lim1, angle_lim0, angle_lim1]
y_lims = [y_lim0, y_lim0, y_lim1, y_lim1]
surfs = []
for i in range(len(num_els)):
    surfs += [create_arc_surf(num_els[i][0], num_els[i][1], 
              p, R, angle_lims[i], y_lims[i])]
    vtk_writer.write('./geometry/surf'+str(i)+'_init.vtk', surfs[i], ref_level=4)
    vtk_writer.write_cp('./geometry/surf'+str(i)+'_cp_init.vtk', surfs[i])

# surf0 = create_arc_surf(num_el0[0], num_el0[1], p, R, angle_lim, y_lim[0:2])
# surf1 = create_arc_surf(num_el1[0], num_el1[1], p, R, angle_lim, y_lim[1:3])
# surfs = [surf0, surf1]

# # plt.figure()
# # ikplot.plot(surf0)
# # ikplot.plot(surf1)
# # plt.show()
# vtk_writer.write("./geometry/surf0.vtk", surf0, ref_level=4)
# vtk_writer.write_cp("./geometry/surf0_cp.vtk", surf0)
# vtk_writer.write("./geometry/surf1.vtk", surf1, ref_level=4)
# vtk_writer.write_cp("./geometry/surf1_cp.vtk", surf1)

E = Constant(1.0e12)
nu = Constant(0.)
h_th = Constant(0.01)
penalty_coefficient = 1.0e3

splines = []
for i in range(len(surfs)):
        spline = ikSurf2tIGArSpline(surfs[i])
        splines += [spline,]
        File("./geometry/para_mesh"+str(i)+".pvd") << splines[i].mesh

# plt.figure()
# plot(splines[0].mesh)
# plot(splines[1].mesh)
# plt.show()

opt_field = [2]
nonmatching_opt_ffd = NonMatchingOptFFD(splines, E, h_th, nu, 
                                        opt_field=opt_field, comm=worldcomm)

# print(nonmatching_opt_ffd.cpsurf_lims)
# Create FFD block in igakit format
ffd_block_num_el = [3,3,1]
cpsurf_lims = nonmatching_opt_ffd.cpsurf_lims
cp_ffd_lims = [[] for i in range(len(cpsurf_lims))]
ffd_edge_ratio = [0.03,0.03,0.2]
for field in [0,1,2]:
    cp_range = cpsurf_lims[field][1] - cpsurf_lims[field][0]
    cp_ffd_lims[field] += [cpsurf_lims[field][0] - ffd_edge_ratio[field]*cp_range]
    cp_ffd_lims[field] += [cpsurf_lims[field][1] + ffd_edge_ratio[field]*cp_range]
FFD_block = create_3D_block(ffd_block_num_el, p_ffd, cp_ffd_lims)

# Set FFD to non-matching optimization instance
nonmatching_opt_ffd.set_shopt_FFD(FFD_block.knots, FFD_block.control)

vtk_writer.write("./geometry/FFD_block_init.vtk", FFD_block, ref_level=4)
vtk_writer.write_cp("./geometry/FFD_cp_init.vtk", FFD_block)

# Update FFD block CP
ffd_block_cpdisp = np.zeros(nonmatching_opt_ffd.shopt_cpffd.shape[0:-1])
for z_ind in range(ffd_block_cpdisp.shape[2]):
    for y_ind in range(ffd_block_cpdisp.shape[1]):
        for x_ind in range(ffd_block_cpdisp.shape[0]):
            if x_ind == 1:
                zx_disp = -0.6
            elif x_ind == 3:
                zx_disp = 0.6
            else:
                zx_disp = 0
            # For y coord
            if y_ind == 1:
                zy_disp = 1
            elif y_ind == 2:
                zy_disp = 1.3
            elif y_ind == 3:
                zy_disp = 1
            # if y_ind == 1:
            #     zy_disp = 1
            # elif y_ind == 3:
            #     zy_disp = -1
            else:
                zy_disp = 0
            # For z coord
            if z_ind == 2:
                zz_disp = 0.2
            # if z_ind == 0:
            #     zz_disp = -0.5
            # elif z_ind == 2:
            #     zz_disp = 0.5
            else:
                zz_disp = 0
            cp_disp = zx_disp + zy_disp + zz_disp
            ffd_block_cpdisp[x_ind, y_ind, z_ind] = cp_disp

new_ffd_cp = FFD_block.control.copy()
new_ffd_cp[...,2] = new_ffd_cp[...,2] + ffd_block_cpdisp

new_ffd_block = NURBS(FFD_block.knots, new_ffd_cp)

vtk_writer.write("./geometry/FFD_block_new.vtk", new_ffd_block, ref_level=4)
vtk_writer.write_cp("./geometry/FFD_cp_new.vtk", new_ffd_block)

# Update tIGAr splines cp funcs
cpiga_flat_old = nonmatching_opt_ffd.get_init_CPIGA()[:,2]

cpffd_flat_new = new_ffd_block.control[...,0:3].transpose(2,1,0,3)\
                 .reshape(-1,3)[:,opt_field[0]]
cpfe_new = nonmatching_opt_ffd.shopt_dcpsurf_fedcpffd*cpffd_flat_new
nonmatching_opt_ffd.update_CPFE(cpfe_new, opt_field[0])
nonmatching_opt_ffd.create_files(save_path="./", folder_name="results/", 
                                 refine_mesh=True, ref_nel=64)
nonmatching_opt_ffd.save_files()

# Compute new CP IGA
cpfe2iga_comp = CPFE2IGAImOperation(nonmatching_opt=nonmatching_opt_ffd)

update_nest_vec(cpfe_new, cpfe2iga_comp.cp_fe_vecs[0])
cpiga_flat_new = cpfe2iga_comp.solve_nonlinear()[0]
cpiga_flat_new_sub = []
start_ind = 0
end_ind = 0
for i in range(len(surfs)):
    end_ind += nonmatching_opt_ffd.splines[i].M_control.size(1)
    # print("start:end", start_ind, end_ind)
    # print(cpiga_flat_new[start_ind:end_ind])
    cpiga_flat_new_sub += [cpiga_flat_new[start_ind:end_ind]]
    start_ind += nonmatching_opt_ffd.splines[i].M_control.size(1)


cpiga_ik_list = []
surf_cp_new_list = []
surfs_new = []
for i in range(len(surfs)):
    cpiga_ik_list += [cpiga_flat_new_sub[i].reshape(surfs[i].control.shape[1],
                      surfs[i].control.shape[0]).transpose()]
    surf_cp_new = surfs[i].control.copy()
    surf_cp_new[...,opt_field[0]] = cpiga_ik_list[i]
    surf_cp_new_list += [surf_cp_new]
    surfs_new += [NURBS(surfs[i].knots, surf_cp_new_list[i])]
    vtk_writer.write('./geometry/surf'+str(i)+'_new.vtk', surfs_new[i], ref_level=5)
    vtk_writer.write_cp('./geometry/surf'+str(i)+'_cp_new.vtk', surfs_new[i])

# # Save new surfaces in igakit format
# cpiga0_ik = cpiga_flat_new_sub[0].reshape(surfs[0].control.shape[1],
#             surfs[0].control.shape[0]).transpose()
# cpiga1_ik = cpiga_flat_new_sub[1].reshape(surfs[1].control.shape[1],
#             surfs[1].control.shape[0]).transpose()

# surf0_cp_new = surf0.control.copy()
# surf0_cp_new[...,opt_field[0]] = cpiga0_ik

# surf1_cp_new = surf1.control.copy()
# surf1_cp_new[...,opt_field[0]] = cpiga1_ik

# surf0_new = NURBS(surf0.knots, surf0_cp_new)
# surf1_new = NURBS(surf1.knots, surf1_cp_new)

# vtk_writer.write("./geometry/surf0_new.vtk", surf0_new, ref_level=4)
# vtk_writer.write_cp("./geometry/surf0_new_cp.vtk", surf0_new)
# vtk_writer.write("./geometry/surf1_new.vtk", surf1_new, ref_level=4)
# vtk_writer.write_cp("./geometry/surf1_new_cp.vtk", surf1_new)