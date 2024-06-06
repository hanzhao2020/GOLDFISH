import numpy as np
import openmdao.api as om
from igakit.cad import *
from igakit.io import VTK
from GOLDFISH.nonmatching_opt_om import *
import meshio

test_ind = 51

save_path = '/home/han/Documents/test_results/'
folder_name = "results"+str(test_ind)+"/"
opt_data_dir = save_path+folder_name+'opt_data/'
# if not os.path.isdir(opt_data_dir):
#     os.mkdir(opt_data_dir)
recorder_name = opt_data_dir+'recorder'+str(test_ind)+'.sql'
shopt_data_name = opt_data_dir+'shopt_ffd_data'+str(test_ind)+'.npz'

# # Load recorder data
cr = om.CaseReader(recorder_name)
# driver_cases = cr.list_cases('driver', out_stream=None)
driver_cases = cr.get_cases('driver', recurse=False)

npz_file = np.load(shopt_data_name, allow_pickle=True)

opt_field = npz_file['opt_field']
major_iter_ind = npz_file['major_iter_ind']
FFD_block_control = npz_file['ffd_control']
FFD_block_knots = npz_file['ffd_knots']
QoI=npz_file['QoI']

FFD_block = NURBS(FFD_block_knots, FFD_block_control)

# shopt_ffd_path = './geometry/FFD_files'+str(test_ind)+'/'
shopt_ffd_path = opt_data_dir+'shopt_ffd_files/'
if not os.path.isdir(shopt_ffd_path):
    os.mkdir(shopt_ffd_path)

ffd_block_name_pre = 'shopt_ffd_block_'
ffd_CP_name_pre = 'shopt_ffd_cp_'
wint_list_major = []
design_var_list_major = []
FFD_block_list = []
for i, major_ind in enumerate(major_iter_ind):
    case = driver_cases[major_ind]
    wint_list_major += [case.get_objectives()['internal_energy_comp.int_E'][0]]
    design_val = []
    for val in case.get_design_vars().values():
        design_val += [val]
    design_var_list_major += [design_val]
    FFD_block_list += [update_FFD_block(FFD_block, design_val, opt_field),]
    # Save files
    ffd_name_ind = '0'*(6-len(str(i)))+str(i)
    ffd_block_name_vtk = shopt_ffd_path + ffd_block_name_pre + ffd_name_ind + '.vtk'
    VTKWriter().write(ffd_block_name_vtk, FFD_block_list[i], ref_level=3)
    ffd_block_name_vtu = shopt_ffd_path + ffd_block_name_pre + ffd_name_ind + '.vtu'
    ffd_block_vtk_file = meshio.read(ffd_block_name_vtk)
    meshio.write(ffd_block_name_vtu, ffd_block_vtk_file)
    ffd_cp_name_vtk = shopt_ffd_path + ffd_CP_name_pre + ffd_name_ind + '.vtk'
    VTKWriter().write_cp(ffd_cp_name_vtk, FFD_block_list[i])
    ffd_cp_name_vtu = shopt_ffd_path + ffd_CP_name_pre + ffd_name_ind + '.vtu'
    ffd_cp_vtk_file = meshio.read(ffd_cp_name_vtk)
    meshio.write(ffd_cp_name_vtu, ffd_cp_vtk_file)

FFD_block_pvd_file = 'shopt_ffd_block.pvd'
FFD_CP_pvd_file = 'shopt_ffd_cp.pvd'
create_pvd(FFD_block_pvd_file, vtu_pre=ffd_block_name_pre, directory=shopt_ffd_path)
create_pvd(FFD_CP_pvd_file, vtu_pre=ffd_CP_name_pre, directory=shopt_ffd_path)