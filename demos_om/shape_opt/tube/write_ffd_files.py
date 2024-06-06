import numpy as np
import openmdao.api as om
from igakit.cad import *
from igakit.io import VTK
from GOLDFISH.nonmatching_opt_om import *
import meshio

test_ind = 7
recorder_name = './opt_data/recorder'+str(test_ind)+'.sql'
FFD_data_name = './opt_data/FFD_data'+str(test_ind)+'.npz'

# # Load recorder data
cr = om.CaseReader(recorder_name)
# driver_cases = cr.list_cases('driver', out_stream=None)
driver_cases = cr.get_cases('driver', recurse=False)

npz_file = np.load(FFD_data_name, allow_pickle=True)

opt_field = npz_file['opt_field']
SLSQP_major_inds = npz_file['major_iter_ind']
FFD_block_control = npz_file['ffd_control']
FFD_block_knots = npz_file['ffd_knots']
QoI=npz_file['QoI']

FFD_block = NURBS(FFD_block_knots, FFD_block_control)

ffd_path = './geometry/FFD_files'+str(test_ind)+'/'
if not os.path.isdir(ffd_path):
    os.mkdir(ffd_path)

ffd_block_name_pre = 'FFD_block_'
ffd_CP_name_pre = 'FFD_CP_'
wint_list_major = []
design_var_list_major = []
FFD_block_list = []
for i, major_ind in enumerate(SLSQP_major_inds):
    case = driver_cases[major_ind]
    wint_list_major += [case.get_objectives()['internal_energy_comp.int_E'][0]]
    design_val = []
    for val in case.get_design_vars().values():
        design_val += [val]
    design_var_list_major += [design_val]
    FFD_block_list += [update_FFD_block(FFD_block, design_val, opt_field),]
    # Save files
    ffd_name_ind = '0'*(6-len(str(i)))+str(i)
    ffd_block_name_vtk = ffd_path + ffd_block_name_pre + ffd_name_ind + '.vtk'
    VTKWriter().write(ffd_block_name_vtk, FFD_block_list[i], ref_level=5)
    ffd_block_name_vtu = ffd_path + ffd_block_name_pre + ffd_name_ind + '.vtu'
    ffd_block_vtk_file = meshio.read(ffd_block_name_vtk)
    meshio.write(ffd_block_name_vtu, ffd_block_vtk_file)
    ffd_cp_name_vtk = ffd_path + ffd_CP_name_pre + ffd_name_ind + '.vtk'
    VTKWriter().write_cp(ffd_cp_name_vtk, FFD_block_list[i])
    ffd_cp_name_vtu = ffd_path + ffd_CP_name_pre + ffd_name_ind + '.vtu'
    ffd_cp_vtk_file = meshio.read(ffd_cp_name_vtk)
    meshio.write(ffd_cp_name_vtu, ffd_cp_vtk_file)

FFD_block_pvd_file = 'FFD_block.pvd'
FFD_CP_pvd_file = 'FFD_CP.pvd'
create_pvd(FFD_block_pvd_file, vtu_pre=ffd_block_name_pre, directory=ffd_path)
create_pvd(FFD_CP_pvd_file, vtu_pre=ffd_CP_name_pre, directory=ffd_path)