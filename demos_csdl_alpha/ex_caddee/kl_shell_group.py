import numpy as np
import csdl_alpha as csdl
from matplotlib import pyplot as plt

import time
from datetime import datetime

from igakit.cad import *
from igakit.io import VTK
from GOLDFISH.nonmatching_opt_csdl import *
# from GOLDFISH.nonmatching_opt_om import *
from PENGoLINS.igakit_utils import *

def return_ik_knots(cd_knot, return_order=False):
    xi0_0_count = 0
    num_int_konts0 = 0
    for i in cd_knot:
        if i == 0:
            xi0_0_count += 1
        elif i == 1:
            break
        else:
            num_int_konts0 += 1
    p0 = xi0_0_count-1
    knot0 = cd_knot[0:int(xi0_0_count*2+num_int_konts0)]
    knot1 = cd_knot[int(xi0_0_count*2+num_int_konts0):]
    xi1_0_count = 0
    for i in knot1:
        if i == 0:
            xi1_0_count += 1
        else:
            break
    p1 = xi1_0_count-1
    if return_order:
        return [knot0, knot1], [p0, p1]
    else:
        return [knot0, knot1]

def clampedBC(spline_generator, side=0, direction=0):
    """
    Apply clamped boundary condition to spline.
    """
    for field in [0,1,2]:
        if field in [0]:
            n_layers = 1
        else:
            n_layers = 2
        scalar_spline = spline_generator.getScalarSpline(field)
        side_dofs = scalar_spline.getSideDofs(direction, side, nLayers=n_layers)
        spline_generator.addZeroDofs(field, side_dofs)

def OCCBSpline2tIGArSpline(surface, num_field=3, quad_deg_const=4, 
                        setBCs=None, side=0, direction=0):
    """
    Generate ExtractedBSpline from OCC B-spline surface.
    """
    quad_deg = surface.UDegree()*quad_deg_const
    spline_mesh = NURBSControlMesh4OCC(surface, useRect=False)
    spline_generator = EqualOrderSpline(worldcomm, num_field, spline_mesh)
    if setBCs is not None:
        setBCs(spline_generator, side, direction)
    spline = ExtractedSpline(spline_generator, quad_deg)
    return spline


class KLShellModel(object):
    def __init__(self, knot_list, cp_list, bc_list=[], recorder=None):
        '''
        bc_list: [[i, dir, side], [j, dir, side], ...]
        '''
        self.knot_list = knot_list
        self.cp_list = cp_list
        self.bc_list = bc_list
        self.recorder = recorder 

        self.ik_knot_list = []
        for knot in self.knot_list:
            self.ik_knot_list += [return_ik_knots(knot),]

        self.ik_surfs = []
        self.occ_surfs = []
        for i in range(len(self.ik_knot_list)):
            self.ik_surfs += [NURBS(self.ik_knot_list[i], self.cp_list[i])]
            self.occ_surfs += [ikNURBS2BSpline_surface(self.ik_surfs[-1])]

        write_geom_file(self.occ_surfs, "./geometry/wing.igs")
        # Geometry preprocessing and surface-surface intersections computation
        self.preprocessor = OCCPreprocessing(self.occ_surfs, 
                            reparametrize=False, 
                            refine=False)

        # self.preprocessor.compute_intersections(
        #                   rtol=1e-6, mortar_refine=2, 
        #                   edge_rel_ratio=1e-3)
        int_data_filename = "wing_int_data.npz"
        self.preprocessor.load_intersections_data(int_data_filename)

        if mpirank == 0:
            print("Total DoFs:", self.preprocessor.total_DoFs)
            print("Number of intersections:", 
                  self.preprocessor.num_intersections_all)

        self.num_surfs = len(self.ik_surfs)

        # Create tIGAr extracted spline instances
        self.splines = []
        surf_wbcs = [ind[0] for ind in self.bc_list]
        for i in range(self.num_surfs):
            if i in surf_wbcs:
                # Apply clamped BC to surfaces near root
                bc_ind = surf_wbcs.index(i)
                bc_dir = self.bc_list[bc_ind][1]
                bc_side = self.bc_list[bc_ind][2]
                spline = OCCBSpline2tIGArSpline(
                        self.preprocessor.BSpline_surfs[i], 
                        setBCs=clampedBC, direction=bc_dir, side=bc_side)
                self.splines += [spline,]
            else:
                spline = OCCBSpline2tIGArSpline(
                        self.preprocessor.BSpline_surfs[i])
                self.splines += [spline,]

    def evaluate(self, shell_forces_list:list, 
                 h_th:csdl.Variable, 
                 E, nu, density, 
                 pressure_inds=None,
                 disp=None, penalty_coefficient=Constant(1.0e3)):
        self.shell_forces_list = shell_forces_list
        self.h_th = h_th
        self.E = E 
        self.nu = nu
        self.density = density
        self.disp = disp
        self.penalty_coefficient = penalty_coefficient
        if pressure_inds is None:
            self.pressure_inds = list(range(self.num_surfs))
        else:
            self.pressure_inds = pressure_inds

        self.load_spaces = []
        self.loads = []
        self.h_th_func = []
        for i in range(self.num_surfs):
            self.h_th_func += [Function(self.splines[i].V_linear)]
            self.load_spaces += [VectorFunctionSpace(
                                 self.splines[i].mesh, 'CG', 1, dim=3)]
            if i in self.pressure_inds:
                local_ind = self.pressure_inds.index(i)
                val = shell_forces_list[local_ind].value[::-1,:].reshape(-1)
                self.loads += [Function(self.load_spaces[-1])]
                self.loads[-1].vector().set_local(val)
            else:
                self.loads += [as_vector([Constant(0.),Constant(0.),Constant(0.)])]

        E = Constant(self.E.value[0])
        nu = Constant(self.nu.value[0])

        # Create non-matching problem
        self.nonmatching_opt = NonMatchingOpt(self.splines, E, self.h_th_func, nu, 
                               comm=worldcomm)
        self.nonmatching_opt.create_mortar_meshes(
                             self.preprocessor.mortar_nels)
        self.nonmatching_opt.set_thickness_opt(var_thickness=False)
        self.nonmatching_opt.update_h_th(self.h_th.value)

        if mpirank == 0:
            print("Setting up mortar meshes...")
        self.nonmatching_opt.mortar_meshes_setup(
                             self.preprocessor.mapping_list, 
                             self.preprocessor.intersections_para_coords, 
                             self.penalty_coefficient)

        self.source_terms = []
        self.residuals = []
        for i in range(self.num_surfs):
            self.source_terms += [inner(self.loads[i], self.nonmatching_opt.splines[i].\
                rationalize(self.nonmatching_opt.spline_test_funcs[i]))*\
                self.nonmatching_opt.splines[i].dx]
            self.residuals += [SVK_residual(self.nonmatching_opt.splines[i], 
                                    self.nonmatching_opt.spline_funcs[i], 
                                    self.nonmatching_opt.spline_test_funcs[i], 
                                    E, nu, self.h_th_func[i], self.source_terms[i])]
        self.nonmatching_opt.set_residuals(self.residuals)
        
        self.output_shape = self.nonmatching_opt.vec_iga_dof

        save_path = "./"
        folder_name = "results/"
        self.nonmatching_opt.create_files(save_path=save_path, folder_name=folder_name)

        self.nonmatching_opt.solve_nonlinear_nonmatching_problem()
        self.nonmatching_opt.save_files()
