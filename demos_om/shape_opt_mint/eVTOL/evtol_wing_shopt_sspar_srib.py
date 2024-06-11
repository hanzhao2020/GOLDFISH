import numpy as np
import matplotlib.pyplot as plt
import openmdao.api as om
from igakit.cad import *
from igakit.io import VTK
from GOLDFISH.nonmatching_opt_om import *

from custom_comps.evtol_sspar_srib_align_comp import CPSurfAlignComp
from create_geom_evtol import preprocessor


set_log_active(False)

class ShapeOptGroup(om.Group):

    def initialize(self):
        self.options.declare('nonmatching_opt')
        self.options.declare('cpdesign2analysis')
        self.options.declare('cp_design_name_pre', default='CP_design')
        self.options.declare('cp_coarse_name_pre', default='CP_coarse')
        self.options.declare('cp_order_ele_name_pre', default='CP_order_ele')
        self.options.declare('cp_analysis_name_pre', default='CP_analysis')

        self.options.declare('cp_regu_name_pre', default='CP_regu')
        self.options.declare('cp_pin_name_pre', default='CP_pin')
        self.options.declare('cp_dist_name_pre', default='CP_dist')

        self.options.declare('int_name', default='int_para')
        self.options.declare('disp_name', default='displacements')
        self.options.declare('int_energy_name', default='int_E')
        self.options.declare('volume_name', default='volume')
        self.options.declare('int_xi_edge_name', default='int_xi_edge')

    def init_parameters(self):
        self.nonmatching_opt = self.options['nonmatching_opt']
        self.cpdesign2analysis = self.options['cpdesign2analysis']
        self.cp_design_name_pre = self.options['cp_design_name_pre']
        self.cp_coarse_name_pre = self.options['cp_coarse_name_pre']
        self.cp_order_ele_name_pre = self.options['cp_order_ele_name_pre']
        self.cp_analysis_name_pre = self.options['cp_analysis_name_pre']

        self.cp_regu_name_pre = self.options['cp_regu_name_pre']
        self.cp_pin_name_pre = self.options['cp_pin_name_pre']
        self.cp_dist_name_pre = self.options['cp_dist_name_pre']

        self.int_name = self.options['int_name']
        self.disp_name = self.options['disp_name']
        self.int_energy_name = self.options['int_energy_name']
        self.volume_name = self.options['volume_name']
        self.int_xi_edge_name = self.options['int_xi_edge_name']

        self.opt_field = self.nonmatching_opt.opt_field

        self.init_cp_design = self.cpdesign2analysis.init_cp_design
        self.input_cp_shapes = [len(cp) for cp in self.init_cp_design]

        self.design_var_lower = [np.array([4.67, 3.9, 4.67, 3.9]), 0.7, 2.7] # x min, z min
        self.design_var_upper = [np.array([5.48, 4.45, 5.48, 4.45]), 5.2, 3.7] # x max, z max

        self.cp_design_name_list = []
        self.cp_coarse_name_list = []
        self.cp_order_ele_name_list = []
        self.cp_analysis_name_list = []
        for field_ind, field in enumerate(self.opt_field):
            self.cp_design_name_list += [self.cp_design_name_pre+str(field)]
            self.cp_coarse_name_list += [self.cp_coarse_name_pre+str(field)]
            self.cp_order_ele_name_list += [self.cp_order_ele_name_pre+str(field)]
            self.cp_analysis_name_list += [self.cp_analysis_name_pre+str(field)]


        self.regu_field = self.cpdesign2analysis.cp_coarse_regu_field
        self.cp_regu_name_list = []
        for field_ind, field in enumerate(self.regu_field):
            self.cp_regu_name_list += [self.cp_regu_name_pre+str(field)]

        self.pin_field = self.cpdesign2analysis.cp_coarse_pin_field
        self.cp_pin_name_list = []
        for field_ind, field in enumerate(self.pin_field):
            self.cp_pin_name_list += [self.cp_pin_name_pre+str(field)]

        self.dist_field = self.cpdesign2analysis.cp_coarse_dist_field
        self.cp_dist_name_list = []
        for field_ind, field in enumerate(self.dist_field):
            self.cp_dist_name_list += [self.cp_dist_name_pre+str(field)]

        self.inputs_comp_name = 'inputs_comp'
        self.cp_align_comp_name = 'CP_design_align_comp'
        self.cp_order_ele_comp_name = 'CP_order_ele_comp'
        self.cp_knot_refine_comp_name = 'CP_knot_refine_comp'

        self.cp_regu_comp_name = 'CP_regu_comp'
        self.cp_pin_comp_name = 'CP_pin_comp'
        self.cp_dist_comp_name = 'CP_dist_comp'

        self.cpiga2xi_comp_name = 'CPIGA2Xi_comp'
        self.disp_states_comp_name = 'disp_states_comp'
        self.int_energy_comp_name = 'internal_energy_comp'
        self.volume_comp_name = 'volume_comp'
        self.int_xi_edge_comp_name = 'int_xi_edge_comp'

        self.diff_vec = []
        for field_ind, field in enumerate(self.opt_field):
            if field == 0:
                self.diff_vec += [np.asarray(self.cpdesign2analysis.init_cp_coarse[field_ind] \
                               - np.dot(self.cpdesign2analysis.cp_coarse_align_deriv_list[field_ind]\
                                 .todense()[:,0:4],
                                 self.cpdesign2analysis.init_cp_design[field_ind]))[0]]
            else:
                self.diff_vec += [self.cpdesign2analysis.init_cp_coarse[field_ind] \
                               - self.cpdesign2analysis.cp_coarse_align_deriv_list[field_ind] \
                                *self.cpdesign2analysis.init_cp_design[field_ind]]
            if field in [1,2]:
                self.diff_vec[field_ind] = np.zeros(self.diff_vec[field_ind].size)
        
    def setup(self):
        # Add inputs comp
        inputs_comp = om.IndepVarComp()
        for field_ind, field in enumerate(self.opt_field):
            inputs_comp.add_output(self.cp_design_name_list[field_ind],
                        shape=self.input_cp_shapes[field_ind],
                        val=self.init_cp_design[field_ind])
        self.add_subsystem(self.inputs_comp_name, inputs_comp)

        self.cp_align_comp = CPSurfAlignComp(
                             cpdesign2analysis=self.cpdesign2analysis,
                             diff_vec=self.diff_vec,
                             input_cp_design_name_pre=self.cp_design_name_pre,
                             output_cp_coarse_name_pre=self.cp_coarse_name_pre)
        self.cp_align_comp.init_parameters()
        self.add_subsystem(self.cp_align_comp_name, self.cp_align_comp)


        self.cp_order_ele_comp = CPSurfOrderElevationComp(
                                 cpdesign2analysis=self.cpdesign2analysis,
                                 input_cp_coarse_name_pre=self.cp_coarse_name_pre,
                                 output_cp_order_ele_name_pre=self.cp_order_ele_name_pre)
        self.cp_order_ele_comp.init_parameters()
        self.add_subsystem(self.cp_order_ele_comp_name, self.cp_order_ele_comp)


        self.cp_knot_refine_comp = CPSurfKnotRefinementComp(
                                   cpdesign2analysis=self.cpdesign2analysis,
                                   input_cp_order_ele_name_pre=self.cp_order_ele_name_pre,
                                   output_cp_fine_name_pre=self.cp_analysis_name_pre)
        self.cp_knot_refine_comp.init_parameters()
        self.add_subsystem(self.cp_knot_refine_comp_name, self.cp_knot_refine_comp)


        if len(self.regu_field) > 0:
            self.cp_regu_comp = CPSurfReguComp(
                                cpdesign2analysis=self.cpdesign2analysis,
                                input_cp_design_name_pre=self.cp_design_name_pre,
                                output_cp_regu_name_pre=self.cp_regu_name_pre)
            self.cp_regu_comp.init_parameters()
            self.add_subsystem(self.cp_regu_comp_name, self.cp_regu_comp)


        if len(self.pin_field) > 0:
            self.cp_pin_comp = CPSurfPinComp(
                                cpdesign2analysis=self.cpdesign2analysis,
                                input_cp_design_name_pre=self.cp_design_name_pre,
                                output_cp_pin_name_pre=self.cp_pin_name_pre)
            self.cp_pin_comp.init_parameters()
            self.add_subsystem(self.cp_pin_comp_name, self.cp_pin_comp)


        if len(self.dist_field) > 0:
            self.cp_dist_comp = CPSurfDistanceComp(
                                cpdesign2analysis=self.cpdesign2analysis,
                                input_cp_design_name_pre=self.cp_design_name_pre,
                                output_cp_dist_name_pre=self.cp_dist_name_pre)
            self.cp_dist_comp.init_parameters()
            self.add_subsystem(self.cp_dist_comp_name, self.cp_dist_comp)


        # Add CPIGA2Xi comp
        self.cpiga2xi_comp = CPIGA2XiComp(
                        nonmatching_opt=self.nonmatching_opt,
                        input_cp_iga_name_pre=self.cp_analysis_name_pre,
                        output_xi_name=self.int_name)
        self.cpiga2xi_comp.init_parameters()
        self.add_subsystem(self.cpiga2xi_comp_name, self.cpiga2xi_comp)

        # Add disp_states_comp
        self.disp_states_comp = DispMintStatesComp(
                           nonmatching_opt=self.nonmatching_opt,
                           input_cp_iga_name_pre=self.cp_analysis_name_pre,
                           input_xi_name=self.int_name,
                           output_u_name=self.disp_name)
        self.disp_states_comp.init_parameters(save_files=save_files,
                                             nonlinear_solver_rtol=1e-3,
                                             nonlinear_solver_max_it=10)
        self.add_subsystem(self.disp_states_comp_name, self.disp_states_comp)

        # Add internal energy comp (objective function)
        self.int_energy_comp = IntEnergyComp(
                          nonmatching_opt=self.nonmatching_opt,
                          input_cp_iga_name_pre=self.cp_analysis_name_pre,
                          input_u_name=self.disp_name,
                          output_wint_name=self.int_energy_name)
        self.int_energy_comp.init_parameters()
        self.add_subsystem(self.int_energy_comp_name, self.int_energy_comp)

        # Add volume comp (constraint)
        self.vol_surf_inds = [5,6,7,8,9,10]
        self.volume_comps = []
        self.vol_val_list = []
        for vol_ind, surf_ind in enumerate(self.vol_surf_inds):
            self.volume_comps += [VolumeComp(
                           nonmatching_opt=self.nonmatching_opt,
                           vol_surf_inds=[surf_ind],
                           input_cp_iga_name_pre=self.cp_analysis_name_pre,
                           output_vol_name=self.volume_name+str(vol_ind))]
            self.volume_comps[vol_ind].init_parameters()
            self.add_subsystem(self.volume_comp_name+str(vol_ind), 
                              self.volume_comps[vol_ind])
            self.vol_val_list += [assemble(self.nonmatching_opt.h_th[surf_ind]
                                  *self.nonmatching_opt.splines[surf_ind].dx)]

        # Add int xi edge comp (constraint)
        self.int_xi_edge_comp = IntXiEdgeComp(
                               nonmatching_opt=self.nonmatching_opt,
                               input_xi_name=self.int_name,
                               output_name=self.int_xi_edge_name)
        self.int_xi_edge_comp.init_parameters()
        self.add_subsystem(self.int_xi_edge_comp_name, self.int_xi_edge_comp)
        self.int_xi_edge_cons_val = np.zeros(self.int_xi_edge_comp.output_shape)

        # Connect names between components
        for field_ind, field in enumerate(self.opt_field):
            # For optimization components
            self.connect(self.inputs_comp_name+'.'
                         +self.cp_design_name_list[field_ind],
                         self.cp_align_comp_name+'.'
                         +self.cp_design_name_list[field_ind])

            self.connect(self.cp_align_comp_name+'.'
                         +self.cp_coarse_name_list[field_ind],
                         self.cp_order_ele_comp_name+'.'
                         +self.cp_coarse_name_list[field_ind])

            self.connect(self.cp_order_ele_comp_name+'.'
                         +self.cp_order_ele_name_list[field_ind],
                         self.cp_knot_refine_comp_name+'.'
                         +self.cp_order_ele_name_list[field_ind])

            self.connect(self.cp_knot_refine_comp_name+'.'
                         +self.cp_analysis_name_list[field_ind],
                         self.cpiga2xi_comp_name+'.'
                         +self.cp_analysis_name_list[field_ind])

            self.connect(self.cp_knot_refine_comp_name+'.'
                         +self.cp_analysis_name_list[field_ind],
                         self.disp_states_comp_name+'.'
                         +self.cp_analysis_name_list[field_ind])

            self.connect(self.cp_knot_refine_comp_name+'.'
                         +self.cp_analysis_name_list[field_ind],
                         self.int_energy_comp_name+'.'
                         +self.cp_analysis_name_list[field_ind])

            for vol_ind, surf_ind in enumerate(self.vol_surf_inds):
                self.connect(self.cp_knot_refine_comp_name+'.'
                             +self.cp_analysis_name_list[field_ind],
                             self.volume_comp_name+str(vol_ind)+'.'
                             +self.cp_analysis_name_list[field_ind])

        self.connect(self.cpiga2xi_comp_name+'.'+self.int_name,
                     self.int_xi_edge_comp_name+'.'+self.int_name)

        self.connect(self.cpiga2xi_comp_name+'.'+self.int_name,
                     self.disp_states_comp_name+'.'+self.int_name)

        self.connect(self.disp_states_comp_name+'.'+self.disp_name,
                     self.int_energy_comp_name+'.'+self.disp_name)

        # For constraints
        if len(self.regu_field) > 0:
            for field_ind, field in enumerate(self.regu_field):
                opt_field_ind = self.opt_field.index(field)
                self.connect(self.inputs_comp_name+'.'
                         +self.cp_design_name_list[opt_field_ind],
                         self.cp_regu_comp_name+'.'
                         +self.cp_design_name_list[opt_field_ind])

        if len(self.pin_field) > 0:
            for field_ind, field in enumerate(self.pin_field):
                opt_field_ind = self.opt_field.index(field)
                self.connect(self.inputs_comp_name+'.'
                         +self.cp_design_name_list[opt_field_ind],
                         self.cp_pin_comp_name+'.'
                         +self.cp_design_name_list[opt_field_ind])

        if len(self.dist_field) > 0:
            for field_ind, field in enumerate(self.dist_field):
                opt_field_ind = self.opt_field.index(field)
                self.connect(self.inputs_comp_name+'.'
                         +self.cp_design_name_list[opt_field_ind],
                         self.cp_dist_comp_name+'.'
                         +self.cp_design_name_list[opt_field_ind])

        # Add design variable, constraints and objective
        for field_ind, field in enumerate(self.opt_field):
            self.add_design_var(self.inputs_comp_name+'.'
                                +self.cp_design_name_list[field_ind],
                                lower=self.design_var_lower[field_ind],
                                upper=self.design_var_upper[field_ind])


        if len(self.regu_field) > 0:
            for field_ind, regu_field in enumerate(self.regu_field):
                self.add_constraint(self.cp_regu_comp_name+'.'
                             +self.cp_regu_name_list[field_ind],
                             upper=0.25, lower=2e-2)

        if len(self.pin_field) > 0:
            for field_ind, pin_field in enumerate(self.pin_field):
                self.add_constraint(self.cp_pin_comp_name+'.'
                             +self.cp_pin_name_list[field_ind],
                             equals=0)

        if len(self.dist_field) > 0:
            for field_ind, dist_field in enumerate(self.dist_field):
                if dist_field == 0:
                    lower_val = 0.1
                else:
                    lower_val = 0.5
                self.add_constraint(self.cp_dist_comp_name+'.'
                             +self.cp_dist_name_list[field_ind],
                             lower=lower_val)

        self.add_constraint(self.int_xi_edge_comp_name+'.'+self.int_xi_edge_name,
                            equals=self.int_xi_edge_cons_val)

        for vol_ind, surf_ind in enumerate(self.vol_surf_inds):
            self.add_constraint(self.volume_comp_name+str(vol_ind)+'.'+self.volume_name+str(vol_ind),
                                upper=1.5*self.vol_val_list[vol_ind])
                                # equals=self.vol_val)

        self.add_objective(self.int_energy_comp_name+'.'
                           +self.int_energy_name,
                           scaler=1)


def clampedBC(spline_generator, side=0, direction=0):
    """
    Apply clamped boundary condition to spline.
    """
    for field in [0,1,2]:
        if field in [1]:
            n_layers = 1
        else:
            n_layers = 2
        scalar_spline = spline_generator.getScalarSpline(field)
        side_dofs = scalar_spline.getSideDofs(direction, side, nLayers=n_layers)
        spline_generator.addZeroDofs(field, side_dofs)

def OCCBSpline2tIGArSpline(surface, num_field=3, quad_deg_const=4, 
                        setBCs=None, side=0, direction=0, index=0):
    """
    Generate ExtractedBSpline from OCC B-spline surface.
    """
    quad_deg = surface.UDegree()*quad_deg_const
    DIR = SAVE_PATH+"spline_data/extraction_"+str(index)+"_init"
    # spline = ExtractedSpline(DIR, quad_deg)
    spline_mesh = NURBSControlMesh4OCC(surface, useRect=False)
    spline_generator = EqualOrderSpline(worldcomm, num_field, spline_mesh)
    if setBCs is not None:
        setBCs(spline_generator, side, direction)
    # spline_generator.writeExtraction(DIR)
    spline = ExtractedSpline(spline_generator, quad_deg)
    return spline

save_files = True
optimizer = 'SNOPT'
save_path = './'
folder_name = "results/"

geom_scale = 2.54e-5  # Convert current length unit to m
E = Constant(68e9)  # Young's modulus, Pa
nu = Constant(0.35)  # Poisson's ratio
penalty_coefficient = 1e3
num_surfs = preprocessor.num_surfs
h_th = [Constant(3.0e-3)]*num_surfs  # Thickness of surfaces, m

if mpirank == 0:
    print("Total DoFs:", preprocessor.total_DoFs)
    print("Number of surfaces:", preprocessor.num_surfs)
    print("Number of intersections:", preprocessor.num_intersections_all)

# # Display B-spline surfaces and intersections using 
# # PythonOCC build-in 3D viewer.
# display, start_display, add_menu, add_function_to_menu = init_display()
# preprocessor.display_surfaces(display, save_fig=False)
# preprocessor.display_intersections(display, color='RED', save_fig=False)

if mpirank == 0:
    print("Creating splines...")
# Create tIGAr extracted spline instances
splines = []
for i in range(num_surfs):
    if i in [0, 1]:
        # Apply clamped BC to surfaces near root
        spline = OCCBSpline2tIGArSpline(
                 preprocessor.BSpline_surfs_refine[i], 
                 setBCs=clampedBC, side=0, direction=0, index=i)
        splines += [spline,]
    else:
        spline = OCCBSpline2tIGArSpline(
                 preprocessor.BSpline_surfs_refine[i], index=i)
        splines += [spline,]

# Create non-matching problem
nonmatching_opt = NonMatchingOpt(splines, E, h_th, nu, comm=worldcomm)

opt_field = [0,1,2]
shopt_surf_inds = [[3,4,5,6,7,8,9,10], [5,6,7,8,9,10], [3,4,5,6,7,8,9,10]]

nonmatching_opt.set_shopt_surf_inds(opt_field=opt_field, 
                                    shopt_surf_inds=shopt_surf_inds)
nonmatching_opt.get_init_CPIGA()
nonmatching_opt.set_geom_preprocessor(preprocessor)
nonmatching_opt.create_mortar_meshes(preprocessor.mortar_nels)

if mpirank == 0:
    print("Setting up mortar meshes...")
nonmatching_opt.mortar_meshes_setup(preprocessor.mapping_list, 
                    preprocessor.intersections_para_coords, 
                    penalty_coefficient, 2)
pressure = Constant(5e2)
f1 = as_vector([Constant(0.), Constant(0.), pressure])
f0 = as_vector([Constant(0.), Constant(0.), Constant(0.)])
force_list = [f1]+[f0]*(num_surfs-1)

source_terms = []
residuals = []
for s_ind in range(nonmatching_opt.num_splines):
    f = force_list[s_ind]
    z = nonmatching_opt.splines[s_ind].rationalize(
        nonmatching_opt.spline_test_funcs[s_ind])
    source_terms += [inner(f, z)*nonmatching_opt.splines[s_ind].dx]
    residuals += [SVK_residual(nonmatching_opt.splines[s_ind], 
                  nonmatching_opt.spline_funcs[s_ind], 
                  nonmatching_opt.spline_test_funcs[s_ind], 
                  E, nu, h_th[s_ind], source_terms[s_ind])]        
nonmatching_opt.set_residuals(residuals)

nonmatching_opt.create_diff_intersections(num_edge_pts=4)

cpsurfd2a = CPSurfDesign2Analysis(preprocessor, opt_field=opt_field, 
            shopt_surf_inds=shopt_surf_inds)

p_list_x = [[1,1],]*len(shopt_surf_inds[0])
p_list_y = [[1,1],]*len(shopt_surf_inds[1])
p_list_z = [[3,1],]*len(shopt_surf_inds[2])
init_p_list = [p_list_x, p_list_y, p_list_z]
init_knots_list = []
for field_ind, field in enumerate(opt_field):
    knots_list = []
    for p in init_p_list[field_ind]:
        knots_list += [[[0]*(p[0]+1)+[1]*(p[0]+1), 
                        [0]*(p[1]+1)+[1]*(p[1]+1)]]
    init_knots_list += [knots_list]


p_ele = [3,3]
p_list_ele = [] 
for field_ind, field in enumerate(opt_field):
    p_list = []
    for i, s_ind in enumerate(shopt_surf_inds[field_ind]):
        p_list += [p_ele]
    p_list_ele += [p_list]

knots_list_ele = []
for field_ind, field in enumerate(opt_field):
    knots_list = []
    for p in p_list_ele[field_ind]:
        knots_list += [[[0]*(p[0]+1)+[1]*(p[0]+1), 
                        [0]*(p[1]+1)+[1]*(p[1]+1)]]
    knots_list_ele += [knots_list]

# Optimize x, y and z direction
cpsurfd2a.set_init_knots_by_field(init_p_list, init_knots_list)
cpsurfd2a.set_order_elevation_by_field(p_list_ele, knots_list_ele)
cpsurfd2a.set_knot_refinement()
cpsurfd2a.get_init_cp_coarse()

cpsurfd2a.set_cp_align(field=0, align_dir_list=[1,1,[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]])
cpsurfd2a.set_cp_align(field=1, align_dir_list=[1,1,1,1,1,1])
cpsurfd2a.set_cp_dist(0, surf_inds=[3,4], rev_dir=False)
cpsurfd2a.set_cp_dist(1, surf_inds=[5,6,7,8,9,10], rev_dir=False)

cpsurfd2a.cp_coarse_dist_deriv_list[0] = coo_matrix(cpsurfd2a.cp_coarse_dist_deriv_list[0].todense()[:,0:4])
cpsurfd2a.init_cp_design[0] = cpsurfd2a.init_cp_design[0][0:4]

# Set up optimization
if save_files:
    nonmatching_opt.create_files(save_path=save_path, folder_name=folder_name, 
                                 refine_mesh=False, ref_nel=64)
model = ShapeOptGroup(nonmatching_opt=nonmatching_opt,
                      cpdesign2analysis=cpsurfd2a)
model.init_parameters()
prob = om.Problem(model=model)


if optimizer.upper() == 'SNOPT':
    prob.driver = om.pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'
    prob.driver.opt_settings['Minor feasibility tolerance'] = 1e-4
    prob.driver.opt_settings['Major feasibility tolerance'] = 1e-4
    prob.driver.opt_settings['Major optimality tolerance'] = 1e-3
    prob.driver.opt_settings['Major iterations limit'] = 50000
    prob.driver.opt_settings['Summary file'] = './SNOPT_summary.out'
    prob.driver.opt_settings['Print file'] = './SNOPT_print.out'
    prob.driver.options['debug_print'] = ['objs', 'desvars']
    prob.driver.options['print_results'] = True
elif optimizer.upper() == 'SLSQP':
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1e-6
    prob.driver.options['disp'] = True
    # prob.driver.options['debug_print'] = ['objs', 'desvars']
    prob.driver.options['maxiter'] = 50000
else:
    raise ValueError("Undefined optimizer: {}".format(optimizer))

prob.setup()
prob.run_driver()

if save_files:
    nonmatching_opt.save_files()