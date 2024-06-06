from GOLDFISH.nonmatching_opt_ffd import *
import openmdao.api as om
from openmdao.api import Problem


class CPFFDReguCompAgg(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('nonmatching_opt_ffd')
        self.options.declare('input_cpffd_name_pre', default='CP_FFD')
        self.options.declare('output_cpregu_name_pre', default='CP_FFD_regu')
        self.options.declare('rho', default=1000)
        self.options.declare('m', default=None)

    def init_parameters(self):
        # print("*"*50)
        self.nonmatching_opt_ffd = self.options['nonmatching_opt_ffd']
        self.input_cpffd_name_pre = self.options['input_cpffd_name_pre']
        self.output_cpregu_name_pre = self.options['output_cpregu_name_pre']

        self.rho = self.options['rho']
        self.m = self.options['m']

        self.opt_field = self.nonmatching_opt_ffd.opt_field
        # self.input_shape = self.nonmatching_opt_ffd.shopt_cpffd_size
        # self.output_shapes = self.nonmatching_opt_ffd.shopt_cpregu_sizes
        # self.derivs = self.nonmatching_opt_ffd.shopt_dcpregudcpffd_list

        self.init_cpffd = []
        if self.nonmatching_opt_ffd.shopt_multiffd:
            self.input_shape = np.sum(self.nonmatching_opt_ffd.shopt_cpffd_size_list)
            self.output_shapes = self.nonmatching_opt_ffd.shopt_cpregu_sizes_multiffd
            self.derivs = self.nonmatching_opt_ffd.shopt_dcpregudcpmultiffd
            for i, field in enumerate(self.opt_field):
                self.init_cpffd += [self.nonmatching_opt_ffd.get_init_CPFFD_multiFFD(field),]
        else:
            self.input_shape = self.nonmatching_opt_ffd.shopt_cpffd_size
            self.output_shapes = self.nonmatching_opt_ffd.shopt_cpregu_sizes
            self.derivs = self.nonmatching_opt_ffd.shopt_dcpregudcpffd_list
            for i, field in enumerate(self.opt_field):
                self.init_cpffd += [self.nonmatching_opt_ffd.shopt_cpffd_flat[:,field]]
        
        if self.m is None:
            self.m = []
            for i, field in enumerate(self.opt_field):
                self.m += [np.max(self.init_cpffd[i])\
                          -np.min(self.init_cpffd[i])]

        self.input_cpffd_name_list = []
        self.output_cpregu_name_list = []
        for i, field in enumerate(self.opt_field):
            self.input_cpffd_name_list += \
                [self.input_cpffd_name_pre+str(field)]
            self.output_cpregu_name_list += \
                [self.output_cpregu_name_pre+str(field)]

        self.cpffd_regu_row_ind = []
        self.cpffd_regu_col_ind = []
        self.cpffd_regu_data = []
        for i, field in enumerate(self.opt_field):
            self.cpffd_regu_row_ind += [self.derivs[i].row]
            self.cpffd_regu_col_ind += [self.derivs[i].col]
            self.cpffd_regu_data += [self.derivs[i].data]

    def setup(self):
        for i, field in enumerate(self.opt_field):
            self.add_input(self.input_cpffd_name_list[i],
                           shape=self.input_shape,
                           val=self.init_cpffd[i])
            self.add_output(self.output_cpregu_name_list[i])
            self.declare_partials(self.output_cpregu_name_list[i],
                                  self.input_cpffd_name_list[i])

    def compute(self, inputs, outputs):
        """
        di_min = -m*(sum(-di/m+1)**(rho+1)/sum(-di/m+1)**rho-1),
        where di = Pi - Pip1
        """
        for i, field in enumerate(self.opt_field):
            cpffd_array = inputs[self.input_cpffd_name_list[i]]
            ip_num = 0
            ip_den = 0
            for regu_ind in range(self.derivs[i].shape[0]):
                dof0 = self.cpffd_regu_col_ind[i][regu_ind*2]
                dof1 = self.cpffd_regu_col_ind[i][regu_ind*2+1]
                coeff0 = self.cpffd_regu_data[i][regu_ind*2]
                coeff1 = self.cpffd_regu_data[i][regu_ind*2+1]
                regu_val = cpffd_array[dof0]*coeff0 + cpffd_array[dof1]*coeff1
                regu_val_norm = -regu_val/self.m[i]+1
                regu_val_norm_rho_pow = regu_val_norm**self.rho
                ip_num += regu_val_norm_rho_pow*regu_val_norm
                ip_den += regu_val_norm_rho_pow
            cpffd_regu_agg = -(ip_num/ip_den-1)*self.m[i]
            outputs[self.output_cpregu_name_list[i]] = cpffd_regu_agg

    def compute_partials(self, inputs, partials):
        for i, field in enumerate(self.opt_field):
            cpffd_array = inputs[self.input_cpffd_name_list[i]]
            ip_pow_rho_m1 = np.zeros(self.derivs[i].shape[0]) # di**(rho-1)
            ip_pow_rho = np.zeros(self.derivs[i].shape[0])    # di**rho
            ip_pow_rho_p1 = np.zeros(self.derivs[i].shape[0]) # di**(rho+1)
            for regu_ind in range(self.derivs[i].shape[0]):
                dof0 = self.cpffd_regu_col_ind[i][regu_ind*2]
                dof1 = self.cpffd_regu_col_ind[i][regu_ind*2+1]
                coeff0 = self.cpffd_regu_data[i][regu_ind*2]
                coeff1 = self.cpffd_regu_data[i][regu_ind*2+1]
                regu_val = cpffd_array[dof0]*coeff0 + cpffd_array[dof1]*coeff1
                regu_val_norm = -regu_val/self.m[i]+1
                ip_pow_rho_m1[regu_ind] = regu_val_norm**(self.rho-1)
                ip_pow_rho[regu_ind] = ip_pow_rho_m1[regu_ind]*regu_val_norm
                ip_pow_rho_p1[regu_ind] = ip_pow_rho[regu_ind]*regu_val_norm

            ip_pow_rho_sum = np.sum(ip_pow_rho)
            ip_pow_rho_p1_sum = np.sum(ip_pow_rho_p1)
            ip_num_prime_vec = np.zeros(self.input_shape)
            ip_den_prime_vec = np.zeros(self.input_shape)
            deriv_vec = np.zeros([self.input_shape])
            for cp_ind in range(self.input_shape):
                regu_col_inds = np.where(self.cpffd_regu_col_ind[i]==cp_ind)[0]
                if len(regu_col_inds) > 0:
                    for j, regu_col_ind in enumerate(regu_col_inds):
                        coeff0 = self.cpffd_regu_data[i][regu_col_ind]
                        regu_row_ind = self.cpffd_regu_row_ind[i][regu_col_ind]
                        coeff1 = (self.rho+1)*(-coeff0/self.m[i])
                        ip_num_prime_vec[cp_ind] += coeff1*ip_pow_rho[regu_row_ind]
                        coeff2 = self.rho*(-coeff0/self.m[i])
                        ip_den_prime_vec[cp_ind] += coeff2*ip_pow_rho_m1[regu_row_ind]
                deriv_vec[cp_ind] = -self.m[i]*\
                    (ip_num_prime_vec[cp_ind]*ip_pow_rho_sum\
                     -ip_den_prime_vec[cp_ind]*ip_pow_rho_p1_sum)\
                    /(ip_pow_rho_sum**2)
            partials[self.output_cpregu_name_list[i],
                     self.input_cpffd_name_list[i]] = deriv_vec


if __name__ == "__main__":
    # from GOLDFISH.tests.test_tbeam import nonmatching_opt
    # # from GOLDFISH.tests.test_slr import nonmatching_opt

    # ffd_block_num_el = [4,4,1]
    # p = 3
    # # Create FFD block in igakit format
    # cp_ffd_lims = nonmatching_opt.cpsurf_lims
    # for field in [2]:
    #     cp_range = cp_ffd_lims[field][1] - cp_ffd_lims[field][0]
    #     cp_ffd_lims[field][0] = cp_ffd_lims[field][0] - 0.2*cp_range
    #     cp_ffd_lims[field][1] = cp_ffd_lims[field][1] + 0.2*cp_range
    # FFD_block = create_3D_block(ffd_block_num_el, p, cp_ffd_lims)
    # nonmatching_opt.set_shopt_FFD(FFD_block.knots, FFD_block.control)
    # nonmatching_opt.set_shopt_regu_CPFFD(shopt_regu_dir=[None, None, None],
    #                                      shopt_regu_side=[None, None, None])

    # prob = Problem()
    # comp = CPFFDReguCompAgg(nonmatching_opt_ffd=nonmatching_opt)
    # comp.init_parameters()
    # prob.model = comp
    # prob.setup()
    # prob.run_model()
    # prob.model.list_outputs()
    # print('check_partials:')
    # prob.check_partials(compact_print=True)

    import sys
    import numpy as np
    import matplotlib.pyplot as plt
    import openmdao.api as om
    from igakit.cad import *
    from igakit.io import VTK
    # from GOLDFISH.nonmatching_opt_om import *

    class SplineBC(object):
        """
        Setting Dirichlet boundary condition to tIGAr spline generator.
        """
        def __init__(self, directions=[0,1], sides=[[0,1],[0,1]], 
                    fields=[[[0,1,2],[0,1,2]],[[0,1,2],[0,1,2]]],
                    n_layers=[[1,1],[1,1]]):
            self.fields = fields
            self.directions = directions
            self.sides = sides
            self.n_layers = n_layers

        def set_bc(self, spline_generator):
            for direction in self.directions:
                for side in self.sides[direction]:
                    for field in self.fields[direction][side]:
                        scalar_spline = spline_generator.getScalarSpline(field)
                        side_dofs = scalar_spline.getSideDofs(direction,
                                    side, nLayers=self.n_layers[direction][side])
                        spline_generator.addZeroDofs(field, side_dofs)

    def OCCBSpline2tIGArSpline(surface, num_field=3, quad_deg_const=3, 
                            spline_bc=None, index=0):
        """
        Generate ExtractedBSpline from OCC B-spline surface.
        """
        quad_deg = surface.UDegree()*quad_deg_const
        # DIR = SAVE_PATH+"spline_data/extraction_"+str(index)+"_init"
        # spline = ExtractedSpline(DIR, quad_deg)
        spline_mesh = NURBSControlMesh4OCC(surface, useRect=False)
        spline_generator = EqualOrderSpline(worldcomm, num_field, spline_mesh)
        if spline_bc is not None:
            spline_bc.set_bc(spline_generator)
        # spline_generator.writeExtraction(DIR)
        spline = ExtractedSpline(spline_generator, quad_deg)
        return spline

    path = '/home/han/OneDrive/github/GOLDFISH/demos_om/shape_opt_mint/T-beam/'
    opt_field = [0, 2]
    filename_igs = path+"geometry/init_Tbeam_geom_curved.igs"
    igs_shapes = read_igs_file(filename_igs, as_compound=False)
    occ_surf_list = [topoface2surface(face, BSpline=True) 
                    for face in igs_shapes]
    occ_surf_data_list = [BSplineSurfaceData(surf) for surf in occ_surf_list]
    num_surfs = len(occ_surf_list)
    p = occ_surf_data_list[0].degree[0]

    # Define material and geometric parameters
    E = Constant(1.0e12)
    nu = Constant(0.)
    h_th = Constant(0.1)
    penalty_coefficient = 1.0e3
    pressure = Constant(1.)

    fields0 = [None, [[0,1,2]],]
    spline_bc0 = SplineBC(directions=[1], sides=[None, [0]],
                        fields=fields0, n_layers=[None, [1]])
    spline_bcs = [spline_bc0,]*2

    # Geometry preprocessing and surface-surface intersections computation
    preprocessor = OCCPreprocessing(occ_surf_list, reparametrize=False, 
                                    refine=False)
    print("Computing intersections...")
    int_data_filename = "int_data.npz"
    # if os.path.isfile(int_data_filename):
    #     preprocessor.load_intersections_data(int_data_filename)
    # else:
    #     preprocessor.compute_intersections(mortar_refine=2)
    #     preprocessor.save_intersections_data(int_data_filename)

    preprocessor.compute_intersections(mortar_refine=2)
    if mpirank == 0:
        print("Total DoFs:", preprocessor.total_DoFs)
        print("Number of intersections:", preprocessor.num_intersections_all)


    if mpirank == 0:
        print("Creating splines...")
    # Create tIGAr extracted spline instances
    splines = []
    for i in range(num_surfs):
            spline = OCCBSpline2tIGArSpline(preprocessor.BSpline_surfs[i], 
                                            spline_bc=spline_bcs[i], index=i)
            splines += [spline,]

    # Create non-matching problem
    nonmatching_opt = NonMatchingOptFFD(splines, E, h_th, nu, 
                                        opt_field=opt_field, comm=worldcomm)
    nonmatching_opt.create_mortar_meshes(preprocessor.mortar_nels)

    if mpirank == 0:
        print("Setting up mortar meshes...")
    nonmatching_opt.mortar_meshes_setup(preprocessor.mapping_list, 
                        preprocessor.intersections_para_coords, 
                        penalty_coefficient, 2)
    pressure = -Constant(1.)
    f = as_vector([Constant(0.), Constant(0.), pressure])
    source_terms = []
    residuals = []
    for s_ind in range(nonmatching_opt.num_splines):
        z = nonmatching_opt.splines[s_ind].rationalize(
            nonmatching_opt.spline_test_funcs[s_ind])
        source_terms += [inner(f, z)*nonmatching_opt.splines[s_ind].dx]
        residuals += [SVK_residual(nonmatching_opt.splines[s_ind], 
                    nonmatching_opt.spline_funcs[s_ind], 
                    nonmatching_opt.spline_test_funcs[s_ind], 
                    E, nu, h_th, source_terms[s_ind])]
    nonmatching_opt.set_residuals(residuals)

    shopt_multi_ffd_inds = [[0], [1]]
    nonmatching_opt.set_shopt_multiFFD_surf_inds(shopt_multi_ffd_inds)

    #################################################
    num_shopt_ffd = nonmatching_opt.num_shopt_ffd
    shopt_ffd_lims_multiffd = nonmatching_opt.shopt_cpsurf_lims_multiffd

    shopt_ffd_num_el = [[1,1,1], [1,2,3]]
    shopt_ffd_p = [2]*num_shopt_ffd
    extrude_dir = [2,0]

    shopt_ffd_block_list = []
    for ffd_ind in range(num_shopt_ffd):
        field = extrude_dir[ffd_ind]
        cp_range = shopt_ffd_lims_multiffd[ffd_ind][field][1]\
                -shopt_ffd_lims_multiffd[ffd_ind][field][0]
        shopt_ffd_lims_multiffd[ffd_ind][field][1] = \
            shopt_ffd_lims_multiffd[ffd_ind][field][1] + 0.1*cp_range
        shopt_ffd_lims_multiffd[ffd_ind][field][0] = \
            shopt_ffd_lims_multiffd[ffd_ind][field][0] - 0.1*cp_range
        shopt_ffd_block_list += [create_3D_block(shopt_ffd_num_el[ffd_ind],
                                        shopt_ffd_p[ffd_ind],
                                        shopt_ffd_lims_multiffd[ffd_ind])]

    # for ffd_ind in range(num_shopt_ffd):
    #     vtk_writer = VTKWriter()
    #     vtk_writer.write("./geometry/tbeam_shopt_ffd_block_init"+str(ffd_ind)+".vtk", 
    #                     shopt_ffd_block_list[ffd_ind])
    #     vtk_writer.write_cp("./geometry/tbeam_shopt_ffd_cp_init"+str(ffd_ind)+".vtk", 
    #                     shopt_ffd_block_list[ffd_ind])


    shopt_ffd_knots_list = [ffd_block.knots for ffd_block 
                            in shopt_ffd_block_list]
    shopt_ffd_control_list = [ffd_block.control for ffd_block 
                            in shopt_ffd_block_list]
    print("Setting multiple thickness FFD blocks ...")
    nonmatching_opt.set_shopt_multiFFD(shopt_ffd_knots_list, 
                                        shopt_ffd_control_list)

    ########### Set constraints info #########
    a0 = nonmatching_opt.set_shopt_regu_CP_multiFFD(shopt_regu_dir_list=[[None, None], 
                                                                    [None, None]], 
                                            shopt_regu_side_list=[[None, None], 
                                                                    [None, None]])
    a1 = nonmatching_opt.set_shopt_pin_CP_multiFFD(0, pin_dir0_list=['whole', None], 
                                            pin_side0_list=None,
                                            pin_dir1_list=None, 
                                            pin_side1_list=None)
    a2 = nonmatching_opt.set_shopt_pin_CP_multiFFD(2, pin_dir0_list=['whole', None], 
                                            pin_side0_list=None,
                                            pin_dir1_list=None, 
                                            pin_side1_list=None)
    a3 = nonmatching_opt.set_shopt_align_CP_multiFFD(shopt_align_dir_list=[1,1])

    prob = Problem()
    comp = CPFFDReguCompAgg(nonmatching_opt_ffd=nonmatching_opt)
    comp.init_parameters()
    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print('check_partials:')
    prob.check_partials(compact_print=True)