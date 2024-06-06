from GOLDFISH.nonmatching_opt_ffd import *
from scipy.sparse import coo_matrix, csr_matrix
from scipy.linalg import block_diag
from tIGAr import BSplines
from cpiga2xi import CPIGA2Xi
from scipy.sparse.linalg import splu

class CPIGA2XiImOperation(CPIGA2Xi):
    def __init__(self, preprocessor, int_indices_diff=None, opt_field=[0,1,2]):
        super().__init__(preprocessor, int_indices_diff, opt_field)

    def apply_nonlinear(self, xi_flat):
        # print("Running apply nonlinear ...")
        return self.residual(xi_flat)

    def solve_nonlinear(self, xi_flat_init):
        # print("Running solve nonlinear ...")
        return self.solve_xi(xi_flat_init)

    def linearize(self, xi_flat, coo=True):
        # print("Running linearize...")
        self.update_occ_surfs()
        self.dRdxi_mat = self.dRdxi(xi_flat, coo=coo)
        self.dRdCP_mat_list = []
        for i, field in enumerate(self.opt_field):
            self.dRdCP_mat_list += [self.dRdCP(xi_flat, field, coo=coo)]

        # print('dRdxi norm:', np.linalg.norm(self.dRdxi_mat.todense()))
        # print('dRdxi det:', np.linalg.det(self.dRdxi_mat.todense()))
        self.lu_fwd = splu(self.dRdxi_mat.tocsc())
        self.lu_rev = splu(self.dRdxi_mat.T.tocsc())
        return self.dRdxi_mat, self.dRdCP_mat_list

    def apply_linear_fwd(self, d_inputs_array_list=None, 
                         d_outputs_array=None, 
                         d_residuals_array=None):
        """
        ``d_inputs_array_list`` is the list of control points in IGA DoFs
        ``d_outputs_array`` is the intersections' parametric coordinates
        ``d_residuals_array`` is the implicit residuals beteen CP and xi
        """
        if d_residuals_array is not None:
            if d_outputs_array is not None:
                dres_array = self.dRdxi_mat*d_outputs_array
                d_residuals_array[:] += dres_array
            if d_inputs_array_list is not None:
                for i, field in enumerate(self.opt_field):
                    dres_array = self.dRdCP_mat_list[i]*d_inputs_array_list[i]
                    d_residuals_array[:] += dres_array
        return d_residuals_array

    def apply_linear_rev(self, d_inputs_array_list=None, 
                         d_outputs_array=None,
                         d_residuals_array=None):
        """
        ``d_inputs_array_list`` is the list of control points in IGA DoFs
        ``d_outputs_array`` is the intersections' parametric coordinates
        ``d_residuals_array`` is the implicit residuals beteen CP and xi
        """
        if d_residuals_array is not None:
            if d_outputs_array is not None:
                dxi_array = self.dRdxi_mat.T*d_residuals_array
                d_outputs_array[:] += dxi_array
            if d_inputs_array_list is not None:
                for i, field in enumerate(self.opt_field):
                    dcp_iga_array = self.dRdCP_mat_list[i].T*d_residuals_array
                    d_inputs_array_list[i][:] += dcp_iga_array
        return d_inputs_array_list, d_outputs_array

    def solve_linear_fwd(self, d_outputs_array, d_residuals_array):
        d_outputs_array[:] = self.lu_fwd.solve(d_residuals_array)
        return d_outputs_array

    def solve_linear_rev(self, d_outputs_array, d_residuals_array):
        d_residuals_array[:] = self.lu_rev.solve(d_outputs_array)
        return d_residuals_array


if __name__ == '__main__':
    # from GOLDFISH.tests.test_tbeam import nonmatching_opt
    # from GOLDFISH.tests.test_slr import nonmatching_opt
    from PENGoLINS.occ_preprocessing import *

    filename_igs = "./geometry/init_Tbeam_geom_moved.igs"
    igs_shapes = read_igs_file(filename_igs, as_compound=False)
    occ_surf_list = [topoface2surface(face, BSpline=True) 
                     for face in igs_shapes]
    occ_surf_data_list = [BSplineSurfaceData(surf) for surf in occ_surf_list]
    num_surfs = len(occ_surf_list)
    p = occ_surf_data_list[0].degree[0]

    # Geometry preprocessing and surface-surface intersections computation
    preprocessor = OCCPreprocessing(occ_surf_list, reparametrize=False, 
                                    refine=False)
    print("Computing intersections...")
    int_data_filename = "int_data.npz"
    if os.path.isfile(int_data_filename):
        preprocessor.load_intersections_data(int_data_filename)
    else:
        preprocessor.compute_intersections(mortar_refine=2)
        preprocessor.save_intersections_data(int_data_filename)

    if mpirank == 0:
        print("Total DoFs:", preprocessor.total_DoFs)
        print("Number of intersections:", preprocessor.num_intersections_all)

    cpiga2xi_imop = CPIGA2XiImOperation(preprocessor)

    int_ind = 0
    xi_flat = cpiga2xi_imop.xis_flat[int_ind]
    cpiga2xi_imop.apply_nonlinear(xi_flat)
    cpiga2xi_imop.solve_nonlinear(xi_flat)
    cpiga2xi_imop.linearize(xi_flat)