# import sys
# sys.path.append("../")
from GOLDFISH.nonmatching_opt_ffd import *
from scipy.sparse import coo_matrix, csr_matrix
from scipy.linalg import block_diag
from tIGAr import BSplines
# from cpiga2xi import CPIGA2Xi
from scipy.sparse.linalg import splu

from GOLDFISH.nonmatching_opt_ffd import *
from scipy.sparse import coo_matrix, csr_matrix
from scipy.linalg import block_diag
from tIGAr import BSplines
from igakit.cad import NURBS
from scipy.optimize import fsolve, newton_krylov
from scipy.sparse import coo_matrix, bmat
# from scipy.linalg import block_diag
from scipy.sparse.linalg import spsolve

from PENGoLINS.occ_preprocessing import *
from PENGoLINS.igakit_utils import *

class CPIGA2XiCoarseImOperation(object):
    def __init__(self, nonmatching_opt):
        self.nonmatching_opt = nonmatching_opt
        self.preprocessor = self.nonmatching_opt.preprocessor
        self.cpiga2xi = self.nonmatching_opt.cpiga2xi
        self.opt_field = self.nonmatching_opt.opt_field

    def apply_nonlinear(self, xi_flat):
        # print("Running apply nonlinear ...")
        return self.cpiga2xi.residual(xi_flat)

    def solve_nonlinear(self, xi_flat_init):
        # print("Running solve nonlinear ...")
        return self.cpiga2xi.solve_xi(xi_flat_init)

    def linearize(self, xi_flat, coo=True):
        # print("Running linearize...")
        self.cpiga2xi.update_occ_surfs()
        self.dRdxi_mat = self.cpiga2xi.dRdxi(xi_flat, coo=coo)
        # print("xi_flat:", xi_flat)
        # print("xi_flat norm:", np.linalg.norm(xi_flat))
        # print("det dRdxi:", np.linalg.det(self.dRdxi_mat.todense()))

        self.dRdCP_mat_list = []
        for i, field in enumerate(self.opt_field):
            self.dRdCP_mat_list += [self.cpiga2xi.dRdCP(xi_flat, 
                                    field, coo=coo)]

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
    from GOLDFISH.tests.test_tbeam_mint import preprocessor, nonmatching_opt

    #################################
    nonmatching_opt.set_xi_diff_info(preprocessor)
    #################################

    cpiga2xi_imop = CPIGA2XiCoarseImOperation(nonmatching_opt)

    int_ind = 0
    xi_flat = cpiga2xi_imop.cpiga2xi.xis_flat[int_ind]
    cpiga2xi_imop.apply_nonlinear(xi_flat)
    cpiga2xi_imop.solve_nonlinear(xi_flat)
    cpiga2xi_imop.linearize(xi_flat)