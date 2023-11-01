import m3l
import array_mapper as am
import csdl
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from lsdo_modules.module.module import Module
import numpy as np
from scipy.sparse.linalg import spsolve
from typing import Tuple, Dict

from klshell_pde import (KLShellPDE, KLShellModule, 
                          RadialBasisFunctions, NodalMap, 
                          memory_usage_psutil)

class LinearKLShellSurface(Module):
    def initialize(self, kwargs):
        self.parameter.declare('meshes', types=dict)
        self.parameter.declare('mesh_units', default='m')


class LinearKLShellCSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('nonmatching_opt')
        self.parameters.declare('klshell_pde')
        self.parameters.declare('shells', default={}) # material properties

    def define(self):
        nonmatching_opt = self.parameters['nonmatching_opt']
        klshell_pde = self.parameters['klshell_pde']
        shells = self.parameters['shells']
        klshell_module = KLShellModule(
                        nonmatching_opt=nonmatching_opt,
                        klshell_pde=klshell_pde,
                        shells=shells)
        klshell_module.init_parameters()
        self.add_module(klshell_module, name='klshell')


class KLShell(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('component')
        # self.parameters.declare('mesh')
        self.parameters.declare('nonmatching_opt')
        self.parameters.declare('klshell_pde')
        self.parameters.declare('shells', default={})

    def assign_attributes(self):
        self.component = self.parameters['component']
        self.nonmatching_opt = self.parameters['nonmatching_opt']
        # self.mesh = self.parameters['mesh']
        self.klshell_pde = self.parameters['klshell_pde']
        self.shells = self.parameters['shells']

    def compute(self) -> csdl.Model:
        shells = self.parameters['shells']
        klshell_pde = self.parameters['klshell_pde']

        csdl_model = LinearKLShellCSDL(
                     module=self,
                     nonmatching_opt=self.nonmatching_opt,
                     klshell_pde=klshell_pde,
                     shells=shells)
        return csdl_model

    def evaluate(self, forces:m3l.Variable=None,
                 moments:m3l.Variable=None,
                 thicknesses:m3l.Variable=None) -> m3l.Variable:

        shell_name = list(self.parameters['shells'].keys())[0]

        self.component = self.parameters['component']
        self.name = f'{self.component.name}_klshell_model'

        mesh_shape = self.klshell_pde.solid_mesh_phys.shape # TODO: figure out this data shape

        self.arguments = {}
        if thicknesses is not None:
            # self.arguments[f'{shell_name}_thicknesses'] = thicknesses
            self.arguments[shell_name+'_hth_design'] = thicknesses
        if forces is not None:
            # self.arguments[f'{shell_name}_forces'] = forces
            self.arguments[shell_name+'_forces'] = forces
        if moments is not None:
            self.arguments[f'{shell_name}_moments'] = moments

        displacements = m3l.Variable(name=f'{shell_name}_displacement',
                        shape=mesh_shape, operation=self)
        stresses = m3l.Variable(name=f'{shell_name}_stress',
                                shape=mesh_shape[0], operation=self)
        # mass = m3l.Variable(name='mass', shape=(1,), operation=self)
        volume = m3l.Variable(name='volume', shape=(1,), operation=self)
        # inertia_tensor = m3l.Variable(name='inertia_tensor',
        #                               shape=(3,3), operation=self)
        return displacements, stresses, volume


class KLShellForces(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('component')
        # self.parameters.declare('mesh')
        self.parameters.declare('nonmatching_opt')
        self.parameters.declare('klshell_pde')
        self.parameters.declare('shells', default={})

    def assign_attributes(self):
        self.component = self.parameters['component']
        self.nonmatching_opt = self.parameters['nonmatching_opt']
        # self.mesh = self.parameters['mesh']
        self.klshell_pde = self.parameters['klshell_pde']
        self.shells = self.parameters['shells']

    def compute(self) -> csdl.Model:
        shell_name = list(self.parameters['shells'].keys())[0]

        self.klshell_pde = self.parameters['klshell_pde']

        nodal_forces = self.arguments['nodal_forces']

        csdl_model = ModuleCSDL()

        mesh_value = self.klshell_pde.solid_mesh_phys # TODO: figure out these two variables 
        mesh_shape = mesh_value.shape

        force_map = self.fmap(oml=self.nodal_forces_mesh.value.reshape((-1,3)))

        flattened_nodal_forces_shape = (np.prod(nodal_forces.shape[:-1]),
                                        nodal_forces.shape[-1])

        nodal_forces_csdl = csdl_model.register_module_input(
                            name='nodal_forces',
                            shape=nodal_forces.shape)

        # print("Test 3 ........")
        # print("flattened_nodal_forces:", flattened_nodal_forces_shape)
        # print("nodal_forces.shape:", nodal_forces.shape)


        flattened_nodal_forces = csdl.reshape(nodal_forces_csdl,
                                 new_shape=flattened_nodal_forces_shape)

        force_map_csdl = csdl_model.create_input(
                         f'nodal_to_{shell_name}_forces_map',
                         val=force_map)

        flatenned_shell_mesh_forces = csdl.matmat(force_map_csdl,
                                      flattened_nodal_forces)

        output_shape = tuple(mesh_shape[:-1]) + (nodal_forces.shape[-1],)

        # print("Test 4 ..........")
        # print("force_map shape", force_map.shape)
        # print("output shape: ", output_shape)

        shell_mesh_forces = csdl.reshape(flatenned_shell_mesh_forces, 
                                         new_shape=output_shape)

        csdl_model.register_module_output(f'{shell_name}_forces', 
                                          -1.0*shell_mesh_forces)

        return csdl_model

    def evaluate(self, nodal_forces:m3l.Variable,
                 nodal_forces_mesh:am.MappedArray) -> m3l.Variable:

        self.component = self.parameters['component']
        self.name = f'{self.component.name}_klshell_force_mapping'

        self.nodal_forces_mesh = nodal_forces_mesh
        shell_name = list(self.parameters['shells'].keys())[0]


        mesh_shape = self.klshell_pde.solid_mesh_phys.shape 
        self.arguments = {'nodal_forces': nodal_forces}
        output_shape = tuple(mesh_shape[:-1]) + (nodal_forces.shape[-1],)
        shell_forces = m3l.Variable(name=f'{shell_name}_forces',
                            shape=output_shape, operation=self)
        return shell_forces

    def fmap(self, oml):

        print("Inspection fmap 0: Memory usage: {:8.2f} MB.\n"\
          .format(memory_usage_psutil()))

        self.G_mat = NodalMap(self.klshell_pde.solid_mesh_phys, 
                              oml, 
                              RBF_width_par=0.05, 
                              RBF_func=RadialBasisFunctions.Gaussian, 
                              column_scaling_vec=self.klshell_pde.bf_sup_sizes)

        rhs_mats = self.G_mat.map.T
        mat_f_sp = self.klshell_pde.compute_sparse_mass_matrix()
        weights = spsolve(mat_f_sp, rhs_mats)

        print("Inspection fmap 1: Memory usage: {:8.2f} MB.\n"\
          .format(memory_usage_psutil()))
        return weights


class KLShellNodalDisplacements(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('component')
        # self.parameters.declare('mesh')
        self.parameters.declare('nonmatching_opt')
        self.parameters.declare('klshell_pde')
        self.parameters.declare('shells', default={})


    def assign_attributes(self):
        self.component = self.parameters['component']
        self.nonmatching_opt = self.parameters['nonmatching_opt']
        # self.mesh = self.parameters['mesh']
        self.klshell_pde = self.parameters['klshell_pde']
        self.shells = self.parameters['shells']

    def compute(self) -> csdl.Model:
        shell_name = list(self.parameters['shells'].keys())[0]
        self.klshell_pde = self.parameters['klshell_pde']
        mesh_value = self.klshell_pde.solid_mesh_phys
        self.klshell_pde.compute_basisfunc_support_size()

        nodal_displacements_mesh = self.nodal_displacements_mesh
        shell_displacements = self.arguments[f'{shell_name}_displacement']

        csdl_model = ModuleCSDL()

        displacement_map = self.umap(oml=nodal_displacements_mesh.value.reshape((-1,3)))

        shell_displacements_csdl = csdl_model.register_module_input(
                                   name=f'{shell_name}_displacement',
                                   shape=shell_displacements.shape)

        displacement_map_csdl = csdl_model.create_input(
                            f'{shell_name}_displacements_to_nodal_displacements',
                            val=displacement_map)

        nodal_displacements = csdl.matmat(displacement_map_csdl,
                                          shell_displacements_csdl)

        csdl_model.register_module_output(f'{shell_name}_nodal_displacement',
                                            nodal_displacements)        
        csdl_model.register_module_output(f'{shell_name}_tip_displacement',
                                          csdl.max(nodal_displacements, rho=1000))
        return csdl_model

    def evaluate(self, shell_displacements:m3l.Variable,
                 nodal_displacements_mesh:am.MappedArray) -> m3l.Variable:
        self.component = self.parameters['component']
        shell_name = list(self.parameters['shells'].keys())[0]   # this is only taking the first mesh added to the solver.
        self.name = f'{self.component.name}_klshell_displacement_map'
        self.arguments = {f'{shell_name}_displacement': shell_displacements}
        self.nodal_displacements_mesh = nodal_displacements_mesh

        nodal_displacements = m3l.Variable(name=f'{shell_name}_nodal_displacement',
                                            shape=nodal_displacements_mesh.shape,
                                            operation=self)
        tip_displacement = m3l.Variable(name=f'{shell_name}_tip_displacement',
                                    shape=(1,),
                                    operation=self)
        return nodal_displacements, tip_displacement

    def umap(self, oml):

        print("Inspection umap 0: Memory usage: {:8.2f} MB.\n"\
                .format(memory_usage_psutil()))
        # Up = W*Us
        G_mat = NodalMap(self.klshell_pde.solid_mesh_phys, 
                         oml, RBF_width_par=10.0,
                         column_scaling_vec=self.klshell_pde.bf_sup_sizes)
        weights = G_mat.map

        print("Inspection umap 1: Memory usage: {:8.2f} MB.\n"\
                .format(memory_usage_psutil()))
        return weights


if __name__ == '__main__':
    pass