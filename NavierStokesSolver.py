import dune.fem
from dune.grid import cartesianDomain, reader
from dune.alugrid import aluConformGrid as leafGridView
from dune.fem import integrate, threading
from dune.fem.view import adaptiveLeafGridView
from dune.fem.space import lagrange, finiteVolume
from dune.common import comm
from dune.fem.scheme import galerkin as solutionScheme
from dune.fem.function import gridFunction
from dune.ufl import Constant, DirichletBC
from ufl import (
    TrialFunction,
    TestFunction,
    SpatialCoordinate,
    div,
    grad,
    inner,
    dx,
    ds,
    FacetNormal,
    nabla_grad,
    sym,
    Identity,
    as_vector,
    outer,
    dot,
    sqrt as ufl_sqrt,
    CellVolume,
)
from math import sqrt as math_sqrt
from gmsh2dgf import gmsh2DGF as mesh2DGF
import os
import numpy as np
from tqdm import tqdm
import pygmsh
from matplotlib import pyplot as plt
from mpi4py import MPI

mpi_comm = MPI.COMM_WORLD
rank = mpi_comm.Get_rank()

# fem.threading.useMax()
# threading.use = 1

if comm.rank == 0:
    print("Running on master process.")
else:
    print(f"Running on worker process with rank {comm.rank}.")


class NavierStokesSolver:
    def __init__(
        self, dt_value: float, H: float, L: float, rho_value: float, mu_value: float
    ):
        self.L = L
        self.H = H
        self.rho = Constant(rho_value, "rho")
        self.mu = Constant(mu_value, "mu")
        self.dt = Constant(dt_value, "dt")
        self.f = as_vector([Constant(0, "fx"), Constant(0, "fy")])
        self.gridView = None
        self.u_prelim = None
        self.u_prev = None
        self.u_h = None
        self.p_h = None
        self.n = None
        self.u = None
        self.v = None
        self.p_prev = None
        self.velocitySpace = None
        self.pressureSpace = None
        self.x = None
        self.dbc_velocity = None
        self.dbc_pressure = None
        self.scheme_1 = None
        self.scheme_2 = None
        self.scheme_3 = None
        self.solution_p = None
        self.solution_u = None
        self.solution_provided = False
        self.inflow_factor = None
        self.inflow_ramp = None

    def buildForms(self, initial_p_lambda=None):
        dim = self.gridView.dimension
        self.velocitySpace = lagrange(
            self.gridView, order=2, dimRange=dim, storage="petsc"
        )
        self.pressureSpace = lagrange(self.gridView, order=1, storage="petsc")

        # verbose info
        print(f"Velocity space has {self.velocitySpace.size} degrees of freedom.")
        print(f"Pressure space has {self.pressureSpace.size} degrees of freedom.")

        u = TrialFunction(self.velocitySpace)
        v = TestFunction(self.velocitySpace)
        p = TrialFunction(self.pressureSpace)
        q = TestFunction(self.pressureSpace)

        def epsilon(u):
            return sym(nabla_grad(u))

        def sigma(u, p):
            return 2.0 * self.mu * epsilon(u) - p * Identity(dim)

        self.x = SpatialCoordinate(self.velocitySpace)

        self.u_prelim = self.velocitySpace.function(name="u_prelim")
        self.u_prev = self.velocitySpace.function(name="u_prev")
        self.u_h = self.velocitySpace.function(name="u_h")
        self.u = u
        self.v = v

        self.p_h = self.pressureSpace.function(name="p_h")
        if initial_p_lambda is not None:
            self.p_prev = self.pressureSpace.interpolate(
                initial_p_lambda, name="p_prev"
            )
        else:
            self.p_prev = self.pressureSpace.function(name="p_prev")

        n = FacetNormal(self.velocitySpace)
        self.n = n

        self.form_1 = (
            self.rho * dot(u - self.u_prev, v) / self.dt * dx
            + inner(sigma(u, self.p_prev), epsilon(v)) * dx
            - dot(self.f, v) * dx
            + self.rho * dot(dot(self.u_prev, nabla_grad(self.u_prev)), v) * dx
        )

        self.form_2 = (
            dot(grad(p), grad(q)) * dx
            - dot(grad(self.p_prev), grad(q)) * dx
            + 1 / self.dt * div(self.u_prelim) * q * dx
        )

        self.form_3 = (
            dot(u, v) * dx
            - dot(self.u_prelim, v) * dx
            + self.dt * dot(grad(self.p_h - self.p_prev), v) * dx
        )

    def buildPoiseuilleFlowBC(self):
        if self.x is None or self.velocitySpace is None or self.pressureSpace is None:
            raise ValueError(
                "Spatial coordinates and function spaces must be defined before setting boundary conditions."
            )

        self.dbc_velocity = [
            DirichletBC(self.velocitySpace, [0, 0], 3),
            DirichletBC(self.velocitySpace, [0, 0], 4),
        ]
        self.dbc_pressure = [
            DirichletBC(self.pressureSpace, 8, 1),
            DirichletBC(self.pressureSpace, 0, 2),
        ]

        # DUne Fem convention apparently (source: chatty)
        left_id = 1
        right_id = 2

        self.form_1 += -dot(
            self.mu * dot(nabla_grad(self.u), self.n) - self.p_prev * self.n, self.v
        ) * ds((left_id, right_id))

    def buildKarmanBC(
        self,
        inflow_ramp_time=1.0,
    ):
        self.inflow_factor = Constant(0.0, "inflow_factor")
        self.inflow_ramp_time = inflow_ramp_time
        inflow_profile = as_vector(
            [self.inflow_factor * 6 * self.x[1] * (self.H - self.x[1]) / self.H**2, 0.0]
        )
        self.dbc_velocity = [
            DirichletBC(self.velocitySpace, inflow_profile, 1),
            DirichletBC(self.velocitySpace, [0, 0], 3),
            DirichletBC(self.velocitySpace, [0, 0], 4),
            DirichletBC(self.velocitySpace, [0, 0], 5),
        ]
        self.dbc_pressure = [
            DirichletBC(self.pressureSpace, 0, 2),
        ]

        # Neumann
        self.form_1 += -dot(
            self.mu * dot(nabla_grad(self.u), self.n) - self.p_prev * self.n, self.v
        ) * ds(2)

    def visualize_boundary_conditions(self):
        @gridFunction(self.gridView, order=0, name="boundary_ids")
        def bnd(element, x):
            # Iterate over all intersections (edges in 2D) of the current element
            for intersection in self.gridView.intersections(element):
                if intersection.boundary:
                    # Return the ID of the first boundary edge we find for this element
                    return intersection.boundarySegmentIndex
            return 0  # Interior elements get 0

        bnd.plot()

    def buildSolutionScheme(
        self,
        solverParameters,
        solver_types=[("istl", "gmres"), ("istl", "gmres"), ("istl", "gmres")],
    ):
        if self.dbc_velocity is None or self.dbc_pressure is None:
            raise ValueError(
                "Boundary conditions must be set before building the solution scheme."
            )

        self.scheme_1 = solutionScheme(
            [self.form_1 == 0, *self.dbc_velocity],
            parameters=solverParameters["solver_1"],
            solver=solver_types[0],
        )
        self.scheme_2 = solutionScheme(
            [self.form_2 == 0, *self.dbc_pressure],
            parameters=solverParameters["solver_2"],
            solver=solver_types[1],
        )
        self.scheme_3 = solutionScheme(
            [self.form_3 == 0, *self.dbc_velocity],
            parameters=solverParameters["solver_3"],
            solver=solver_types[2],
        )

    def buildSolutionsPoiseuille(self):
        self.solution_u = as_vector([4 * self.x[1] * (1 - self.x[1]), 0])
        self.solution_p = 8 * (1 - self.x[0])
        self.solution_provided = True

    def adapt(self, indicator, maxLevel, expr):
        indicator.interpolate(expr)
        scalar = np.max(indicator.as_numpy) - np.min(indicator.as_numpy)
        if scalar <= 0:
            return

        dune.fem.mark(
            indicator,
            refineTolerance=0.75 * scalar,
            coarsenTolerance=0.1 * 0.75 * scalar,
            maxLevel=maxLevel,
        )
        dune.fem.adapt([self.u_h, self.p_h, self.u_prev, self.p_prev, self.u_prelim])
        dune.fem.loadBalance(
            [self.u_h, self.p_h, self.u_prev, self.p_prev, self.u_prelim]
        )

    def solve(
        self, T=10.0, plot_results=False, adaptive=False, adaptStep=5, maxLevel=5, analysis=False, write_vtk=False, interactive_plot=False
    ):
        if self.scheme_1 is None or self.scheme_2 is None or self.scheme_3 is None:
            raise ValueError("Solution schemes must be built before solving.")

        steady_tolerance = 1e-8
        t = 0
        total_steps = max(int(round(T / self.dt.value)), 1)
        steady_time = None
        error_history = []

        fvspc = finiteVolume(self.gridView, dimRange=1)
        indicator = fvspc.function(name="indicator")

        omega = grad(self.u_h[1])[0] - grad(self.u_h[0])[1]  # curl of u
        expr = ufl_sqrt(
            omega * omega
        )  # vorticity magnitude used as an adaptive indicator
        # omega = self.u_h[1].dx(0) - self.u_h[0].dx(1)
        # expr = ufl_sqrt(CellVolume(self.velocitySpace) * omega * omega)

        if interactive_plot:
            fig = self.initialize_visualisation()

        if write_vtk:
            os.makedirs("out", exist_ok=True)
            vtk_basename = f"out/solution_dt_{self.dt.value:.4f}"

            vtkwriter = self.gridView.sequencedVTK(
                vtk_basename,
                pointdata={"velocity": self.u_h, "pressure": self.p_h},
                subsampling=0,
            )

        iterator = range(1, total_steps + 1)
        if comm.rank == 0:
            iterator = tqdm(iterator)

        scheme_1_linear_iterations = np.zeros(total_steps, dtype=int)
        scheme_2_linear_iterations = np.zeros(total_steps, dtype=int)
        scheme_3_linear_iterations = np.zeros(total_steps, dtype=int)
        scheme_1_convergence = np.zeros(total_steps, dtype=bool)
        scheme_2_convergence = np.zeros(total_steps, dtype=bool)
        scheme_3_convergence = np.zeros(total_steps, dtype=bool)
        scheme_1_solve_time = np.zeros(total_steps)
        scheme_2_solve_time = np.zeros(total_steps)
        scheme_3_solve_time = np.zeros(total_steps)
        scheme_1_assembly_time = np.zeros(total_steps)
        scheme_2_assembly_time = np.zeros(total_steps)
        scheme_3_assembly_time = np.zeros(total_steps)
        scheme_1_return = None
        scheme_2_return = None
        scheme_3_return = None


        for step in iterator:
            t_new = step * self.dt.value

            if self.inflow_factor is not None:
                self.inflow_factor.value = min(1.0, t_new / self.inflow_ramp_time)

            scheme_1_results = self.scheme_1.solve(target=self.u_prelim)
            scheme_2_results = self.scheme_2.solve(target=self.p_h)
            scheme_3_results = self.scheme_3.solve(target=self.u_h)

            if not scheme_1_results["converged"]:
                print(f"Warning: Scheme 1 did not converge at step {step}, t={t_new:.4f}.")
            if not scheme_2_results["converged"]:
                print(f"Warning: Scheme 2 did not converge at step {step}, t={t_new:.4f}.")
            if not scheme_3_results["converged"]:
                print(f"Warning: Scheme 3 did not converge at step {step}, t={t_new:.4f}.")
            
            if analysis:
                scheme_1_linear_iterations[step - 1] = scheme_1_results["linear_iterations"]
                scheme_2_linear_iterations[step - 1] = scheme_2_results["linear_iterations"]
                scheme_3_linear_iterations[step - 1] = scheme_3_results["linear_iterations"]
                scheme_1_convergence[step - 1] = scheme_1_results["converged"]
                scheme_2_convergence[step - 1] = scheme_2_results["converged"]
                scheme_3_convergence[step - 1] = scheme_3_results["converged"]
                scheme_1_solve_time[step - 1] = scheme_1_results["timing"][2]
                scheme_2_solve_time[step - 1] = scheme_2_results["timing"][2]
                scheme_3_solve_time[step - 1] = scheme_3_results["timing"][2]
                scheme_1_assembly_time[step - 1] = scheme_1_results["timing"][1]
                scheme_2_assembly_time[step - 1] = scheme_2_results["timing"][1]
                scheme_3_assembly_time[step - 1] = scheme_3_results["timing"][1]

                if self.solution_provided:
                    velocity_l2_error, pressure_l2_error = self.calculate_l2_errors()
                    error_history.append((t, velocity_l2_error, pressure_l2_error))

                temporal_update_l2 = self.calculate_temporal_update()
                if steady_time is None and temporal_update_l2 < steady_tolerance:
                    steady_time = t + self.dt.value

            self.p_prev.assign(self.p_h)
            self.u_prev.assign(self.u_h)
            t += self.dt.value

            if step % max(total_steps // 100, 1) == 0:
                if write_vtk:
                    vtkwriter()
                if interactive_plot:
                    self.refresh_visualization(t, fig, step)

            if adaptive and step % adaptStep == 0:
                self.adapt(indicator, maxLevel, expr)

        if interactive_plot:
            plt.ioff()
            plt.show()
        if plot_results:
            self.u_h.plot()
            self.p_h.plot()

        if analysis:
            scheme_1_return = (scheme_1_linear_iterations, scheme_1_convergence, scheme_1_solve_time, scheme_1_assembly_time)
            scheme_2_return = (scheme_2_linear_iterations, scheme_2_convergence, scheme_2_solve_time, scheme_2_assembly_time)
            scheme_3_return = (scheme_3_linear_iterations, scheme_3_convergence, scheme_3_solve_time, scheme_3_assembly_time)

            if self.solution_provided:
                self.print_solution_error_message(error_history)

            if steady_time is None:
                print(
                    f"Steady state criterion not reached: "
                    f"combined temporal update remained {temporal_update_l2:.6e} "
                    f"> {steady_tolerance:.1e} at T={T}."
                )
            else:
                print(
                    f"Steady state reached at "
                    f"t={steady_time:.4f} with criterion "
                    f"sqrt(||u^n-u^(n-1)||_L2^2 + ||p^n-p^(n-1)||_L2^2) "
                    f"< {steady_tolerance:.1e}."
                )

        return error_history, scheme_1_return, scheme_2_return, scheme_3_return

    def print_solution_error_message(self, error_history):
        final_time, final_u_error, final_p_error = error_history[-1]
        print(
                f"Final L2 errors at "
                f"t={final_time:.4f}: ||u_h-u_exact||_L2={final_u_error:.6e}, "
                f"||p_h-p_exact||_L2={final_p_error:.6e}"
            )

    def initialize_visualisation(self):
        plt.ion()
        fig = plt.figure(figsize=(12, 5))
        plt.show(block=False)
        return fig

    def refresh_visualization(self, t, fig, step):
        fig.clf()  # clear existing figure instead of opening new windows

        self.u_h.plot(figure=(fig, 121))
        self.p_h.plot(figure=(fig, 122))

        fig.suptitle(f"step={step}, t={t:.4f}")
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.001)

    def calculate_l2_errors(self):
        velocity_l2_error = math_sqrt(
            integrate(
                inner(self.u_h - self.solution_u, self.u_h - self.solution_u),
                gridView=self.gridView,
                order=6,
            )
        )
        pressure_l2_error = math_sqrt(
            integrate(
                (self.p_h - self.solution_p) ** 2,
                gridView=self.gridView,
                order=4,
            )
        )

        return velocity_l2_error, pressure_l2_error

    def calculate_temporal_update(self):
        velocity_update_l2 = math_sqrt(
            integrate(
                inner(self.u_h - self.u_prev, self.u_h - self.u_prev),
                gridView=self.gridView,
                order=6,
            )
        )
        pressure_update_l2 = math_sqrt(
            integrate(
                (self.p_h - self.p_prev) ** 2,
                gridView=self.gridView,
                order=4,
            )
        )
        temporal_update_l2 = math_sqrt(velocity_update_l2**2 + pressure_update_l2**2)
        return temporal_update_l2

    def create_task_A_gridview(self, STRUCTURED_CELLS):
        domain = cartesianDomain([0, 0], [self.L, self.H], STRUCTURED_CELLS)
        gridView = leafGridView(domain)
        gridView = adaptiveLeafGridView(gridView)
        self.gridView = gridView
        print(
            f"Created structured grid with {gridView.size(0)} elements"
        )
        self.gridView.plot(gridLines="black")

    def create_task_B_gridview(self, mesh_size):
        with pygmsh.occ.Geometry() as geom:
            geom.add_rectangle([0, 0, 0], self.L, self.H, mesh_size=mesh_size)
            mesh = geom.generate_mesh()
            points, cells = mesh.points, mesh.cells_dict
            eps = 1e-8  # tolerance
            # dictionary containing id and a list containing the lower left and upper right corner of the bounding box
            bndDomain = {
                1: [[-eps, -eps], [eps, self.H + eps]],  # left
                2: [[self.L - eps, -eps], [self.L + eps, self.H + eps]],  # right
                3: [[-eps, -eps], [self.L + eps, eps]],  # bottom
                4: [[-eps, self.H - eps], [self.L + eps, self.H + eps]],  # top
                5: "default",  # top and bottom wall,
                # which are all other segments not contained in the above bounding boxes
            }

            # return dgf string which can be read by DGF parser or written to file for later use
            dgf = mesh2DGF(points, cells, bndDomain=bndDomain, dim=2)
            domain2d = (reader.dgfString, dgf)
        gridView = leafGridView(domain2d, dimgrid=2)
        gridView = adaptiveLeafGridView(gridView)
        self.gridView = gridView
        self.gridView.plot(gridLines="black")

        print(f"number of elements in unstructured grid: {gridView.size(0)}")

    def create_karman_gridView(
        self, mesh_size, cylinder_center, cylinder_r, coarse=False
    ):
        dgf = None
        if comm.rank == 0:
            outside_size = 0.08 if coarse else mesh_size

            def local_size(x, y):
                radius2 = (x - cylinder_center[0]) ** 2 + (y - cylinder_center[1]) ** 2
                return min(0.01 + 0.6 * radius2, outside_size)

            with pygmsh.occ.Geometry() as geom:
                geom.set_mesh_size_callback(
                    lambda dim, tag, x, y, z, lc: local_size(x, y)
                )
                rectangle = geom.add_rectangle([0, 0, 0], self.L, self.H)
                cylinder = geom.add_disk(
                    [cylinder_center[0], cylinder_center[1], 0.0],
                    cylinder_r,
                )
                geom.boolean_difference(rectangle, cylinder)
                mesh = geom.generate_mesh()
                points, cells = mesh.points, mesh.cells_dict
                eps = 0.01  # tolerance
                # dictionary containing id and a list containing the lower left and upper right corner of the bounding box
                bndDomain = {
                    1: [[-eps, -eps], [eps, self.H + eps]],  # left
                    2: [[self.L - eps, -eps], [self.L + eps, self.H + eps]],  # right
                    3: [[-eps, -eps], [self.L + eps, eps]],  # bottom
                    4: [[-eps, self.H - eps], [self.L + eps, self.H + eps]],  # top
                    5: "default",  # hole boundary, which are all other segments not contained in the above bounding boxes
                }

                # return dgf string which can be read by DGF parser or written to file for later use
                dgf = mesh2DGF(points, cells, bndDomain=bndDomain, dim=2)

        dgf = mpi_comm.bcast(dgf, root=0)
        domain2d = (reader.dgfString, dgf)
        # fig = pyplot.figure()
        # boundaryFunction( gridView2d).plot(gridLines="white",linewidth=2,figure=fig)
        # fig.get_axes()[0].set_facecolor("lightgray")
        # visualize the grid with the boundary function plotted on top to check if the boundary conditions are correctly identified

        gridView = leafGridView(domain2d, dimgrid=2, lbMethod=14)
        gridView = adaptiveLeafGridView(gridView)
        self.gridView = gridView

        # verbose info
        print(
            f"Created grid with {gridView.size(0)} elements."
        )

        # plot the grid to check if it looks correct
        fig = plt.figure(figsize=(8, 4))
        self.gridView.plot(gridLines="black", figure=fig)
        # fig.get_axes()[0].set_facecolor("lightgray")
        fig.suptitle("Grid visualization")
        plt.show()
