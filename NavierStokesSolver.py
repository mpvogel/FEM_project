# type: ignore
from dune.grid import cartesianDomain, reader
from dune.alugrid import aluConformGrid as leafGridView
from dune.fem import integrate, threading
from dune.fem.view import adaptiveLeafGridView
from dune.fem.space import lagrange
from dune.fem.scheme import galerkin as solutionScheme
from math import sqrt
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
)
from dune.ufl import Constant, DirichletBC
# from dune.fem.function import boundaryFunction
from gmsh2dgf import gmsh2DGF as mesh2DGF

from tqdm import tqdm
import pygmsh
import dune.fem as fem

from matplotlib import pyplot as plt

fem.threading.useMax()


class NavierStokesSolver:
    def __init__(
        self, dt_value: float = 0.01, H: float = 1, L: float = 1, rho_value=1, mu_value=1
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
        self.inflow_factor = None
        self.inflow_ramp = None

    def buildForms(self, initial_p_lambda = None):
        dim = self.gridView.dimension
        self.velocitySpace = lagrange(self.gridView, order=2, dimRange=dim, storage="petsc")
        self.pressureSpace = lagrange(self.gridView, order=1, storage="petsc")

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
            self.p_prev = self.pressureSpace.interpolate(initial_p_lambda, name="p_prev")
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

        self.form_1 += - dot(self.mu * dot(nabla_grad(self.u), self.n) - self.p_prev * self.n, self.v) * ds((left_id, right_id))

    def buildKarmanBC(
        self,
        cyclinder_c,
        cylinder_r,
        inflow_ramp_time=1.0,
    ):
        # TODO - add option for time-dependent ramp function bc bc are not consitsten in the beginning.
        self.inflow_factor = Constant(0.0, "inflow_factor")
        self.inflow_ramp_time = inflow_ramp_time
        inflow_profile = as_vector([self.inflow_factor*6*self.x[1]*(self.H - self.x[1])/self.H**2, 0.0])
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
        self.form_1 += - dot(self.mu * dot(nabla_grad(self.u), self.n) - self.p_prev * self.n, self.v) * ds(2)

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

    def solve(self, T=10.0, plot_results=False):
        if self.scheme_1 is None or self.scheme_2 is None or self.scheme_3 is None:
            raise ValueError("Solution schemes must be built before solving.")

        steady_tolerance = 1e-8
        t = 0
        total_steps = max(int(round(T / self.dt.value)), 1)
        steady_time = None
        error_history = []

        if plot_results:
            plt.ion()
            fig = plt.figure(figsize=(12, 5))
            plt.show(block=False)

        for step in tqdm(range(1, total_steps + 1)):
            t_new = step * self.dt.value
            if self.inflow_factor is not None:
                self.inflow_factor.value = min(1.0, t_new / self.inflow_ramp_time)

            self.scheme_1.solve(target=self.u_prelim)
            self.scheme_2.solve(target=self.p_h)
            self.scheme_3.solve(target=self.u_h)

            if self.solution_u is not None or self.solution_p is not None:
                velocity_l2_error = sqrt(
                    integrate(
                        inner(self.u_h - self.solution_u, self.u_h - self.solution_u),
                        gridView=self.gridView,
                        order=6,
                    )
                )
                pressure_l2_error = sqrt(
                    integrate(
                        (self.p_h - self.solution_p) ** 2,
                        gridView=self.gridView,
                        order=4,
                    )
                )

            velocity_update_l2 = sqrt(
                integrate(
                    inner(self.u_h - self.u_prev, self.u_h - self.u_prev),
                    gridView=self.gridView,
                    order=6,
                )
            )
            pressure_update_l2 = sqrt(
                integrate(
                    (self.p_h - self.p_prev) ** 2,
                    gridView=self.gridView,
                    order=4,
                )
            )
            temporal_update_l2 = sqrt(velocity_update_l2**2 + pressure_update_l2**2)
            # error_history.append(
            #     (
            #         t + self.dt.value,
            #         velocity_l2_error,
            #         pressure_l2_error,
            #         temporal_update_l2,
            #     )
            # )
            if steady_time is None and temporal_update_l2 < steady_tolerance:
                steady_time = t + self.dt.value

            self.p_prev.assign(self.p_h)
            self.u_prev.assign(self.u_h)
            t += self.dt.value

            if plot_results and step % max(total_steps // 100, 1) == 0:
                fig.clf()   # clear existing figure instead of opening new windows

                self.u_h.plot(figure=(fig, 121))
                self.p_h.plot(figure=(fig, 122))

                fig.suptitle(f"step={step}, t={t:.4f}")
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.001)
        
        if plot_results:
            plt.ioff()
            plt.show()

        self.u_h.plot()
        self.p_h.plot()

        # final_time, final_u_error, final_p_error, final_update = error_history[-1]
        # print(
        #     f"Final L2 errors at "
        #     f"t={final_time:.4f}: ||u_h-u_exact||_L2={final_u_error:.6e}, "
        #     f"||p_h-p_exact||_L2={final_p_error:.6e}"
        # )
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

        # return {
        #     "gridView": self.gridView,
        #     "u_h": self.u_h,
        #     "p_h": self.p_h,
        #     "error_history": error_history,
        #     "steady_time": steady_time,
        # }


    def create_task_A_gridview(self, STRUCTURED_CELLS):
        domain = cartesianDomain([0, 0], [self.L, self.H], STRUCTURED_CELLS)
        gridView = leafGridView(domain)
        gridView = adaptiveLeafGridView(gridView)
        self.gridView = gridView
        print(f"Created structured grid with {gridView.size(0)} vertices and {gridView.size(1)} cells.")


    def create_task_B_gridview(self, mesh_size):
        with pygmsh.occ.Geometry() as geom:
            geom.add_rectangle([0, 0, 0], self.L, self.H, mesh_size=mesh_size)
            mesh = geom.generate_mesh()
            points, cells = mesh.points, mesh.cells_dict
            eps = 1e-8 # tolerance
            # dictionary containing id and a list containing the lower left and upper right corner of the bounding box
            bndDomain = {1: [[-eps, -eps], [eps, self.H + eps]],  # left
                        2: [[self.L - eps, -eps], [self.L + eps, self.H + eps]],  # right
                        3: [[-eps, -eps], [self.L + eps, eps]],  # bottom
                        4: [[-eps, self.H - eps], [self.L + eps, self.H + eps]],  # top
                        5: "default"  # top and bottom wall,
                        # which are all other segments not contained in the above bounding boxes
                        }

            # return dgf string which can be read by DGF parser or written to file for later use
            dgf = mesh2DGF(points, cells, bndDomain=bndDomain, dim=2)
            domain2d = (reader.dgfString, dgf)
        gridView = leafGridView(domain2d, dimgrid=2)
        gridView = adaptiveLeafGridView(gridView)
        self.gridView = gridView


    def create_karman_gridView(
        self, mesh_size, cylinder_center, cylinder_r, coarse=False
    ):
        outside_size = 0.08 if coarse else mesh_size

        def local_size(x, y):
            radius2 = (x - cylinder_center[0]) ** 2 + (y - cylinder_center[1]) ** 2
            return min(0.01 + 0.6 * radius2, outside_size)

        with pygmsh.occ.Geometry() as geom:
            geom.set_mesh_size_callback(lambda dim, tag, x, y, z, lc: local_size(x, y))
            rectangle = geom.add_rectangle([0, 0, 0], self.L, self.H)
            cylinder = geom.add_disk(
                [cylinder_center[0], cylinder_center[1], 0.0],
                cylinder_r,
            )
            geom.boolean_difference(rectangle, cylinder)
            mesh = geom.generate_mesh()
            points, cells = mesh.points, mesh.cells_dict
            eps = 0.01 # tolerance
            # dictionary containing id and a list containing the lower left and upper right corner of the bounding box
            bndDomain = {1: [[-eps, -eps], [eps, self.H + eps]],  # left
                        2: [[self.L - eps, -eps], [self.L + eps, self.H + eps]],  # right
                        3: [[-eps, -eps], [self.L + eps, eps]],  # bottom
                        4: [[-eps, self.H - eps], [self.L + eps, self.H + eps]],  # top
                        5: "default"  # hole boundary, which are all other segments not contained in the above bounding boxes
                        }

            # return dgf string which can be read by DGF parser or written to file for later use
            dgf = mesh2DGF(points, cells, bndDomain=bndDomain, dim=2)
            domain2d = (reader.dgfString, dgf)
        # fig = pyplot.figure()
        # boundaryFunction( gridView2d).plot(gridLines="white",linewidth=2,figure=fig)
        # fig.get_axes()[0].set_facecolor("lightgray")
        # visualize the grid with the boundary function plotted on top to check if the boundary conditions are correctly identified
        
        gridView = leafGridView(domain2d, dimgrid=2, lbMethod=14)
        gridView = adaptiveLeafGridView(gridView)
        self.gridView = gridView


if __name__ == "__main__":
    L = 1.0
    H = 1.0
    T = 5.0
    DT = 0.001
    STRUCTURED_CELLS = [16, 16]
    UNSTRUCTURED_MESH_SIZE = 0.08
    CYLINDER_L = 2.2
    CYLINDER_H = 0.41
    CYLINDER_CENTER = (0.2, 0.2)
    CYLINDER_RADIUS = 0.05
    CYLINDER_T = 5.0
    CYLINDER_DT = CYLINDER_T / 1000
    CYLINDER_MESH_SIZE = 0.045
    INFLOW_RAMP_TIME = 1.0

    steady_tolerance = 1e-8
    plot_results = True
    threading.use = 1

    solver_1_Parameters = {
        "nonlinear.tolerance": 1e-8,
        "nonlinear.verbose": False,
        "linear.tolerance": 1e-9,
        "linear.preconditioning.method": "ilu",
        "linear.verbose": False,
        "linear.maxiterations": 1000,
    }

    solver_2_Parameters = {
        "nonlinear.tolerance": 1e-8,
        "nonlinear.verbose": False,
        "linear.tolerance": 1e-9,
        "linear.preconditioning.method": "ilu",
        "linear.verbose": False,
        "linear.maxiterations": 1000,
    }

    solver_3_Parameters = {
        "nonlinear.tolerance": 1e-8,
        "nonlinear.verbose": False,
        "linear.tolerance": 1e-9,
        "linear.preconditioning.method": "ilu",
        "linear.verbose": False,
        "linear.maxiterations": 1000,
    }

    solverParameters = {
        "solver_1": solver_1_Parameters,
        "solver_2": solver_2_Parameters,    
        "solver_3": solver_3_Parameters
    }


    # TASK A structured mesh
    # solver = NavierStokesSolver(dt_value=DT, H=H, L=L)
    # solver.create_task_A_gridview(STRUCTURED_CELLS)

    # TASK B unstructured mesh
    # solver = NavierStokesSolver(dt_value=DT, H=H, L=L)
    # solver.create_task_B_gridview(UNSTRUCTURED_MESH_SIZE)

    # TASK C Karman vortex street
    L = CYLINDER_L
    H = CYLINDER_H
    solver = NavierStokesSolver(dt_value=DT, H=H, L=L, mu_value= 1e-3, rho_value = 1)
    solver.create_karman_gridView(
        mesh_size=CYLINDER_MESH_SIZE,
        cylinder_center=CYLINDER_CENTER,
        cylinder_r=CYLINDER_RADIUS,
        coarse=False,
    )


    solver.buildForms()
    # solver.buildPoiseuilleFlowBC()
    solver.buildKarmanBC(
        CYLINDER_CENTER,
        CYLINDER_RADIUS,
        inflow_ramp_time=1.0,
    )

    solver_lib = "petsc"
    solver.buildSolutionScheme(
        solverParameters, solver_types=[(solver_lib, "gmres"), (solver_lib, "cg"), (solver_lib, "cg")]
    )
    # solver.buildSolutionsPoiseuille()
    results = solver.solve(T=T, plot_results=True)
