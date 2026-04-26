# type: ignore 
from dune.grid import cartesianDomain
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
from dune.ufl import Constant, DirichletBC, NeumannBC
from tqdm import tqdm
import pygmsh


class NavierStokesSolver:
    def __init__(self, gridView=None, dt_value: float = 0.01, H: float = 1, L: float = 1):
        self.gridView = gridView
        self.L = L
        self.H = H
        self.rho = Constant(1, "rho")
        self.mu = Constant(1, "mu")
        self.dt = Constant(dt_value, "dt")
        self.f = as_vector([Constant(0, "fx"), Constant(0, "fy")])
        self.u_prelim = None
        self.u_prev = None
        self.u_h = None
        self.p_h = None
        self.p_prev = None
        self.velocitySpace = None
        self.pressureSpace = None
        self.x = None
        self.dbc_velocity = None
        self.dbc_pressure = None
        self.scheme_1 = None
        self.scheme_2 = None
        self.scheme_3 = None

        self.buildForms()

    def buildForms(self):
        dim = self.gridView.dimension
        self.velocitySpace = lagrange(self.gridView, order=2, dimRange=dim)
        self.pressureSpace = lagrange(self.gridView, order=1)

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

        self.p_h = self.pressureSpace.function(name="p_h")
        self.p_prev = self.pressureSpace.interpolate(lambda x: 8 * (1 - x[0]), name="p_prev")

        n = FacetNormal(self.velocitySpace)

        self.form_1 = (
            self.rho * dot(u - self.u_prev, v) / self.dt * dx
            + inner(sigma(u, self.p_prev), epsilon(v)) * dx
            - dot(self.mu * dot(nabla_grad(u), n) - self.p_prev * n, v) * ds
            - dot(self.f, v) * dx
            + self.rho * dot(dot(self.u_prev, nabla_grad(self.u_prev)), v) * dx
        )

        self.form_2 = (
            dot(grad(p), grad(q)) * dx
            - dot(grad(self.p_prev), grad(q)) * dx
            + 1 / self.dt * div(self.u_prelim) * q * dx
        )

        self.form_3 = (
            dot(u, v) * dx - dot(self.u_prelim, v) * dx + self.dt * dot(grad(self.p_h - self.p_prev), v) * dx
        )

    def buildPoiseuilleFlowBC(self):
        if self.x is None or self.velocitySpace is None or self.pressureSpace is None:
            raise ValueError("Spatial coordinates and function spaces must be defined before setting boundary conditions.")
        
        self.dbc_velocity = [
            DirichletBC(self.velocitySpace, [0, 0], abs(self.x[1]) < 1e-10),
            DirichletBC(self.velocitySpace, [0, 0], abs(self.x[1] - self.H) < 1e-10),
        ]
        self.dbc_pressure = [
            DirichletBC(self.pressureSpace, 8, abs(self.x[0]) < 1e-10),
            DirichletBC(self.pressureSpace, 0, abs(self.x[0] - self.L) < 1e-10),
        ]

    def buildKarmanBC(self, cyclinder_c, cylinder_h, cylinder_l, cylinder_r, inflow_ramp_time, inflow_ramp):
        # TODO - add option for time-dependent ramp function bc bc are not consitsten in the beginning.
        inflow_profile = as_vector(
            [inflow_ramp * 6 * self.x[1] * (cylinder_h - self.x[1]) / cylinder_h**2, 0]
        )
        cylinder_boundary = (
            abs(
                (self.x[0] - cyclinder_c[0]) ** 2
                + (self.x[1] - cyclinder_c[1]) ** 2
                - cylinder_r**2
            )
            < 1e-5
        )
        self.dbc_velocity = [
            DirichletBC(self.velocitySpace, inflow_profile, abs(self.x[0]) < 1e-10),
            DirichletBC(self.velocitySpace, [0, 0], abs(self.x[1]) < 1e-10),
            DirichletBC(self.velocitySpace, [0, 0], abs(self.x[1] - cylinder_h) < 1e-10),
            DirichletBC(self.velocitySpace, [0, 0], cylinder_boundary),
            NeumannBC(self.velocitySpace, [0, 0], abs(self.x[0] - cylinder_l) < 1e-10),
        ]
        self.dbc_pressure = [
            DirichletBC(self.pressureSpace, 0, abs(self.x[0] - cylinder_l) < 1e-10),
        ]

    def buildSolutionScheme(self, solverParameters, solver_types=[("istl", "gmres"), ("istl", "gmres"), ("istl", "gmres")]):
        if self.dbc_velocity is None or self.dbc_pressure is None:
            raise ValueError("Boundary conditions must be set before building the solution scheme.")
        
        self.scheme_1 = solutionScheme(
            [self.form_1 == 0, *self.dbc_velocity],
            parameters=solverParameters,
            solver=solver_types[0],
        )
        self.scheme_2 = solutionScheme(
            [self.form_2 == 0, *self.dbc_pressure],
            parameters=solverParameters,
            solver=solver_types[1],
        )
        self.scheme_3 = solutionScheme(
            [self.form_3 == 0, *self.dbc_velocity],
            parameters=solverParameters,
            solver=solver_types[2],
        )

    def buildSolutionsPoiseuille(self):
        self.solution_u = as_vector([4 * self.x[1] * (1 - self.x[1]), 0])
        self.solution_p = 8 * (1 - self.x[0])

    def solve(self, T=10.0):
        if self.scheme_1 is None or self.scheme_2 is None or self.scheme_3 is None:
            raise ValueError("Solution schemes must be built before solving.")

        steady_tolerance = 1e-8
        t = 0
        total_steps = max(int(round(T / self.dt.value)), 1)
        steady_time = None
        error_history = []
        for step in tqdm(range(1, total_steps + 1)):
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


def create_task_A_gridview(L, H, STRUCTURED_CELLS):
    domain = cartesianDomain([0, 0], [L, H], STRUCTURED_CELLS)
    gridView = leafGridView(domain)
    gridView = adaptiveLeafGridView(gridView)
    return gridView


def create_task_B_gridview(L, H, mesh_size):
    with pygmsh.occ.Geometry() as geom:
        geom.add_rectangle([0, 0, 0], L, H, mesh_size=mesh_size)
        mesh = geom.generate_mesh()
        points, cells = mesh.points, mesh.cells_dict
        domain = {
            "vertices": points[:, :2].astype(float),
            "simplices": cells["triangle"].astype(int),
        }
    gridView = leafGridView(domain, dimgrid=2)
    gridView = adaptiveLeafGridView(gridView)
    return gridView


def make_karman_domain(mesh_size, cylinder_center, cylinder_l, cylinder_h, cylinder_r, coarse=False):
    outside_size = 0.08 if coarse else mesh_size

    def local_size(x, y):
        radius2 = (x - cylinder_center[0]) ** 2 + (y - cylinder_center[1]) ** 2
        return min(0.01 + 0.6 * radius2, outside_size)

    with pygmsh.occ.Geometry() as geom:
        geom.set_mesh_size_callback(
            lambda dim, tag, x, y, z, lc: local_size(x, y)
        )
        rectangle = geom.add_rectangle([0, 0, 0], cylinder_l, cylinder_h)
        cylinder = geom.add_disk(
            [cylinder_center[0], cylinder_center[1], 0.0],
            cylinder_r,
        )
        geom.boolean_difference(rectangle, cylinder)
        mesh = geom.generate_mesh()
        points, cells = mesh.points, mesh.cells_dict
        domain = {
            "vertices": points[:, :2].astype(float),
            "simplices": cells["triangle"].astype(int),
        }
    gridView = leafGridView(domain, dimgrid=2, lbMethod=14)
    gridView = adaptiveLeafGridView(gridView)
    return gridView


if __name__ == "__main__":
    L = 1.0
    H = 1.0
    T = 10.0
    DT = 0.02
    STRUCTURED_CELLS = [16, 16]
    UNSTRUCTURED_MESH_SIZE = 0.08
    CYLINDER_L = 2.2
    CYLINDER_H = 0.41
    CYLINDER_CENTER = (0.2, 0.2)
    CYLINDER_RADIUS = 0.05
    CYLINDER_T = 5.0
    CYLINDER_DT = CYLINDER_T / 1000
    CYLINDER_MESH_SIZE = 0.025
    INFLOW_RAMP_TIME = 1.0

    steady_tolerance = 1e-8
    plot_results = True
    threading.use = 1

    solverParameters = {
        "nonlinear.tolerance": 1e-10,
        "nonlinear.verbose": False,
        "linear.tolerance": 1e-14,
        "linear.preconditioning.method": "ilu",
        "linear.verbose": False,
        "linear.maxiterations": 1000,
    }

    # TASK A structured mesh
    # gridView = create_task_A_gridview(L, H, STRUCTURED_CELLS)

    # TASK B unstructured mesh
    # gridView = create_task_B_gridview(L, H, UNSTRUCTURED_MESH_SIZE)

    # TASK C Karman vortex street
    gridView = make_karman_domain(CYLINDER_MESH_SIZE, CYLINDER_CENTER, CYLINDER_L, CYLINDER_H, CYLINDER_RADIUS, coarse=False)
    L = CYLINDER_L
    H = CYLINDER_H

    solver = NavierStokesSolver(gridView, dt_value=DT, H=H, L=L)
    solver.buildForms()
    # solver.buildPoiseuilleFlowBC()
    solver.buildKarmanBC(CYLINDER_CENTER, CYLINDER_H, CYLINDER_L, CYLINDER_RADIUS, INFLOW_RAMP_TIME, inflow_ramp=0.0)
    solver.buildSolutionScheme(solverParameters, solver_types=[("istl", "cg"), ("istl", "cg"), ("istl", "cg")])
    # solver.buildSolutionsPoiseuille()
    results = solver.solve(T=T)
