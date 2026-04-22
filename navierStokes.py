# %% [markdown]
#
# # The incompressible Navier-Stokes equations with IPCS
#
# Please have a look at the separate *navierstokes-intro* file for explanations.
#
# At your choice you may replace the IPCS splitting scheme by another
# splitting scheme of your choice. However, this has an impact on the
# boundary conditions that can be applied. In particular, the
# Peaceman-Racheford scheme does not allow you to impose boundary
# conditions for the pressure.
#
# # Task A
#
# Implement the Incremental Pressure Correction Scheme (IPCS) desribed
# in the Navier-Stokes intro by expressing the variational
# formulations in UFL. Use a Taylor-Hood finite element type to
# discretize, e.g.:

# %%

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
from dune.ufl import Constant, DirichletBC
from tqdm import tqdm
import pygmsh

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

# Choose which exercise blocks are executed when running this file.
# Valid entries: "A", "B", "C". Example: RUN_TASKS = ["A", "C"]
RUN_TASKS = ["C"]

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


def epsilon(u):
    return sym(nabla_grad(u))


def solve_poiseuille_on_grid(gridView, label, L=1.0, H=1.0, T=10.0, dt_value=0.02):
    dim = gridView.dimension

    velocitySpace = lagrange(gridView, order=2, dimRange=dim)
    pressureSpace = lagrange(gridView, order=1)

    u = TrialFunction(velocitySpace)
    v = TestFunction(velocitySpace)
    p = TrialFunction(pressureSpace)
    q = TestFunction(pressureSpace)

    rho = Constant(1, "rho")
    mu = Constant(1, "mu")
    dt = Constant(dt_value, "dt")

    def local_sigma(u, p):
        return 2 * mu * epsilon(u) - p * Identity(dim)

    x = SpatialCoordinate(velocitySpace)
    solution_u = as_vector([4 * x[1] * (1 - x[1]), 0])
    solution_p = 8 * (1 - x[0])

    name = label.lower().replace(" ", "_")
    u_prelim = velocitySpace.function(name=f"{name}_u_prelim")
    u_prev = velocitySpace.function(name=f"{name}_u_prev")
    u_h = velocitySpace.function(name=f"{name}_u_h")

    p_h = pressureSpace.function(name=f"{name}_p_h")
    p_prev = pressureSpace.interpolate(lambda x: 8 * (1 - x[0]), name=f"{name}_p_prev")

    n = FacetNormal(velocitySpace)
    f = as_vector([Constant(0, "fx"), Constant(0, "fy")])

    form_1 = (
        rho * dot(u - u_prev, v) / dt * dx
        + inner(local_sigma(u, p_prev), epsilon(v)) * dx
        - dot(mu * dot(nabla_grad(u), n) - p_prev * n, v) * ds
        - dot(f, v) * dx
        + rho * dot(dot(u_prev, nabla_grad(u_prev)), v) * dx
    )

    form_2 = (
        dot(grad(p), grad(q)) * dx
        - dot(grad(p_prev), grad(q)) * dx
        + 1 / dt * div(u_prelim) * q * dx
    )

    form_3 = (
        dot(u, v) * dx
        - dot(u_prelim, v) * dx
        + dt * dot(grad(p_h - p_prev), v) * dx
    )

    dbc_velocity = [
        DirichletBC(velocitySpace, [0, 0], abs(x[1]) < 1e-10),
        DirichletBC(velocitySpace, [0, 0], abs(x[1] - H) < 1e-10),
    ]
    dbc_pressure = [
        DirichletBC(pressureSpace, 8, abs(x[0]) < 1e-10),
        DirichletBC(pressureSpace, 0, abs(x[0] - L) < 1e-10),
    ]

    scheme_1 = solutionScheme(
        [form_1 == 0, *dbc_velocity],
        parameters=solverParameters,
        solver=("istl", "gmres"),
    )
    scheme_2 = solutionScheme(
        [form_2 == 0, *dbc_pressure],
        parameters=solverParameters,
        solver=("istl", "gmres"),
    )
    scheme_3 = solutionScheme(
        [form_3 == 0, *dbc_velocity],
        parameters=solverParameters,
        solver=("istl", "gmres"),
    )

    print(f"\n{label}: {gridView.size(0)} elements")
    t = 0
    total_steps = max(int(round(T / dt.value)), 1)
    steady_time = None
    error_history = []
    for step in tqdm(range(1, total_steps + 1), desc=label):
        scheme_1.solve(target=u_prelim)
        scheme_2.solve(target=p_h)
        scheme_3.solve(target=u_h)

        velocity_l2_error = sqrt(
            integrate(
                inner(u_h - solution_u, u_h - solution_u),
                gridView=gridView,
                order=6,
            )
        )
        pressure_l2_error = sqrt(
            integrate(
                (p_h - solution_p) ** 2,
                gridView=gridView,
                order=4,
            )
        )
        velocity_update_l2 = sqrt(
            integrate(
                inner(u_h - u_prev, u_h - u_prev),
                gridView=gridView,
                order=6,
            )
        )
        pressure_update_l2 = sqrt(
            integrate(
                (p_h - p_prev) ** 2,
                gridView=gridView,
                order=4,
            )
        )
        temporal_update_l2 = sqrt(velocity_update_l2**2 + pressure_update_l2**2)
        error_history.append(
            (
                t + dt.value,
                velocity_l2_error,
                pressure_l2_error,
                temporal_update_l2,
            )
        )
        if steady_time is None and temporal_update_l2 < steady_tolerance:
            steady_time = t + dt.value

        p_prev.assign(p_h)
        u_prev.assign(u_h)
        t += dt.value

    final_time, final_u_error, final_p_error, final_update = error_history[-1]
    print(
        f"{label}: Final L2 errors at "
        f"t={final_time:.4f}: ||u_h-u_exact||_L2={final_u_error:.6e}, "
        f"||p_h-p_exact||_L2={final_p_error:.6e}"
    )
    if steady_time is None:
        print(
            f"{label}: Steady state criterion not reached: "
            f"combined temporal update remained {final_update:.6e} "
            f"> {steady_tolerance:.1e} at T={T}."
        )
    else:
        print(
            f"{label}: Steady state reached at "
            f"t={steady_time:.4f} with criterion "
            f"sqrt(||u^n-u^(n-1)||_L2^2 + ||p^n-p^(n-1)||_L2^2) "
            f"< {steady_tolerance:.1e}."
        )

    if plot_results:
        u_h.plot()
        p_h.plot()

    return {
        "gridView": gridView,
        "u_h": u_h,
        "p_h": p_h,
        "error_history": error_history,
        "steady_time": steady_time,
    }


def run_task_a():
    domain = cartesianDomain([0, 0], [L, H], STRUCTURED_CELLS)
    gridView = leafGridView(domain)
    gridView = adaptiveLeafGridView(gridView)
    return solve_poiseuille_on_grid(gridView, "Task A structured", L=L, H=H, T=T, dt_value=DT)


def make_unstructured_channel_domain(mesh_size=UNSTRUCTURED_MESH_SIZE):
    with pygmsh.occ.Geometry() as geom:
        geom.add_rectangle([0, 0, 0], L, H, mesh_size=mesh_size)
        mesh = geom.generate_mesh()
        points, cells = mesh.points, mesh.cells_dict
        domain = {
            "vertices": points[:, :2].astype(float),
            "simplices": cells["triangle"].astype(int),
        }
    return domain


def run_task_b():
    domain = make_unstructured_channel_domain()
    print("Task B unstructured mesh elements:", len(domain["simplices"]))
    gridView = leafGridView(domain, dimgrid=2)
    gridView = adaptiveLeafGridView(gridView)
    if plot_results:
        gridView.plot()
    return solve_poiseuille_on_grid(gridView, "Task B unstructured", L=L, H=H, T=T, dt_value=DT)


def make_cylinder_domain(mesh_size=CYLINDER_MESH_SIZE, coarse=False):
    outside_size = 0.08 if coarse else mesh_size

    def local_size(x, y):
        radius2 = (x - CYLINDER_CENTER[0]) ** 2 + (y - CYLINDER_CENTER[1]) ** 2
        return min(0.01 + 0.6 * radius2, outside_size)

    with pygmsh.occ.Geometry() as geom:
        geom.set_mesh_size_callback(
            lambda dim, tag, x, y, z, lc: local_size(x, y)
        )
        rectangle = geom.add_rectangle([0, 0, 0], CYLINDER_L, CYLINDER_H)
        cylinder = geom.add_disk(
            [CYLINDER_CENTER[0], CYLINDER_CENTER[1], 0.0],
            CYLINDER_RADIUS,
        )
        geom.boolean_difference(rectangle, cylinder)
        mesh = geom.generate_mesh()
        points, cells = mesh.points, mesh.cells_dict
        domain = {
            "vertices": points[:, :2].astype(float),
            "simplices": cells["triangle"].astype(int),
        }
    return domain


def set_constant_value(constant, value):
    try:
        constant.value = value
    except AttributeError:
        constant.assign(value)


def solve_cylinder_flow(
    gridView,
    label="Task C cylinder",
    T=CYLINDER_T,
    dt_value=CYLINDER_DT,
    ramp_time=INFLOW_RAMP_TIME,
):
    dim = gridView.dimension
    velocitySpace = lagrange(gridView, order=2, dimRange=dim)
    pressureSpace = lagrange(gridView, order=1)

    u = TrialFunction(velocitySpace)
    v = TestFunction(velocitySpace)
    p = TrialFunction(pressureSpace)
    q = TestFunction(pressureSpace)

    rho = Constant(1, "rho")
    mu = Constant(1e-3, "mu")
    dt = Constant(dt_value, "dt")
    inflow_ramp = Constant(0.0, "inflow_ramp")

    def local_sigma(u, p):
        return 2 * mu * epsilon(u) - p * Identity(dim)

    x = SpatialCoordinate(velocitySpace)
    inflow_profile = as_vector(
        [inflow_ramp * 6 * x[1] * (CYLINDER_H - x[1]) / CYLINDER_H**2, 0]
    )

    u_prelim = velocitySpace.function(name="task_c_u_prelim")
    u_prev = velocitySpace.function(name="task_c_u_prev")
    u_h = velocitySpace.function(name="task_c_u_h")

    p_h = pressureSpace.function(name="task_c_p_h")
    p_prev = pressureSpace.function(name="task_c_p_prev")

    f = as_vector([Constant(0, "fx"), Constant(0, "fy")])

    form_1 = (
        rho * dot(u - u_prev, v) / dt * dx
        + inner(local_sigma(u, p_prev), epsilon(v)) * dx
        - dot(f, v) * dx
        + rho * dot(dot(u_prev, nabla_grad(u_prev)), v) * dx
    )

    form_2 = (
        dot(grad(p), grad(q)) * dx
        - dot(grad(p_prev), grad(q)) * dx
        + 1 / dt * div(u_prelim) * q * dx
    )

    form_3 = (
        dot(u, v) * dx
        - dot(u_prelim, v) * dx
        + dt * dot(grad(p_h - p_prev), v) * dx
    )

    cylinder_boundary = (
        abs(
            (x[0] - CYLINDER_CENTER[0]) ** 2
            + (x[1] - CYLINDER_CENTER[1]) ** 2
            - CYLINDER_RADIUS**2
        )
        < 1e-5
    )
    dbc_velocity = [
        DirichletBC(velocitySpace, inflow_profile, abs(x[0]) < 1e-10),
        DirichletBC(velocitySpace, [0, 0], abs(x[1]) < 1e-10),
        DirichletBC(velocitySpace, [0, 0], abs(x[1] - CYLINDER_H) < 1e-10),
        DirichletBC(velocitySpace, [0, 0], cylinder_boundary),
    ]
    dbc_pressure = [
        DirichletBC(pressureSpace, 0, abs(x[0] - CYLINDER_L) < 1e-10),
    ]

    scheme_1 = solutionScheme(
        [form_1 == 0, *dbc_velocity],
        parameters=solverParameters,
        solver=("istl", "gmres"),
    )
    scheme_2 = solutionScheme(
        [form_2 == 0, *dbc_pressure],
        parameters=solverParameters,
        solver=("istl", "gmres"),
    )
    scheme_3 = solutionScheme(
        [form_3 == 0, *dbc_velocity],
        parameters=solverParameters,
        solver=("istl", "gmres"),
    )

    print(f"\n{label}: {gridView.size(0)} elements")
    total_steps = max(int(round(T / dt.value)), 1)
    output_every = max(total_steps // 20, 1)
    t = 0
    for step in tqdm(range(1, total_steps + 1), desc=label):
        next_t = t + dt.value
        set_constant_value(inflow_ramp, min(next_t / ramp_time, 1.0))

        scheme_1.solve(target=u_prelim)
        scheme_2.solve(target=p_h)
        scheme_3.solve(target=u_h)

        p_prev.assign(p_h)
        u_prev.assign(u_h)
        t = next_t

        if plot_results and (step % output_every == 0 or step == total_steps):
            u_h.plot()
            p_h.plot()

    print(f"{label}: finished at t={t:.4f} with dt={dt.value:.2e}.")
    return {"gridView": gridView, "u_h": u_h, "p_h": p_h}


def run_task_c():
    domain = make_cylinder_domain()
    print("Task C cylinder mesh elements:", len(domain["simplices"]))
    gridView = leafGridView(domain, dimgrid=2, lbMethod=14)
    gridView = adaptiveLeafGridView(gridView)
    if plot_results:
        gridView.plot()
    return solve_cylinder_flow(gridView)


def main():
    results = {}
    for task in RUN_TASKS:
        task = task.upper()
        if task == "A":
            results["A"] = run_task_a()
        elif task == "B":
            results["B"] = run_task_b()
        elif task == "C":
            results["C"] = run_task_c()
        else:
            raise ValueError(f"Unknown task {task!r}; use 'A', 'B', and/or 'C'.")
    return results

# %% [markdown]
# time step by solving __Step 1__, __Step 2__, and then __Step
# 3__. Test your implementation using the Poiseuille flow problem.
#
# ## Poiseuille flow
#
# Let $\Omega = [0,L] \times [0,H] \subset \mathbb{R}^2$ be the
# computational domain where $H$ is the distance of two plates from
# each other and $L$ the length of the channel. Define the boundaries
# as
#
# \begin{align*}
# \Gamma_{left}   &= 0 \times [0,H],\\
# \Gamma_{right}  &= L \times [0,H],  \\
# \Gamma_{top}    &= [0,L] \times H,  \\
# \Gamma_{bottom} &= [0,L] \times 0.
# \end{align*}
#
# Solve the Navier-Stokes equations (1-2) with $\mu = 1$, $\rho = 1$,
# and $f=0$ for the following initial conditions:
#
# $$
# u(0,x) = 0 \quad \forall x \in \Omega
# $$
# and boundary conditions
# $$
# u(t,x) = 0 \quad \forall x \in \Gamma_{top} \cup \Gamma_{bottom}.
# $$
#
# In order to be able to solve Poisson's equation for the pressure in
# sub-step 2 you may for simplicity just impose Dirichlet conditions for the pressure though this in general is not feasible:
#
# \begin{align*}
# p(t,x) &= 8 \quad \forall x \in \Gamma_{left},\\
# p(t,x) &= 0 \quad \forall x \in \Gamma_{right}.
# \end{align*}
#
# In order for this to work you have to impose Neumann boundary
# conditions on the velocity, as detailed in the `navierstokes-intro`
# slides, i.e. an appropriate boundary integral has to be added to the
# equation of step 1.
#
# Run the simulation until $T=10$ and use a time step size small
# enough, e.g. $\triangle t = 0.02$ with the above $16$ cells in each
# direction.
#
# A steady state solution to this problem is given by $u(x) =
# 4x_2(1−x_2)$ and $p(x)=8(1 - x_1)$ with $x = (x_1,x_2) \in
# \mathbb{R}^2$. Compare your numerical results to the exact solution
# and compute the $L_2$ error between your computed solution and the
# steady state solution. At which time $t \in (0,T]$ is the steady
# state reached? Discuss the impact of mesh refinement on the
# difference between the discrete steady state and the exact steady
# state.
#
# # Task B
#
# Repeat the problem from __Task A__ with an unstructured grid.
# Create a triangular grid with __pygmsh__, and integrate that into your code.

# %%

if __name__ == "__main__":
    results = main()

# %% [markdown]
# # Task C
#
# Flow around a cylinder: __Karman Vortex Street__.
#
# Consider the domain $\Omega = [0,L] \times [0,H] \setminus B_r(0.2,
# 0.2) \subset \mathbb{R}^2$ with $L=2.2$, $H=0.41$ and
# $r=0.05$. Define the boundaries as
#
# \begin{align*}
# \Gamma_{left}   &= 0 \times [0,H],\\
# \Gamma_{right}  &= L \times [0,H],  \\
# \Gamma_{top}    &= [0,L] \times H,  \\
# \Gamma_{bottom} &= [0,L] \times 0.
# \end{align*}
#
# Solve the Navier-Stokes equations (1-2) with $\mu = 10^{-3}$, $\rho
# = 1$, and $f=0$. Use the following boundary condition on the
# "unproblematic" lower, upper and right boundaries:
# \begin{align*}
# u(t,x) &= 0 \quad \forall x \in \Gamma_{top} \cup \Gamma_{bottom} \cup \partial B_r,\\
# \partial_n u(t,x) &= 0 \quad \forall x \in \Gamma_{right},
# p(t,x) &= 0 \quad \forall x \in \Gamma_{right}.
# \end{align*}
#
# Note that the Navier-Stokes equation in principle do not allow to
# impose boundary values for the pressure. Nevertheless
# pressure-correction schemes are quite popular because they are
# computationally very performant.
#
# For the initial conditions and the conditions on the inflow boundary
# $\Gamma_{left}$ one has to be careful. When imposing non-zero inflow
# right from the start then an initial value $u=0$ is incompatible
# with the inflow condition on $\Gamma_{left}$. There are at least two
# possible ways to cope with this:
#
# - gradually increase the inflow profile: use the initial condition
# $$
# u(0,x) = 0 \quad \forall x \in \Omega
# $$
# but only gradually increase the inflow starting from zero up the the target inflow profile
# $$
# u(t,x) = \left( \frac{6 x_2(H−x_2)}{H^2}, 0 \right) \quad \forall x \in \Gamma_{left}
# $$
#
# - numerically compute compatible initial values: start witht the
# target inflow profile right from time $0$ and use an initial value
# which is the solution of the stationary stokes equation which
# satisfies all boundary conditions for $u$, including the non-zero
# inflow profile. For this you can, e.g., use the Uzawa algorithm for
# the (Quasi-)Stokes problem that you have implemented in the second
# week of the course (or use the corresponding Python code from the
# ILIAS page).
#
# Create an appropriate grid using __pygmsh__ and run the simulation
# until $T=5$ and use a time step size small enough, e.g. $\triangle t
# = T / 10^5$. This number depends on the grid size of your mesh
# (i.e. the smallest edge length present).
#
# A sketch of the domain and a picture of expected results can be
# found
# [here](https://www.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark2_re100.html).
# Please note that the DFG-benchmark problem uses a different
# formulation of the Navier-Stokes equations where the Neumann
# boundary conditions on the free boundary for the velocity $u$ are
# just "natural" i.e. the boundary integral resulting from integration
# by parts is just left out in the weak formulation.
#
# Note: please have a look at the folder "Tools and Examples" of the
# ILIAS page for the course; in particular for example code concerning
# the construction of the computational domain.
#
# # Task D
#
# Test different preconditioning options available and report the observed results.
#
# __Note__: Not all preconditioning methods work with parallelization. For parallel tests use the PETSc options.
#
# Please use at least three different preconditioners. "observed results"
# should include the number of iterations needed to solve the
# sub-problems as well as the actual time needed to carry out the simulation.
#
# ## Task E -- Parallel implementation
#
# Adjust your program such that it can run in parallel using MPI,
# e.g. on $4$ cores (or use more, but at least more than $1$ core!) using
#
# ```
# mpirun -np 4 python your_program.py
# ```
#
# Please keep in mind that on your laptop parallelization may gain
# little to no speedup. You will be given access to a compute-server
# of the Maths department. Run speed-up tests on that server. For
# meaningful results you have to control the number of threads used
# for linear algebra, e.g.
#

# %% [markdown]
# ## Task F -- Adaptivity
#
# Run the Karman Vortex Street example using adaptive mesh refinement
# and coarsening. As an indicator use the curl of $u$ also known as
# the vorticity. More precise: use the $L^2$-norm of the curl on each
# element. Refine the grid, where the vorticity is high, and coarsen,
# where the vorticity is low.
#
# Consult the tutorial for the appropriate adaptivity cycle, in
# particular the solution-code for the Cahn-Hilliard and Heat-Robin
# examples. Both can be found in the ILIAS.
#
