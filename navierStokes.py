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
import os
import pygmsh
import time

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
CYLINDER_BOUNDARY_TOL = 1e-4
CYLINDER_T = 5.0
CYLINDER_DT = CYLINDER_T / 1000
CYLINDER_MESH_SIZE = 0.025
INFLOW_RAMP_TIME = 1.0
TASK_D_PRECONDITIONERS = ["ilu", "jacobi", "ssor"]
TASK_D_T = 0.1
TASK_D_DT = 0.02
TASK_D_CELLS = [16, 16]
TASK_E_PRECONDITIONER = "jacobi"
TASK_E_SOLVER = ("petsc", "gmres")
TASK_E_T = 0.1
TASK_E_DT = 0.02
TASK_E_CELLS = [32, 32]

# Choose which exercise blocks are executed when running this file.
# Valid entries: "A", "B", "C", "D", "E". Example: RUN_TASKS = ["A", "C"]
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


def mpi_rank():
    for key in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK", "MV2_COMM_WORLD_RANK"):
        if key in os.environ:
            return int(os.environ[key])
    return 0


def mpi_size():
    for key in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE", "MV2_COMM_WORLD_SIZE"):
        if key in os.environ:
            return int(os.environ[key])
    return 1


def is_root_process():
    return mpi_rank() == 0


def print_root(*args, **kwargs):
    if is_root_process():
        print(*args, **kwargs)


def make_solver_parameters(preconditioner="ilu", linear_verbose=False):
    parameters = dict(solverParameters)
    parameters["linear.preconditioning.method"] = preconditioner
    parameters["linear.verbose"] = linear_verbose
    return parameters


def epsilon(u):
    return sym(nabla_grad(u))


def make_ipcs_problem(
    gridView,
    name,
    mu_value,
    rho_value,
    dt_value,
    make_velocity_bcs,
    make_pressure_bcs,
    p_initial=None,
    include_pressure_boundary_term=False,
    solver_parameters=None,
    solver_backend=("istl", "gmres"),
):
    dim = gridView.dimension
    velocitySpace = lagrange(gridView, order=2, dimRange=dim)
    pressureSpace = lagrange(gridView, order=1)

    u = TrialFunction(velocitySpace)
    v = TestFunction(velocitySpace)
    p = TrialFunction(pressureSpace)
    q = TestFunction(pressureSpace)

    rho = Constant(rho_value, "rho")
    mu = Constant(mu_value, "mu")
    dt = Constant(dt_value, "dt")
    x = SpatialCoordinate(velocitySpace)

    solution_name = name.lower().replace(" ", "_")
    u_prelim = velocitySpace.function(name=f"{solution_name}_u_prelim")
    u_prev = velocitySpace.function(name=f"{solution_name}_u_prev")
    u_h = velocitySpace.function(name=f"{solution_name}_u_h")
    p_h = pressureSpace.function(name=f"{solution_name}_p_h")
    if p_initial is None:
        p_prev = pressureSpace.function(name=f"{solution_name}_p_prev")
    else:
        p_prev = pressureSpace.interpolate(p_initial, name=f"{solution_name}_p_prev")

    def local_sigma(u_value, p_value):
        return 2 * mu * epsilon(u_value) - p_value * Identity(dim)

    n = FacetNormal(velocitySpace)
    f = as_vector([Constant(0, "fx"), Constant(0, "fy")])

    form_1 = (
        rho * dot(u - u_prev, v) / dt * dx
        + inner(local_sigma(u, p_prev), epsilon(v)) * dx
        - dot(f, v) * dx
        + rho * dot(dot(u_prev, nabla_grad(u_prev)), v) * dx
    )
    if include_pressure_boundary_term:
        form_1 -= dot(mu * dot(nabla_grad(u), n) - p_prev * n, v) * ds

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

    dbc_velocity = make_velocity_bcs(velocitySpace, x)
    dbc_pressure = make_pressure_bcs(pressureSpace, x)
    solver_parameters = solver_parameters or solverParameters

    scheme_1 = solutionScheme(
        [form_1 == 0, *dbc_velocity],
        parameters=solver_parameters,
        solver=solver_backend,
    )
    scheme_2 = solutionScheme(
        [form_2 == 0, *dbc_pressure],
        parameters=solver_parameters,
        solver=solver_backend,
    )
    scheme_3 = solutionScheme(
        [form_3 == 0, *dbc_velocity],
        parameters=solver_parameters,
        solver=solver_backend,
    )

    return {
        "gridView": gridView,
        "velocitySpace": velocitySpace,
        "pressureSpace": pressureSpace,
        "x": x,
        "dt": dt,
        "u_prelim": u_prelim,
        "u_prev": u_prev,
        "u_h": u_h,
        "p_h": p_h,
        "p_prev": p_prev,
        "scheme_1": scheme_1,
        "scheme_2": scheme_2,
        "scheme_3": scheme_3,
    }


def solve_poiseuille_on_grid(
    gridView,
    label,
    L=1.0,
    H=1.0,
    T=10.0,
    dt_value=0.02,
    solver_parameters=None,
    collect_stats=False,
    evaluate_diagnostics=True,
    solver_backend=("istl", "gmres"),
):
    def make_velocity_bcs(velocitySpace, x):
        return [
            DirichletBC(velocitySpace, [0, 0], abs(x[1]) < 1e-10),
            DirichletBC(velocitySpace, [0, 0], abs(x[1] - H) < 1e-10),
        ]

    def make_pressure_bcs(pressureSpace, x):
        return [
            DirichletBC(pressureSpace, 8, abs(x[0]) < 1e-10),
            DirichletBC(pressureSpace, 0, abs(x[0] - L) < 1e-10),
        ]

    problem = make_ipcs_problem(
        gridView,
        label,
        mu_value=1,
        rho_value=1,
        dt_value=dt_value,
        make_velocity_bcs=make_velocity_bcs,
        make_pressure_bcs=make_pressure_bcs,
        p_initial=lambda x: 8 * (1 - x[0]),
        include_pressure_boundary_term=True,
        solver_parameters=solver_parameters,
        solver_backend=solver_backend,
    )
    gridView = problem["gridView"]
    x = problem["x"]
    dt = problem["dt"]
    u_prelim = problem["u_prelim"]
    u_prev = problem["u_prev"]
    u_h = problem["u_h"]
    p_h = problem["p_h"]
    p_prev = problem["p_prev"]
    scheme_1 = problem["scheme_1"]
    scheme_2 = problem["scheme_2"]
    scheme_3 = problem["scheme_3"]
    solution_u = as_vector([4 * x[1] * (1 - x[1]), 0])
    solution_p = 8 * (1 - x[0])

    print_root(f"\n{label}: {gridView.size(0)} elements")
    t = 0
    total_steps = max(int(round(T / dt.value)), 1)
    steady_time = None
    error_history = []
    solve_stats = []
    for step in tqdm(range(1, total_steps + 1), desc=label, disable=not is_root_process()):
        step_start = time.perf_counter()
        info_1 = scheme_1.solve(target=u_prelim)
        info_2 = scheme_2.solve(target=p_h)
        info_3 = scheme_3.solve(target=u_h)
        step_elapsed = time.perf_counter() - step_start
        if collect_stats:
            solve_stats.append(
                {
                    "step": step,
                    "elapsed": step_elapsed,
                    "step_1": info_1,
                    "step_2": info_2,
                    "step_3": info_3,
                }
            )

        if evaluate_diagnostics:
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

    if evaluate_diagnostics:
        final_time, final_u_error, final_p_error, final_update = error_history[-1]
        print_root(
            f"{label}: Final L2 errors at "
            f"t={final_time:.4f}: ||u_h-u_exact||_L2={final_u_error:.6e}, "
            f"||p_h-p_exact||_L2={final_p_error:.6e}"
        )
        if steady_time is None:
            print_root(
                f"{label}: Steady state criterion not reached: "
                f"combined temporal update remained {final_update:.6e} "
                f"> {steady_tolerance:.1e} at T={T}."
            )
        else:
            print_root(
                f"{label}: Steady state reached at "
                f"t={steady_time:.4f} with criterion "
                f"sqrt(||u^n-u^(n-1)||_L2^2 + ||p^n-p^(n-1)||_L2^2) "
                f"< {steady_tolerance:.1e}."
            )
    else:
        print_root(f"{label}: completed {total_steps} steps to t={t:.4f}.")

    if plot_results and is_root_process():
        u_h.plot()
        p_h.plot()

    return {
        "gridView": gridView,
        "u_h": u_h,
        "p_h": p_h,
        "error_history": error_history,
        "steady_time": steady_time,
        "solve_stats": solve_stats,
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
    print_root("Task B unstructured mesh elements:", len(domain["simplices"]))
    gridView = leafGridView(domain, dimgrid=2)
    gridView = adaptiveLeafGridView(gridView)
    if plot_results and is_root_process():
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
    inflow_ramp = Constant(0.0, "inflow_ramp")

    def make_velocity_bcs(velocitySpace, x):
        inflow_profile = as_vector(
            [inflow_ramp * 6 * x[1] * (CYLINDER_H - x[1]) / CYLINDER_H**2, 0]
        )
        cylinder_boundary = (
            abs(
                (x[0] - CYLINDER_CENTER[0]) ** 2
                + (x[1] - CYLINDER_CENTER[1]) ** 2
                - CYLINDER_RADIUS**2
            )
            < CYLINDER_BOUNDARY_TOL
        )
        return [
            DirichletBC(velocitySpace, inflow_profile, abs(x[0]) < 1e-10),
            DirichletBC(velocitySpace, [0, 0], abs(x[1]) < 1e-10),
            DirichletBC(velocitySpace, [0, 0], abs(x[1] - CYLINDER_H) < 1e-10),
            DirichletBC(velocitySpace, [0, 0], cylinder_boundary),
        ]

    def make_pressure_bcs(pressureSpace, x):
        return [
            DirichletBC(pressureSpace, 0, abs(x[0] - CYLINDER_L) < 1e-10),
        ]

    problem = make_ipcs_problem(
        gridView,
        label,
        mu_value=1e-3,
        rho_value=1,
        dt_value=dt_value,
        make_velocity_bcs=make_velocity_bcs,
        make_pressure_bcs=make_pressure_bcs,
        include_pressure_boundary_term=True,
        solver_parameters=solverParameters,
        solver_backend=("istl", "gmres"),
    )
    gridView = problem["gridView"]
    dt = problem["dt"]
    u_prelim = problem["u_prelim"]
    u_prev = problem["u_prev"]
    u_h = problem["u_h"]
    p_h = problem["p_h"]
    p_prev = problem["p_prev"]
    scheme_1 = problem["scheme_1"]
    scheme_2 = problem["scheme_2"]
    scheme_3 = problem["scheme_3"]

    print_root(f"\n{label}: {gridView.size(0)} elements")
    total_steps = max(int(round(T / dt.value)), 1)
    output_every = max(total_steps // 20, 1)
    t = 0
    for step in tqdm(range(1, total_steps + 1), desc=label, disable=not is_root_process()):
        next_t = t + dt.value
        set_constant_value(inflow_ramp, min(next_t / ramp_time, 1.0))

        scheme_1.solve(target=u_prelim)
        scheme_2.solve(target=p_h)
        scheme_3.solve(target=u_h)

        p_prev.assign(p_h)
        u_prev.assign(u_h)
        t = next_t

        if plot_results and is_root_process() and (step % output_every == 0 or step == total_steps):
            u_h.plot()
            p_h.plot()

    print_root(f"{label}: finished at t={t:.4f} with dt={dt.value:.2e}.")
    return {"gridView": gridView, "u_h": u_h, "p_h": p_h}


def run_task_c():
    domain = make_cylinder_domain()
    print_root("Task C cylinder mesh elements:", len(domain["simplices"]))
    gridView = leafGridView(domain, dimgrid=2, lbMethod=14)
    gridView = adaptiveLeafGridView(gridView)
    if plot_results and is_root_process():
        gridView.plot()
    return solve_cylinder_flow(gridView)


def info_number(info, key):
    if not isinstance(info, dict):
        return None
    value = info.get(key)
    if isinstance(value, (int, float)):
        return value
    return None


def summarize_solve_stats(solve_stats):
    summary = {}
    for substep in ("step_1", "step_2", "step_3"):
        linear_values = [
            info_number(step[substep], "linear_iterations")
            for step in solve_stats
            if info_number(step[substep], "linear_iterations") is not None
        ]
        nonlinear_values = [
            info_number(step[substep], "iterations")
            for step in solve_stats
            if info_number(step[substep], "iterations") is not None
        ]
        summary[substep] = {
            "linear_total": sum(linear_values) if linear_values else None,
            "linear_average": (
                sum(linear_values) / len(linear_values) if linear_values else None
            ),
            "nonlinear_total": sum(nonlinear_values) if nonlinear_values else None,
            "nonlinear_average": (
                sum(nonlinear_values) / len(nonlinear_values) if nonlinear_values else None
            ),
        }
    summary["elapsed_total"] = sum(step["elapsed"] for step in solve_stats)
    return summary


def format_number(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def print_benchmark_table(results):
    if not is_root_process():
        return

    print("\nTask D preconditioner benchmark")
    print(
        "preconditioner | status | wall time [s] | "
        "step1 lin it | step2 lin it | step3 lin it"
    )
    print("-" * 86)
    for result in results:
        if result["status"] != "ok":
            print(
                f"{result['preconditioner']:14s} | failed | "
                f"{result['elapsed']:.3f} | {result['error']}"
            )
            continue

        summary = result["summary"]
        print(
            f"{result['preconditioner']:14s} | ok     | "
            f"{result['elapsed']:.3f} | "
            f"{format_number(summary['step_1']['linear_total']):>12s} | "
            f"{format_number(summary['step_2']['linear_total']):>12s} | "
            f"{format_number(summary['step_3']['linear_total']):>12s}"
        )

    print("\nAverage linear iterations per time step")
    print("preconditioner | step1 | step2 | step3")
    print("-" * 44)
    for result in results:
        if result["status"] != "ok":
            continue
        summary = result["summary"]
        print(
            f"{result['preconditioner']:14s} | "
            f"{format_number(summary['step_1']['linear_average']):>5s} | "
            f"{format_number(summary['step_2']['linear_average']):>5s} | "
            f"{format_number(summary['step_3']['linear_average']):>5s}"
        )


def run_preconditioner_benchmark(
    preconditioners,
    label,
    cells,
    T,
    dt_value,
    solver_backend=("istl", "gmres"),
    parallel_hint=False,
):
    global plot_results

    old_plot_results = plot_results
    plot_results = False
    results = []

    print_root(
        f"{label}: running on MPI size {mpi_size()} with "
        f"threading.use={threading.use} and solver={solver_backend}"
    )
    if parallel_hint and is_root_process():
        print(
            "Task E command example: "
            "mpirun -np 4 ../.venv/bin/python navierStokes.py"
        )

    try:
        for preconditioner in preconditioners:
            parameters = make_solver_parameters(preconditioner=preconditioner)
            domain = cartesianDomain([0, 0], [L, H], cells)
            gridView = adaptiveLeafGridView(leafGridView(domain))
            start = time.perf_counter()
            try:
                result = solve_poiseuille_on_grid(
                    gridView,
                    f"{label} {preconditioner}",
                    L=L,
                    H=H,
                    T=T,
                    dt_value=dt_value,
                    solver_parameters=parameters,
                    collect_stats=True,
                    evaluate_diagnostics=False,
                    solver_backend=solver_backend,
                )
                elapsed = time.perf_counter() - start
                results.append(
                    {
                        "preconditioner": preconditioner,
                        "status": "ok",
                        "elapsed": elapsed,
                        "summary": summarize_solve_stats(result["solve_stats"]),
                        "result": result,
                    }
                )
            except Exception as exc:
                elapsed = time.perf_counter() - start
                results.append(
                    {
                        "preconditioner": preconditioner,
                        "status": "failed",
                        "elapsed": elapsed,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print_root(f"{label} {preconditioner}: failed with {exc!r}")
    finally:
        plot_results = old_plot_results

    print_benchmark_table(results)
    return results


def run_task_d():
    return run_preconditioner_benchmark(
        TASK_D_PRECONDITIONERS,
        "Task D",
        TASK_D_CELLS,
        TASK_D_T,
        TASK_D_DT,
    )


def run_task_e():
    return run_preconditioner_benchmark(
        [TASK_E_PRECONDITIONER],
        "Task E MPI",
        TASK_E_CELLS,
        TASK_E_T,
        TASK_E_DT,
        solver_backend=TASK_E_SOLVER,
        parallel_hint=True,
    )


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
        elif task == "D":
            results["D"] = run_task_d()
        elif task == "E":
            results["E"] = run_task_e()
        else:
            raise ValueError(f"Unknown task {task!r}; use 'A', 'B', 'C', 'D', and/or 'E'.")
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
