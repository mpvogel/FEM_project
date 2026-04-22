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
from dune.fem.view import adaptiveLeafGridView
from dune.fem.space import lagrange
from dune.fem.scheme import galerkin as solutionScheme
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

L = 10
H = 1
T = 1

domain = cartesianDomain([0, 0], [L, H], [16, 16])
gridView = leafGridView(domain)
gridView = adaptiveLeafGridView(gridView)
dim = gridView.dimension

# velocity space (vector valued)
velocitySpace = lagrange(gridView, order=2, dimRange=gridView.dimension)

# pressure space
pressureSpace = lagrange(gridView, order=1)

u = TrialFunction(velocitySpace)
v = TestFunction(velocitySpace)
p = TrialFunction(pressureSpace)
q = TestFunction(pressureSpace)

rho = Constant(1, "rho")
mu = Constant(1, "mu")
dt = Constant(0.02, "dt")


def epsilon(u):
    return sym(nabla_grad(u))


def sigma(u, p):
    return 2 * mu * epsilon(u) - p * Identity(dim)


u_prelim_fun = velocitySpace.function(name="u_prelim")
u_prev_fun = velocitySpace.function(name="u_prev")
# u_prev_fun.assign(0)  # initial condition: u=0 at time t=0
u_h_fun = velocitySpace.function(name="u_h")
u_prelim = u_prelim_fun
u_prev = u_prev_fun
u_h = u_h_fun
p_h_fun = pressureSpace.function(name="p_h")
p_prev_fun = pressureSpace.function(name="p_prev")
p_h = p_h_fun
p_prev = p_prev_fun
n = FacetNormal(velocitySpace)

fx = Constant(0, "fx")
fy = Constant(0, "fy")
f = as_vector([fx, fy])  # right-hand side

# IPCS weak UFL:
form_1 = (
    rho * dot(u - u_prev, v) / dt * dx
    + inner(sigma(u, p_prev), epsilon(v)) * dx
    - dot(mu * dot(grad(u), n) - p_prev * n, v) * ds
    - dot(f, v) * dx
    + rho * dot(dot(u_prev, grad(u_prev)), v) * dx
)
# solve after u which has the role of u_n+1/2

form_2 = (
    dot(grad(p), grad(q)) * dx
    - dot(grad(p_prev), grad(q)) * dx
    + 1 / dt * div(u_prelim) * q * dx
)

form_3 = dot(u, v) * dx - dot(u_prelim, v) * dx + dt * dot(grad(p_h - p_prev), v) * dx


x = SpatialCoordinate(velocitySpace)
x_p = SpatialCoordinate(pressureSpace)
dbc_velocity_1 = DirichletBC(
    velocitySpace, [0, 0], abs(x[1]) < 1e-10
)  # zero velocity at bottom
dbc_velocity_2 = DirichletBC(
    velocitySpace, [0, 0], abs(x[1] - H) < 1e-10
)  # zero velocity at top
dbc_pressure_1 = DirichletBC(
    pressureSpace, 8, abs(x_p[0]) < 1e-10
)  # pressure at left boundary
dbc_pressure_2 = DirichletBC(
    pressureSpace, 0, abs(x_p[0] - L) < 1e-10
)  # pressure at right boundary

dbc_pressure = [dbc_pressure_1, dbc_pressure_2]
dbc_velocity = [dbc_velocity_1, dbc_velocity_2]


solverParameters = {
    "nonlinear.tolerance": 1e-10,
    "nonlinear.verbose": False,
    "linear.tolerance": 1e-14,
    "linear.preconditioning.method": "ilu",
    "linear.verbose": False,
    "linear.maxiterations": 1000,
}

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


i = 0
t = 0
while t < T:
    # Solve for new (u,eta)
    info = scheme_1.solve(target=u_prelim_fun)

    info2 = scheme_2.solve(target=p_h_fun)

    info3 = scheme_3.solve(target=u_h_fun)
    p_prev_fun.assign(p_h_fun)
    u_prev_fun.assign(u_h_fun)

    # increment time
    t += dt.value
    i += 1

u_h_fun.plot()
p_h_fun.plot()

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

import pygmsh

with pygmsh.occ.Geometry() as geom:
    # add rectangle with length 1 in x and 1 in y direction.
    L, H = 1.0, 1.0
    rectangle = geom.add_rectangle([0, 0, 0], L, H)
    mesh = geom.generate_mesh()
    points, cells = mesh.points, mesh.cells_dict
    # convert to dictionary understood by DUNE
    domain = {
        "vertices": points[:, :2].astype(float),
        "simplices": cells["triangle"].astype(int),
    }
    print("Number of elements: ", len(domain["simplices"]))

gridView = leafGridView(domain, dimgrid=2)
gridView.plot()

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

# %%
from dune.fem import threading

threading.use = 1
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
