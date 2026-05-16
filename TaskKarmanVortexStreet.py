from NavierStokesSolver import NavierStokesSolver
from mpi4py import MPI

comm = MPI.COMM_WORLD
L = 2.2
H = 0.41
CYLINDER_CENTER = (0.2, 0.2)
CYLINDER_RADIUS = 0.05
T = 5.0
DT = 0.001
MESH_SIZE = 0.045
INFLOW_RAMP_TIME = 1.0
solverParameters = {
    "solver_1": {
        "nonlinear.tolerance": 1e-8,
        "nonlinear.verbose": False,
        "linear.tolerance": 1e-9,
        "linear.preconditioning.method": "none",
        "linear.petsc.blockedmode": False,  # makes it more robust
        "linear.verbose": False,
        "linear.maxiterations": 1000,
    },
    "solver_2": {
        "nonlinear.tolerance": 1e-8,
        "nonlinear.verbose": False,
        "linear.tolerance": 1e-9,
        "linear.preconditioning.method": "none",
        "linear.petsc.blockedmode": False,
        "linear.verbose": False,
        "linear.maxiterations": 1000,
    },
    "solver_3": {
        "nonlinear.tolerance": 1e-8,
        "nonlinear.verbose": False,
        "linear.tolerance": 1e-9,
        "linear.preconditioning.method": "none",
        "linear.petsc.blockedmode": False,
        "linear.verbose": False,
        "linear.maxiterations": 1000,
    },
}

solver_lib = "petsc"

if __name__ == "__main__":
    solver = NavierStokesSolver(dt_value=DT, H=H, L=L, mu_value=1e-3, rho_value=1)
    solver.create_karman_gridView(
        mesh_size=MESH_SIZE,
        cylinder_center=CYLINDER_CENTER,
        cylinder_r=CYLINDER_RADIUS,
        coarse=False,
    )
    solver.buildForms()
    solver.buildKarmanBC(inflow_ramp_time=1.0,)
    solver.visualize_boundary_conditions()
    solver.buildSolutionScheme(
        solverParameters, solver_types=[(solver_lib, "gmres"), (solver_lib, "cg"), (solver_lib, "cg")]
    )
    solver.solve(T=T, plot_results=True)
