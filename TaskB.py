from NavierStokesSolver import NavierStokesSolver
from TaskA import L, H, T, DT, solver_lib

UNSTRUCTURED_MESH_SIZE = 0.08
solverParameters = {
    "solver_1": {
        "nonlinear.tolerance": 1e-8,
        "nonlinear.verbose": False,
        "linear.tolerance": 1e-9,
        "linear.preconditioning.method": "none",
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
        "linear.verbose": False,
        "linear.maxiterations": 1000,
    },
}

if __name__ == "__main__":
    solver = NavierStokesSolver(dt_value=DT, H=H, L=L, mu_value=1, rho_value=1)
    solver.create_task_B_gridview(UNSTRUCTURED_MESH_SIZE)
    solver.buildForms()
    solver.buildPoiseuilleFlowBC()
    solver.visualize_boundary_conditions()
    solver.buildSolutionScheme(
        solverParameters, solver_types=[(solver_lib, "gmres"), (solver_lib, "cg"), (solver_lib, "cg")]
    )
    solver.buildSolutionsPoiseuille()
    results = solver.solve(T=T, plot_results=True)