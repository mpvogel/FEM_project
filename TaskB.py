from NavierStokesSolver import NavierStokesSolver, solverParameters
from TaskA import L, H, T, DT, solver_lib

UNSTRUCTURED_MESH_SIZE = 0.08

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