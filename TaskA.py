from NavierStokesSolver import NavierStokesSolver, solverParameters

L = 1.0
H = 1.0
T = 10.0
DT = 0.02
STRUCTURED_CELLS = [16, 16]
solver_lib = "petsc"

if __name__ == "__main__":
    solver = NavierStokesSolver(dt_value=DT, H=H, L=L, mu_value= 1, rho_value = 1)
    solver.create_task_A_gridview(STRUCTURED_CELLS)
    solver.buildForms()
    solver.buildPoiseuilleFlowBC()
    solver.visualize_boundary_conditions()
    solver.buildSolutionScheme(
        solverParameters, solver_types=[(solver_lib, "gmres"), (solver_lib, "cg"), (solver_lib, "cg")]
    )
    solver.buildSolutionsPoiseuille()
    results = solver.solve(T=T, plot_results=True)