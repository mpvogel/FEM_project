from NavierStokesSolver import NavierStokesSolver
import numpy as np
import sys
import time
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

solver_precond_types_list = [
    # Baselines
    [("gmres", "none"),         ("cg",    "none"),    ("cg",    "none")],
    [("gmres", "jacobi"),       ("cg",    "jacobi"),  ("cg",    "jacobi")],

    # Einfache Glätter
    [("gmres", "gauss-seidel"), ("gmres", "gauss-seidel"), ("gmres", "gauss-seidel")],
    [("gmres", "sor"),          ("gmres", "sor"),          ("gmres", "sor")],
    [("gmres", "ssor"),         ("cg",    "ssor"),         ("cg",    "ssor")],

    # Lokale Faktorisierung
    [("gmres", "ilu"),          ("gmres", "ilu"),     ("gmres", "ilu")],

    # Domain-Decomposition / Schwarz
    [("gmres", "oas"),          ("gmres", "oas"),     ("cg",    "oas")],

    # AMG-Varianten
    [("gmres", "pcgamg"),       ("cg",    "pcgamg"),  ("cg",    "pcgamg")],
    [("gmres", "hypre"),        ("cg",    "hypre"),   ("cg",    "hypre")],
    [("gmres", "ml"),           ("cg",    "ml"),      ("cg",    "ml")],

    # Direkte Referenz, nur kleine Gitter
    [("gmres", "lu"),           ("gmres", "lu"),      ("gmres", "lu")],

    # Sinnvolle gemischte Varianten für IPCS
    [("gmres", "oas"),          ("cg",    "pcgamg"),  ("cg",    "jacobi")],
    [("gmres", "oas"),          ("cg",    "hypre"),   ("cg",    "jacobi")],
    [("gmres", "ilu"),          ("cg",    "pcgamg"),  ("cg",    "jacobi")],
    [("gmres", "ilu"),          ("cg",    "hypre"),   ("cg",    "jacobi")],
    [("gmres", "pcgamg"),       ("cg",    "pcgamg"),  ("cg",    "jacobi")],
    [("gmres", "hypre"),        ("cg",    "hypre"),   ("cg",    "jacobi")],
]

solver_precond_types_list_parallel_all = [
    # Baselines
    [("gmres", "none"),   ("cg",    "none"),   ("cg", "none")],
    [("gmres", "jacobi"), ("cg",    "jacobi"), ("cg", "jacobi")],

    # Einfache parallele Glätter
    [("gmres", "sor"),    ("gmres", "sor"),    ("gmres", "sor")],
    [("gmres", "ssor"),   ("cg",    "ssor"),   ("cg",    "ssor")],

    # Domain-Decomposition
    [("gmres", "oas"),    ("gmres", "oas"),    ("cg", "jacobi")],

    # AMG-Varianten
    [("gmres", "pcgamg"), ("cg", "pcgamg"), ("cg", "jacobi")],
    [("gmres", "hypre"),  ("cg", "hypre"),  ("cg", "jacobi")],
    [("gmres", "ml"),     ("cg", "ml"),     ("cg", "jacobi")],

    # AMG auch für Step 3 testen, falls ihr sehen wollt, ob es sich lohnt
    # Erwartung: meist nicht, weil Step 3 nur Mass-Matrix ist.
    [("gmres", "pcgamg"), ("cg", "pcgamg"), ("cg", "pcgamg")],
    [("gmres", "hypre"),  ("cg", "hypre"),  ("cg", "hypre")],
    [("gmres", "ml"),     ("cg", "ml"),     ("cg", "ml")],

    # Mixed: Step 1 lokal/DD, Step 2 AMG, Step 3 billig
    [("gmres", "oas"),    ("cg", "pcgamg"), ("cg", "jacobi")],
    [("gmres", "oas"),    ("cg", "hypre"),  ("cg", "jacobi")],
    [("gmres", "oas"),    ("cg", "ml"),     ("cg", "jacobi")],

    # Mixed: Step 1 einfacher Glätter, Step 2 AMG
    [("gmres", "jacobi"), ("cg", "pcgamg"), ("cg", "jacobi")],
    [("gmres", "jacobi"), ("cg", "hypre"),  ("cg", "jacobi")],
    [("gmres", "jacobi"), ("cg", "ml"),     ("cg", "jacobi")],

    [("gmres", "sor"),    ("cg", "pcgamg"), ("cg", "jacobi")],
    [("gmres", "sor"),    ("cg", "hypre"),  ("cg", "jacobi")],
    [("gmres", "sor"),    ("cg", "ml"),     ("cg", "jacobi")],

    [("gmres", "ssor"),   ("cg", "pcgamg"), ("cg", "jacobi")],
    [("gmres", "ssor"),   ("cg", "hypre"),  ("cg", "jacobi")],
    [("gmres", "ssor"),   ("cg", "ml"),     ("cg", "jacobi")],

    # Mixed: Step 1 AMG, Step 2 anderer AMG
    [("gmres", "pcgamg"), ("cg", "hypre"),  ("cg", "jacobi")],
    [("gmres", "hypre"),  ("cg", "pcgamg"), ("cg", "jacobi")],
    [("gmres", "pcgamg"), ("cg", "ml"),     ("cg", "jacobi")],
    [("gmres", "ml"),     ("cg", "pcgamg"), ("cg", "jacobi")],
    [("gmres", "hypre"),  ("cg", "ml"),     ("cg", "jacobi")],
    [("gmres", "ml"),     ("cg", "hypre"),  ("cg", "jacobi")],
]

solver_precond_types_list_parallel_main = [
    # 1. Keine Vorkonditionierung: nur Referenz auf kleinem Problem
    [("gmres", "none"),   ("cg", "none"),   ("cg", "none")],

    # 2. Einfache parallele Baseline
    [("gmres", "jacobi"), ("cg", "jacobi"), ("cg", "jacobi")],

    # 3. Domain-Decomposition-Preconditioner
    [("gmres", "oas"),    ("gmres", "oas"), ("cg", "jacobi")],

    # 4. PETSc GAMG
    [("gmres", "pcgamg"), ("cg", "pcgamg"), ("cg", "jacobi")],

    # 5. hypre/BoomerAMG
    [("gmres", "hypre"),  ("cg", "hypre"),  ("cg", "jacobi")],

    # 6. Optional: ML AMG, falls es stabil läuft
    [("gmres", "ml"),     ("cg", "ml"),     ("cg", "jacobi")],

    # 7. Wahrscheinlich beste praktische Mixed-Varianten
    [("gmres", "oas"),    ("cg", "pcgamg"), ("cg", "jacobi")],
    [("gmres", "oas"),    ("cg", "hypre"),  ("cg", "jacobi")],
]

if __name__ == "__main__":
    # get argument for parallel or serial run
    save_name = "None"
    analysis = None
    save_detailed_results = False
    if len(sys.argv) > 1:
        if sys.argv[1] == "parallel":
            solver_precond_types_list = solver_precond_types_list_parallel_all
            save_name = "karman_results_parallel.npy"
            analysis = False
        elif sys.argv[1] == "serial":
            solver_precond_types_list = solver_precond_types_list
            save_name = "karman_results_serial.npy"
            analysis = True
            save_detailed_results = True
        elif sys.argv[1] == "parallel_main":
            solver_precond_types_list = solver_precond_types_list_parallel_main
            save_name = "karman_results_parallel_main.npy"
            analysis = False
        else:
            print("Invalid argument. Use 'parallel', 'serial', or 'parallel_main'.")
            sys.exit(1)
        
    results_list = []
    results_minimal_list = []
    for solver_precond_types in solver_precond_types_list:
        precond_types = [precond for _, precond in solver_precond_types]
        solver_types = [(solver_lib, solver) for solver, _ in solver_precond_types]
        
        solverParameters["solver_1"]["linear.preconditioning.method"] = precond_types[0]
        solverParameters["solver_2"]["linear.preconditioning.method"] = precond_types[1]
        solverParameters["solver_3"]["linear.preconditioning.method"] = precond_types[2]

        solver = NavierStokesSolver(dt_value=DT, H=H, L=L, mu_value= 1e-3, rho_value = 1)
        solver.create_karman_gridView(
            mesh_size=MESH_SIZE,
            cylinder_center=CYLINDER_CENTER,
            cylinder_r=CYLINDER_RADIUS,
            coarse=False,
        )


        solver.buildForms()
        solver.buildKarmanBC(
            inflow_ramp_time=1.0,
        )

        solver.visualize_boundary_conditions()
        runtime = 0
        try:
            solver.buildSolutionScheme(
                solverParameters, solver_types
            )
            comm.Barrier()
            start_time = MPI.Wtime()
            _, scheme_1_return, scheme_2_return, scheme_3_return = solver.solve(T=T, plot_results=False, adaptive=False, analysis=analysis, maxLevel=solver.gridView.hierarchicalGrid.maxLevel) # type: ignore
            runtime = MPI.Wtime() - start_time

            # Allreduce: take Max of sim time over ranks
            runtime = comm.allreduce(runtime, op=MPI.MAX)
            
        except Exception as e:
            print(f"Error with preconditioner combination {solver_precond_types}: {e}")
            scheme_1_return = scheme_2_return = scheme_3_return = "Error"

        if save_detailed_results:
            save_results = {
                "T": T,
                "total_runtime": runtime,
                "tolerance1": solverParameters["solver_1"]["linear.tolerance"],
                "maxiter1": solverParameters["solver_1"]["linear.maxiterations"],
                "tolerance2": solverParameters["solver_2"]["linear.tolerance"],
                "maxiter2": solverParameters["solver_2"]["linear.maxiterations"],
                "tolerance3": solverParameters["solver_3"]["linear.tolerance"],
                "maxiter3": solverParameters["solver_3"]["linear.maxiterations"],
                "precond1": precond_types[0],
                "precond2": precond_types[1],
                "precond3": precond_types[2],
                "solver_1": solver_types[0][1],
                "solver_2": solver_types[1][1],
                "solver_3": solver_types[2][1],
                "scheme_1_return": scheme_1_return,
                "scheme_2_return": scheme_2_return,
                "scheme_3_return": scheme_3_return,
            }

            results_list.append(save_results)

        minimal_results = {
            "precond1": precond_types[0],
            "precond2": precond_types[1],
            "precond3": precond_types[2],
            "solver_1": solver_types[0][1],
            "solver_2": solver_types[1][1],
            "solver_3": solver_types[2][1],
            "T": T,
            "total_runtime": runtime,
            "number of ranks": comm.Get_size(),
        }

    if save_detailed_results:
        np.save(save_name, results_list)

    if comm.rank == 0:
        np.save(f"minresults_{save_name}", results_minimal_list)
        
