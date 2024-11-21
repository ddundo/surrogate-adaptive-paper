import argparse
import os
import time

import numpy as np
from animate.adapt import adapt
from animate.metric import RiemannianMetric
from diagnostics import BubbleShearDiagnostics
from firedrake import *
from firedrake.petsc import PETSc
from goalie.metric import space_time_normalise
from goalie.time_partition import TimePartition

period = 6.0
simulation_end_time = period / 2
timestep_size = 0.005


@PETSc.Log.EventDecorator()
def update_velocity_numpy(mesh, t, u):
    coords = mesh.coordinates.dat.data
    x, y = coords[:, 0], coords[:, 1]
    u_x = (
        2
        * np.sin(np.pi * x) ** 2
        * np.sin(2 * np.pi * y)
        * np.cos(2 * np.pi * float(t) / period)
    )
    u_y = (
        -np.sin(2 * np.pi * x)
        * np.sin(np.pi * y) ** 2
        * np.cos(2 * np.pi * float(t) / period)
    )
    u.dat.data[:, 0] = u_x
    u.dat.data[:, 1] = u_y


@PETSc.Log.EventDecorator()
def get_initial_condition(mesh):
    x, y = SpatialCoordinate(mesh)
    ball_r, ball_x0, ball_y0 = 0.15, 0.5, 0.65
    r = sqrt(pow(x - ball_x0, 2) + pow(y - ball_y0, 2))
    c0 = Function(FunctionSpace(mesh, "CG", 1), name="c")
    c0.interpolate(conditional(r < ball_r, 1.0, 0.0))
    return c0


@PETSc.Log.EventDecorator()
def get_variational_form(mesh, sol_fspace, sol_, u, u_):
    R = FunctionSpace(mesh, "R", 0)

    # SUPG stabilisation
    D = Function(R).assign(0.1)  # diffusivity coefficient
    h = CellSize(mesh)  # mesh cell size
    U = sqrt(dot(u, u))  # velocity magnitude
    tau = 0.5 * h / U
    tau = min_value(tau, U * h / (6 * D))

    # Apply SUPG stabilisation to the test function
    phi = TestFunction(sol_fspace)
    phi += tau * dot(u, grad(phi))

    # Time-stepping parameters
    dt = Function(R).assign(timestep_size)  # timestep size
    theta = Function(R).assign(0.5)  # Crank-Nicolson implicitness

    # Variational form of the advection equation
    trial = TrialFunction(sol_fspace)
    a = inner(trial, phi) * dx + dt * theta * inner(dot(u, grad(trial)), phi) * dx
    L = inner(sol_, phi) * dx - dt * (1 - theta) * inner(dot(u_, grad(sol_)), phi) * dx

    return a, L


@PETSc.Log.EventDecorator()
def run_simulation(mesh, t_start, c0):
    Q = FunctionSpace(mesh, "CG", 1)
    V = VectorFunctionSpace(mesh, "CG", 1)

    # project initial condition onto the current mesh if it is on a different mesh
    c_ = Function(Q).project(c0) if c0.ufl_domain() != mesh else Function(Q).assign(c0)
    c = Function(Q, name="c")  # solution at current timestep

    u_ = Function(V)  # velocity field at previous timestep
    update_velocity_numpy(mesh, t_start, u_)
    u = Function(V)  # velocity field at current timestep

    a, L = get_variational_form(mesh, Q, c_, u, u_)
    lvp = LinearVariationalProblem(a, L, c, bcs=DirichletBC(Q, 0.0, "on_boundary"))
    lvs = LinearVariationalSolver(lvp)

    # Integrate from t_start to t_end
    t = t_start + timestep_size
    while True:
        # Update the background velocity field at the current timestep
        update_velocity_numpy(mesh, t, u)

        # Solve the advection equation
        lvs.solve()

        yield c, t

        # Update the solution at the previous timestep
        c_.assign(c)
        u_.assign(u)
        t += timestep_size


@PETSc.Log.EventDecorator()
def adapt_classical(mesh, c, metric_params):
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    metric = RiemannianMetric(P1_ten)
    metric.set_parameters(metric_params)
    metric.compute_hessian(c)
    metric.normalise()
    adapted_mesh = adapt(mesh, metric)
    return adapted_mesh


@PETSc.Log.EventDecorator()
def adapt_metric_advection(mesh, t_start, t_end, c, metric_params):
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    m = RiemannianMetric(P1_ten)  # metric at current timestep
    m_ = RiemannianMetric(P1_ten)  # metric at previous timestep
    metric_intersect = RiemannianMetric(P1_ten)

    # Compute the Hessian metric at t_start
    for mtrc in [m, m_, metric_intersect]:
        mtrc.set_parameters(metric_params)
    m_.compute_hessian(c)
    m_.normalise()

    # Set the boundary condition for the metric tensor
    h_bc = Function(P1_ten)
    h_max = metric_params["dm_plex_metric"]["h_max"]
    h_bc.interpolate(Constant([[1.0 / h_max**2, 0.0], [0.0, 1.0 / h_max**2]]))

    V = VectorFunctionSpace(mesh, "CG", 1)
    u_ = Function(V)
    update_velocity_numpy(mesh, t_start, u_)
    u = Function(V)

    a, L = get_variational_form(mesh, P1_ten, m_, u, u_)
    bcs = DirichletBC(P1_ten, h_bc, "on_boundary")
    lvp = LinearVariationalProblem(a, L, m, bcs=bcs)
    lvs = LinearVariationalSolver(lvp)

    # Integrate from t_start to t_end
    t = t_start + timestep_size
    while t < t_end + 0.5 * timestep_size:
        update_velocity_numpy(mesh, t, u)

        lvs.solve()

        # Intersect metrics at every timestep
        m.enforce_spd(restrict_sizes=True, restrict_anisotropy=True)
        metric_intersect.intersect(m)

        # Update fields at the previous timestep
        m_.assign(m)
        u_.assign(u)
        t += timestep_size

    metric_intersect.normalise()
    adapted_mesh = adapt(mesh, metric_intersect)
    return adapted_mesh


@PETSc.Log.EventDecorator()
def adapt_global(mesh_seq, solutions, metric_parameters):
    tp = TimePartition(
        simulation_end_time,
        len(mesh_seq),
        timestep_size,
        "c",
        num_timesteps_per_export=1,
    )
    metrics = []
    for i, mesh in enumerate(mesh_seq):
        P1_ten = TensorFunctionSpace(mesh, "CG", 1)
        metric = RiemannianMetric(P1_ten)
        for sol in solutions[i]:
            hessian = RiemannianMetric(P1_ten)
            hessian.compute_hessian(sol)
            hessian.enforce_spd(restrict_sizes=True, restrict_anisotropy=True)
            metric.intersect(hessian)
        metrics.append(metric)
    space_time_normalise(metrics, tp, metric_parameters)
    adapted_mesh_seq = [adapt(mesh_seq[i], metrics[i]) for i in range(len(mesh_seq))]
    return adapted_mesh_seq


@PETSc.Log.EventDecorator()
def solve_subinterval(
    t0,
    num_timesteps,
    mesh,
    c0,
    global_export_freq=None,
    vtk_export_freq=None,
    vtk_export_file=None,
):
    solutions = []
    sim_generator = run_simulation(mesh, t0, c0)
    if global_export_freq is not None:
        for tstep in range(num_timesteps):
            c, _ = next(sim_generator)
            if tstep % global_export_freq == 0:
                solutions.append(c.copy(deepcopy=True))
    elif vtk_export_file is not None:
        if t0 == 0:
            vtk_export_file.write(c0, time=t0)
        for tstep in range(num_timesteps):
            c, t = next(sim_generator)
            if tstep % vtk_export_freq == 0:
                vtk_export_file.write(c, time=round(t, 3))
    else:
        for _ in range(num_timesteps):
            c, _ = next(sim_generator)
    return c, solutions


@PETSc.Log.EventDecorator()
def main(
    mesh_resolution,
    num_subintervals,
    adaptation_method,
    metric_parameters,
    global_export_freq=5,
    vtk_export_dir=None,
    vtk_export_freq=None,
    surrogate_model_path=None,
):
    if vtk_export_dir is not None:
        vtk_export_file = VTKFile(os.path.join(vtk_export_dir, "c.pvd"), adaptive=True)
    else:
        vtk_export_file = None
    interval_length = simulation_end_time / num_subintervals
    num_timesteps_per_subinterval = int(interval_length / timestep_size)
    target_complexity = metric_parameters["dm_plex_metric"]["target_complexity"]

    if adaptation_method == "global":
        solutions = []
        meshes = [
            UnitSquareMesh(mesh_resolution, mesh_resolution)
            for _ in range(num_subintervals)
        ]
        mesh = meshes[0]
    elif adaptation_method == "hybrid-classical":
        solutions = []
        meshes = []
        mesh = UnitSquareMesh(mesh_resolution, mesh_resolution)
    else:
        mesh = UnitSquareMesh(mesh_resolution, mesh_resolution)
        mesh_numVertices = []
    c = get_initial_condition(mesh)

    cpu_start_time = time.time()
    for i in range(num_subintervals):
        t0 = i * interval_length
        if adaptation_method in ("global", "hybrid-classical"):
            if adaptation_method == "hybrid-classical":
                mesh = adapt_classical(mesh, c, metric_parameters)
                meshes.append(mesh)
            c, subinterval_solutions = solve_subinterval(
                t0,
                num_timesteps_per_subinterval,
                meshes[i],
                c,
                global_export_freq=global_export_freq,
            )
            solutions.append(subinterval_solutions)
            if i != num_subintervals - 1:
                continue
            else:
                # After the last subinterval, adapt mesh sequence and solve again
                adapted_meshes = adapt_global(meshes, solutions, metric_parameters)
                mesh_numVertices = [mesh.num_vertices() for mesh in adapted_meshes]

                c = get_initial_condition(adapted_meshes[0])
                for ii in range(num_subintervals):
                    t0 = ii * interval_length
                    mesh = adapted_meshes[ii]
                    c, _ = solve_subinterval(
                        t0,
                        num_timesteps_per_subinterval,
                        mesh,
                        c,
                        vtk_export_freq=vtk_export_freq,
                        vtk_export_file=vtk_export_file,
                    )
                break

        elif adaptation_method == "classical":
            mesh = adapt_classical(mesh, c, metric_parameters)
        elif adaptation_method == "metric-advection":
            t1 = (i + 1) * interval_length
            mesh = adapt_metric_advection(mesh, t0, t1, c, metric_parameters)
        mesh_numVertices.append(mesh.num_vertices())
        c, _ = solve_subinterval(
            t0,
            num_timesteps_per_subinterval,
            mesh,
            c,
            vtk_export_freq=vtk_export_freq,
            vtk_export_file=vtk_export_file,
        )

    cpu_total_time = time.time() - cpu_start_time
    diagnostics = BubbleShearDiagnostics()
    c0 = get_initial_condition(mesh)
    rel_error = diagnostics.compute_rel_error(c0, c)
    diagnostics.save_results(
        mesh_numVertices,
        rel_error,
        cpu_total_time,
        adaptation_method,
        target_complexity,
        mesh_resolution,
        vtk_export_file is not None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial_resolution", type=int)
    parser.add_argument("--num_subintervals", type=int)
    parser.add_argument(
        "--adaptation_method",
        type=str,
        choices=[
            "uniform",
            "classical",
            "metric-advection",
            "global",
            "hybrid-classical",
            "surrogate",
        ],
    )
    parser.add_argument("--target_complexity", type=float)
    parser.add_argument("--export_dir", type=str)
    parser.add_argument("--export_frequency", type=int)
    parser.add_argument("--surrogate_model_path", type=str)

    args = parser.parse_known_args()[0]

    metric_params = {
        "dm_plex_metric": {
            "target_complexity": args.target_complexity,
            "p": 2.0,  # normalisation order
            "h_min": 1e-04,  # minimum allowed edge length
            "h_max": 1.0,  # maximum allowed edge length
        }
    }

    main(
        args.initial_resolution,
        args.num_subintervals,
        args.adaptation_method,
        metric_params,
        vtk_export_dir=args.export_dir,
        vtk_export_freq=args.export_frequency,
        surrogate_model_path=args.surrogate_model_path,
    )
