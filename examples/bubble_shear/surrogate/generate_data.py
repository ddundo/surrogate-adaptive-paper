import argparse
import os
import sys
sys.path.append("../../../scripts")
from utils import UnstructuredSquareMeshGenerator

import numpy as np
from firedrake import *
from firedrake.petsc import PETSc

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
def get_initial_condition(mesh, train=True):
    x, y = SpatialCoordinate(mesh)
    c0 = Function(FunctionSpace(mesh, "CG", 1), name="c")
    if train:
        c0.interpolate(sin(pi * x) * sin(pi * y))
    else:
        ball_r, ball_x0, ball_y0 = 0.15, 0.5, 0.65
        r = sqrt(pow(x - ball_x0, 2) + pow(y - ball_y0, 2))
        c0.interpolate(conditional(r < ball_r, 1.0, 0.0))
    return c0


@PETSc.Log.EventDecorator()
def run_simulation(mesh, t_start, c0):
    Q = FunctionSpace(mesh, "CG", 1)
    V = VectorFunctionSpace(mesh, "CG", 1)
    R = FunctionSpace(mesh, "R", 0)

    # project initial condition onto the current mesh if it is on a different mesh
    c_ = Function(Q).project(c0) if c0.ufl_domain() != mesh else Function(Q).assign(c0)
    c = Function(Q, name="c")  # solution at current timestep

    u_ = Function(V)  # velocity field at previous timestep
    update_velocity_numpy(mesh, t_start, u_)
    u = Function(V)  # velocity field at current timestep

    # SUPG stabilisation
    D = Function(R).assign(0.1)  # diffusivity coefficient
    h = CellSize(mesh)  # mesh cell size
    U = sqrt(dot(u, u))  # velocity magnitude
    tau = 0.5 * h / U
    tau = min_value(tau, U * h / (6 * D))

    # Apply SUPG stabilisation to the test function
    phi = TestFunction(Q)
    phi += tau * dot(u, grad(phi))

    # Time-stepping parameters
    dt = Function(R).assign(timestep_size)  # timestep size
    theta = Function(R).assign(0.5)  # Crank-Nicolson implicitness

    # Variational form of the advection equation
    trial = TrialFunction(Q)
    a = inner(trial, phi) * dx + dt * theta * inner(dot(u, grad(trial)), phi) * dx
    L = inner(c_, phi) * dx - dt * (1 - theta) * inner(dot(u_, grad(c_)), phi) * dx

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
def main(
    mesh_resolution,
    export_dir,
    train=True,
    unstructured=False,
):
    dirname = f"{"train" if train else "valid"}_res{mesh_resolution}"
    dirname += "us" if unstructured else "s"
    export_dir = os.path.join(export_dir, dirname)
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    if not unstructured:
        mesh = UnitSquareMesh(mesh_resolution, mesh_resolution)
    else:
        mesh_gen = UnstructuredSquareMeshGenerator()
        mesh = mesh_gen.generate_mesh(res=1 / mesh_resolution, remove_file=True)
        print(f"Generated mesh has {mesh.num_cells()} cells and {mesh.num_vertices()} vertices")
    num_timesteps = int(simulation_end_time / timestep_size)

    u = Function(VectorFunctionSpace(mesh, "CG", 1))
    update_velocity_numpy(mesh, 0, u)
    c = get_initial_condition(mesh, train=train)

    sim_generator = run_simulation(mesh, 0, c)
    with CheckpointFile(os.path.join(export_dir, "outputs.h5"), "w") as afile:
        afile.save_function(c, name="c", idx=0)
        afile.save_function(u, name="u", idx=0)
        for i in range(1, num_timesteps + 1):
            c, t = next(sim_generator)

            update_velocity_numpy(mesh, t, u)
            afile.save_function(c, name="c", idx=i)
            afile.save_function(u, name="u", idx=i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--export_dir", type=str, default="data")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--unstructured", action="store_true")

    args = parser.parse_known_args()[0]

    main(
        args.resolution,
        args.export_dir,
        train=args.train,
        unstructured=args.unstructured,
    )
