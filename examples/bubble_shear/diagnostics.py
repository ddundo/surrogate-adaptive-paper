import os

import firedrake as fd


class BubbleShearDiagnostics:
    def __init__(self):
        pass

    def compute_rel_error(self, c_init, c_final):
        init_L2_norm = fd.norm(c_init, norm_type="L2")
        abs_L2_error = fd.errornorm(c_init, c_final, norm_type="L2")
        rel_L2_error = 100 * abs_L2_error / init_L2_norm
        return rel_L2_error

    def save_results(
        self,
        mesh_numVertices,
        error,
        cpu_time,
        method,
        target_complexity,
        initial_resolution,
        export,
        filename="results.csv",
    ):
        num_subintervals = len(mesh_numVertices)
        avg_numVertices = sum(mesh_numVertices) / num_subintervals
        if not os.path.exists(filename):
            with open(filename, "w") as file:
                file.write(
                    "initial_resolution,num_subintervals,method,target_complexity,"
                    "avg_numVertices,error,cpu_time,export\n"
                )
        with open(filename, "a") as file:
            file.write(
                f"{initial_resolution},{num_subintervals},{method},{target_complexity},"
                f"{avg_numVertices:.1f},{error:.2f},{cpu_time:.2f},{int(export)}\n"
            )
