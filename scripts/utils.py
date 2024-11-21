import os

import gmsh
from firedrake.mesh import Mesh
import vtk
from vtk.util.numpy_support import vtk_to_numpy

__all__ = ["read_vtu"]


def read_vtu(vtu_fpath, field_name):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtu_fpath)
    reader.Update()

    block = reader.GetOutput()

    points = block.GetPoints()
    vtk_points = points.GetData()
    np_points = vtk_to_numpy(vtk_points)
    x = np_points[:, 0]
    y = np_points[:, 1]

    point_data = block.GetPointData()
    f_array = point_data.GetArray(field_name)
    f = vtk_to_numpy(f_array)

    return x, y, f


class UnstructuredSquareMeshGenerator():
    """
    Base class for mesh generators.
    """

    def __init__(self, scale=1.0, mesh_type=2):
        """
        :kwarg scale: overall scale factor for the domain size (default: 1.0)
        :type scale: float
        :kwarg mesh_type: Gmsh algorithm number (default: 2)
        :type mesh_type: int
        """
        self.scale = scale
        # TODO: More detail on Gmsh algorithm number (#50)
        self.mesh_type = mesh_type
        self._mesh = None

    @property
    def corners(self):
        """
        Property defining the coordinates of the corner vertices of the domain to be
        meshed.

        :returns: coordinates of the corner vertices of the domain
        :rtype: tuple
        """
        return ((0, 0), (self.scale, 0), (self.scale, self.scale), (0, self.scale))

    def generate_mesh(self, res=0.1, output_filename="./temp.msh", remove_file=False):
        """
        Generate a mesh at a given resolution level.

        :kwarg res: mesh resolution (element diameter) (default: 0.1, suitable for mesh
            with scale 1.0)
        :type res: float
        :kwarg output_filename: filename for saving the mesh, including the path and .msh
            extension (default: './temp.msh')
        :type output_filename: str
        :kwarg remove_file: should the .msh file be removed after generation? (default:
            False)
        :type remove_file: bool
        :returns: mesh generated
        :rtype: :class:`firedrake.mesh.MeshGeometry`
        """
        gmsh.initialize()
        gmsh.model.add("t1")
        self.lc = res
        self._points = [
            gmsh.model.geo.addPoint(*corner, 0, self.lc) for corner in self.corners
        ]
        self._lines = [
            gmsh.model.geo.addLine(point, point_next)
            for point, point_next in zip(
                self._points, self._points[1:] + [self._points[0]]
            )
        ]
        gmsh.model.geo.addCurveLoop([i + 1 for i in range(len(self._points))], 1)
        gmsh.model.geo.addPlaneSurface([1], 1)
        gmsh.model.geo.synchronize()
        gmsh.option.setNumber("Mesh.Algorithm", self.mesh_type)
        for i, line_tag in enumerate(self._lines):
            gmsh.model.addPhysicalGroup(1, [line_tag], i + 1)
            gmsh.model.setPhysicalName(1, i + 1, "Boundary " + str(i + 1))
        gmsh.model.addPhysicalGroup(2, [1], name="My surface")
        gmsh.model.mesh.generate(2)
        gmsh.write(output_filename)
        gmsh.finalize()
        self.num_boundary = len(self._lines)
        self._mesh = Mesh(output_filename)
        if remove_file:
            os.remove(output_filename)
        return self._mesh

    def load_mesh(self, filename):
        """
        Load a mesh from a file saved in .msh format.

        :arg filename: filename including path and the .msh extension
        :type filename: str
        :returns: mesh loaded from file
        :rtype: :class:`firedrake.mesh.MeshGeometry`
        """
        self._mesh = Mesh(filename)
        return self._mesh
