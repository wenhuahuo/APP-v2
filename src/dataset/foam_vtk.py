"""读取 foamToVTK 导出的 VTK 文件：网格、体积、体场、等值面与点插值。"""

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


def _read_unstructured(path):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    if not reader.CanReadFile(str(path)):
        raise RuntimeError(f"无法读取 {path}")
    reader.Update()
    return reader.GetOutput()


def read_internal(path, fields):
    """读 internal.vtu，返回 vtk 网格本体与指定的单元场。"""
    grid = _read_unstructured(path)

    data = {}
    for name in fields:
        array = grid.GetCellData().GetArray(name)
        if array is None:
            raise KeyError(f"{path} 缺少单元场 {name}")
        data[name] = vtk_to_numpy(array).astype(np.float32)
    return {"grid": grid, "fields": data}


def read_patch(path, fields):
    """读 boundary/<patch>.vtp，返回 (points, offsets, connectivity, {field: (Nf,)})。"""
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    if not reader.CanReadFile(str(path)):
        raise RuntimeError(f"无法读取 {path}")
    reader.Update()
    poly = reader.GetOutput()

    points = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
    offsets = vtk_to_numpy(poly.GetPolys().GetOffsetsArray()).astype(np.int64)
    connectivity = vtk_to_numpy(poly.GetPolys().GetConnectivityArray()).astype(np.int64)

    data = {}
    for name in fields:
        array = poly.GetCellData().GetArray(name)
        if array is None:
            raise KeyError(f"{path} 缺少面场 {name}")
        data[name] = vtk_to_numpy(array).astype(np.float64)
    return points, offsets, connectivity, data


def read_free_surface(grid, value=0.5, field="alpha.water"):
    """抽 alpha.water = value 等值面的点，返回 (N,3) float64。

    先把单元量插值到顶点：多面体单元的单元量直接送 vtkContourFilter 会在网格内部
    产生伪面，插值到顶点后等值面才是真实自由面。
    """
    to_points = vtk.vtkCellDataToPointData()
    to_points.SetInputData(grid)
    to_points.PassCellDataOn()
    to_points.Update()

    contour = vtk.vtkContourFilter()
    contour.SetInputData(to_points.GetOutput())
    contour.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, field)
    contour.SetValue(0, value)
    contour.Update()
    points = contour.GetOutput().GetPoints()
    if points is None:
        raise RuntimeError(f"{field} = {value} 等值面为空")
    return vtk_to_numpy(points.GetData()).astype(np.float64)


def probe_cell_fields(grid, points, fields):
    """在任意点上插值单元场，返回 ({field: (N,)}, valid (N,))。

    用 vtkProbeFilter 在包含该点的单元内插值，不做跨单元平均，因此界面附近的
    速度不会被空气相污染。落在网格之外的 query 点由 valid 掩除。
    """
    poly = vtk.vtkPolyData()
    coordinates = vtk.vtkPoints()
    coordinates.SetData(numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float64), deep=True))
    poly.SetPoints(coordinates)

    probe = vtk.vtkProbeFilter()
    probe.SetInputData(poly)
    probe.SetSourceData(grid)
    probe.Update()
    output = probe.GetOutput().GetPointData()

    valid = vtk_to_numpy(output.GetArray("vtkValidPointMask")).astype(bool)
    data = {}
    for name in fields:
        array = output.GetArray(name)
        if array is None:
            raise KeyError(f"internal.vtu 缺少单元场 {name}")
        data[name] = vtk_to_numpy(array).astype(np.float64)
    return data, valid
