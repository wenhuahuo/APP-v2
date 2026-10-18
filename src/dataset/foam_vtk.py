"""读取 foamToVTK 导出的 VTK 文件：单元中心与体场、边界面场。"""

from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def read_internal(path, fields):
    """读 internal.vtu，返回 (cell_points (N,3) float32, {field: array})。

    单元中心取 foamToVTK 导出的 C 场（OpenFOAM 自己写的单元中心）。vtkCellCenters
    对少数多面体单元会给出错误的中心位置，不能用于数据集几何。
    """
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    if not reader.CanReadFile(str(path)):
        raise RuntimeError(f"无法读取 {path}")
    reader.Update()
    grid = reader.GetOutput()

    centres = grid.GetCellData().GetArray("C")
    if centres is None:
        raise KeyError(f"{path} 缺少单元场 C，导出时需先运行 writeCellCentres")
    points = vtk_to_numpy(centres).astype(np.float32)

    data = {}
    for name in fields:
        array = grid.GetCellData().GetArray(name)
        if array is None:
            raise KeyError(f"{path} 缺少单元场 {name}")
        data[name] = vtk_to_numpy(array).astype(np.float32)
    return points, data


def read_patch(path, fields):
    """读 boundary/<patch>.vtp，返回 (points, offsets, connectivity, {field: (Nf,)})。"""
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    if not reader.CanReadFile(str(path)):
        raise RuntimeError(f"无法读取 {path}")
    reader.Update()
    poly = reader.GetOutput()

    points = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float32)
    offsets = vtk_to_numpy(poly.GetPolys().GetOffsetsArray()).astype(np.int32)
    connectivity = vtk_to_numpy(poly.GetPolys().GetConnectivityArray()).astype(np.int32)

    data = {}
    for name in fields:
        array = poly.GetCellData().GetArray(name)
        if array is None:
            raise KeyError(f"{path} 缺少面场 {name}")
        data[name] = vtk_to_numpy(array).astype(np.float32)
    return points, offsets, connectivity, data


def free_surface(path):
    """从 internal.vtu 抽 alpha.water = 0.5 等值面，返回 (points, offsets, connectivity)。

    先做单元量到顶点量的插值：多面体单元的单元量直接送 vtkContourFilter
    会在网格内部产生伪面，插值到顶点后等值面才是真实自由面。
    """
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    if not reader.CanReadFile(str(path)):
        raise RuntimeError(f"无法读取 {path}")
    reader.Update()

    to_points = vtk.vtkCellDataToPointData()
    to_points.SetInputData(reader.GetOutput())
    to_points.PassCellDataOn()
    to_points.Update()

    contour = vtk.vtkContourFilter()
    contour.SetInputData(to_points.GetOutput())
    contour.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "alpha.water")
    contour.SetValue(0, 0.5)
    contour.Update()
    surface = contour.GetOutput()

    points = vtk_to_numpy(surface.GetPoints().GetData()).astype(np.float32)
    offsets = vtk_to_numpy(surface.GetPolys().GetOffsetsArray()).astype(np.int32)
    connectivity = vtk_to_numpy(surface.GetPolys().GetConnectivityArray()).astype(np.int32)
    return points, offsets, connectivity
