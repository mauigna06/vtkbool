#!/usr/bin/env python
# *-* coding: UTF-8 *-*

# Copyright 2012-2025 Ronald Römer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# tests of the real boolean operations (union, intersection, difference, difference2)
#
# every result must be a closed, consistently oriented surface with the correct volume

import sys
sys.path.extend(['/home/zippy/vtkbool/build/lib/python3.13/site-packages/vtkbool'])

import math
import pytest
from collections import Counter

from vtkmodules.vtkCommonCore import vtkCommand, VTK_STRING
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkCommonExecutionModel import vtkTrivialProducer, vtkAlgorithm
from vtkmodules.vtkFiltersCore import vtkCleanPolyData
from vtkmodules.vtkFiltersSources import vtkCubeSource, vtkCylinderSource, vtkSphereSource
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.util.misc import calldata_type

from vtkbool import vtkPolyDataBooleanFilter

from test_filter import check_result, write_result, extrude_polygon, create_polydata

OPER_NONE, OPER_UNION, OPER_INTERSECTION, OPER_DIFFERENCE, OPER_DIFFERENCE2 = range(5)

MODES = {
    'union': OPER_UNION,
    'intersection': OPER_INTERSECTION,
    'difference': OPER_DIFFERENCE,
    'difference2': OPER_DIFFERENCE2
}

def get_pd(src):
    if isinstance(src, vtkPolyData):
        return src
    src.Update()
    return src.GetOutput()

def producer(src):
    if isinstance(src, vtkPolyData):
        prod = vtkTrivialProducer()
        prod.SetOutput(src)
        return prod
    return src

def run_boolean(srcA, srcB, mode, matrices=None):
    bf = vtkPolyDataBooleanFilter()
    bf.SetInputConnection(0, producer(srcA).GetOutputPort())
    bf.SetInputConnection(1, producer(srcB).GetOutputPort())
    bf.SetOperMode(mode)

    if matrices is not None:
        for i, m in enumerate(matrices):
            if m is not None:
                bf.SetMatrix(i, m)

    errors = []

    @calldata_type(VTK_STRING)
    def on_error(obj, event, msg):
        errors.append(msg)

    bf.AddObserver(vtkCommand.ErrorEvent, on_error)
    bf.AddObserver(vtkCommand.WarningEvent, on_error)

    bf.Update()

    return bf, errors

def diagonal(pd):
    b = pd.GetBounds()
    return math.sqrt(sum((b[2*i+1]-b[2*i])**2 for i in range(3)))

def merged(pd, tol):
    clean = vtkCleanPolyData()
    clean.SetInputData(pd)
    clean.PointMergingOn()
    clean.ToleranceIsAbsoluteOn()
    clean.SetAbsoluteTolerance(tol)
    clean.ConvertPolysToLinesOff()
    clean.ConvertLinesToPointsOff()
    clean.ConvertStripsToPolysOff()
    clean.Update()
    return clean.GetOutput()

def surface_props(pd, tol=None):
    """returns (signed volume, list of defective edges) of the merged surface"""

    if tol is None:
        tol = 1e-6*max(1., diagonal(pd))

    pd = merged(pd, tol)

    pts = pd.GetPoints()

    b = pd.GetBounds()
    c = [(b[0]+b[1])/2, (b[2]+b[3])/2, (b[4]+b[5])/2]

    def pt(i):
        x, y, z = pts.GetPoint(i)
        return (x-c[0], y-c[1], z-c[2])

    edges = Counter()
    vol = 0

    polys = pd.GetPolys()
    polys.InitTraversal()

    it = polys.NewIterator()

    while not it.IsDoneWithTraversal():
        cell = it.GetCurrentCell()

        ids = [ cell.GetId(i) for i in range(cell.GetNumberOfIds()) ]

        # aufeinanderfolgende gleiche punkte entfernen
        ids = [ a for a, b in zip(ids, ids[1:]+ids[:1]) if a != b ]

        if len(ids) > 2:
            p0 = pt(ids[0])

            for a, b in zip(ids[1:-1], ids[2:]):
                pa = pt(a)
                pb = pt(b)

                vol += (p0[0]*(pa[1]*pb[2]-pa[2]*pb[1])
                    + p0[1]*(pa[2]*pb[0]-pa[0]*pb[2])
                    + p0[2]*(pa[0]*pb[1]-pa[1]*pb[0]))/6

        for a, b in zip(ids, ids[1:]+ids[:1]):
            edges[(a, b)] += 1

        it.GoToNextCell()

    # jede kante muss gleich oft in beiden richtungen vorkommen (geschlossen und konsistent orientiert),
    # mehr als einmal nur dort, wo sich teilkörper entlang einer kante berühren
    bad = [ (e, n) for e, n in edges.items() if n != edges.get(e[::-1], 0) ]

    return vol, bad

def volume(src):
    vol, bad = surface_props(get_pd(src))
    assert not bad, 'input is not closed'
    return vol

def check_solid(bf, errors, expected=None, scale=1., rel_tol=1e-6):
    assert not errors, errors

    pd = bf.GetOutput(0)

    if expected is None or abs(expected) < rel_tol*scale:
        # leeres ergebnis ist erlaubt, die volumen-identitäten fangen falsche leere ergebnisse ab
        if pd.GetNumberOfCells() == 0:
            return 0
    else:
        assert pd.GetNumberOfCells() > 0, 'result is empty'

    assert pd.GetNumberOfLines() == 0
    assert pd.GetNumberOfStrips() == 0

    vol, bad = surface_props(pd)

    assert not bad, f'{len(bad)} defective edges, e.g. {bad[:5]}'
    assert vol > -rel_tol*scale, f'negative volume {vol}'

    if expected is not None:
        assert vol == pytest.approx(expected, rel=0, abs=rel_tol*scale)

    return vol

def check_all_modes(srcA, srcB, expected=None, rel_tol=1e-6, d=None, none=True, matrices=None):
    """
    expected: dict mode-name -> volume, oder None

    prüft zusätzlich die volumen-identitäten
    """

    if matrices is None:
        volA = volume(srcA)
        volB = volume(srcB)
    else:
        volA = volume(transformed(srcA, matrices[0]))
        volB = volume(transformed(srcB, matrices[1]))

    scale = max(abs(volA), abs(volB))

    vols = {}

    for name, mode in MODES.items():
        bf, errors = run_boolean(srcA, srcB, mode, matrices)

        if d is not None:
            path = d / name
            path.mkdir()
            write_result(bf, path)

        exp = None if expected is None else expected.get(name)

        try:
            vols[name] = check_solid(bf, errors, exp, scale, rel_tol)
        except AssertionError as e:
            raise AssertionError(f'{name}: {e}') from e

    tol = rel_tol*scale

    assert vols['union']+vols['intersection'] == pytest.approx(volA+volB, rel=0, abs=tol)
    assert vols['difference']+vols['intersection'] == pytest.approx(volA, rel=0, abs=tol)
    assert vols['difference2']+vols['intersection'] == pytest.approx(volB, rel=0, abs=tol)

    if none:
        bf, errors = run_boolean(srcA, srcB, OPER_NONE, matrices)
        assert not errors, errors
        check_result(bf)

    return vols

def transformed(src, matrix):
    if matrix is None:
        return src

    tra = vtkTransform()
    tra.SetMatrix(matrix)

    tf = vtkTransformPolyDataFilter()
    tf.SetTransform(tra)
    tf.SetInputConnection(producer(src).GetOutputPort())
    tf.SetOutputPointsPrecision(vtkAlgorithm.DOUBLE_PRECISION)

    return tf

def box(x1, x2, y1, y2, z1, z2):
    cube = vtkCubeSource()
    cube.SetBounds(x1, x2, y1, y2, z1, z2)
    cube.SetOutputPointsPrecision(vtkAlgorithm.DOUBLE_PRECISION)
    return cube

def sphere(r=.5, center=(0, 0, 0), res=24):
    s = vtkSphereSource()
    s.SetRadius(r)
    s.SetCenter(*center)
    s.SetThetaResolution(res)
    s.SetPhiResolution(res)
    s.SetOutputPointsPrecision(vtkAlgorithm.DOUBLE_PRECISION)
    return s

def cylinder(r=.5, h=1, center=(0, 0, 0), res=24):
    c = vtkCylinderSource()
    c.SetRadius(r)
    c.SetHeight(h)
    c.SetCenter(*center)
    c.SetResolution(res)
    c.SetOutputPointsPrecision(vtkAlgorithm.DOUBLE_PRECISION)
    return c

def box_volumes(a, b):
    """analytische volumen zweier achsparalleler quader"""

    vol = lambda q: (q[1]-q[0])*(q[3]-q[2])*(q[5]-q[4])

    inter = 1
    for i in range(3):
        inter *= max(0, min(a[2*i+1], b[2*i+1])-max(a[2*i], b[2*i]))

    return {
        'union': vol(a)+vol(b)-inter,
        'intersection': inter,
        'difference': vol(a)-inter,
        'difference2': vol(b)-inter
    }

def check_boxes(a, b, d=None, **kwargs):
    return check_all_modes(box(*a), box(*b), box_volumes(a, b), d=d, **kwargs)

# level 1: einfache boolesche operationen

def test_offset_boxes(tmp_path):
    check_boxes((0, 1, 0, 1, 0, 1), (.5, 1.5, .25, 1.25, .3, 1.3), tmp_path)

def test_box_sphere(tmp_path):
    check_all_modes(box(-.5, .5, -.5, .5, -.5, .5), sphere(.6, (.3, .2, .1)), d=tmp_path)

def test_spheres(tmp_path):
    check_all_modes(sphere(.5), sphere(.5, (.3, .1, .05), 20), d=tmp_path)

def test_drilled_box(tmp_path):
    check_all_modes(box(-.5, .5, -.5, .5, -.5, .5), cylinder(.3, 2, res=32), d=tmp_path)

def test_readme(tmp_path):
    cube = vtkCubeSource()
    cube.SetYLength(.5)

    cyl = vtkCylinderSource()
    cyl.SetResolution(32)
    cyl.SetHeight(.5)
    cyl.SetCenter(0, .5, 0)

    check_all_modes(cube, cyl, d=tmp_path)

# level 2: koplanare und sich berührende flächen

UNIT = (0, 1, 0, 1, 0, 1)

coplanar_boxes = {
    'face_touch': (1, 2, 0, 1, 0, 1),
    'face_touch_partial': (1, 2, .5, 1.5, .25, 1.25),
    'face_touch_inner': (1, 2, .25, .75, .25, .75),
    'shared_4_faces': (.5, 1.5, 0, 1, 0, 1),
    'shared_2_faces': (.5, 1.5, .5, 1.5, 0, 1),
    'shared_1_face': (.5, 1.5, .5, 1.5, .5, 1),
    'flush_corner': (0, .5, 0, .5, 0, .5),
    'flush_face': (0, .5, .25, .75, .25, .75),
    'flush_through': (0, 1, .25, .75, .25, .75),
    'slab': (0, 1, 0, 1, .25, .75),
    'identical': UNIT,
    'contained_shared': (0, 2, 0, 1, 0, 1),
    'cross': (.25, .75, -1, 2, 0, 1)
}

@pytest.mark.parametrize('name', coplanar_boxes.keys())
def test_coplanar_boxes(name, tmp_path):
    check_boxes(UNIT, coplanar_boxes[name], tmp_path)

@pytest.mark.parametrize('r', [.3, .5])
def test_cylinder_caps(r, tmp_path):
    # deckflächen koplanar zu den seiten des würfels, bei r=.5 berühren sich die mantelflächen
    check_all_modes(box(-.5, .5, -.5, .5, -.5, .5), cylinder(r, 1, res=32), d=tmp_path)

def test_identical_spheres(tmp_path):
    s = sphere(.5, res=16)
    vol = volume(s)
    check_all_modes(s, sphere(.5, res=16), {'union': vol, 'intersection': vol, 'difference': 0, 'difference2': 0}, d=tmp_path)
