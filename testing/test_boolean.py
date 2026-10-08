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

from vtkmodules.vtkCommonCore import vtkCommand, VTK_STRING, vtkOutputWindow, vtkStringOutputWindow
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

    # meldungen der intern verwendeten objekte
    win = vtkStringOutputWindow()

    prev = vtkOutputWindow.GetInstance()
    prev.Register(None)

    vtkOutputWindow.SetInstance(win)

    try:
        bf.Update()
    finally:
        vtkOutputWindow.SetInstance(prev)
        prev.UnRegister(None)

    if win.GetOutput():
        errors.append(win.GetOutput())

    return bf, errors

def diagonal(pd):
    b = pd.GetBounds()
    return math.sqrt(sum((b[2*i+1]-b[2*i])**2 for i in range(3)))

def internal_tol(pdA, pdB):
    # der filter rechnet mit 1e-5 in koordinaten, in denen die diagonale der gemeinsamen bounding-box in [1, 2) liegt

    a = pdA.GetBounds()
    b = pdB.GetBounds()

    diag = math.sqrt(sum((max(a[2*i+1], b[2*i+1])-min(a[2*i], b[2*i]))**2 for i in range(3)))

    _, e = math.frexp(diag)

    return 1e-5*2**(e-1)

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
        tol = 1e-6*diagonal(pd)

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
        check_result(bf, tol=internal_tol(get_pd(srcA), get_pd(srcB)))

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

# level 3: fast entartete fälle

def rotated(src, angle, axis=(0, 0, 1), center=(0, 0, 0), translate=(0, 0, 0)):
    tra = vtkTransform()
    tra.PostMultiply()
    tra.Translate(*[ -c for c in center ])
    tra.RotateWXYZ(angle, *axis)
    tra.Translate(*center)
    tra.Translate(*translate)

    tf = vtkTransformPolyDataFilter()
    tf.SetTransform(tra)
    tf.SetInputConnection(producer(src).GetOutputPort())
    tf.SetOutputPointsPrecision(vtkAlgorithm.DOUBLE_PRECISION)

    return tf

EPSILONS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10]

# flächen, die näher als die interne toleranz (1e-5) beieinander liegen, werden als koplanar behandelt,
# das volumen darf sich dabei um fläche*abstand ändern
SNAP_TOL = 2e-5

@pytest.mark.parametrize('eps', EPSILONS)
def test_near_coplanar(eps, tmp_path):
    check_boxes(UNIT, (.5, 1.5, .25, 1.25, eps, 1+eps), tmp_path, rel_tol=SNAP_TOL)

@pytest.mark.parametrize('eps', EPSILONS)
def test_near_coplanar_2(eps, tmp_path):
    check_boxes(UNIT, (.5+eps, 1.5+eps, -eps, 1-eps, eps, 1+eps), tmp_path, rel_tol=SNAP_TOL)

# liegen die flächen in mehreren richtungen knapp innerhalb der toleranz neben denen des anderen körpers,
# sind die eingerasteten schnittpunkte bis zu eps*sqrt(2) voneinander entfernt und lassen sich nicht mehr
# konsistent zusammenfassen (grenze des toleranzbasierten ansatzes)

@pytest.mark.xfail(strict=True, reason='offsets between tol/sqrt(2) and tol in several directions')
@pytest.mark.parametrize('f', [.75, .85, .95, .999])
def test_near_coplanar_band(f):
    eps = f*internal_tol(get_pd(box(*UNIT)), get_pd(box(.5, 1.5, 0, 1, 0, 1)))
    check_boxes(UNIT, (.5+eps, 1.5+eps, -eps, 1-eps, eps, 1+eps), rel_tol=SNAP_TOL)

@pytest.mark.parametrize('eps', EPSILONS)
def test_near_touching(eps, tmp_path):
    # berühren fast, überlappen minimal
    check_boxes(UNIT, (1-eps, 2, .25, .75, .25, .75), tmp_path, rel_tol=SNAP_TOL)

@pytest.mark.parametrize('k', range(1, 8))
def test_sphere_rings(k, tmp_path):
    # die deckfläche des quaders geht genau durch einen breitenkreis der kugel
    z = .5*math.cos(k*math.pi/8)
    check_all_modes(sphere(.5, res=9), box(-.3, .3, -.3, .3, -1, z), d=tmp_path)

def test_sphere_meridian(tmp_path):
    # die seitenfläche y=0 enthält die kanten eines meridians
    check_all_modes(sphere(.5, res=8), box(-1, 1, 0, 1, -.3, .3), d=tmp_path)

def test_sphere_meridian_2(tmp_path):
    check_all_modes(sphere(.5, res=8), box(-1, 1, 0, 1, -1, 1), d=tmp_path)

@pytest.mark.parametrize('angle', [1e-4, 1e-6, 1e-8])
def test_tiny_rotation(angle, tmp_path):
    a = box(0, 1, 0, 1, 0, 1)
    b = rotated(box(.5, 1.5, .5, 1.5, 0, 1), math.degrees(angle), center=(1, 1, .5))
    check_all_modes(a, b, d=tmp_path)

@pytest.mark.parametrize('angle', [1e-4, 1e-6, 1e-8])
def test_tiny_rotation_2(angle, tmp_path):
    a = box(0, 1, 0, 1, 0, 1)
    b = rotated(box(0, 1, 0, 1, 0, 1), math.degrees(angle), axis=(1, 1, 1), center=(.5, .5, .5))
    check_all_modes(a, b, d=tmp_path)

# level 4: drehungen, auch über SetMatrix

def rotation_matrix(angle, axis, center=(0, 0, 0)):
    tra = vtkTransform()
    tra.PostMultiply()
    tra.Translate(*[ -c for c in center ])
    tra.RotateWXYZ(angle, *axis)
    tra.Translate(*center)
    return tra.GetMatrix()

AXES = {'z': (0, 0, 1), 'xy': (1, 1, 0), 'xyz': (1, 1, 1), 'skew': (.3, -.7, .2)}

def check_rotated(srcA, srcB, angle, axis, center, rel_tol=1e-6):
    # einmal vorab transformiert, einmal über die matrix des filters

    m = rotation_matrix(angle, axis, center)

    vols = check_all_modes(srcA, transformed(srcB, m), rel_tol=rel_tol)
    vols2 = check_all_modes(srcA, srcB, rel_tol=rel_tol, matrices=[None, m], none=False)

    scale = max(volume(srcA), volume(srcB))

    for name in MODES:
        assert vols2[name] == pytest.approx(vols[name], rel=0, abs=rel_tol*scale), name

@pytest.mark.parametrize('axis', AXES.keys())
@pytest.mark.parametrize('angle', range(0, 91, 3))
def test_rotated_boxes(angle, axis):
    check_rotated(box(*UNIT), box(.4, 1.4, .3, 1.3, .2, 1.2), angle, AXES[axis], (.9, .8, .7))

@pytest.mark.parametrize('axis', AXES.keys())
@pytest.mark.parametrize('angle', range(0, 91, 15))
def test_rotated_boxes_centered(angle, axis):
    # gleicher mittelpunkt, viele koplanare und kollineare lagen
    check_rotated(box(*UNIT), box(.25, .75, .25, .75, -.5, 1.5), angle, AXES[axis], (.5, .5, .5))

@pytest.mark.parametrize('angle', range(0, 181, 20))
def test_rotated_spheres(angle):
    check_rotated(sphere(.5, res=18), sphere(.45, (.2, .1, 0), 13), angle, (.2, 1, .4), (0, 0, 0))

@pytest.mark.parametrize('angle', range(0, 91, 10))
def test_rotated_cylinder(angle):
    check_rotated(box(-.5, .5, -.5, .5, -.5, .5), cylinder(.3, 2, res=20), angle, (1, 0, 0), (0, 0, 0))

# level 5: maßstab und lage

def scaled(src, s, t=(0, 0, 0)):
    tra = vtkTransform()
    tra.PostMultiply()
    tra.Scale(s, s, s)
    tra.Translate(*t)

    tf = vtkTransformPolyDataFilter()
    tf.SetTransform(tra)
    tf.SetInputConnection(producer(src).GetOutputPort())
    tf.SetOutputPointsPrecision(vtkAlgorithm.DOUBLE_PRECISION)

    return tf

SCALES = [1e-4, 1e-3, 1e-2, 1e2, 1e3, 1e4, 1e6]

scale_cases = {
    'offset_boxes': lambda: (box(*UNIT), box(.5, 1.5, .25, 1.25, .3, 1.3)),
    'box_sphere': lambda: (box(-.5, .5, -.5, .5, -.5, .5), sphere(.6, (.3, .2, .1))),
    'drilled_box': lambda: (box(-.5, .5, -.5, .5, -.5, .5), cylinder(.3, 2, res=32)),
    'shared_2_faces': lambda: (box(*UNIT), box(.5, 1.5, .5, 1.5, 0, 1)),
    'flush_through': lambda: (box(*UNIT), box(0, 1, .25, .75, .25, .75)),
    'cylinder_caps': lambda: (box(-.5, .5, -.5, .5, -.5, .5), cylinder(.3, 1, res=32)),
    'rotated_boxes': lambda: (box(*UNIT), transformed(box(.4, 1.4, .3, 1.3, .2, 1.2), rotation_matrix(30, (1, 1, 1), (.9, .8, .7))))
}

@pytest.mark.parametrize('s', SCALES)
@pytest.mark.parametrize('name', scale_cases.keys())
def test_scaled(name, s):
    a, b = scale_cases[name]()
    check_all_modes(scaled(a, s), scaled(b, s))

@pytest.mark.parametrize('t', [1e3, 1e4, 1e6])
@pytest.mark.parametrize('name', scale_cases.keys())
def test_translated(name, t):
    a, b = scale_cases[name]()
    check_all_modes(scaled(a, 1, (t, -t/2, t/3)), scaled(b, 1, (t, -t/2, t/3)))
