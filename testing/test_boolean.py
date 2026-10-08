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
from vtkmodules.vtkFiltersCore import vtkCleanPolyData, vtkTriangleFilter, vtkAppendPolyData
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
    return src.GetOutputDataObject(0)

def connect(alg, src, port=0):
    # ein temporärer vtkTrivialProducer wäre nach GetOutputPort() schon wieder freigegeben
    if isinstance(src, vtkPolyData):
        alg.SetInputData(port, src)
    else:
        alg.SetInputConnection(port, src.GetOutputPort())

def run_boolean(srcA, srcB, mode, matrices=None):
    bf = vtkPolyDataBooleanFilter()
    connect(bf, srcA, 0)
    connect(bf, srcB, 1)
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

    if pd.GetNumberOfStrips() > 0:
        tri = vtkTriangleFilter()
        tri.SetInputData(pd)
        tri.Update()
        pd = tri.GetOutput()

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
    connect(tf, src)
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
    connect(tf, src)
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
    connect(tf, src)
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

# level 6: topologie und komplexität

from vtkmodules.vtkFiltersSources import vtkPolyLineSource
from vtkmodules.vtkFiltersModeling import vtkRotationalExtrusionFilter

def star(n=5, r1=1, r2=.4):
    pts = []
    for i in range(2*n):
        r = r1 if i%2 == 0 else r2
        a = math.pi/2+i*math.pi/n
        pts.append([r*math.cos(a), r*math.sin(a)])
    return pts

def comb(teeth=5):
    # steg [0, 2*teeth-1]x[0, 1], darauf zähne der breite 1 und höhe 3 im abstand 2
    pts = [[0, 0], [2*teeth-1, 0]]

    for i in reversed(range(teeth)):
        pts += [[2*i+1, 4], [2*i, 4]]

        if i > 0:
            pts += [[2*i, 1], [2*i-1, 1]]

    return pts

def torus(R=1, r=.3, n=24, m=12, center=(0, 0, 0)):
    line = vtkPolyLineSource()
    line.SetNumberOfPoints(m+1)

    for i in range(m+1):
        a = 2*math.pi*i/m
        line.SetPoint(i, R+r*math.cos(a), 0, r*math.sin(a))

    extr = vtkRotationalExtrusionFilter()
    extr.SetInputConnection(line.GetOutputPort())
    extr.SetResolution(n)

    return scaled(extr, 1, center)

def appended(*srcs):
    app = vtkAppendPolyData()
    for src in srcs:
        if isinstance(src, vtkPolyData):
            app.AddInputData(src)
        else:
            app.AddInputConnection(src.GetOutputPort())
    return app

def test_comb_polygon():
    # das polygon selbst muss gültig sein
    pd = get_pd(extrude_polygon(comb(), 1))
    assert surface_props(pd)[1] == []
    assert volume(pd) == pytest.approx(9+5*3, rel=1e-9)

def test_star(tmp_path):
    check_all_modes(extrude_polygon(star(), 1), box(-.6, .6, -.6, .6, .25, .75), d=tmp_path)

def test_star_2(tmp_path):
    check_all_modes(extrude_polygon(star(), 1), box(-.2, .2, -2, 2, -1, 2), d=tmp_path)

@pytest.mark.parametrize('z', [(-1, 2), (.25, .75), (0, 1), (-1, .5)])
def test_comb(z, tmp_path):
    # ein loop je zahn
    check_all_modes(extrude_polygon(comb(), 1), box(-1, 10, 2, 3, *z), d=tmp_path)

def test_comb_2(tmp_path):
    check_all_modes(extrude_polygon(comb(), 1), transformed(extrude_polygon(comb(), 1), rotation_matrix(180, (0, 0, 1), (4.5, 3.5, 0))), d=tmp_path)

def test_torus_box(tmp_path):
    check_all_modes(torus(), box(-2, 2, -.1, .1, -1, 1), d=tmp_path)

def test_torus_cylinder(tmp_path):
    # der zylinder schneidet die innenseite des rings
    cyl = transformed(cylinder(.8, 2, res=32), rotation_matrix(90, (1, 0, 0)))
    check_all_modes(torus(), cyl, d=tmp_path)

def test_tori(tmp_path):
    t2 = transformed(torus(), rotation_matrix(90, (1, 0, 0), (0, 0, 0)))
    check_all_modes(torus(), scaled(t2, 1, (.5, 0, 0)), d=tmp_path)

def test_holes(tmp_path):
    cyls = [ transformed(cylinder(.1, 2, center=(x, 0, z), res=16), rotation_matrix(0, (1, 0, 0))) for x, z in [(-.25, -.25), (.25, -.25), (-.25, .25), (.25, .25), (0, 0)] ]
    check_all_modes(box(-.5, .5, -.5, .5, -.5, .5), appended(*cyls), d=tmp_path)

@pytest.mark.parametrize('res', [64, 128])
def test_high_res(res, tmp_path):
    check_all_modes(sphere(.5, res=res), sphere(.45, (.2, .1, .05), res-3), none=False)

def test_slivers(tmp_path):
    s = vtkSphereSource()
    s.SetThetaResolution(300)
    s.SetPhiResolution(5)
    s.SetOutputPointsPrecision(vtkAlgorithm.DOUBLE_PRECISION)
    check_all_modes(s, box(-.3, .3, -.3, .3, -.2, .7), d=tmp_path)

def test_lat_long_quads(tmp_path):
    s = sphere(.5, res=16)
    s.LatLongTessellationOn()
    check_all_modes(s, box(-.3, .3, -.3, .3, -.2, .7), d=tmp_path)

def twisted_box(angle, h=1):
    # die deckfläche ist verdreht, die seitenflächen sind nicht eben
    a = math.radians(angle)
    bottom = [(-.5, -.5, 0), (.5, -.5, 0), (.5, .5, 0), (-.5, .5, 0)]
    top = [ (x*math.cos(a)-y*math.sin(a), x*math.sin(a)+y*math.cos(a), h) for x, y, _ in bottom ]
    pts = bottom+top
    polys = [[3, 2, 1, 0], [4, 5, 6, 7]]+[ [i, (i+1)%4, (i+1)%4+4, i+4] for i in range(4) ]
    return create_polydata(pts, polys)

@pytest.mark.parametrize('angle', [1, 10, 30])
def test_twisted(angle, tmp_path):
    check_all_modes(twisted_box(angle), box(-.3, .3, -.3, .3, .3, 1.3), d=tmp_path)

@pytest.mark.parametrize('angle', [1, 10, 30])
def test_twisted_2(angle, tmp_path):
    check_all_modes(twisted_box(angle), box(0, 1, 0, 1, .3, .7), d=tmp_path)

def octagon_box():
    # quader, dessen seitenflächen die kantenmitten enthalten (kollineare punkte)
    pts = []
    ids = {}
    def pt(p):
        if p not in ids:
            ids[p] = len(pts)
            pts.append(p)
        return ids[p]
    faces = []
    for axis in range(3):
        for side in (0, 1):
            u, v = [ i for i in range(3) if i != axis ]
            corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
            if side == 0:
                corners = corners[::-1]
            if axis == 1:
                corners = corners[::-1]
            ring = []
            for (a, b), (c, d) in zip(corners, corners[1:]+corners[:1]):
                for p in [(a, b), ((a+c)/2, (b+d)/2)]:
                    q = [0, 0, 0]
                    q[axis] = side
                    q[u], q[v] = p
                    ring.append(pt(tuple(q)))
            faces.append(ring)
    return create_polydata(pts, faces)

def test_octagon_box_valid():
    pd = get_pd(octagon_box())
    assert surface_props(pd)[1] == []
    assert volume(pd) == pytest.approx(1)

@pytest.mark.parametrize('b', [(.5, 1.5, .5, 1.5, .5, 1.5), (.5, 1.5, .25, .75, .25, .75), (.25, .75, .5, 1.5, -1, 2), (.5, 1.5, 0, 1, 0, 1)])
def test_octagon_box(b, tmp_path):
    check_all_modes(octagon_box(), box(*b), box_volumes(UNIT, b), d=tmp_path)

# level 7: verkettung und api

def boolean_output(srcA, srcB, mode, matrices=None):
    bf, errors = run_boolean(srcA, srcB, mode, matrices)
    assert not errors, errors

    pd = vtkPolyData()
    pd.DeepCopy(bf.GetOutput(0))

    return pd

def test_drill_sequence(tmp_path):
    # drei sich kreuzende bohrungen, jede durchquert die vorherigen
    vol = volume(box(-.5, .5, -.5, .5, -.5, .5))
    current = get_pd(box(-.5, .5, -.5, .5, -.5, .5))

    cyls = [
        cylinder(.2, 2, res=24),
        transformed(cylinder(.15, 2, res=20), rotation_matrix(90, (1, 0, 0))),
        transformed(cylinder(.25, 2, res=28), rotation_matrix(90, (0, 0, 1))),
        transformed(cylinder(.1, 2, center=(.3, 0, .3), res=12), rotation_matrix(45, (0, 0, 1)))
    ]

    for i, cyl in enumerate(cyls):
        vols = check_all_modes(current, cyl, none=False)
        assert vols['difference'] < vol
        vol = vols['difference']
        current = boolean_output(current, cyl, OPER_DIFFERENCE)

def test_union_chain(tmp_path):
    current = get_pd(box(0, 1, 0, 1, 0, 1))

    for i in range(1, 6):
        b = transformed(box(0, 1, 0, 1, 0, 1), rotation_matrix(10*i, (0, 0, 1), (.5, .5, .5)))
        b = scaled(b, 1, (.4*i, .1*i, .05*i))
        check_all_modes(current, b, none=False)
        current = boolean_output(current, b, OPER_UNION)

def test_union_chain_coplanar(tmp_path):
    # alle quader liegen auf der gleichen ebene und überlappen sich
    rects = [(0, 1, 0, 1)]

    current = get_pd(box(0, 1, 0, 1, 0, 1))

    for i in range(1, 6):
        r = (.5*i, .5*i+1, .25*(i%2), .25*(i%2)+1)
        rects.append(r)

        b = box(*r, 0, 1)
        check_all_modes(current, b, none=False)
        current = boolean_output(current, b, OPER_UNION)

    # exakte fläche der vereinigung auf einem raster der weite .25
    area = sum(.25**2 for i in range(16) for j in range(8)
        if any(r[0] < (i+.5)*.25 < r[1] and r[2] < (j+.5)*.25 < r[3] for r in rects))

    assert volume(current) == pytest.approx(area, rel=1e-9)

def test_rerun_oper_mode():
    a, b = box(*UNIT), sphere(.6, (.8, .7, .6))

    bf, errors = run_boolean(a, b, OPER_UNION)

    for name, mode in MODES.items():
        bf.SetOperMode(mode)
        bf.Update()

        fresh = boolean_output(a, b, mode)

        assert surface_props(bf.GetOutput(0))[0] == pytest.approx(surface_props(fresh)[0], abs=1e-9), name
        assert bf.GetOutput(0).GetNumberOfCells() == fresh.GetNumberOfCells(), name

@pytest.mark.parametrize('i', [0, 1])
def test_rerun_matrix(i):
    a, b = box(*UNIT), box(.4, 1.4, .3, 1.3, .2, 1.2)

    m = rotation_matrix(0, (1, 0, 0))

    bf = vtkPolyDataBooleanFilter()
    bf.SetInputConnection(0, a.GetOutputPort())
    bf.SetInputConnection(1, b.GetOutputPort())
    bf.SetOperModeToDifference()
    bf.SetMatrix(i, m)

    for angle in [0, 20, 45, 70]:
        tra = vtkTransform()
        tra.PostMultiply()
        tra.Translate(-.9, -.8, -.7)
        tra.RotateWXYZ(angle, 1, 2, 3)
        tra.Translate(.9, .8, .7)

        # die matrix wird an ort und stelle geändert
        m.DeepCopy(tra.GetMatrix())

        bf.Update()

        mats = [None, None]
        mats[i] = tra.GetMatrix()

        srcs = [a, b]
        srcs[i] = transformed(srcs[i], tra.GetMatrix())

        fresh = boolean_output(*srcs, OPER_DIFFERENCE)

        vol, bad = surface_props(bf.GetOutput(0))

        assert not bad
        assert vol == pytest.approx(surface_props(fresh)[0], abs=1e-9), angle

def test_rerun_input():
    a = box(*UNIT)
    b = sphere(.5, (1, 1, 1))

    bf, errors = run_boolean(a, b, OPER_DIFFERENCE)

    for c in [(.9, .8, .7), (.2, .3, .4), (1.2, 1.1, 1)]:
        b.SetCenter(*c)
        bf.Update()

        fresh = boolean_output(a, sphere(.5, c), OPER_DIFFERENCE)

        vol, bad = surface_props(bf.GetOutput(0))

        assert not bad
        assert vol == pytest.approx(surface_props(fresh)[0], abs=1e-9), c

def test_orig_cell_ids():
    from vtkmodules.vtkCommonCore import vtkIntArray

    a, b = get_pd(box(*UNIT)), get_pd(sphere(.6, (.8, .7, .6)))

    # gleicher name, da vtkAppendPolyData nur gemeinsame arrays übernimmt
    for pd, offset in [(a, 1000), (b, 2000)]:
        arr = vtkIntArray()
        arr.SetName('label')
        for i in range(pd.GetNumberOfCells()):
            arr.InsertNextValue(offset+i)
        pd.GetCellData().AddArray(arr)

    for name, mode in MODES.items():
        out = boolean_output(a, b, mode)

        idsA = out.GetCellData().GetArray('OrigCellIdsA')
        idsB = out.GetCellData().GetArray('OrigCellIdsB')

        assert idsA is not None and idsB is not None

        label = out.GetCellData().GetArray('label')

        for i in range(out.GetNumberOfCells()):
            ia = idsA.GetValue(i)
            ib = idsB.GetValue(i)

            assert (ia < 0) != (ib < 0)

            src, j = (a, ia) if ia >= 0 else (b, ib)

            # die zelle muss in der ebene der ursprünglichen zelle liegen
            cell = src.GetCell(j)
            p0 = src.GetPoint(cell.GetPointId(0))
            n = [0, 0, 0]
            from vtkmodules.vtkCommonDataModel import vtkPolygon
            vtkPolygon.ComputeNormal(cell.GetPoints(), n)

            c = out.GetCell(i)
            for k in range(c.GetNumberOfPoints()):
                p = out.GetPoint(c.GetPointId(k))
                assert abs(sum((p[m]-p0[m])*n[m] for m in range(3))) < 1e-6, (name, i)

            if ia >= 0:
                assert label.GetValue(i) == 1000+ia
            else:
                assert label.GetValue(i) == 2000+ib

@pytest.mark.parametrize('which', ['A', 'B', 'both'])
def test_empty_input(which):
    empty = vtkPolyData()
    a = empty if which in ('A', 'both') else get_pd(box(*UNIT))
    b = empty if which in ('B', 'both') else get_pd(box(*UNIT))

    for mode in range(5):
        bf, errors = run_boolean(a, b, mode)

        # eine fehlermeldung ist in ordnung, ein absturz nicht
        assert bf.GetOutput(0).GetNumberOfCells() == 0 or not errors

def test_lines_only():
    line = vtkPolyLineSource()
    line.SetNumberOfPoints(2)
    line.SetPoint(0, -1, .5, .5)
    line.SetPoint(1, 2, .5, .5)

    for mode in range(5):
        bf, errors = run_boolean(box(*UNIT), line, mode)

def test_disjoint():
    # keine berührung: bekannte einschränkung, aber kein absturz
    bf, errors = run_boolean(box(*UNIT), box(2, 3, 2, 3, 2, 3), OPER_UNION)
    assert errors

# level 8: aus zufälligen fällen gewonnene tests

@pytest.mark.parametrize('s', [.75, 1.5, 193])
def test_contact_near_vertex(s):
    # ein schnittpunkt liegt genau auf einer kante und innerhalb der toleranz einer anderen kante der gleichen ecke,
    # er muss der nächsten kante zugeordnet werden (aus test_fuzz[53])

    a = torus(0.9266030039270914, 0.1527190191896381, 22, 15)

    b = scaled(extrude_polygon(star(7, 0.9684906516891164, 0.43823622049112826), 0.8455704932930983), 1, (0, 0, -.5))
    b = scaled(transformed(b, rotation_matrix(45, (1, 0, 0))), 1, (0, -.25, 0))

    check_all_modes(scaled(a, s), scaled(b, s))

@pytest.mark.parametrize('s', [.3, .75, 1, 1.5, 3])
def test_vertex_near_face(s):
    # eine ecke der kugel liegt knapp innerhalb der toleranz unter einer seite des quaders, die schnittlinie schneidet
    # die polygone um diese ecke in stücken, die kürzer als die toleranz sind (aus test_fuzz[32])

    x, y, z = 1.4117970783735823, 0.47357408079321683, 1.1380247613806755

    a = box(-x/2, x/2, -y/2, y/2, -z/2, z/2)

    b = transformed(sphere(0.5977514953508448, res=29), rotation_matrix(90, (0, 1, 0)))
    b = scaled(b, 1, (-.5, .5, .25))

    check_all_modes(scaled(a, s), scaled(b, s))

def test_collapsing_capture_points():
    # zwei punkte der schnittlinie liegen knapp mehr als die toleranz auseinander, rasten aber an der gleichen stelle
    # einer kante ein (aus test_fuzz[503])

    tra = vtkTransform()
    tra.PostMultiply()
    tra.RotateWXYZ(12.4011390522013, 1.749295983601171, 0.2079123312026314, -0.07539880503236884)
    tra.Translate(0.0573241467934712, 0.39778485494180604, -0.07983220395294721)

    a = sphere(0.39105171272436057, res=29)
    b = transformed(sphere(0.5264984889817228, res=10), tra.GetMatrix())

    s = 23.33704284428023

    check_all_modes(scaled(a, s), scaled(b, s))

# zufällige fälle (anzahl über VTKBOOL_FUZZ_CASES)

import os
import random

from vtkmodules.vtkFiltersModeling import vtkSelectEnclosedPoints

def random_primitive(rng):
    kind = rng.choice(['box', 'sphere', 'cylinder', 'star', 'torus'])

    if kind == 'box':
        x, y, z = [ rng.uniform(.3, 1.5) for _ in range(3) ]
        return kind, box(-x/2, x/2, -y/2, y/2, -z/2, z/2)

    if kind == 'sphere':
        return kind, sphere(rng.uniform(.3, .9), res=rng.randint(6, 30))

    if kind == 'cylinder':
        return kind, cylinder(rng.uniform(.2, .7), rng.uniform(.5, 2), res=rng.randint(5, 40))

    if kind == 'star':
        r1 = rng.uniform(.5, 1)
        return kind, scaled(extrude_polygon(star(rng.randint(3, 8), r1, rng.uniform(.2, .9)*r1), rng.uniform(.3, 1.5)), 1, (0, 0, -.5))

    R = rng.uniform(.5, 1)
    return kind, torus(R, rng.uniform(.1, .4)*R, rng.randint(6, 30), rng.randint(4, 16))

def random_pose(rng):
    if rng.random() < .4:
        # eingerastete lagen erzeugen koplanare und kollineare situationen
        axis = rng.choice([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
        angle = rng.choice([0, 45, 90, 180])
        t = [ rng.randint(-2, 2)*.25 for _ in range(3) ]
    else:
        axis = [ rng.gauss(0, 1) for _ in range(3) ]
        angle = rng.uniform(0, 360)
        t = [ rng.uniform(-.5, .5) for _ in range(3) ]

    tra = vtkTransform()
    tra.PostMultiply()
    tra.RotateWXYZ(angle, *axis)
    tra.Translate(*t)

    return tra.GetMatrix(), (axis, angle, t)

def enclosed(pdA, pdB):
    # anteil der punkte von pdA in pdB
    sel = vtkSelectEnclosedPoints()
    sel.SetInputData(pdA)
    sel.SetSurfaceData(pdB)
    sel.Update()

    n = pdA.GetNumberOfPoints()

    return sum(sel.IsInside(i) for i in range(n))/n

def fuzz_case(seed, plain=False):
    rng = random.Random(seed)

    kindA, a = random_primitive(rng)
    kindB, b = random_primitive(rng)

    m, pose = random_pose(rng)

    s = 10**rng.uniform(-3, 3)
    t = [ s*10**rng.uniform(0, 4)*rng.choice([-1, 1]) if rng.random() < .3 else 0 for _ in range(3) ]

    if plain:
        # ohne skalierung und verschiebung
        s, t = 1, [0, 0, 0]

    a = scaled(a, s, t)
    b = scaled(transformed(b, m), s, t)

    return a, b, f'{kindA} {kindB} pose={pose} scale={s:.3g} t={t}'

FUZZ_CASES = int(os.environ.get('VTKBOOL_FUZZ_CASES', 50))

@pytest.mark.parametrize('seed', range(FUZZ_CASES))
def test_fuzz(seed):
    a, b, desc = fuzz_case(seed)

    pdA, pdB = get_pd(a), get_pd(b)

    bf, errors = run_boolean(pdA, pdB, OPER_UNION)

    if any('There is no contact.' in e for e in errors):
        # nur erlaubt, wenn sich die oberflächen tatsächlich nicht schneiden
        inA = enclosed(pdB, pdA)
        inB = enclosed(pdA, pdB)

        if inA in (0, 1) and inB in (0, 1):
            pytest.skip(f'no contact: {desc}')

    try:
        check_all_modes(pdA, pdB)
    except AssertionError as e:
        raise AssertionError(f'seed {seed}: {desc}: {e}') from e
