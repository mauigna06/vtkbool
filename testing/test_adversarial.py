import pytest

from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkCommonMath import vtkMatrix4x4

from test_boolean import (
    UNIT, MODES, OPER_NONE, box, box_volumes, check_all_modes, check_solid,
    check_embedding, get_pd, internal_tol, run_boolean, surface_props, transformed,
    fuzz_case_grid, check_fuzz,
)


@pytest.mark.parametrize('port', [0, 1])
@pytest.mark.parametrize('mode', MODES.values())
def test_removing_matrix(port, mode):
    a = box(*UNIT)
    bounds_b = (.4, 1.4, .3, 1.3, .2, 1.2)
    b = box(*bounds_b)
    transform = vtkTransform()
    transform.Translate(.1, .05, -.05)
    matrices = [None, None]
    matrices[port] = transform.GetMatrix()
    bf, errors = run_boolean(a, b, mode, matrices)
    check_solid(bf, errors)

    bf.SetMatrix(port, None)
    bf.Update()
    name = next(name for name, value in MODES.items() if value == mode)
    check_solid(bf, errors, box_volumes(UNIT, bounds_b)[name])


@pytest.mark.parametrize('port', [0, 1])
def test_replacing_matrix_with_older_matrix(port):
    old = vtkTransform()
    old.Translate(.1, .05, -.05)
    old_matrix = vtkMatrix4x4()
    old_matrix.DeepCopy(old.GetMatrix())
    current = vtkTransform()
    current.Translate(-.1, -.05, .05)
    a, b = box(*UNIT), box(.4, 1.4, .3, 1.3, .2, 1.2)
    matrices = [None, None]
    matrices[port] = current.GetMatrix()
    bf, errors = run_boolean(a, b, MODES['difference'], matrices)
    check_solid(bf, errors)

    bf.SetMatrix(port, old_matrix)
    bf.Update()
    matrices[port] = old_matrix
    fresh, fresh_errors = run_boolean(a, b, MODES['difference'], matrices)
    expected = check_solid(fresh, fresh_errors)
    check_solid(bf, errors, expected)


@pytest.mark.parametrize('scale', [1e-6, 1e-3, 1, 1e3, 1e6])
@pytest.mark.parametrize('translation', [0, 1e6])
def test_matrix_normalization(scale, translation):
    transform = vtkTransform()
    transform.PostMultiply()
    transform.Scale(scale, scale, scale)
    transform.Translate(translation, -translation, translation)
    a, b = box(*UNIT), box(.4, 1.4, .3, 1.3, .2, 1.2)
    expected = {name: vol*scale**3 for name, vol in box_volumes(
        UNIT, (.4, 1.4, .3, 1.3, .2, 1.2)).items()}
    check_all_modes(a, b, expected, matrices=[transform.GetMatrix()]*2, none=False,
                    rel_tol=1e-3 if translation and scale == 1e-6 else 1e-6)


@pytest.mark.parametrize('mode', [OPER_NONE, *MODES.values()])
def test_matrix_scale_rerun(mode):
    a, b = box(*UNIT), box(.4, 1.4, .3, 1.3, .2, 1.2)
    transform = vtkTransform()
    matrix = vtkMatrix4x4()
    bf, errors = run_boolean(a, b, mode, [matrix, matrix])
    assert not errors

    for scale in [1e-6, 1e6, 1]:
        transform.Identity()
        transform.Scale(scale, scale, scale)
        matrix.DeepCopy(transform.GetMatrix())
        bf.Update()
        assert not errors
        if mode == OPER_NONE:
            for port in [0, 1]:
                vol, bad = surface_props(bf.GetOutput(port))
                assert not bad
                assert vol == pytest.approx(scale**3, rel=1e-6)
            pd_a = get_pd(transformed(a, matrix))
            pd_b = get_pd(transformed(b, matrix))
            check_embedding(bf, internal_tol(pd_a, pd_b))
        else:
            name = next(name for name, value in MODES.items() if value == mode)
            expected = box_volumes(UNIT, (.4, 1.4, .3, 1.3, .2, 1.2))[name]*scale**3
            check_solid(bf, errors, expected, 1e-6*scale**3)


@pytest.mark.parametrize('seed', [518, 719, 1285])
def test_grid_star_contacts(seed):
    a, b, desc = fuzz_case_grid(seed)
    check_fuzz(get_pd(a), get_pd(b), desc, seed)


@pytest.mark.parametrize('ports', [(0,), (1,), (0, 1)])
@pytest.mark.parametrize('mode', [OPER_NONE, *MODES.values()])
def test_reflected_matrix(ports, mode):
    a, b = get_pd(box(*UNIT)), get_pd(box(.4, 1.4, .3, 1.3, .2, 1.2))
    original = [[pd.GetPoint(i) for i in range(pd.GetNumberOfPoints())] for pd in [a, b]]
    transform = vtkTransform()
    transform.PostMultiply()
    transform.Scale(-1, 1, 1)
    transform.Translate(1, 0, 0)
    matrices = [transform.GetMatrix() if i in ports else None for i in range(2)]
    bf, errors = run_boolean(a, b, mode, matrices)
    assert not errors
    if mode == OPER_NONE:
        for port in [0, 1]:
            vol, bad = surface_props(bf.GetOutput(port))
            assert not bad
            assert vol == pytest.approx(1)
    else:
        bounds_b = (-.4, .6, .3, 1.3, .2, 1.2) if 1 in ports else (.4, 1.4, .3, 1.3, .2, 1.2)
        name = next(name for name, value in MODES.items() if value == mode)
        check_solid(bf, errors, box_volumes(UNIT, bounds_b)[name])
    for pd, points in zip([a, b], original):
        assert [pd.GetPoint(i) for i in range(pd.GetNumberOfPoints())] == points
