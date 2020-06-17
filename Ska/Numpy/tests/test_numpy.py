# Licensed under a 3-clause BSD style license - see LICENSE.rst
from pathlib import Path
import pytest
import numpy as np
import Ska.Numpy

# Check that this test file is in the same package as the imported Ska.Numpy.
# Due to subtleties with pytest test collection and native namespace pacakges,
# running `pytest Ska/Numpy` in the git repo will end up getting the installed
# Ska.Numpy not the local one.  Use `python setup.py test` instead.
assert Path(__file__).parent.parent == Path(Ska.Numpy.__file__).parent

ra = np.rec.fromrecords(((1,   3.4, 's1'),
                         (-1,  4.3, 'hey'),
                         (1,   3.4, 'there'),
                         (100, 3, 'string'),
                         (10,  8.4, 'col sdfas')),
                        names=('icol', 'fcol', 'scol'))


def test_filter():
    assert len(Ska.Numpy.filter(ra, 'icol > 80')) == 1
    b = Ska.Numpy.filter(ra, '_row_ <= 2')
    assert len(b) == 3
    assert b[-1]['scol'] == 'there'

    b = Ska.Numpy.filter(ra, ['scol < m', 'fcol < 5'])
    assert len(b) == 1
    assert b[0]['icol'] == -1


def test_filter_bad_colname():
    with pytest.raises(ValueError):
        Ska.Numpy.filter(ra, 'badcol == 10')


def test_filter_bad_syntax():
    with pytest.raises(ValueError):
        Ska.Numpy.filter(ra, 'icol = 10')


def test_structured_array():
    vals = {'icol': ra['icol'].copy(),
            'fcol': ra['fcol'].copy(),
            'scol': ra['scol'].copy()}
    ra.dtype.names
    dat = Ska.Numpy.structured_array(vals)
    assert dat.dtype.names == ('fcol', 'icol', 'scol')
    assert np.all(dat['icol'] == ra['icol'])


def test_search_both_sorted():
    a = np.linspace(1, 10, 1_000_000)
    v = np.linspace(0, 11, 1_000_000)
    i_np = np.searchsorted(a, v)
    i_sbs = Ska.Numpy.search_both_sorted(a, v)
    assert np.all(i_np == i_sbs)

    a = np.ones(100)
    v = np.ones(100)
    i_np = np.searchsorted(a, v)
    i_sbs = Ska.Numpy.search_both_sorted(a, v)
    assert np.all(i_np == i_sbs)

    a = np.linspace(1, 10, 1_000_000)
    v = np.linspace(0, 11, 100)
    i_np = np.searchsorted(a, v)
    i_sbs = Ska.Numpy.search_both_sorted(a, v)
    assert np.all(i_np == i_sbs)

    a = np.sort(np.random.random(100))
    v = np.sort(np.random.random(100))
    i_np = np.searchsorted(a, v)
    i_sbs = Ska.Numpy.search_both_sorted(a, v)
    assert np.all(i_np == i_sbs)


def test_interpolate_sorted_cython():
    """Verify that the new cython versions give the same answer as the
    legacy (numpy vectorized) code.  This hits both interpolation
    and implicitly search_sorted.
    """
    n = 1000
    xin = np.sort(np.random.random(n))
    xout = np.linspace(-0.2, 1.2, n)
    for dtype in (np.float32, np.float64, np.int, 'S4'):
        yin = np.asarray(np.random.random(n) + 0.5, dtype=dtype)
        for method in ('nearest', 'linear'):
            if dtype == 'S4' and method == 'linear':
                continue
            ys = []
            for cython in (True, False):
                for sorted_ in (True, False):
                    y = Ska.Numpy.interpolate(yin, xin, xout, method=method,
                                           sorted=sorted_, cython=cython)
                    ys.append(y)

            for yc in ys[1:]:
                if yin.dtype.kind == 'f':
                    assert np.allclose(ys[0], yc)
                else:
                    assert np.all(ys[0] == yc)
