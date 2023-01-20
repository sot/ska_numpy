# Licensed under a 3-clause BSD style license - see LICENSE.rst
from pathlib import Path
import pytest
import numpy as np
import ska_numpy

# Check that this test file is in the same package as the imported ska_numpy.
# Due to subtleties with pytest test collection and native namespace pacakges,
# running `pytest Ska/Numpy` in the git repo will end up getting the installed
# ska_numpy not the local one.  Use `python setup.py test` instead.
assert Path(__file__).parent.parent == Path(ska_numpy.__file__).parent

ra = np.rec.fromrecords(((1,   3.4, 's1'),
                         (-1,  4.3, 'hey'),
                         (1,   3.4, 'there'),
                         (100, 3, 'string'),
                         (10,  8.4, 'col sdfas')),
                        names=('icol', 'fcol', 'scol'))


def test_filter():
    assert len(ska_numpy.filter(ra, 'icol > 80')) == 1
    b = ska_numpy.filter(ra, '_row_ <= 2')
    assert len(b) == 3
    assert b[-1]['scol'] == 'there'

    b = ska_numpy.filter(ra, ['scol < m', 'fcol < 5'])
    assert len(b) == 1
    assert b[0]['icol'] == -1


def test_filter_bad_colname():
    with pytest.raises(ValueError):
        ska_numpy.filter(ra, 'badcol == 10')


def test_filter_bad_syntax():
    with pytest.raises(ValueError):
        ska_numpy.filter(ra, 'icol = 10')


def test_structured_array():
    vals = {'icol': ra['icol'].copy(),
            'fcol': ra['fcol'].copy(),
            'scol': ra['scol'].copy()}
    ra.dtype.names
    dat = ska_numpy.structured_array(vals)
    assert dat.dtype.names == ('fcol', 'icol', 'scol')
    assert np.all(dat['icol'] == ra['icol'])


def test_search_both_sorted():
    a = np.linspace(1, 10, 1_000_000)
    v = np.linspace(0, 11, 1_000_000)
    i_np = np.searchsorted(a, v)
    i_sbs = ska_numpy.search_both_sorted(a, v)
    assert np.all(i_np == i_sbs)

    a = np.ones(100)
    v = np.ones(100)
    i_np = np.searchsorted(a, v)
    i_sbs = ska_numpy.search_both_sorted(a, v)
    assert np.all(i_np == i_sbs)

    a = np.linspace(1, 10, 1_000_000)
    v = np.linspace(0, 11, 100)
    i_np = np.searchsorted(a, v)
    i_sbs = ska_numpy.search_both_sorted(a, v)
    assert np.all(i_np == i_sbs)

    a = np.sort(np.random.random(100))
    v = np.sort(np.random.random(100))
    i_np = np.searchsorted(a, v)
    i_sbs = ska_numpy.search_both_sorted(a, v)
    assert np.all(i_np == i_sbs)


def test_interpolate_sorted_cython():
    """Verify that the new cython versions give the same answer as the
    legacy (numpy vectorized) code.  This hits both interpolation
    and implicitly search_sorted.
    """
    n = 1000
    xin = np.sort(np.random.random(n))
    xout = np.linspace(-0.2, 1.2, n)
    for dtype in (np.float32, np.float64, int, 'S4'):
        yin = np.asarray(np.random.random(n) + 0.5, dtype=dtype)
        for method in ('nearest', 'linear'):
            if dtype == 'S4' and method == 'linear':
                continue
            ys = []
            for cython in (True, False):
                for sorted_ in (True, False):
                    y = ska_numpy.interpolate(yin, xin, xout, method=method,
                                           sorted=sorted_, cython=cython)
                    ys.append(y)

            for yc in ys[1:]:
                if yin.dtype.kind == 'f':
                    assert np.allclose(ys[0], yc)
                else:
                    assert np.all(ys[0] == yc)
