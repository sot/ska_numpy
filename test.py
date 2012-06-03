import nose.tools as nt
import numpy as np
import Ska.Numpy
print Ska.Numpy.__file__

ra = np.rec.fromrecords(((1,   3.4, 's1'),
                         (-1,  4.3, 'hey'),
                         (1,   3.4, 'there'),
                         (100, 3, 'string'),
                         (10,  8.4, 'col sdfas')), names=('icol', 'fcol', 'scol'))

def test_filter():
    assert len(Ska.Numpy.filter(ra, 'icol > 80')) == 1
    b = Ska.Numpy.filter(ra, '_row_ <= 2')
    assert len(b) == 3
    assert b[-1]['scol'] == 'there'

    b = Ska.Numpy.filter(ra, ['scol < m', 'fcol < 5'])
    assert len(b) == 1
    assert b[0]['icol'] == -1
    
@nt.raises(ValueError)
def test_filter_bad_colname():
    a = Ska.Numpy.filter(ra, 'badcol == 10')
    
@nt.raises(ValueError)
def test_filter_bad_syntax():
    a = Ska.Numpy.filter(ra, 'icol = 10')
    
def test_structured_array():
    vals = {'icol': ra['icol'].copy(),
            'fcol': ra['fcol'].copy(),
            'scol': ra['scol'].copy()}
    names = ra.dtype.names
    dat = Ska.Numpy.structured_array(vals)
    assert dat.dtype.names == ('fcol', 'icol', 'scol')
    assert np.all(dat['icol'] == ra['icol'])

def test_search_both_sorted():
    a = np.linspace(1, 10, 1e6)
    v = np.linspace(0, 11, 1e6)
    i_np = np.searchsorted(a, v)
    i_sbs = Ska.Numpy.search_both_sorted(a, v)
    assert np.all(i_np == i_sbs)

    a = np.ones(100)
    v = np.ones(100)
    i_np = np.searchsorted(a, v)
    i_sbs = Ska.Numpy.search_both_sorted(a, v)
    assert np.all(i_np == i_sbs)

    a = np.linspace(1, 10, 1e6)
    v = np.linspace(0, 11, 1e2)
    i_np = np.searchsorted(a, v)
    i_sbs = Ska.Numpy.search_both_sorted(a, v)
    assert np.all(i_np == i_sbs)

    a = np.sort(np.random.random(100))
    v = np.sort(np.random.random(100))
    i_np = np.searchsorted(a, v)
    i_sbs = Ska.Numpy.search_both_sorted(a, v)
    assert np.all(i_np == i_sbs)
