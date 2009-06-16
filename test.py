import nose.tools as nt
import numpy as np
import Ska.Numpy

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
    
