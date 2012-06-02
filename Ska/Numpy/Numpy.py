"""Provide useful utilities for numpy."""

import numpy as np
import re
import operator
import sys

__docformat__ = "restructuredtext en"

def add_column(recarray, name, val, index=None):
    """
    Add a column ``name`` with value ``val`` to ``recarray`` and return a new
    record array.

    :param recarray: Input record array
    :param name: Name of the new column
    :param val: Value of the new column (np.array or list)
    :param index: Add column before index (default: append at end)

    :rtype: New record array with column appended
    """
    if len(val) != len(recarray):
        raise ValueError('Length mismatch: recarray, val = (%d, %d)' % (len(recarray), len(val)))

    arrays = [recarray[x] for x in recarray.dtype.names]
    dtypes = recarray.dtype.descr
    
    if index is None:
        index = len(recarray.dtype.names)

    if not hasattr(val, 'dtype'):
        val = np.array(val)
    valtype = val.dtype.str

    arrays.insert(index, val)
    dtypes.insert(index, (name, valtype))

    return np.rec.fromarrays(arrays, dtype=dtypes)

def match(recarray, filters):
    """
    Apply the list of ``filters`` to the numpy record array ``recarray`` and
    return the corresponding boolean mask array.
    
    Each filter is a string with a simple boolean comparison of the form::

      colname op value

    where ``colname`` is a column name in ``recarray``, ``op`` is an operator
    (e.g. == or < or >= etc), and ``value`` is a value.  String values can
    optionally be enclosed in single or double quotes.

    The pseudo-column name '_row_' can be used to filter on the row number.

    :param recarray: Input numpy record array
    :param filters: List of filters or string with one filter
    :rtype: list of strings
    """
    re_filter_expr = re.compile(r'\s* (\w+) \s* ([<>=!]+) \s* (\S.*)', re.VERBOSE)
    
    # Convert filters input to a list if string was supplied
    try:
        filters = [filters + '']
    except TypeError:
        pass

    matches = np.ones(len(recarray), dtype=np.bool)

    for filtr in filters:
        filtr = filtr.strip()           # No leading/trailing whitespace

        if not filtr:
            continue

        # Parse the filter expression
        m = re_filter_expr.match(filtr)
        if m:
            colname, op, val = m.groups()
        else:
            raise ValueError('Filter expression "%s" is not valid.' % filtr)
        
        # Strip off up to one set of matched quotes.
        m = re.match(r'([\'"])(.*)\1$', val)
        if m:
            val = m.group(2)

        # Set column values for comparison and convert string 'val' to correct
        # type for the column.  Pseudo-column #row is the row number.
        if colname in recarray.dtype.names:
            colvals = recarray[colname]
            val = recarray[colname].dtype.type(val)
        elif colname == '_row_':
            colvals = np.arange(len(recarray), dtype=int)
            val = int(val)
        else:
            raise ValueError('Column', colname, 'is not in', recarray.dtype.names)

        # Set up operator to do the comparison specified by the filtr expression
        compare_ops = { '>':  operator.gt,
                        '>=': operator.ge,
                        '!=': operator.ne,
                        '==': operator.eq,
                        '<':  operator.lt,
                        '<=': operator.le }
        try:
            compare = compare_ops[op]
        except KeyError:
            raise ValueError('Comparison operator "%s" in filter expression "%s" is not valid.' % (op, filtr))

        # And finally do the matching comparison
        ok = compare(colvals, val)
        matches = matches & ok

    # return rows filtered by matches
    return matches
        
def filter(recarray, filters):
    """
    Apply the list of ``filters`` to the numpy record array ``recarray`` and
    return the filtered recarray.  See L{match} for description of the
    filter syntax.

    :param recarray: Input numpy record array
    :param filters: List of filters

    :rtype: Filtered record array
    """
    if filters:
        return recarray[match(recarray, filters)]
    else:
        return recarray

def interpolate(yin, xin, xout, method='linear'):
    """
    Interpolate the curve defined by (xin, yin) at points xout.  The array
    xin must be monotonically increasing.  The output has the same data type as
    the input yin.

    :param yin: y values of input curve
    :param xin: x values of input curve
    :param xout: x values of output interpolated curve
    :param method: interpolation method ('linear' | 'nearest')

    @:rtype: numpy array with interpolated curve
    """
    yout = np.empty(len(xout), dtype=yin.dtype)
    lenxin = len(xin)

    i1 = np.searchsorted(xin, xout)
    i1[ i1==0 ] = 1
    i1[ i1==lenxin ] = lenxin-1

    x0 = xin[i1-1]
    x1 = xin[i1]
    y0 = yin[i1-1]
    y1 = yin[i1]

    if method == 'linear':
        return (xout - x0) / (x1 - x0) * (y1 - y0) + y0
    elif method == 'nearest':
        return np.where(np.abs(xout - x0) < np.abs(xout - x1), y0, y1)
    else:
        raise ValueError('Invalid interpolation method: %s' % method)

def smooth(x, window_len=10, window='hanning'):
    """
    Smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    Example::

      t = linspace(-2, 2, 50)
      y = sin(t) + randn(len(t)) * 0.1
      ys = Ska.Numpy.smooth(y)
      plot(t, y, t, ys)
    
    See also::

      numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
      scipy.signal.lfilter

    :param x: input signal 
    :param window_len: dimension of the smoothing window
    :param window: type of window ('flat', 'hanning', 'hamming', 'bartlett', 'blackman')

    :rtype: smoothed signal
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[2*x[0]-x[window_len:1:-1],x,2*x[-1]-x[-1:-window_len:-1]]

    if window == 'flat': #moving average
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)

    y=np.convolve(w/w.sum(),s,mode='same')
    return y[window_len-1:-window_len+1]

def compress(recarray, delta=None, indexcol=None, diff=None, avg=None, colnames=None):
    """
    Compress ``recarray`` rows into intervals where adjacent rows are similar.

    In addition to the original column names, the output recarray will have
    these columns:

      ``<indexcol>_start``
         start value of the ``indexcol`` column.  
      ``<indexcol>_stop``
         stop value of the ``indexcol`` column (inclusive up to the next interval).
      ``samples``
         number of samples in interval

    If ``indexcol`` is None (default) then the table row index will be used and
    the output columns will be row_start and row_stop.

    ``delta`` is a dict mapping column names to a delta value defining whether a
    column is sufficiently different to break the interval.  These are used
    when generating the default ``diff`` functions for numerical columns
    (i.e. those for which abs(x) succeeds).

    ``diff`` is a dict mapping column names to functions that take as input two
    values and return a boolean indicating whether the values are sufficiently
    different to break the interval.  Default diff functions will be generated
    if ``diff`` is None or for columns without an entry.

    ``avg`` is a dict mapping column names to functions that calculate the
    average of a numpy array of values for that column.  Default avg functions
    will be generated if ``avg`` is None or for columns without an entry.

    Example::

      a = ((1, 2, 'hello', 2.),
           (1, 4, 'hello', 3.),
           (1, 2, 'hello', 4.),
           (1, 2, 'hi there', 5.),
           (1, 2, 'hello', 6.),
           (3, 2, 'hello', 7.),
           (1, 2, 'hello', 8.),
           (2, 2, 'hello', 9.))
      arec = numpy.rec.fromrecords(a, names=('col1','col2','greet','time'))
      acomp = compress(arec, indexcol='time', delta={'col1':1.5})

    :param delta: dict of delta thresholds defining when to break interval
    :param indexcol: name of column to report start and stop values for interval.
    :param diff: dict of functions defining the diff of 2 vals for that column name.
    :param avg: dict of functions defining the average value for that column name.
    :param colnames: list of column names to include (default = all).

    :rtype: record array of compressed values
    """

    def _numdiff(colname):
        d = delta.get(colname, 0)
        def _diff(x, y):
            return (False if d is None else np.abs(x-y) > d)
        return _diff


    if delta is None: delta = {}
    if diff is None: diff = {}
    if avg is None: avg = {}
    if colnames is None: colnames = recarray.dtype.names

    i0 = 0
    mins = {}
    maxs = {}
    break_interval = False
    end_of_data = False
    intervals = []
    nrec = len(recarray)

    colnames = [x for x in colnames if x != indexcol]

    # Set up the difference functions
    row = recarray[0]
    for colname in set(colnames) - set(diff):
        try:
            diff[colname] = _numdiff(colname)
            diff[colname](row[colname], row[colname])
        except TypeError:
            diff[colname] = lambda x, y: x != y

    # Set up averaging functions
    for colname in set(colnames) - set(avg):
        try:
            avg[colname] = lambda x: x.mean().tolist()
            avg[colname](row[colname])
        except TypeError:
            avg[colname] = lambda x: x[0].tolist()

    for i in range(nrec+1):
        if i < nrec:
            row = recarray[i]
            for colname in colnames:
                # calc running min/max
                val = row[colname]
                if val < mins.setdefault(colname, val):
                    mins[colname] = val
                if val > maxs.setdefault(colname, val):
                    maxs[colname] = val

                if diff[colname](mins[colname], maxs[colname]):
                    break_interval = True
                    break
        else:
            break_interval = True

        if break_interval:
            i1 = i
            vals = tuple(avg[x](recarray[x][i0:i1]) for x in colnames)
            samples = (i1-i0, )
            indexvals = (recarray[i0][indexcol], row[indexcol]) if indexcol else (i0, i1) 
            intervals.append(indexvals + samples + vals)
            i0 = i
            mins = dict((x, row[x]) for x in colnames)
            maxs = dict((x, row[x]) for x in colnames)
            break_interval = False

    if indexcol is None:
        indexcol = 'row'
    names = [indexcol+'_start', indexcol+'_stop', 'samples'] + colnames
    return np.rec.fromrecords(intervals, names=names)

def pprint(recarray, fmt=None, out=sys.stdout):
    """
    Print a nicely-formatted version of ``recarray`` to ``out`` file-like object.
    If ``fmt`` is provided it should be a dict of ``colname:fmt_spec`` pairs where
    ``fmt_spec`` is a format specifier (e.g. '%5.2f').

    :param recarray: input record array
    :param fmt: dict of format specifiers (optional)
    :param out: output file-like object

    :rtype: None
    """

    # Define a dict of pretty-print functions for each column in fmt
    if fmt is None:
        pprint = {}
    else:
        pprint = dict((colname, lambda x: fmt[colname] % x) for colname in fmt)

    colnames = recarray.dtype.names

    # Pretty-print all columns and turn into another recarray made of strings
    str_recarray = []
    for row in recarray:
        str_recarray.append([pprint.get(x, str)(row[x]) for x in colnames])
    str_recarray = np.rec.fromrecords(str_recarray, names=colnames)

    # Parse the descr fields of str_recarray recarray to get field width
    colfmt = {}
    for descr in str_recarray.dtype.descr:
        colname, coldescr = descr
        collen = max(int(re.search(r'\d+', coldescr).group()), len(colname))
        colfmt[colname] = '%-' + str(collen) + 's'
    
    # Finally print everything to out
    print >>out, ' '.join(colfmt[x] % x for x in colnames)
    for row in str_recarray:
        print >>out, ' '.join(colfmt[x] % row[x] for x in colnames)

def pformat(recarray, fmt=None):
    """Light wrapper around Ska.Numpy.pprint to return a string instead of
    printing to a file.

    :param recarray: input record array
    :param fmt: dict of format specifiers (optional)

    :rtype: string
    """
    import StringIO
    out = StringIO.StringIO()
    pprint(recarray, fmt, out)
    return out.getvalue()

def structured_array(vals, colnames=None):
    """Create a numpy structured array (ndarray) given a dict of numpy arrays.
    The arrays can be multidimensional but must all have the same length (same
    size of the first dimension). 

    :param vals: dict of numpy ndarrays
    :param colnames: column names (default=sorted vals keys)
    """
    if colnames is None:
        colnames = sorted(vals.keys())
        
    lens = set(len(vals[x]) for x in colnames)
    if len(lens) != 1:
        raise ValueError('Inconsistent length of input arrays')

    dtypes = [(x, vals[x].dtype, vals[x].shape[1:]) for x in colnames]
    dat = np.ndarray(lens.pop(), dtype=dtypes)
    for colname in colnames:
        dat[colname] = vals[colname]

    return dat
