"""Provide useful utilities for numpy."""

import numpy 
import re
import operator

def add_column(recarray, name, val, index=None):
    """Add a column C{name} with value C{val} to C{recarray} and return a new
    record array.

    @param recarray: Input record array
    @param name: Name of the new column
    @param val: Value of the new column (numpy.array or list)
    @param index: Add column before index (default: append at end)

    @return: New record array with column appended
    """
    if len(val) != len(recarray):
        raise ValueError('Length mismatch: recarray, val = (%d, %d)' % (len(recarray), len(val)))

    arrays = [recarray[x] for x in recarray.dtype.names]
    dtypes = recarray.dtype.descr
    
    if index is None:
        index = len(recarray.dtype.names)

    if not hasattr(val, 'dtype'):
        val = numpy.array(val)
    valtype = val.dtype.str

    arrays.insert(index, val)
    dtypes.insert(index, (name, valtype))

    return numpy.rec.fromarrays(arrays, dtype=dtypes)

def match(recarray, filters):
    """Apply the list of C{filters} to the numpy record array C{recarray} and
    return the corresponding boolean mask array.
    
    Each filter is a string with a simple boolean comparison of the form::

      colname op value

    where C{colname} is a column name in C{recarray}, C{op} is an operator
    (e.g. == or < or >= etc), and C{value} is a value.  String values can
    optionally be enclosed in single or double quotes.  

    @param recarray: Input numpy record array
    @param filters: List of filters or string with one filter

    @return: numpy boolean mask array of rows that match the filters.
    """
    re_filter_expr = re.compile(r'\s* (\w+) \s* ([<>=!]+) \s* (\S.*)', re.VERBOSE)
    
    # Convert filters input to a list if string was supplied
    try:
        filters = [filters + '']
    except TypeError:
        pass

    matches = numpy.ones(len(recarray), dtype=numpy.bool)

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

        if colname not in recarray.dtype.names:
            raise ValueError('Column', colname, 'is not in', recarray.dtype.names)

        # Convert string 'val' to correct numpy type for the column
        val = recarray[colname].dtype.type(val)

        # Set up operator to do the comparison specified by the filtr expression
        compare = { '>':  operator.gt,
                    '>=': operator.ge,
                    '!=': operator.ne,
                    '==': operator.eq,
                    '<':  operator.lt,
                    '<=': operator.le }[op]

        # And finally do the matching comparison
        print colname, type(recarray[colname]), recarray[colname].dtype, val, type(val)
        ok = compare(recarray[colname], val)
        matches = matches & ok

    # return rows filtered by matches
    return matches
        
def filter(recarray, filters):
    """Apply the list of C{filters} to the numpy record array C{recarray} and
    return the filtered recarray.  See L{match} for description of the
    filter syntax.

    @param recarray: Input numpy record array
    @param filters: List of filters

    @return: Filtered record array
    """
    if filters:
        return recarray[match(recarray, filters)]
    else:
        return recarray

def interpolate(yin, xin, xout, method='linear'):
    """Interpolate the curve defined by (xin, yin) at points xout.  The array
    xin must be monotonically increasing.  The output has the same data type as
    the input yin.

    @param yin: y values of input curve
    @param xin: x values of input curve
    @param xout: x values of output interpolated curve
    @param method: interpolation method ('linear' | 'nearest')

    @return: numpy array with interpolated curve
    """
    yout = numpy.empty(len(xout), dtype=yin.dtype)
    lenxin = len(xin)

    i1 = numpy.searchsorted(xin, xout)
    i1[ i1==0 ] = 1
    i1[ i1==lenxin ] = lenxin-1

    x0 = xin[i1-1]
    x1 = xin[i1]
    y0 = yin[i1-1]
    y1 = yin[i1]

    if method == 'linear':
        return (xout - x0) / (x1 - x0) * (y1 - y0) + y0
    elif method == 'nearest':
        return numpy.where(numpy.abs(xout - x0) < numpy.abs(xout - x1), y0, y1)
    else:
        raise ValueError('Invalid interpolation method: %s' % method)

def smooth(x,window_len=10,window='hanning'):
    """Smooth the data using a window with requested size.
    
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

    @param x: input signal 
    @param window_len: dimension of the smoothing window
    @param window: type of window ('flat', 'hanning', 'hamming', 'bartlett', 'blackman')

    @return: smoothed signal
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=numpy.r_[2*x[0]-x[window_len:1:-1],x,2*x[-1]-x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='same')
    return y[window_len-1:-window_len+1]

