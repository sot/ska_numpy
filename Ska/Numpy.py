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

