"""!
@file
Functions to set context for the logging and error checking.
"""

def setContext(_context=None):
    """!
    Set the context in which PyPO is run.

    If the context is set to "S", PyPO assumes it is run in scripting mode and use the regular output streams and methods.
    If the context is set to "G", PyPO assumes it is run in GUI mode and will not output to the standard output. 
    Also, PyPO will use special GUI methods.
    """

    global context
    context = "S" if _context is None else _context

def setOverride(_override=None):
    """!
    Set the override parameter.

    If the parameter is set to True, PyPO will overwrite entries with identical names.
    If False, PyPO will append the number of occurences of the name to the name, so that there are no duplicates anymore.
    """

    global override
    override = True if _override is None else _override
