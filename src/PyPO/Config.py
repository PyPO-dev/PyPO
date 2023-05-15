##
# @file
# Functions to set context for the logging and error checking.

def setContext(_context=None):
    global context
    context = "S" if _context is None else _context

def setOverride(_override=None):
    global override
    override = True if _override is None else _override
