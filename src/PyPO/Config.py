def initPrint(redirect):
    global print
    if redirect != None:
        print = redirect

    else:
        print = print

def setContext(_context):
    global context
    context = "S" if _context is None else _context
