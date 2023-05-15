import matplotlib.pyplot as pt
import matplotlib
from distutils.spawn import find_executable

##
# @file
# File containing the definitions for plotting style.
# Also enables LaTeX backend if present.

if find_executable('latex'):

    matplotlib.rcParams.update({
        'font.size': 15,
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsmath}',
        'text.latex.preamble': r'\usepackage{amssymb}',
        'text.latex.preamble': r'\usepackage[utf8]{inputenc}',
        'text.latex.preamble': r'\usepackage{siunitx}'
    })

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

pt.rcParams['xtick.top'] = True
pt.rcParams['ytick.right'] = True

pt.rcParams['xtick.direction'] = "in"
pt.rcParams['ytick.direction'] = "in"

pt.rcParams['xtick.minor.visible'] = True
pt.rcParams['ytick.minor.visible'] = True
