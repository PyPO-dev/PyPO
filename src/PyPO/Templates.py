##
# @file
# Templates for commonly used PyPO dictionaries.
#
# Here, the templates for reflDicts, (G)RTDicts and PODicts are provided.

##
# Template for a reflDict. Note that some fields are only relevent when a certain gmode or 
reflDict = {
        "name"      : "Reflector name (string)",
        "pmode"     : "Direct abc definition, or vertex & foc(ii) (string). Only for quadric surfaces",
        "gmode"     : "xy gridding or polar uv gridding (string). For planar surfaces also includes Azimuth over Elevation (AoE)",
        "flip"      : "Flip reflector normal surfaces (bool)",
        "coeffs"    : "a, b and c parameters (list of 3 reals, pmode='manual' only)",
        "vertex"    : "Vertex co-ordinate (parabola only, len-3 Numpy array, pmode='focus' only)",
        "focus_1"   : "First focal point co-ordinate (len-3 Numpy array, pmode='focus' only)",
        "focus_2"   : "Second focal point co-ordinate (len-3 Numpy array, pmode='focus' only, hyperbola & ellipse only)",
        "ecc"       : "Eccentricity (hyperbola & ellipse only)",
        "lims_x"    : "Upper and lower x-limit (gmode=xy, Numpy array of 2 reals)",
        "lims_y"    : "Upper and lower y-limit (gmode=xy, Numpy array of 2 reals)",
        "lims_u"    : "Aperture and vertex radii (gmode=uv, Numpy array of 2 reals)",
        "lims_v"    : "Upper and lower rotation angle (gmode=uv, Numpy array of 2 reals)",
        "ecc_uv"    : "Eccentricity of uv-generated xy_grid (uv only)",
        "rot_uv"    : "Position angle of uv-generated xy grid (uv only)",
        "gridsize"  : "Number of cells along x(u) and y(v) axes (Numpy array of 2 ints)"
        }

# Manual raytracer
TubeRTDict = {
        "nRays"     : "Number of rays in a ray-trace ring (int)",
        "nRing"     : "Number of concentric ray-trace rings (int)",
        "angx"      : "Opening angle in x-direction, degrees (real)",
        "angy"      : "Opening angle in y-direction, degrees (real)",
        "a"         : "Radius of outer ring along x-axis (real)",
        "b"         : "Radius of outer ring along y-axis (real)",
        "tChief"    : "Tilt w.r.t. positive z-axis of chief ray (len-3 Numpy array)",
        "oChief"    : "origin of chief ray (len-3 Numpy array)"
        }

# Gaussian beam dict, contains definitions of beam
GDict = {
        "lam"       :       "Wavelength of light (in mm, real)",
        "w0"        :       4,
        "n"         :       1,
        "E0"        :       1,
        "z"         :       0,
        "pol"       :       np.array([1, 0, 0])
        }

# Aperture dict, to be passed to plotter.plotBeam2D (or not)
PlotAper = {
        "plot"      : False,
        "center"    : np.zeros(2),
        "r_out"     : 1,
        "r_in"      : 0
        }

runPODict = {
        "t_name"    : "Name of target surface (string)",
        "s_current" : "Name of current object in system.currents (string)",
        "s_field"   : "Scalar complex field for scalar propagation only (Numpy array)",
        "epsilon"   : "Relative permittivity of source surface (real)",
        "exp"       : "Divergent or convergent beam (string)",
        "nThreads"  : "Number of CPU/GPU threads (int)",
        "device"    : "Device to use for calculation",
        "mode"      : "Determine return object (string)"
        }

runRTDict = {
        "fr_in"     : "Name of input frame",
        "fr_out"    : "Name of output frame",
        "t_name"    : "Name of target surface",
        "tol"       : "Tolerance of ray-tracer",
        "nThreads"  : "Number of CPU/GPU threads (int)",
        "t0"        : "Initial guess for propagation",
        "device"    : "Device to use for calculation",
        "nexus"     : "Reflect or transmit the field at surface"
        "epsilon"   : "Relative permittivity of nexus medium",
        }
