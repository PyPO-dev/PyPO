# Refl params template. Import or just copy-paste into your script
Refl = {
        "name"      : "Reflector name (string)",
        "pmode"     : "Direct abc definition, or vertex & foc(ii) (string)",
        "gmode"     : "xy gridding or polar uv gridding (string)",
        "flip"      : "Flip reflector normal surfaces (bool)",
        "coeffs"    : "a, b and c parameters (list of 3 reals)"
        "vertex"    : "Vertex co-ordinate (parabola only, len-3 np array)",
        "focus_1"   : "First focal point co-ordinate (len-3 np array)",
        "focus_2"   : "Second focal point co-ordinate (len-3 np array)",
        "ecc"       : "Eccentricity (hyperbola & ellipse only)",
        "lims_x"    : "Upper and lower x-limit (gmode=xy, list of 2 reals)",
        "lims_y"    : "Upper and lower y-limit (gmode=xy, list of 2 reals)",
        "lims_u"    : "Aperture and vertex radii (gmode=uv, list of 2 reals)",
        "lims_v"    : "Upper and lower rotation angle (gmode=uv, list of 2 reals)",
        "gridsize"  : "Number of cells along x(u) and y(v) axes (list of 2 ints)"
        }

# Manual raytracer
RTDict = {
        "nRays"     : "Number of rays in a ray-trace ring (int)",
        "nRing"     : "Number of concentric ray-trace rings (int)",
        "angx"      : "Opening angle in x-direction, degrees (real)",
        "angy"      : "Opening angle in y-direction, degrees (real)",
        "a"         : "Radius of outer ring along x-axis (real)",
        "b"         : "Radius of outer ring along y-axis (real)",
        "tChief"    : "Tilt w.r.t. z-axis of chief ray (len-3 np array)",
        "oChief"    : "origin of chief ray (len-3 np array)"
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
        "radius"    : 1
        }

PODict = {
        "s_name"    : "",
        "t_name"    : "",
        "s_current" : "JM-current object"
        }
