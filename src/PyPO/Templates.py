##
# @file
# Templates for commonly used PyPO dictionaries.
#
# For each key, a short summary of the key is given.

##
# Template for a reflDict, containing reflector parameters. Note that some fields are only relevent when a certain reflectortype, gmode or pmode is chosen.
# This is signified in the key description.
# 
# @ingroup public_api_templates
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
        "orient"    : "Orient long axis along z ('z') or x ('x') axis. Only relevant for ellipsoids with 'pmode' set to 'focus'"
        "lims_x"    : "Upper and lower x-limit (gmode=xy, Numpy array of 2 reals)",
        "lims_y"    : "Upper and lower y-limit (gmode=xy, Numpy array of 2 reals)",
        "lims_u"    : "Aperture and vertex radii (gmode=uv, Numpy array of 2 reals)",
        "lims_v"    : "Upper and lower rotation angle (gmode=uv, Numpy array of 2 reals)",
        "ecc_uv"    : "Eccentricity of uv-generated xy_grid (uv only)",
        "rot_uv"    : "Position angle of uv-generated xy grid (uv only)",
        "gridsize"  : "Number of cells along x(u) and y(v) axes (Numpy array of 2 ints)"
        }

##
# Template for a TubeRTDict, containing parameters for constructing a tubular ray-trace frame.
# 
# Using the tube, one can create any distribution in between a point source and a collimated beam.
# The tube is cosntructed from concentrical elliptical rings, spaced equally apart. If the number of rings is zero, only the chief ray is generated.
# The tube can be given a semi-major and semi-minor axis. These are the used to construct the outer ring in the tube. 
# In addition, opening angles along the semi-major and semi-minor axis can be specified.
# 
# @ingroup public_api_templates
TubeRTDict = {
        "name"      : "Name of tubular ray-trace frame (string)",
        "nRays"     : "Number of rays in a ray-trace ring (int)",
        "nRing"     : "Number of concentric ray-trace rings (int)",
        "angx0"     : "Opening angle in x-direction, degrees (real)",
        "angy0"     : "Opening angle in y-direction, degrees (real)",
        "x0"        : "Radius of outer ring along x-axis in mm (positive real)",
        "y0"        : "Radius of outer ring along y-axis in mm (positive real)"
        }

##
# Template for a GRTDict, containing parameters for constructing a Gaussian ray-trace frame.
# 
# The Gaussian beam is constructed by rejection sampling of a Gaussian position and direction distribution.
# The divergence angles are calculated from n, lam, and x0/y0.
# The focus of the beam is always initialised at z = 0. After generation, the frame can be freely translated and rotated. 
# 
# @ingroup public_api_templates
GRTDict = {
        "name"      : "Name of Gaussian ray-trace frame (string)",
        "lam"       : "Wavelength of Gaussian beam in mm (positive real)",
        "n"         : "Refractive index of medium (positive real)",
        "nRays"     : "Number of rays in Gaussian beam (positive int)",
        "x0"        : "Beamwaist along x-axis in mm (positive real)",
        "y0"        : "Beamwaidt along y-axis in mm (positive real)",
        "seed"      : "Seed for rejection sampling of Gaussian beam (positive int)"
        }

##
# Template for a GPODict, containing parameters for constructing a Gaussian physical optics beam.
# 
# The beam is always initialised along the positive z-axis with the x focus at z = 0.
# Evaluation of the beam, however, can be on an arbitrary oriented/positioned plane along the beam.
# The beam can have elliptical contours, an arbitrary position angle and general astigmatism.
# Note that the x focus is always at z = 0, and the y focus is at -dxyz, with dxyz the astigmatic distance. 
# The template for a scalar beam is similar, except that the polarisation is not needed.
# 
# @ingroup public_api_templates
GPODict = {
        "name"      : "Name of Gaussian beam",
        "lam"       : "Wavelength of Gaussian beam in mm (positive real)",
        "w0x"       : "Focal beamwaist of beam along x-axis in mm (positive real)",
        "w0y"       : "Focal beamwaist of beam along y-axis in mm (positive real)",
        "n"         : "Refractive index of medium",
        "E0"        : "Peak amplitude (real)",
        "dxyz"      : "Astigmatic distance between x and y focus in mm (real)",
        "pol"       : "Polarisation of beam (Numpy array of length 3)"
        }

##
# Template for a point source dictionary. The point source is generated on the accompanying source surface.
# The template for a scalar beam is similar, except that the polarisation is not needed.
# The uniform source is generated using the same dictionary as the point source.
#
# @ingroup public_api_templates
PSDict = {
        "name"      : "Name of point source (string)",
        "lam"       : "Wavelength of pint source in mm (positive real)",
        "E0"        : "Peak amplitude (real)",
        "phase"     : "Phase of point source",
        "pol"       : "Polarisation of point source (Numpy array of length 3)"
        }

##
# Template for an aperture dictionary. The aperture dictionary is used for efficiency calculations and plotting purposes.
# Because efficiencies are calculated in the restframe of a surface, the center and radius parameters should be interpreted as
# laying in the xy plane.
# 
# @ingroup public_api_templates
aperDict = {
        "plot"      : "Whether to include the aperDict in a plot or not (boolean)",
        "center"    : "Center of aperture, with respect to the origin of the xy plane, in mm (Numpy array of length 2)",
        "outer"     : "Outer semi-major (x) and minor (y) axes of aperture, in mm (Numpy array of length 2)",
        "inner"     : "Inner semi-major (x) and minor (y) axes of aperture, in mm (Numpy array of length 2)",
        }

##
# Template for a physical optics propagation dictionary. The dictionary specifies several important parameters, such as the source currents, target surface and others.
# 
# @ingroup public_api_templates
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

##
# Template for a ray-trace propagation dictionary. The dictionary specifies several important parameters, such as the input frame, target surface and others.
# 
# @ingroup public_api_templates
runRTDict = {
        "fr_in"     : "Name of input frame",
        "fr_out"    : "Name of output frame",
        "t_name"    : "Name of target surface",
        "tol"       : "Tolerance of ray-tracer",
        "nThreads"  : "Number of CPU/GPU threads (int)",
        "t0"        : "Initial guess for propagation",
        "device"    : "Device to use for calculation"
        }

##
# Template for a hybrid RT/PO propagation.
# 
# @ingroup public_api_templates
hybridDict = {
        "fr_in"     : "Name of input frame",
        "field_in"  : "Source reflected fields",
        "t_name"    : "Name of target surface",
        "fr_out"    : "Name of output frame",
        "field_out" : "Name of output field",
        "start"     : "Starting point of rays, for spherical attenuation",
        "interp"    : "Whether to interpolate resulting field on surface",
        "comp"      : "If interp is True, which component to interpolate"
        }
