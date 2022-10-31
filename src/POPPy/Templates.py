# Refl params template. Import or just copy-paste into your script
Refl = {
        "name"      :       "p1",
        "pmode"     :       "manual",
        "pmode"     :       "focus",
        "gmode"     :       "uv",
        "flip"      :       False,
        "coeffs"    :       [1, 1, -1],
        "vertex"    :       np.zeros(3),
        "focus_1"   :       np.array([0,0,3.5e3]),
        "lims_x"    :       [-5000,5000],
        "lims_y"    :       [-5000,5000],
        "lims_u"    :       [200,5e3],
        "lims_v"    :       [0,360],
        "gridsize"  :       [501,501]
        }

# Manual raytracer
Refl = {
        "nRays"     :       "p1",
        "nRing"     :       "manual",
        "angx"      :       "focus",
        "angy"      :       "uv",
        "a"         :       False,
        "b"         :       [1, 1, -1],
        "tChief"    :       np.zeros(3),
        "oChief"    :       np.array([0,0,3.5e3])
        }
