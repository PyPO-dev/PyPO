# PyPO GUI Tutorial: more advanced optical systems and ray-tracing options.

In the previous tutorial we built a simple optical system consisting of a paraboloid. We then performed a ray-trace from the initial frame of rays to the paraboloid surface, and from the paraboloid we found the focus of the frame by calling the s.findRTfocus method. We did all of this using the tubular ray-trace frame input.

In this tutorial, we will introduce Gaussian ray-trace frames. We will then create an optical setup that is slightly more advanced, where we try to generate a collimated beam from a Gaussian ray-trace beam.   


## Initializing a gaussian ray trace frame

In the *Ray-Trace* menu select *Make Frame* > *Gaussian* and we fill in the following parameters:

<img src="README_Resources/initGauss.png" alt="System with one paraboloid reflector" width="400px"/>

We will translate the frame upwards by 100 units.

<img src="README_Resources/gaussTranslation.png" alt="System with one paraboloid reflector" width="400px"/>

If we now plot this frame in the x,y plane, it will look like this:

<img src="README_Resources/GaussPlot.png" alt="System with one paraboloid reflector" width="400px"/>


Next we will define a paraboloid and ellipsoid and two planes with the following parameters


|||
|-----------|-----------|
|<img  width="400px" src="README_Resources/t2ellip.png" alt="System with one paraboloid reflector" />|<img  src="README_Resources/t2para.png" alt="System with one paraboloid reflector" width="400px" />|
|<img  width="400px" src="README_Resources/t2Pl.png" alt="System with one paraboloid reflector" />|<img  width="400px" src="README_Resources/t2Plout.png" alt="System with one paraboloid reflector" />|

Then we do the following transformations:

|||
|-----------|-----------|
|<img  width="400px" src="README_Resources/pltRot.png" alt="System with one paraboloid reflector" />|<img  width="400px" src="README_Resources/ploutTrans.png" alt="System with one paraboloid reflector" />|
|<img  width="400px" src="README_Resources/ploutRot.png" alt="System with one paraboloid reflector" />||

In the *Ray-tracer* menu we select *Propagate rays* and we do the following propagations. Note that each propagation generates a new frame. To be able to select this frame as input frame for a next propagation, we have to reopen the form from the menu bar.


|||
|-----------|-----------|
|<img  width="400px" src="README_Resources/10.png" alt="System with one paraboloid reflector" />|<img  width="400px" src="README_Resources/11.png" alt="System with one paraboloid reflector" />|
|<img  width="400px" src="README_Resources/13.png" alt="System with one paraboloid reflector" />|<img  width="400px" src="README_Resources/14.png" alt="System with one paraboloid reflector" />| 

Now plotting the rayTrace will show the following plot:

<img  width="600" src="README_Resources/15.png" alt="System with one paraboloid reflector" />





