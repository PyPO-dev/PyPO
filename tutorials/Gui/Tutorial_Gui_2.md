# PyPO GUI Tutorial: more advanced optical systems and ray-tracing options.

In the previous tutorial we built a simple optical system consisting of a paraboloid. We then performed a ray-trace from the initial frame of rays to the paraboloid surface, and from the paraboloid we found the focus of the frame by calling the s.findRTfocus method. We did all of this using the tubular ray-trace frame input.

In this tutorial, we will introduce Gaussian ray-trace frames. We will then create an optical setup that is slightly more advanced, where we try to generate a collimated beam from a Gaussian ray-trace beam.   

## Setup
As a setup we will define a paraboloid, an ellipsoid and two planes with the following parameters


|||
|-----------|-----------|
|<img  width="400px" src="README_Resources/ResourcesT2/t2ellip.png" alt="" />|<img  src="README_Resources/ResourcesT2/t2para.png" alt="" width="400px" />|
|<img  width="400px" src="README_Resources/ResourcesT2/t2Pl.png" alt="" />|<img  width="400px" src="README_Resources/ResourcesT2/t2Plout.png" alt="" />|

Then we do the following transformations:

|||
|-----------|-----------|
|<img  width="400px" src="README_Resources/ResourcesT2/pltRot.png" alt="" />|<img  width="400px" src="README_Resources/ResourcesT2/ploutRot.png" alt="" />|
|<img  width="400px" src="README_Resources/ResourcesT2/ploutTrans.png" alt="" />||

Note that we do the rotation on do the rotation on *plane_out* before the translation, as this does not result in the same system as the other way around

This is how our system will look like.

<img src="README_Resources/ResourcesT2/system.png" alt="" width="400px"/>

We will save this system as we will use it in the next tutorial as well. We can do so from the *systems menu > save system*. A form will show up allowing us to give a name to the saved file.

<img src="README_Resources/ResourcesT2/saveSys.png" alt="" width="400px"/>

## Initializing a gaussian ray trace frame

In the *Ray-Trace* menu select *Make Frame* > *Gaussian* and we fill in the following parameters:

<img src="README_Resources/ResourcesT2/initGauss.png" alt="" width="400px"/>

We will translate the frame upwards by 100 units.

<img src="README_Resources/ResourcesT2/gaussTranslation.png" alt="" width="400px"/>

If we now plot this frame in the x,y plane, it will look like this:

<img src="README_Resources/ResourcesT2/GaussPlot.png" alt="" width="400px"/>



In the *Ray-tracer* menu we select *Propagate rays* and we do the following propagations. Note that each propagation generates a new frame. To be able to select this frame as input frame for a next propagation, we have to reopen the form from the menu bar.


|||
|-----------|-----------|
|<img  width="400px" src="README_Resources/ResourcesT2/prop1.png" alt="" />|<img  width="400px" src="README_Resources/ResourcesT2/prop2.png" alt="" />|
|<img  width="400px" src="README_Resources/ResourcesT2/prop3.png" alt="" />|<img  width="400px" src="README_Resources/ResourcesT2/prop4.png" alt="" />| 

Now plotting the rayTrace will show the following plot:

<img  width="600" src="README_Resources/ResourcesT2/rePlot.png" alt="" />


<!-- TODO: focus finder did not generate the same image so I left it out -->


