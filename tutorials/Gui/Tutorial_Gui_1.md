# PyPO GUI Manual

In this tutorial we wil, step by step, build a simple reflector system and propagate a ray trace trough it.
<!--![Blank app](README_Resources/blankGui.png)-->

The gui consists of 3 widgets:

* Workspace (left)
* Plot screen
* Console

For most interactions a form will show up next to the workspace.


<figure><img src="README_Resources/blankGui.png" alt="A blank GUI window at startup" width="800px"/><figcaption>A blank GUI window at startup</figcaption></figure>



## Creating reflectors
From the elements menu select *Add Reflector* > *Quadric surface*. 

A form will show up with one dropdown selector. By selecting options, more options will show up. By filling in the parameters and clicking *add*, a reflector has been added to the system. If successful, a widget will show up in the workspace as such:

<!--![System with one paraboloid reflector](README_Resources/pri_form.png)-->

<figure><img src="README_Resources/pri_form.png" alt="System with one paraboloid reflector" width="700px"/><figcaption>System with one paraboloid reflector</figcaption></figure>


If adding the reflector did not succeed a message will be logged in the console with information about what went wrong.

<figure><img src="README_Resources/emptyFieldErr.png" alt="text saying: Error - Empty field at Focus xyz" width="600px"/></figure>


## Plotting
Now to see the reflector we just defined we need to plot it. 
By clicking on the three dots on the right side of the element widget the element options menu will open. The first option is plot. 

<figure><img src="README_Resources/options_plot.png" width="300px"/><figcaption>Selecting Plot from the element options menu</figcaption></figure>


This will show the plot on the plot screen.

<figure><img src="README_Resources/pri_plot.png" width="600px"/><figcaption>Plot of the paraboloid reflector</figcaption></figure>


## Defining ray trace frames
Now we will define a ray trace frame. We can do that from the menu bar by selecting Ray-trace > Make frame > Tube.

Again, a form will show up. We fill it with the next parameters.


<figure><img src="README_Resources/frame_form.png" alt="Selecting make tube frame" /><figcaption>Frame parameter form</figcaption></figure>

When the frame is added a widget will show up in the *Frames* tab of the workspace. By clicking *plot* in the frame options menu we are presented with a form that allows us to choose the plane in which we want to plot this frame. If we select x and y we get the following plot. 

<figure><img src="README_Resources/frame_plot.png" width="800px" /><figcaption>Plot of the frame in xy plane</figcaption></figure>

Now we've defined a frame on the xy plane with z-coordinate 0. We can translate this frame upwards by selecting the transform option from the frame options as shown below

<figure><img src="README_Resources/frame_transrform.png" width="400px"/><figcaption>Frame transformation form</figcaption></figure>


Now if we plot the frame again from the side (e.g. in the xz plane). After zooming in with the zoom tool in the plot, we can see that all of the points have z coordinate 3000.

<figure><img src="README_Resources/frame_plot2.png" width="800px" /><figcaption>Plot of frame in xy plane</figcaption></figure>

## Propagating rays

We can propagate the rays of the frame we just created by clicking *Ray trace > Propagate Rays*. Again, a form will show up. We fill it with these parameters: 



<figure><img src="README_Resources/prop_frame_form.png"/><figcaption>For for trace propagation</figcaption></figure>
<figure><figcaption></figcaption></figure>

This will create a new frame *fr_pri*. If we plot this frame in th xy plane, it will look exactly the same as the previous frame. But looking from the side (e.g. plotting in the xz plane) we can see that the points have been projected onto the paraboloid.

<figure><img src="README_Resources/fr2_plot.png" alt="Plot *fr_pri* in xz plane" style="width:50%"><figcaption>Plot *fr_pri* in xy plane</figcaption></figure>



<figure><img src="README_Resources/fr2_plot2.png" alt="Plot *fr_pri* in xz plane" style="width:50%"><figcaption>Plot *fr_pri* in xz plane</figcaption></figure>


## Focus Finding

After the rays were propagated onto the paraboloid reflector, they can be propagated to the focus of the reflector. This can be done with the *focus finder*; in the menubar we select *Tools > Focus finder*. Then we select the frame we want to find the focus of and click *Find focus*. This will automatically generate a new frame. If we plot this frame in *xy* we see that the numbers along the axes are very small. The frame has been propagated to the focal plane.


<figure><img src="README_Resources/foc_frame_plot.png" alt="Selecting make tube frame" /><figcaption>Frame plot in focus</figcaption></figure>




