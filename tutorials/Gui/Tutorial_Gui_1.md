# PyPO GUI Manual

## 
![Blank app](README_Resources/blankGui.png)
The gui consists of 3 widgets
* Workspace (left)
* Plot screen
* Console

For most interactions a form will show up next to the workspace.

In this tutorial we wil, step by step, build a simple reflector system and propagate a ray trace trough it.

## Creating reflectors
From the elements menu select *Add Reflector* > *Quadric surface*. 

A form will show up with one dropdown selector. By selecting options, more options will show up. By filling in the parameters and clicking *add*, a reflector has been added to the system. If widget will show up in the workspace as such:

<!--![System with one paraboloid reflector](README_Resources/pri_form.png)-->
<img src="README_Resources/pri_form.png" alt="System with one paraboloid reflector" width="80%"/>

If adding the reflector did not succeed a message will be logged in the console with information about what went wrong.

<img src="README_Resources/emptyFieldErr.png" alt="text saying: Error - Empty field at Focus xyz" width="55%"/>

## Plotting
Now to see the reflector we just defined we need to plot it. 
By clicking on the three dots on the right side of the element widget the element options menu will open. The first option is plot. 

<img src="README_Resources/options_plot.png" alt="text saying: Error - Empty field at Focus xyz" width="40%"/>

This will show the plot on the plot screen.

<img src="README_Resources/pri_plot.png" alt="text saying: Error - Empty field at Focus xyz" width="80%"/>

## Defining ray trace frames
Now we will define a ray trace frame. We can do that from the menu bar by selecting Ray-trace > Make frame > Tube.

Again, a form will show up. We fill it with the next parameters.

<img src="README_Resources/frame_form.png" alt="Selecting make tube frame" />

When the frame is added a widget will show up in the *Frames* tab of the work space. If we plot this frame in the xy plane, it will look like this:

<img src="README_Resources/frame_plot.png" alt="Selecting make tube frame" />


Now we've defined a frame on the xy plane with z-coordinate 0. We can translate this frame upwards by selecting the transform option from the frame options as shown below

<img src="README_Resources/frame_transrform.png" alt="Selecting make tube frame" />

Now if we plot the frame again from the side (e.g. in the xz plane). After zooming in with the zoom tool in the plot, we can see that all of the points have z coordinate 3000.
<img src="README_Resources/frame_plot2.png" alt="Selecting make tube frame" />

## Propagating rays

We can propagate the rays of the frame we just created by clicking *Ray trace > Propagate Rays*. Again, a form will show up. We fill it with these parameters: 


<img src="README_Resources/prop_frame_form.png" alt="Selecting make tube frame" />

This will create a new frame *fr_pri*. If we plot this frame in th xy plane, it will look exactly the same as the previous form. But looking from the side (e.g. plotting in the x-z plane) we can see that the points have been projected onto the paraboloid.

<img src="README_Resources/fr2_plot.png" alt="Selecting make tube frame" />
<img src="README_Resources/fr2_plot2.png" alt="Selecting make tube frame" />

## Focus Finding

After the rays were propagated onto the paraboloid reflector, they can be propagated propagated to the focus of the parabola. This can be done with the *focus finder*. In the menubar select Tools > focus finder. We select the frame we want to find the focus of the click *find focus*. This will automatically generate a new frame. If we plot this frame in xy, we see that the numbers along the axes are very small. The frame has been propagated

<img src="README_Resources/foc_frame_plot.png" alt="Selecting make tube frame" />



## Functionalities
 Defining reflectors
* Grouping reflectors
* Transforming reflectors and groups
* Defining ray trace frames
* 
* Defining physical optics beams
