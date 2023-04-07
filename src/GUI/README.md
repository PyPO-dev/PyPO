# PyPO GUI Manual

## 
![Blank app](README_Resources/blankGui.png)
The gui consists of 3 widgets
* Workspace (left)
* Plot screen
* Console

For most interactions a form will show up next to the workspace.

In this tutorial we wil, step by step, build a simple reflector system and propagate a ray trace trough it.

## Creating reflector
From the elements menu select *Add Reflector* > *Quadric surface*. 

A form will show up with one dropdown selector. By selecting options, more options will show up. By filling in the parameters and clicking *add*, a reflector has been added to the system. If widget will show up in the workspace as such:

<img src="README_Resources/parabolaGenerated.png" alt="System with one paraboloid reflector" width="80%"/>

If adding the reflector did not succeed a message will be logged in the console 
<img src="README_Resources/emptyFieldErr.png" alt="text saying: Error - Empty field at Focus xyz" width="80%"/>
<!-- 
## Functionalities
* Defining reflectors
* Grouping reflectors
* Transforming reflectors and groups
* Defining ray trace frames
* Defining physical optics beams -->
