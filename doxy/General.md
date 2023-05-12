<h1>Welcome to the Python Physical Optics (PyPO) package!</h1>

In this page we will explain how to install `PyPO` using the Build.py script supplied with the package.

<h2>Prerequisites</h2>
A fully functioning [Python 3.8](https://docs.python-guide.org/starting/install3/linux/) installation.
If the PIP package manager was not automatically installed with your Python install, it can be installed [manually](https://pip.pypa.io/en/stable/installation/).

`PyPO` and prerequisites are installed through the Build.py script. 
For an overview of build options, run:
```
./Build.py -h
```

To install the prerequisites for PyPO, navigate to the main folder and run:
```
./Build.py -p
```
To install the optional packages, such as nose2 for the unittests and PySide2 for the GUI, run
```
./Build.py -po
```
These instructions should install all prerequisites except for CUDA on a Linux/MacOS machine.

On Windows, things are slightly more complicated. First, install [Visual Studio](https://visualstudio.microsoft.com/#2010-Visual-CPP). 
This is necessary as Visual Studio contains CMake and the build tool [NMake](https://learn.microsoft.com/en-us/cpp/build/reference/nmake-reference?view=msvc-170), which are important for building PyPO.
During installation, select the 'Community' install.
You will be prompted to install components and/or workloads. 
Under 'Desktop & Mobile', tick 'Desktop development with C++' and proceed. 
This is the minimum requirement, and more components/workloads can be installed, if so desired. After installation of Visual Studio, proceed with regular installation.

PyPO is capable of producing figures using LaTeX typesetting. For this, a [LaTeX installation](https://www.tug.org/texlive/quickinstall.html) should be present on the machine.
This not installed through the Build.py script with -o flag. Instead, the LaTeX libraries can be installed by the user. For example:
```
sudo apt install texlive-latex-extra texlive-fonts-recommended texlive-science
```
This installs the TeX Live distribution.

The GPU version of PyPO needs a [CUDA installation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) and a CUDA-compatible NVIDIA graphics card. 
These are not installed through the Build.py interface and should be installed manually. On Linux, it is relatively straightforward to do this using the [apt repository](https://linuxconfig.org/how-to-install-cuda-on-ubuntu-20-04-focal-fossa-linux). 
For other platforms, please refer to the NVIDIA documentation on how to install NVIDIA drivers and CUDA.

<h2>Installation</h2>
On Linux and MacOs, the following instructions can be run in the regular terminal.
On Windows, they should be run in the 'x64_x86 Cross Tools Command Promp for VS <year>'.
Navigate to the PyPO root folder and configure PyPO by running:
```
./Build.py -f
```
This will check if you have CUDA installed. It will also generate the makefiles necessary for the build.
Then install by running:
```
./Build.py -m
```

To include PyPO from anywhere, the following line should be added to your .bashrc file:
``` 
export PYTHONPATH=${PYTHONPATH}:<absolue/path/to/PyPO>
```
and source the script.
On Windows, do the following:
```
My Computer > Properties > Advanced System Settings > Environment Variables >
```
and select 'Edit System Variable'. Create a new variable called 'PYTHONPATH' and give it the following value:
```
<absolue/path/to/PyPO>
```
Select 'OK' and select 'OK' again when in 'Environment Variables'.
On Mac OS, add the following to your ~/.bash_profile:
```
PYTHONPATH="<absolue/path/to/PyPO>:$PYTHONPATH"
export PYTHONPATH
```
and source the script.

<h2>Testing</h2>
To run the unittests, navigate to the PyPO root directory and from there run:
```
./Build.py -t
```

<h2>Documentation</h2>
The following instructions are for people who would like to develop `PyPO` and generate documentation along the way.
PyPO uses [Doxygen](https://www.doxygen.nl/download.html) for generating documentation for the source code. 
Please refer to the link for download and installation instructions.
If the installation fails on Linux, try:
```
sudo apt install flex bison pandoc
```
On Windows, please see the Doxygen website.
Generate the documentation by running:
```
./Build.py -d
```
which should place the documentation in the docs/ folder.
