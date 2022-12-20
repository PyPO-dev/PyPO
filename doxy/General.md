<h1>Welcome to the Python Physical Optics (PyPO) package!</h1>

<h2>Prerequisites</h2>
A fully functioning [Python 3.8](https://docs.python-guide.org/starting/install3/linux/) installation. At the moment, the GUI only supports a Python version of 3.9 or lower.
If the PIP package manager was not automatically installed with your Python install, it can be installed [manually](https://pip.pypa.io/en/stable/installation/).

To install the prerequisites for PyPO, navigate to the main folder and run:
```
python Build.py --prereqs
```
Alternatively, the prerequisites can be installed manually on Linux by running:
```
sudo apt install gcc build-essential cmake qtcreator
python3 -m pip install numpy matplotlib scipy setuptools nose PyQt5
```
On Mac OS, the prereqs are installed using:
```
brew install gcc cmake
xcode-select --install
python3 -m pip install numpy matplotlib scipy setuptools nose PyQt5
```
On Windows, things are slightly more complicated. First, install [Visual Studio](https://visualstudio.microsoft.com/#2010-Visual-CPP). 
This is necessary as Visual Studio contains CMake and the build tool [NMake](https://learn.microsoft.com/en-us/cpp/build/reference/nmake-reference?view=msvc-170), which are important for building PyPO.
During installation, select the 'Community' install.
You will be prompted to install components and/or workloads. 
Under 'Desktop & Mobile', tick 'Desktop development with C++' and proceed. 
This is the minimum requirement, and more components/workloads can be installed, if so desired. After installation, proceed with regular installation.

PyPO is capable of producing figures using LaTeX typesetting. For this, a [LaTeX installation](https://www.tug.org/texlive/quickinstall.html) should be present on the machine. The LaTeX prerequisites will be installed automatically upon running Build.py but can also be installed manually:
```
sudo apt install cm-super dvipng texlive-latex-extra texlive-fonts-recommended texlive-science
```
The GPU version of PyPO needs a [CUDA installation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) and a CUDA-compatible NVIDIA graphics card. 
These are not installed through the Build.py interface and should be installed manually. Please refer to the NVIDIA documentation on how to install NVIDIA drivers and CUDA on your specific Platform.

<h2>Installation</h2>
On Linux and MacOs, the following instructions can be run in the regular terminal.
On Windows, they should be run in the 'x64_x86 Cross Tools Command Promp for VS <year>'.
Navigate to the PyPO root folder and configure PyPO by running:
```
python Build.py --config
```
This will check if you have CUDA installed. It will also generate the build instructions.
Then install by running:
```
python Build.py --make
```
For an overview of build options, run:
```
python Build.py --help
```

To include PyPO from anywhere, the following two lines should be added to your .bashrc file:
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
To run the Python unittests, go to the main PyPO directory and from there run:
```
nosetests --exe
```
<h2>Documentation</h2>
The following instructions are for people who would like to develop PyPO and generate documentation along the way.
PyPO uses [Doxygen](https://www.doxygen.nl/download.html) for generating documentation for the source code. 
Please refer to the link for download and installation instructions.
If the installation fails on Linux, try:
```
sudo apt install flex
sudo apt install bison
```
Generate the documentation by running:
```
python Build.py --docs
```
which should place the documentation in the docs/ folder.
