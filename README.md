Welcome to the Physical OPtics for Python (POPPy) package!

-- PREREQS --\
A fully functioning [Python 3.8](https://docs.python-guide.org/starting/install3/linux/) installation.
If the PIP package manager was not automatically installed with your Python install, it can be installed [manually](https://pip.pypa.io/en/stable/installation/).

To install the prerequisites for POPPy, navigate to the main folder and run:
```
python Build.py --prereq
```
Alternatively, the prerequisites can be installed manually on Linux by running:
```
sudo apt install cm-super dvipng gcc build-essential cmake
python3 -m pip install numpy matplotlib scipy setuptools nose PyQt5
```
On Windows or Mac OS, [CMake](https://cmake.org/install/) can be installed manually. The Python prerequisites can still be installed manually from the command line. The cm-super and dvipng packages are included on most TeX installations.
POPPy is capable of producing figures using LaTeX typesetting. For this, a [LaTeX installation](https://www.tug.org/texlive/quickinstall.html) should be present on the machine.
The GPU version of POPPy needs a [CUDA installation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) and a CUDA-compatible NVIDIA graphics card. These are not installed through the Build.py interface and should be installed manually. Please refer to the NVIDIA documentation on how to install NVIDIA drivers and CUDA.

-- INSTALLATION --\
Configure POPPy by running:
```
python Build.py --make
```
This will check if you have CUDA installed. It will also generate the build instructions.
Then install by running:
```
python Build.py --install
```
For an overview of build options, run:
```
python Build.py --help
```

To include POPPy from anywhere, the following two lines should be added to your .bashrc file:
``` 
export PYTHONPATH=${PYTHONPATH}:<absolue/path/to/POPPy>
```
and source the script.
On Windows, do the following:
```
My Computer > Properties > Advanced System Settings > Environment Variables >
```
and select 'Edit System Variable'. Create a new variable called 'PYTHONPATH' and give it the following value:
```
<absolue/path/to/POPPy>
```
Select 'OK' and select 'OK' again when in 'Environment Variables'.
On Mac OS, add the following to your ~/.bash_profile:
```
PYTHONPATH="<absolue/path/to/POPPy>:$PYTHONPATH"
export PYTHONPATH
```
and source the script.

-- TESTING --\
To run the Python unittests, go to the main POPPy directory and from there run:
```
nosetests --exe
```
