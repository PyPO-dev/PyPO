Welcome to the Physical OPtics for Python (POPPy) package!

-- PREREQS --\
Any Unix-like operating system should work.
A fully functioning [Python 3.8](https://docs.python-guide.org/starting/install3/linux/) installation.
If the PIP package manager was not automatically installed with your Python install, it can be installed [manually](https://pip.pypa.io/en/stable/installation/).
Run the following command to install the necessary Python packages:
```
    python3 -m pip install numpy matplotlib scipy setuptools nose PyQt5
```
POPPy is capable of producing figures using LaTeX typesetting. For this, a LaTeX installation should be present on the machine. To install the proper packages for the backend, run:
```
    sudo apt install cm-super dvipng
```
The [g++](https://gcc.gnu.org/install/) compiler, capable of compiling code written in the C++11 standard, and [GNU make](https://www.gnu.org/software/make/). Both can be installed using:
```
    sudo apt-get install gcc build-essential
```
The GPU version of POPPy needs a [CUDA installation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) and a CUDA-compatible NVIDIA graphics card. Please refer to the NVIDIA documentation on how to install NVIDIA drivers and CUDA.

-- INSTALLATION --\
The physical optics calculations in POPPy can be run on the CPU or GPU.
In order to build the program for both CPU and GPU, run the following two commands from the POPPy root directory:
```
    cd ./src/C++
    make all
```
This will create the executables for CPU and GPU. If, for example, you only want to install the CPU parallel version, run:
```
    make cpu
```
The GPU version is installed separately by running:
```
    make gpu
```

-- TESTING --\
To run the Python unittests, go to the main POPPy directory and from there run:
```
    nosetests --exe
```
