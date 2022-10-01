Welcome to the Physical OPtics for Python (POPPy) package!

-- PREREQS --\
Any Unix-like operating system should work.
A fully functioning [Python 3.8](https://docs.python-guide.org/starting/install3/linux/) installation.
If the PIP package manager was not automatically installed with your Python install, it can be installed [manually](https://pip.pypa.io/en/stable/installation/).
Run the following command to install the necessary Python packages:
```
    $ python3 -m pip install numpy matplotlib scipy setuptools nose psutil PyQt5
```
The [g++](https://gcc.gnu.org/install/) compiler, capable of compiling code written in the C++11 standard, and [GNU make](https://www.gnu.org/software/make/). Both can be installed using:
```
    $ sudo apt-get install gcc build-essential
```
The GPU version of POPPy needs a [CUDA installation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) and a CUDA-compatible NVIDIA graphics card. For POPPy, only the CUDA-toolkit is important, as this holds the necessary libraries and the nvcc compiler. It is installed as such:
```
    $ sudo apt install nvidia-cuda-toolkit
```
To check whether it installed correctly, run:
```
    $ nvcc --version
```

There are no real system requirements, but the more RAM and CPU threads the better.

-- INSTALLATION --\
The physical optics calculations in POPPy can be run on the CPU or GPU.
In order to build the program for both CPU and GPU, run the following two commands from the POPPy root directory:
```
    $ cd ./src/C++
    $ make all
```
This will create the executables for CPU and GPU. If, for example, you only want to install the CPU parallel version, run:
```
    $ make cpu
```
The GPU version is installed separately by running:
```
    $ make gpu
```

-- TESTING --\
To run the Python unittests, first go to the main POPPy directory and from there run:
```
    $ nosetests
```
