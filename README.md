Welcome to the Physical OPtics for Python (POPPy) package!

-- PREREQS --\
Any Unix-like operating system should work.
A fully functioning [Python 3.8](https://docs.python-guide.org/starting/install3/linux/) installation.
If the PIP package manager was not automatically installed with your Python install, it can be installed [manually](https://pip.pypa.io/en/stable/installation/).

To install the prerequisites for POPPy, navigate to the main folder and run:
```
python Build.py --prereq
```
Alternatively, the prerequisites can be installed manually by running:
```
sudo apt install cm-super dvipng gcc build-essential
python3 -m pip install numpy matplotlib scipy setuptools nose PyQt5
```
POPPy is capable of producing figures using LaTeX typesetting. For this, a [LaTeX installation](https://www.tug.org/texlive/quickinstall.html) should be present on the machine.
The GPU version of POPPy needs a [CUDA installation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) and a CUDA-compatible NVIDIA graphics card. These are not installed through the Build.py interface and should be installed manually. Please refer to the NVIDIA documentation on how to install NVIDIA drivers and CUDA.

-- INSTALLATION --\
The CPU version for POPPy is created by running:
```
python Build.py
```
To compile the CUDA version of POPPy, run:
```
python Build.py --enable-cuda
```

-- TESTING --\
To run the Python unittests, go to the main POPPy directory and from there run:
```
    nosetests --exe
```
