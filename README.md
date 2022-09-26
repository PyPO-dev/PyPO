Welcome to the Physical OPtics for Python (POPPy) package!

-- PREREQS --\
A fully functioning [Python 3.8](https://docs.python-guide.org/starting/install3/linux/) installation.
If the PIP package manager was not automatically installed with your Python install, it can be installed [manually](https://pip.pypa.io/en/stable/installation/).
Run the following command to install the necessary Python packages:
```
    $ python3 -m pip install numpy matplotlib scipy setuptools nose psutil
```
The [g++](https://gcc.gnu.org/install/) compiler, capable of compiling code written in the C++11 standard, and [GNU make](https://www.gnu.org/software/make/). Both can be installed using:
```
    $ sudo apt-get install gcc build-essential
```

-- SYSTEM REQS --\
CPU: One core/thread (minimal). As many cores as possible (recommended).\
RAM: 4 Gigabytes (minimal). As much as possible (recommended).\
OS: Any Unix-like operating system.\

-- TESTING --\
To run the Python unittests, first go to the main POPPy directory and from there run:
```
    $ nosetests
```
