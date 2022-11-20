import sys
import os

def BuildPOPPy():
    pathToBuild = os.path.join("src", "C++")
    # Parse command line input

    helpf   = sys.argv.count("--help") or sys.argv.count("-h")
    cuda    = sys.argv.count("--enable-cuda") or sys.argv.count("-ec")
    verbose = sys.argv.count("--verbose") or sys.argv.count("-v")
    clean   = sys.argv.count("--clean") or sys.argv.count("-c")
    prereq  = sys.argv.count("--prereqs") or sys.argv.count("-p")


    if verbose:
        stream = ""
    else:
        stream = " > /dev/null"

    if helpf:
        print("POPPy build interface list of options:")
        print("'--help', '-h'           : get build options.")
        print("'--enable-cuda', '-ec'   : enable CUDA compilation.")
        print("'--verbose', '-v'        : enable verbose compiler output.")
        print("'--clean', '-c'          : remove POPPy objects and libraries.")
        print("'--prereqs', '-p'        : install POPPy prerequisites.")
        return 0

    if prereq:
        try:
            print("Installing prerequisites...")
            os.system("python3 -m pip install numpy matplotlib scipy setuptools nose PyQt5" + stream)
            os.system("sudo apt-get install cm-super dvipng gcc build-essential" + stream)
            print("Succesfully installed POPPy prerequisites!")
            print("WARNING: CUDA not installed. Install CUDA manually to enable POPPy on GPU.")
            return 0
        except:
            return 1
            

    if clean:
        try:
            print("Removing POPPy objects and libraries...")
            os.chdir(pathToBuild)
            os.system("make clean" + stream)
            print("Succesfully removed POPPy objects and libraries!")
            return 0
        except:
            return 1

    if cuda: 
        try:
            print("Compiling POPPy, enabling CUDA...")
            
            os.chdir(pathToBuild)
            os.system("make all" + stream)
            print("Succesfully compiled POPPy, CUDA enabled!")
            return 0
        except:
            return 1

    else: 
        try:
            print("Compiling POPPy, disabling CUDA...")
            
            os.chdir(pathToBuild)
            os.system("make cpu" + stream)
            print("Succesfully compiled POPPy, CUDA disabled!")
            return 0
        except:
            return 1

    return 0

if __name__ == "__main__":
	BuildPOPPy()
