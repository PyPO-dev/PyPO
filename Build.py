import sys
import os

def BuildPOPPy():
    pathToBuild = os.path.join("src")
    # Parse command line input

    helpf   = sys.argv.count("--help") or sys.argv.count("-h")
    cuda    = sys.argv.count("--enable-cuda") or sys.argv.count("-ec")
    verbose = sys.argv.count("--verbose") or sys.argv.count("-v")
    clean   = sys.argv.count("--clean") or sys.argv.count("-c")
    prereq  = sys.argv.count("--prereqs") or sys.argv.count("-p")
    cmake   = sys.argv.count("--cmake")

    if verbose:
        stream = ""
    else:
        stream = " > /dev/null"

    if cmake:
        print("Building POPPy using CMake...")
        dir_lists = os.path.join(os.getcwd(), "src")
        dir_build = os.path.join(dir_lists, "out", "build")

        if not os.path.exists(dir_build):
            os.makedirs(dir_build)

        os.system("cmake -S{} -B{}".format(dir_lists, dir_build))

        return 0

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
