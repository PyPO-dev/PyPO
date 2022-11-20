import sys
import os

def BuildPOPPy():
    pathToBuild = os.path.join("src", "C++")
    # Parse command line input

    cuda_flag = sys.argv.count("--enable-cuda") or sys.argv.count("-ec")
    verbose = sys.argv.count("--verbose") or sys.argv.count("-v")
    clean = sys.argv.count("--clean") or sys.argv.count("-c")

    if verbose:
        stream = ""
    else:
        stream = " > /dev/null"

    if clean:
        try:
            print("Removing POPPy objects and libraries...")
            os.chdir(pathToBuild)
            os.system("make clean" + stream)
            print("Succesfully removed POPPy objects and libraries!")
            return 0

        except:
            return 1

    if cuda_flag: 
        try:
            print("Compiling POPPy, enabling CUDA...")
            
            os.chdir(pathToBuild)
            os.system("make cpu" + stream)
            os.system("make gpu" + stream)
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

if __name__ == "__main__":
	BuildPOPPy()
