import sys
import os
import shutil

def BuildPOPPy():
    pathToBuild = os.path.join("src")
    # Parse command line input

    helpf   = sys.argv.count("--help") or sys.argv.count("-h")
    prereq  = sys.argv.count("--prereqs") or sys.argv.count("-p")
    config   = sys.argv.count("--config") or sys.argv.count("-f")
    cmake = sys.argv.count("--make") or sys.argv.count("-m")
    cmakec  = sys.argv.count("--clean") or sys.argv.count("-c")
    
    if prereq:
        if os.name == "posix":
            try:
                print("Installing prerequisites...")
                os.system("python3 -m pip install numpy matplotlib scipy setuptools nose PyQt5")
                os.system("sudo apt-get install cm-super dvipng gcc build-essential")
                print("Succesfully installed POPPy prerequisites!")
                print("WARNING: CUDA not installed. Install CUDA manually to enable POPPy on GPU.")
                return 0
            except:
                return 1

        elif os.name == "nt": 
            try:
                print("Installing prerequisites...")
                os.system("pip install numpy matplotlib scipy setuptools nose PyQt5")
                print("Succesfully installed POPPy Python prerequisites! Refer to README for CMake installation.")
                print("WARNING: CUDA not installed. Install CUDA manually to enable POPPy on GPU.")
                return 0
            except:
                return 1

    if cmakec:
        print("Cleaning build directory...")
        dir_build = os.path.join(os.getcwd(), "out", "build")
        shutil.rmtree(dir_build)
        return 0

    if config:
        print("Configuring POPPy...")
        dir_lists = os.path.join(os.getcwd(), "src")
        dir_build = os.path.join(os.getcwd(), "out", "build")

        if not os.path.exists(dir_build):
            os.makedirs(dir_build)


        if os.name == "posix":
            os.system(f"cmake -S {dir_lists} -B {dir_build} -DCMAKE_BUILD_TYPE=Release")

        elif os.name == "nt":
            os.system(f"cmake -S {dir_lists} -B {dir_build}")
        
        return 0

    if cmake:
        print("Building POPPy...")
        dir_lists = os.path.join(os.getcwd(), "src")
        dir_build = os.path.join(os.getcwd(), "out", "build")

        if not os.path.exists(dir_build):
            os.makedirs(dir_build)

        os.system(f"cmake --build {dir_build}")
        return 0

    if helpf:
        print("POPPy build interface list of options:")
        print("'--help', '-h'           : get build options.")
        print("'--clean', '-c'          : remove POPPy objects and libraries.")
        print("'--prereqs', '-p'        : install POPPy prerequisites.")
        print("'--config', '-f'         : configure POPPy.")
        print("'--make', '-m'           : build POPPy libraries.")
        return 0
 
if __name__ == "__main__":
	BuildPOPPy()
