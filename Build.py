import sys
import os
import shutil

def BuildPOPPy():
    pathToBuild = os.path.join("src")
    # Parse command line input

    helpf   = sys.argv.count("--help") or sys.argv.count("-h")
    prereq  = sys.argv.count("--prereqs") or sys.argv.count("-p")
    cmake   = sys.argv.count("--make")
    cinstall = sys.argv.count("--install")
    cmakec  = sys.argv.count("--clean")
    
    if cmakec:
        print("Cleaning CMake build directory...")
        dir_build = os.path.join(os.getcwd(), "out", "build")
        shutil.rmtree(dir_build)
        return 0

    if cmake:
        print("Configuring POPPy using CMake...")
        dir_lists = os.path.join(os.getcwd(), "src")
        dir_build = os.path.join(os.getcwd(), "out", "build")

        if not os.path.exists(dir_build):
            os.makedirs(dir_build)

        os.system(f"cmake -S {dir_lists} -B {dir_build} -DCMAKE_BUILD_TYPE=Release")#.format(dir_lists, dir_build))

        return 0

    if cinstall:
        print("Building POPPy using CMake...")
        dir_lists = os.path.join(os.getcwd(), "src")
        dir_build = os.path.join(os.getcwd(), "out", "build")

        if not os.path.exists(dir_build):
            os.makedirs(dir_build)

        os.system(f"cmake --build {dir_build}")#.format(dir_lists, dir_build))

        return 0
    if helpf:
        print("POPPy build interface list of options:")
        print("'--help', '-h'           : get build options.")
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
            
if __name__ == "__main__":
	BuildPOPPy()
