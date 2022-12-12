#!/usr/bin/env python3

import sys
import os
import shutil
import platform

def BuildPyPO():
    pathToBuild = os.path.join("src")
    # Parse command line input

    helpf   = sys.argv.count("--help") or sys.argv.count("-h")
    prereq  = sys.argv.count("--prereqs") or sys.argv.count("-p")
    config   = sys.argv.count("--config") or sys.argv.count("-f")
    cmake = sys.argv.count("--make") or sys.argv.count("-m")
    cmakec  = sys.argv.count("--clean") or sys.argv.count("-c")
    docs    = sys.argv.count("--docs") or sys.argv.count("-d")

    if prereq:
        print("Installing prerequisites...")
        if platform.system() == "Linux":
            os.system("sudo apt-get install cm-super dvipng gcc build-essential cmake")
            os.system("python3 -m pip install numpy matplotlib scipy setuptools nose PyQt5")
        elif platform.system() == "Darwin":
            os.system("brew install gcc cmake qt5")
            os.system("python3 -m pip install numpy matplotlib scipy setuptools nose")
            os.system("xcode-select --install")
            
        elif platform.system() == "Windows":
            os.system("py -m pip install numpy matplotlib scipy setuptools nose PyQt5")
            
        print("Succesfully installed PyPO Python prerequisites! Refer to README for CMake installation.")
        print("WARNING: CUDA not installed. Install CUDA manually to enable PyPO on GPU.")
    
    if cmakec:
        print("Cleaning build directory...")
        dir_build = os.path.join(os.getcwd(), "out", "build")
        shutil.rmtree(dir_build)
        return 0

    if config:
        print("Configuring PyPO...")
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
        print("Building PyPO...")
        dir_lists = os.path.join(os.getcwd(), "src")
        dir_build = os.path.join(os.getcwd(), "out", "build")

        if not os.path.exists(dir_build):
            os.makedirs(dir_build)

        os.system(f"cmake --build {dir_build}")
        return 0

    if docs:
        print("Generating PyPO documentation...")
        try:
            try:
                shutil.rmtree("docs")
            except:
                pass

            os.system(f"doxygen Doxyfile")

        except:
            print("ERROR: failed to generate documentation!")
        return 0

    if helpf:
        print("PyPO build interface list of options:")
        print("'--help', '-h'           : get build options.")
        print("'--clean', '-c'          : remove PyPO objects and libraries.")
        print("'--prereqs', '-p'        : install PyPO prerequisites.")
        print("'--config', '-f'         : configure PyPO.")
        print("'--make', '-m'           : build PyPO libraries.")
        print("'--docs', '-d'           : generate PyPO documentation. Needs doxygen!")
        return 0
 
if __name__ == "__main__":
	BuildPyPO()
