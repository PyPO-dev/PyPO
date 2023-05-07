#!/usr/bin/env python3
import sys
import os
import shutil
import platform
import argparse

from src.PyPO.CustomLogger import CustomLogger

##
# @file
# PyPO build script.
#
# Configures CMake, generates makefiles and runs them.
# Also contains functions to install prerequisites and clean build directories.
# There is one flag for generating documentation, but most users (probably) won't use this option.
# For an overview of the possible flags, run in a terminal:
#```
# python Build.py --help
#```
def BuildPyPO():
    clog_mgr = CustomLogger(os.path.basename(__file__))
    clog = clog_mgr.getCustomLogger()

    pathToBuild = os.path.join("src")
    
    parser = argparse.ArgumentParser(description="build and test interface script for PyPO")
    parser.add_argument("-p", "--prereqs", help="install PyPO prerequisites", action="store_true")
    parser.add_argument("-o", "--optional", help="install optional packages", action="store_true")
    parser.add_argument("-f", "--config", help="configure PyPO build scripts", action="store_true")
    parser.add_argument("-m", "--make", help="build PyPO libraries", action="store_true")
    parser.add_argument("-c", "--clean", help="remove PyPO build directory", action="store_true")
    parser.add_argument("-d", "--docs", help="generate PyPO documentation with doxygen", action="store_true")
    parser.add_argument("-t", "--test", help="run PyPO unittests", action="store_true")
    args = parser.parse_args()

    if args.prereqs:
        clog.info("Installing PyPO prerequisites...")
        path_to_reqs = os.path.join("out", "requirements", "requirements.txt")
        path_to_reqs_opt = os.path.join("out", "requirements", "requirements_opt.txt")
        if platform.system() == "Linux":
            os.system("sudo apt install cm-super dvipng gcc build-essential cmake")
            
            if args.optional:
                os.system("sudo apt install qtbase5-dev qt5-qmake qtbase5-dev-tools")

        elif platform.system() == "Darwin":
            os.system("xcode-select --install")
            os.system("brew install gcc cmake")
            
            if args.optional:
                os.system("brew install qt5 qtbase5-dev qt5-qmake qtbase5-dev-tools")
            
        #elif platform.system() == "Windows":
        #    os.system("py -m pip install numpy matplotlib scipy setuptools nose PySide2 tqdm inquirer attrs")
            
        if args.optional:
            os.system(f"pip install -U -r {path_to_reqs_opt}")
        
        else:
            os.system(f"pip install -U -r {path_to_reqs}")
        
        clog.info("Succesfully installed PyPO prerequisites.")
        clog.warning("Install CUDA manually to enable PyPO on GPU.")
    
    if args.clean:
        try:
            clog.info("Cleaning build directory...")
            dir_build = os.path.join(os.getcwd(), "out", "build")
            shutil.rmtree(dir_build)
            clog.info("Succesfully cleaned build directory.")
        except:
            clog.warning("Nothing to clean.")

    if args.config:
        dir_lists = os.path.join(os.getcwd(), "src")
        dir_build = os.path.join(os.getcwd(), "out", "build")

        if not os.path.exists(dir_build):
            os.makedirs(dir_build)
        
        try:
            clog.info("Configuring PyPO...")
            if os.name == "posix":
                os.system(f"cmake -S {dir_lists} -B {dir_build} -DCMAKE_BUILD_TYPE=Release")

            elif os.name == "nt":
                os.system(f"cmake -S {dir_lists} -B {dir_build}")
            clog.info("Succesfully configured PyPO.")
        
        except:
            clog.error("Could not configure PyPO. Is CMake installed?")

    if args.make:
        try:
            clog.info("Building PyPO...")
            dir_lists = os.path.join(os.getcwd(), "src")
            dir_build = os.path.join(os.getcwd(), "out", "build")

            if not os.path.exists(dir_build):
                os.makedirs(dir_build)

            os.system(f"cmake --build {dir_build}")
            clog.info("Succesfully built PyPO.")

        except:
            clog.error("Could not build PyPO.")

    if args.docs:
        try:
            try:
                shutil.rmtree("docs")
            except:
                pass
            clog.info("Generating PyPO documentation...")
            os.system("doxygen doxy/Doxyfile")
            
            # Read html to set default detail level of menus
            annotated_path = os.path.join("docs", "annotated.html")
            filelist_path = os.path.join("docs", "files.html")
            
            with open(annotated_path, 'r') as file :
                filedata = file.read()
                filedata = filedata.replace('init_search(); });', 'init_search(); toggleLevel(2); });')
            
            with open(annotated_path, 'w') as file:
                file.write(filedata)
            
            with open(filelist_path, 'r') as file :
                filedata = file.read()
                filedata = filedata.replace('init_search(); });', 'init_search(); toggleLevel(2); });')
            
            with open(filelist_path, 'w') as file:
                file.write(filedata)

            clog.info("Succesfully generated PyPO documentation.")
        
        except:
            clog.error("Failed to generate documentation. Is doxygen installed?")
    
    if args.test:
        try:
            clog.info("Running PyPO unittests...")
            dir_tests = os.path.join(os.getcwd(), "tests")

            os.system(f"nose2 -v")

        except:
            clog.error("Failed to test PyPO.")

 
if __name__ == "__main__":
	BuildPyPO()
