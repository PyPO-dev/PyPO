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
    parser.add_argument("-f", "--config", help="configure PyPO build scripts", action="store_true")
    parser.add_argument("-m", "--make", help="build PyPO libraries", action="store_true")
    parser.add_argument("-c", "--clean", help="remove PyPO build directory", action="store_true")
    parser.add_argument("-d", "--docs", help="generate PyPO documentation with doxygen", action="store_true")
    parser.add_argument("-t", "--test", help="run PyPO unittests", action="store_true")
    parser.add_argument("-fm", "--full-monty", help="clean, configure and make PyPO in one go", action="store_true")
    args = parser.parse_args()

    if args.prereqs:
        clog.info("Installing PyPO prerequisites...")
        if platform.system() == "Linux":
            os.system("sudo apt-get install cm-super dvipng gcc build-essential cmake")
            os.system("sudo apt install qtbase5-dev qt5-qmake qtbase5-dev-tools")
            os.system("python3 -m pip install numpy matplotlib scipy setuptools nose PyQt5 tqdm inquirer attrs")
        elif platform.system() == "Darwin":
            os.system("brew install gcc cmake qt5")
            os.system("brew install qtbase5-dev qt5-qmake qtbase5-dev-tools")
            os.system("python3 -m pip install numpy matplotlib scipy setuptools nose tqdm inquirer attrs")
            os.system("xcode-select --install")
            
        elif platform.system() == "Windows":
            os.system("py -m pip install numpy matplotlib scipy setuptools nose PyQt5 tqdm inquirer attrs")
            
        clog.info("Succesfully installed PyPO prerequisites.")
        clog.warning("Install CUDA manually to enable PyPO on GPU.")
    
    if args.clean or args.full_monty:
        try:
            clog.info("Cleaning build directory...")
            dir_build = os.path.join(os.getcwd(), "out", "build")
            shutil.rmtree(dir_build)
            clog.info("Succesfully cleaned build directory.")
        except:
            clog.warning("Nothing to clean.")

    if args.config or args.full_monty:
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
            clog.error("Could not configure PyPO. Is CMAKE installed?")

    if args.make or args.full_monty:
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
