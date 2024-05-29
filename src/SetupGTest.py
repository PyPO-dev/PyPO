"""!
@file
File for building, making and running the C++ unittests.
"""

import os

import shutil
import traceback
import argparse

def SetupGTest():
    """
    Create build environment for C++/CUDA unittest using Google test.
    """
    
    parser = argparse.ArgumentParser(description="setup for making C++/CUDA unittests")
    parser.add_argument("-b", "--build", help="generate build files", action="store_true")
    parser.add_argument("-m", "--make", help="make from build files", action="store_true")
    parser.add_argument("-c", "--clean", help="clean build files", action="store_true")
    parser.add_argument("-r", "--run", help="run all unittests", action="store_true")
    args = parser.parse_args()

    buildPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "build")
    binPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bin")
    

    if args.clean:
        try:
            shutil.rmtree(buildPath)
        except Exception as err:
            print(traceback.format_exc())
        
        try:
            shutil.rmtree(binPath)
        except Exception as err:
            print(traceback.format_exc())

    if args.build:
        try:
            cwd = os.getcwd()
            try:
                os.mkdir(buildPath)
            
            except Exception as err:
                print(traceback.format_exc())
                
            os.chdir(buildPath)
            os.system(f"cmake ..")
            os.chdir(cwd)
        
        except Exception as err:
            print(traceback.format_exc())

    if args.make:
        try:
            cwd = os.getcwd()
            os.chdir(buildPath)
            os.system("make")
            os.chdir(cwd)
        except Exception as err:
            print(traceback.format_exc())

    if args.run:
        try:
            exePath = os.path.join(binPath, "runTests")
            os.system(exePath)
        except Exception as err:
            print(traceback.format_exc())
        
        try:
            exePath = os.path.join(binPath, "runCUDATests")
            os.system(exePath)
        except Exception as err:
            print(traceback.format_exc())


if __name__ == "__main__":
    SetupGTest()
