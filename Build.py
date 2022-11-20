import sys
import os

def BuildPOPPy():
    pathToBuild = os.path.join("src", "C++")
    print(pathToBuild)
    # Parse command line input
    if sys.argv[-1] == "--enable-gpu": 
        try:
            print("Attempting to compile POPPy on GPU...")
            
            os.chdir(pathToBuild)
            os.system("make gpu")
        except:
            pass

if __name__ == "__main__":
	BuildPOPPy()
