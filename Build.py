#!/usr/bin/env python3
import sys
import os
import shutil
import platform
import argparse
import traceback

##
# @file
# PyPO build and utilities script.
#
# This script contains options to build distributions for PyPO and is mostly meant for convenience when developing.
# Also, documentation can be built through this script. 
# However, for this function to work properly, you should also have cloned the PyPO-docs and PyPO-tutorials repos into the same root as PyPO.
# The unittests can also be run from this script.
# For an overview of the possible flags, run in a terminal:
#```
# python Build.py --help
#```
def BuildPyPO():
    parser = argparse.ArgumentParser(description="documenting and test interface script for PyPO")
    parser.add_argument("-s", "--sdist", help="generate a PyPO source distribution into dist folder", action="store_true")
    parser.add_argument("-b", "--bdist", help="generate a PyPO binary wheel into dist folder. EXPERIMENTAL.", action="store_true")
    parser.add_argument("-u", "--upload", help="upload dist folder to test-pypi using twine. Will ask for username and password", action="store_true")
    parser.add_argument("-d", "--docs", help="generate PyPO documentation with doxygen", action="store_true")
    parser.add_argument("-t", "--test", help="run PyPO unittests", action="store_true")
    args = parser.parse_args()

    if args.sdist:
        os.system("python3 setup.py sdist")

    if args.bdist:
        os.system("python3 setup.py bdist_wheel")

    if args.upload:
        os.system("twine upload --repository testpypi dist/*")

    if args.docs:
        doc_path = os.path.join("..", "PyPO-docs", "docs")
        tut_path = os.path.join("..", "PyPO-tutorials", "tutorials")

        try:
            try:
                shutil.rmtree("docs")
            except Exception as err:
                print(traceback.format_exc())
            
            try:
                shutil.rmtree(doc_path)
            except Exception as err:
                print(traceback.format_exc())
            
            os.system(f"doxygen {os.path.join('doxy', 'Doxyfile')}")
            
            # Convert regular tutorials to html format for inclusion in the documentation.
            for (dirpath, dirnames, filenames) in os.walk(tut_path):
                dest_path = os.path.join("docs", "tutorials")
                os.mkdir(dest_path)
                for file in filenames:
                    if file.split('.')[1] != "ipynb":
                        continue

                    _path = os.path.join(tut_path, file)
                    html_path = os.path.join(tut_path, f"{file.split('.')[0]}.html")
                    html_dest_path = os.path.join(dest_path, f"{file.split('.')[0]}.html")

                    os.system(f"jupyter nbconvert --to html --template lab --theme dark {_path}")
                    os.rename(html_path, html_dest_path)
                
                break
            
            # Convert md for GUI tutorials to html and copy to /docs
            guitut_path = os.path.join(tut_path, "Gui")
            
            file_md = []
            for (dirpath, dirnames, filenames) in os.walk(guitut_path):
                for file in filenames:
                    if file.split(".")[1] == "md":
                        file_md.append(file)
                break
            
            for file in file_md:
                filename = file.split(".")[0]
                filename_html = filename + ".html"

                os.system(f"pandoc {os.path.join(guitut_path, file)} -t html -o {os.path.join(guitut_path, filename_html)}")

            dest_path = os.path.join("docs", "Gui")
            shutil.copytree(guitut_path, dest_path, ignore=shutil.ignore_patterns("*.md"))

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
            
            shutil.move("docs", doc_path)
        
        except Exception as err:
            print(traceback.format_exc())
    
    if args.test:
        try:
            dir_tests = os.path.join(os.getcwd(), "tests")

            os.system(f"nose2 -v")

        except Exception as err:
            print(err)

 
if __name__ == "__main__":
	BuildPyPO()
