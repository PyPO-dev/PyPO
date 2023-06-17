import sys
import os
import shutil
import platform
import argparse
import traceback

##
# @file
# PyPO docs generator.
#
# For this script to work properly, you should have installed the docs prerequisites.
def GenerateDocs():
        tut_path = "tutorials"

        try:
            try:
                shutil.rmtree("docs")
            except Exception as err:
                print(traceback.format_exc())

            # Convert regular tutorials to html format for inclusion in the documentation.
            for (dirpath, dirnames, filenames) in os.walk(tut_path):
                dest_path = os.path.join("docs", "tutorials")
                os.makedirs(dest_path)
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
            dest_path = os.path.join("docs", "Gui")
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

            
            os.system(f"doxygen {os.path.join('doxy', 'Doxyfile')}")
            shutil.copytree(guitut_path, dest_path, ignore=shutil.ignore_patterns("*.md"))
            
            filelist_path = os.path.join("docs", "files.html")
            with open(filelist_path, 'r') as file :
                filedata = file.read()
                filedata = filedata.replace('File List', 'Full Software Documentation')
                filedata = filedata.replace("Here is a list of all documented files with brief descriptions:",
                                            "Here is a list containing the full software documentation. The structure of this page reflects the source code hierarchy.")
            with open(filelist_path, 'w') as file:
                file.write(filedata)
            
        except Exception as err:
            print(traceback.format_exc())
    
if __name__ == "__main__":
    GenerateDocs()
