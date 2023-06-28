import os
import shutil
import traceback
import argparse

##
# @file
# PyPO docs generator.
#
# For this script to work properly, you should have installed the docs prerequisites.
def GenerateDocs():
    docPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "docs")
    tutPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tutorials")
    demoPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "demos")
    doxyPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "doxy")
    
    parser = argparse.ArgumentParser(description="options for generating docs")
    parser.add_argument("-t", "--tutorials", help="generate docs from tutorials", action="store_true")
    parser.add_argument("-g", "--guitutorials", help="generate GUI tutorial docs", action="store_true")
    parser.add_argument("-d", "--demos", help="generate docs from demos", action="store_true")
    args = parser.parse_args()
    
    try:
        try:
            shutil.rmtree(docPath)
        except Exception as err:
            print(traceback.format_exc())

        if args.tutorials:
            # Convert regular tutorials to html format for inclusion in the documentation.
            for (dirpath, dirnames, filenames) in os.walk(tutPath):
                destPath = os.path.join(docPath, "tutorials")
                os.makedirs(destPath)
                for file in filenames:
                    if file.split('.')[1] != "ipynb":
                        continue

                    _path = os.path.join(tutPath, file)
                    htmlPath = os.path.join(tutPath, f"{file.split('.')[0]}.html")
                    html_destPath = os.path.join(destPath, f"{file.split('.')[0]}.html")

                    os.system(f"jupyter nbconvert --to html --template lab --theme dark {_path}")
                    os.rename(htmlPath, html_destPath)
                
                break
       
        if args.demos:
            # Convert demos to html format for inclusion in the documentation.
            for (dirpath, dirnames, filenames) in os.walk(demoPath):
                destPath = os.path.join(docPath, "demos")
                os.makedirs(destPath)
                for file in filenames:
                    if file.split('.')[1] != "ipynb":
                        continue

                    _path = os.path.join(demoPath, file)
                    htmlPath = os.path.join(demoPath, f"{file.split('.')[0]}.html")
                    html_destPath = os.path.join(destPath, f"{file.split('.')[0]}.html")

                    os.system(f"jupyter nbconvert --to html --template lab --theme dark {_path}")
                    os.rename(htmlPath, html_destPath)
                
                break
        
        if args.guitutorials:
            # Convert md for GUI tutorials to html and copy to /docs
            destPath = os.path.join(docPath, "Gui")
            guitutPath = os.path.join(tutPath, "Gui")


            file_md = []
            for (dirpath, dirnames, filenames) in os.walk(guitutPath):
                for file in filenames:
                    if file.split(".")[1] == "md":
                        file_md.append(file)
                break
            
            for file in file_md:
                filename = file.split(".")[0]
                filename_html = filename + ".html"

                os.system(f"pandoc {os.path.join(guitutPath, file)} -t html -o {os.path.join(guitutPath, filename_html)}")
       
                if os.path.isdir(destPath):
                    shutil.rmtree(destPath)

                shutil.copytree(guitutPath, destPath, ignore=shutil.ignore_patterns("*.md"))

        os.system(f"doxygen {os.path.join(doxyPath, 'Doxyfile')}")
        
        filelistPath = os.path.join(docPath, "files.html")
        with open(filelistPath, 'r') as file :
            filedata = file.read()
            filedata = filedata.replace('File List', 'Full Software Documentation')
            filedata = filedata.replace("Here is a list of all documented files with brief descriptions:",
                                        "Here is a list containing the full software documentation. The structure of this page reflects the source code hierarchy.")
        with open(filelistPath, 'w') as file:
            file.write(filedata)
        
    except Exception as err:
        print(traceback.format_exc())
    
if __name__ == "__main__":
    GenerateDocs()
