
import os
from unittest.mock import patch
def ListFilesToTxt(dir,file,wildcard,recursion):
    patch="crowdhuman/images/val/"
    exts = wildcard.split(" ")
    for root, subdirs, files in os.walk(dir):
        for name in files:
            for ext in exts:
                if(name.endswith(ext)):
                    file.write(patch + name + "\n")
                    break
        if(not recursion):
            break
def Test():
 
  dir="/RMOT/dataset/Mask_data/JPEGImages/"
  outfile="binaries.txt"
  wildcard = ".jpg"
 
  file = open(outfile,"w")
  if not file:
    print ("cannot open the file %s for writing" % outfile)
  ListFilesToTxt(dir,file,wildcard, 0)
 
  file.close()
Test()