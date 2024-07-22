#!/bin/sh

rm -f *png
./plot2d.py rho.txt
#./plot2d.py vLat.txt
./plot2d.py vXi.txt
./plot2d.py vNu.txt
