#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import argparse
import matplotlib.cm as cm

# ----------------------------------------------------------------------
# Function to parse input arguments
# ----------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(description = 'Plot advection model results')
    parser.add_argument('filein', metavar = 'filein', nargs = 1, \
                        help = 'input file')

    args = parser.parse_args()

    return args

def read_file(file):

    fpin = open(file, 'r')
    lines = fpin.readlines()
    aline = lines[0].split()
    nX = int(aline[0])
    nY = int(aline[1])

    nLines = len(lines)-1
    nTimes = int(nLines / nX)
    print(nX, nY, nLines, nTimes)
    
    values = np.zeros((nTimes, nX, nY))

    for iTime in range(nTimes):
        for i in range(nX):
            iL = iTime * nX + i + 1
            line = lines[iL]
            aline = line.split()
            v = np.array(aline).astype(float)
            values[iTime, i, :] = v

    return values

def plot_data(values, fileout, mini, maxi, lats = 0, lons = 0, vSave = 0):

    if (mini < 0):
        cmap = cm.bwr
    else:
        cmap = cm.plasma

    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    if (len(lats) < 2):
        ax.pcolor(v, vmin = mini, vmax = maxi, cmap=cmap)
    else:
        ax.scatter(lons, lats, c = values, vmin = mini, vmax = maxi, cmap=cmap)
        iHalf = int(len(lats[0,:])/2)
        ax2.plot(lons[:,iHalf], values[:,iHalf])
        if (len(vSave) > 2):
            ax2.plot(lons[:,iHalf], vSave[:,iHalf])

    fig.savefig(fileout)
    plt.close()
        

args = parse_args()
filein = args.filein
if (not np.isscalar(filein)):
    filein = filein[0]

values = read_file(filein)
lats = read_file('lat.txt') * 180.0 / np.pi
lons = read_file('lon.txt') * 180.0 / np.pi

n = len(values)
if (n > 20):
    nSkip = int(n / 10)
else:
    nSkip = 1

if (nSkip > 10):
    nSkip = 10

mini = np.min(values)
maxi = np.max(abs(values))

if (mini < 0):
    mini = -maxi

print(mini,maxi)

for i in range(0, n, nSkip):
    print("Plotting time : ", i, " of ", n)
    v = values[i]
    if (i == 0):
        vSave = v
    fileout = filein+'.%04d.png' % i
    plot_data(v, fileout, mini, maxi, lats = lats[0], lons = lons[0], vSave = vSave)
    


