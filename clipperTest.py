import pyclipper
import matplotlib.pyplot as plt
import numpy as np

from visHullTwoD import MyPolygon

def drawPolygon(poly, colourInterior, colourBoundary):
    (x, y) = poly.getSeparateXYs()
    plt.fill(x,y, colourInterior)
    plt.plot(x,y, colourBoundary)
        
clipPts = [(1, 7), (3, 7), (5, 9), (4, 11), (4, 9), (1, 8), (2, 10), (0, 10)] #[(190, 210), (240, 210), (240, 130), (190, 130)]

polygon1 = [(0, 7), (2.25, 5), (1.25, 4), (5, 5)] # [(0, 0), (2.25, 0.5), (1.25, 2.3)] # [(0,3),(1,1),(3,0),(4,0),(3,4)]
polygon2 = [(1.15, -3.15), (4, -4), (2, -7), (0.9, -5.25)] #[(1.15, 3.15), (4, 4), (0.9, 5.25)] # [(1,4),(2,5),(2,1),(1,3)]
polygon3 = [(3, 1), (3, 0.0), (4.85, 0.75), (4.85, 2.4), (5,4)] #[(3, 0.7), (4.85, 1.75), (4.85, 3.4)]
polygon4 = [(-0.5, -1), (-0.5, 1.0), (0.5, 1), (0.5, -1)] #[(3, 0.7), (4.85, 1.75), (4.85, 3.4)]

subj = []
scaledSubj = []
for sPoints in [polygon1, polygon2, polygon3, polygon4]:
    sPoints = [(p[0], p[1] + 5) for p in sPoints]
    sPolygon = MyPolygon(sPoints)
    if not sPolygon.isClockwise():
        sPolygon.changeOrientation()
    drawPolygon(sPolygon,"#A0A0A0", "#505050")
    scaledSubj.append(pyclipper.scale_to_clipper(sPolygon.getCoords()))

clipPolygon = MyPolygon(clipPts)
if not clipPolygon.isClockwise():
    clipPolygon.changeOrientation()

drawPolygon(clipPolygon, "#A0000077", "#50000077")    
scaledClip = pyclipper.scale_to_clipper(clipPolygon.getCoords())

plt.show()

pc = pyclipper.Pyclipper()
pc.AddPath(scaledClip, pyclipper.PT_SUBJECT, True)
pc.AddPaths(scaledSubj, pyclipper.PT_CLIP, True)

pcModes = [pyclipper.PFT_EVENODD, pyclipper.PFT_NEGATIVE, pyclipper.PFT_NONZERO, pyclipper.PFT_POSITIVE]

pcModeSubj = pcModes[0]
pcModeClip = pcModes[0]

# CT_INTERSECTION
# CT_DIFFERENCE
solution0 = pc.Execute(pyclipper.CT_INTERSECTION, pcModeSubj, pcModeClip)
solution1 = pc.Execute(pyclipper.CT_DIFFERENCE, pcModeSubj, pcModeClip)

scaledSolution0 = []
for soln in solution0:
    scaledSoln = pyclipper.scale_from_clipper(soln)
    scaledSolution0.append(scaledSoln)
    drawPolygon(MyPolygon(scaledSoln), "r", "g")

plt.show()

scaledSolution1 = []
for soln in solution1:
    scaledSoln = pyclipper.scale_from_clipper(soln)
    scaledSolution1.append(scaledSoln)
    drawPolygon(MyPolygon(scaledSoln), "r", "g")

plt.show()

# solution (a list of paths): [[[240, 200], [190, 200], [190, 150], [240, 150]], [[200, 190], [230, 190], [215, 160]]]
