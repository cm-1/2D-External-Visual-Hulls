from shapely.geometry import Polygon, LineString
import matplotlib.pyplot as plt
import numpy as np
import random
import math

# This is basically a struct for the line intersection algorithm to return.
class MyIntersection:
    def __init__(self, doMeet, meetT, meetS, meetPt):
        self.doMeet = doMeet
        self.meetT = meetT
        self.meetS = meetS
        self.meetPt = meetPt
        

class MyLine:
    # By taking x0 and x1 as arguments, can create both lines and segments with same object
    def __init__(self, p0, p1, isSegment):
        self.p0 = np.array(p0)
        self.p1 = np.array(p1)
        self.isSegment = isSegment
        
        self.dir = (self.p1 - self.p0)/np.linalg.norm(self.p1 - self.p0)
        
    def intersection (self, other):
        deltaX = self.p0[0] - other.p0[0]
        deltaY = self.p0[1] - other.p0[1]
        
        dx = self.dir[0]
        dy = self.dir[1]
        ex = other.dir[0]
        ey = other.dir[1]
        
        # If lines parallel, no intersection.
        if dy*ex == ey*dx:
            return MyIntersection(False, 0, 0, (0,0)) 
        
        # If dx is 0, we need to switch x and y
        # This change in coordinates won't affect s or t
        if (dx == 0):
            deltaX, deltaY = deltaY, deltaX
            dx, dy = dy, dx
            ex, ey = ey, ex
            
            
        s = (dy*deltaX - dx*deltaY)/(dy*ex - ey*dx)
        t = -deltaX/dx + (ex/dx)*s
        
        return MyIntersection(True, t, s, (deltaX + dx*t, deltaY + dy*t))
        

class Scene:
        
    def __init__(self):
        self.polygons = []
        self.lines = []
        self.vertices = np.empty((0, 2))
        
        self.minX = math.inf
        self.maxX = -math.inf
        
        self.minY = math.inf
        self.maxY = -math.inf
        
    def addPolygon(self, pts):
        self.polygons.append(Polygon(pts))
        newVertices = np.array(pts)
        xs = newVertices[:, 0]
        ys = newVertices[:, 1]
        newMinX = xs.min()
        newMaxX = xs.max()
        newMinY = ys.min()
        newMaxY = ys.max()
        
        if newMinX < self. minX:
            self.minX = newMinX
        if newMaxX > self. maxX:
            self.maxX = newMaxX
        if newMinY < self. minY:
            self.minY = newMinY
        if newMaxY > self. maxY:
            self.maxY = newMaxY
            
        print("self.vertices: ", self.vertices.shape)
        print("new vertices: " , newVertices.shape)
            
        self.vertices = np.concatenate((self.vertices, newVertices))
        
    def addLine(self, p0, p1):
        self.lines.append(MyLine(p0, p1, False))
        
    
    def drawScene(self):
        
        
        for obj in self.polygons:
            x,y = obj.exterior.xy
            plt.fill(x,y, "r")
            plt.plot(x,y, "b")
        
        for ln in self.lines:
            tForwardX = math.inf
            tForwardY = math.inf
            tBackwardX = -math.inf
            tBackwardY = -math.inf
            forwardXHit = self.maxX
            forwardYHit = self.maxY
            backwardXHit = self.minX
            backwardYHit = self.minY
            if ln.dir[0] < 0:
                forwardXHit, backwardXHit = backwardXHit, forwardXHit
            if ln.dir[1] < 0:
                forwardYHit, backwardYHit = backwardYHit, forwardYHit
                
            if ln.dir[0] != 0:
                tForwardX = (forwardXHit - ln.p0[0])/ln.dir[0]
                tBackwardX = (ln.p0[0] - backwardXHit)/ln.dir[0]
            if ln.dir[1] != 0:
                tForwardY = (forwardYHit - ln.p0[1])/ln.dir[1]
                tBackwardY = (ln.p0[1] - backwardYHit)/ln.dir[1]
                
            tForward = min(tForwardX, tForwardY)
            tBackward = -min(tBackwardX, tBackwardY)
            
            newP0 = ln.p0 + tBackward*ln.dir
            newP1 = ln.p0 + tForward*ln.dir
            
            plt.plot([newP0[0], newP1[0]], [newP0[1], newP1[1]], "g")
        
        #plt.plot([self.minX, self.maxX], [self.minY, self.maxY], 'bo')
        plt.show()

def sortedPoints(pts):
    retPts = []
    
world = Scene()

polygon1 = [(0,3),(1,1),(3,0),(4,0),(3,4)]

world.addPolygon(polygon1)

polygon2 = [(0,4),(1,5),(1,3),(0,3)]

world.addPolygon(polygon2)

world.addLine((0, 4.2), (3, 1.5))

#line1 = LineString([(0,2), (2, 3)])


#x,y = line1.xy
#plt.plot(x,y)

world.drawScene()