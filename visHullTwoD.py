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
        
        
        # Normalized direction vec from p0 to p1
        self.dir = (self.p1 - self.p0)/np.linalg.norm(self.p1 - self.p0)
        
        
    # Math here basically came from setting up an augmented matrix
    # for the 2D case of line intersection and solving it.
    # So commenting may not be the best here.
    def intersection (self, other):
        # Delta is the difference in "P0's" between lines.
        deltaX = self.p0[0] - other.p0[0]
        deltaY = self.p0[1] - other.p0[1]
        
        # d is the direction vector for self
        dx = self.dir[0]
        dy = self.dir[1]
        # e is the direction vector for the other line
        ex = other.dir[0]
        ey = other.dir[1]
        
        # If lines parallel, no intersection.
        if dy*ex == ey*dx:
            return MyIntersection(False, 0, 0, (0,0)) 
        
        # If dx is 0, we need to switch x and y.
        # Otherwise, we'd be dividing by 0 later on.
        # This change in coordinates won't affect s or t.
        if (dx == 0):
            deltaX, deltaY = deltaY, deltaX
            dx, dy = dy, dx
            ex, ey = ey, ex
            
        # Math checks out here when done on paper, solving augmented matrix.
        s = (dy*deltaX - dx*deltaY)/(dy*ex - ey*dx)
        t = -deltaX/dx + (ex/dx)*s
        
        # Return the struct-like object.
        return MyIntersection(True, t, s, self.p0 + t*self.dir)
        

class Scene:
        
    def __init__(self):
        self.polygons = []
        self.lines = []
        
        # In addition to keeping track of individual polygons,
        # we also keep track of ALL vertices in the scene.
        self.vertices = np.empty((0, 2))
        
        # Boundaries for the scene.
        self.minX = math.inf
        self.maxX = -math.inf
        
        self.minY = math.inf
        self.maxY = -math.inf
        
        
    def addPolygon(self, pts):
        self.polygons.append(Polygon(pts))
        newVertices = np.array(pts)
        
        # Separate the x and y values for the new vertices.
        xs = newVertices[:, 0]
        ys = newVertices[:, 1]
        
        # Get the min/max x & y for this polygon.
        newMinX = xs.min()
        newMaxX = xs.max()
        newMinY = ys.min()
        newMaxY = ys.max()
        
        # Update the world's min/max x & y if necessary.
        if newMinX < self. minX:
            self.minX = newMinX
        if newMaxX > self. maxX:
            self.maxX = newMaxX
        if newMinY < self. minY:
            self.minY = newMinY
        if newMaxY > self. maxY:
            self.maxY = newMaxY
            
        # Once that is done, update the vertices list.
        self.vertices = np.concatenate((self.vertices, newVertices))
        
    def addLine(self, p0, p1):
        # At this time, I'm assuming all lines added to the scene are 
        # full "lines", not "segments".
        self.lines.append(MyLine(p0, p1, False))
        
    
    def drawScene(self):
        # Plot all polygons.
        for obj in self.polygons:
            x,y = obj.exterior.xy
            plt.fill(x,y, "r") # red fill
            plt.plot(x,y, "b") # blue edges/outline
        
        for ln in self.lines:
            # We want the lines in the scene to be rendered 
            # such that they extend all the way to the scene's bounding box.
            # To calculate where intersections with said box occur,
            # the following calculations are done.
            # First, initializing maxes/mins to inf.
            tForwardX = math.inf
            tForwardY = math.inf
            tBackwardX = math.inf
            tBackwardY = math.inf
            # We don't know, in each direction, if we'll hit a vertical
            # or horizontal border first, so we have to test both
            # x and y. 
            forwardXHit = self.maxX
            forwardYHit = self.maxY
            backwardXHit = self.minX
            backwardYHit = self.minY
            # If the direction vector has a negative component,
            # then "forward" along it points to the min borders, not max ones.
            if ln.dir[0] < 0:
                forwardXHit, backwardXHit = backwardXHit, forwardXHit
            if ln.dir[1] < 0:
                forwardYHit, backwardYHit = backwardYHit, forwardYHit
                
            # If the direction vector is not vertical, see where it hits the x borders.
            if ln.dir[0] != 0:
                tForwardX = (forwardXHit - ln.p0[0])/ln.dir[0]
                tBackwardX = (ln.p0[0] - backwardXHit)/ln.dir[0]
            # If the direction vector is not horizontal, see where it hits the y borders.
            if ln.dir[1] != 0:
                tForwardY = (forwardYHit - ln.p0[1])/ln.dir[1]
                tBackwardY = (ln.p0[1] - backwardYHit)/ln.dir[1]
                
            # First hits get chosen.
            tForward = min(tForwardX, tForwardY)
            tBackward = -min(tBackwardX, tBackwardY)
            
            # Endpoints for the lines at these intersections created.
            newP0 = ln.p0 + tBackward*ln.dir
            newP1 = ln.p0 + tForward*ln.dir
            
            # Line is drawn
            plt.plot([newP0[0], newP1[0]], [newP0[1], newP1[1]], "g")
            
        intersections = np.empty((0,2))
        for ln in self.lines:
            for obj in self.polygons:
                pts = obj.exterior.coords
                numPts = len(pts)
                for i in range(numPts-1):
                    edgeLine = MyLine(pts[i], pts[(i+1)], True)
                    intersection = ln.intersection(edgeLine)
                    if intersection.doMeet:
                        intersections = np.append(intersections, [intersection.meetPt], axis=0)
                        
        plt.plot(intersections[:, 0], intersections[:, 1], 'bo')
        plt.show()

#def sortedPoints(pts):
#    retPts = []
    
world = Scene()

polygon1 = [(0,3),(1,1),(3,0),(4,0),(3,4)]

world.addPolygon(polygon1)

polygon2 = [(1,4),(2,5),(2,1),(1,3)]

world.addPolygon(polygon2)

world.addLine((0, 2.5), (3, 2.5))


world.drawScene()