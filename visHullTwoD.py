from shapely.geometry import Polygon, LineString
import matplotlib.pyplot as plt
import numpy as np
import random
import math

EQUAL_THRESHOLD = 0.0001 # Threshold for considering certain fp numbers equal below.

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
        self.length = np.linalg.norm(self.p1 - self.p0)
        
        # Normalized direction vec from p0 to p1
        self.dir = (self.p1 - self.p0)/self.length
        
        
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
        # TECHNICALLY, they could infinitely intersect, i.e. be the same line
        # But for now, I'm just going to treat it as false.
        if abs(dy*ex - ey*dx) < EQUAL_THRESHOLD:
            return MyIntersection(False, 0, 0, (0,0)) 
        
        # If dx is 0, we need to switch x and y.
        # Otherwise, we'd be dividing by 0 later on.
        # This change in coordinates won't affect s or t.
        if (abs(dx) < EQUAL_THRESHOLD):
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
        self.cwList = []

        self.lines = []
        
        
        # In addition to keeping track of individual polygons,
        # we also keep track of ALL vertices in the scene.
        self.vertices = np.empty((0, 2))
        
        # Boundaries for the scene.
        self.minX = math.inf
        self.maxX = -math.inf
        
        self.minY = math.inf
        self.maxY = -math.inf
        
    def whichCaseIsIt(self):
        return "Still need to implement this."
        # If the two vertices form an edge, then it's the first case.
        
        
        # Otherwise, need to determine which side of the line the two vertices "triangles" are on.
        # I'm going to use the inward-pointing bisector of each vertex's angle represent the direction pointing "inside" the triangle from the vertex.
        # The reason for using the bisector, rather than just one of the edges, is because
        # it is possible for one of the edges to lie on the line, but the bisector
        # never will.
        
        # Then, we will consider a local coordinate frame where the free line is the up vector
        # From this, the "right" vector will be [y, -x]
        # The matrix to bring vectors into this local coord frame will be:
        # | y  -x |
        # | x   y | 
        
        # We convert the bisectors into this coord frame and see if their x values have the same sign.
        
    def addPolygon(self, pts):
        newVertices = np.array(pts, dtype=np.float64)
        #newVertices[:, 0] = newVertices[:, 0] - 2.33

        self.polygons.append(Polygon(pts)) #newVertices.tolist()))
        
        # Find out whether the polygon is clockwise or counterclockwise.
        # Will be using the technique given in StackOverflow user Beta's answer
        # to user Stécy's question "How to determine if a list of polygon points are in clockwise order?"
        # asked on 2009-07-22. Link: https://stackoverflow.com/a/1165943/11295586
        # An explanation is given at: https://web.archive.org/web/20200812125342/https://www.element84.com/blog/determining-the-winding-of-a-polygon-given-as-a-set-of-ordered-points
        # Basically, sum up (x2 − x1)(y2 + y1) over all edges
        # Curve is clockwise iff sum is positive.
        
        xy1s = newVertices
        xy2s = np.roll(xy1s, -1, axis=0) # Shift vertices by 1 index
        terms = (xy2s[:, 0] - xy1s[:, 0])*(xy2s[:, 1] + xy1s[:, 1])
        twiceArea = terms.sum() # As described in the links above, the sum is twice the area
        self.cwList.append( (twiceArea > 0) )
        
        
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
        
    def calcFreeLines(self):
        for i in range(len(self.vertices)):
            for j in range(len(self.vertices)):
                if i == j:
                    continue
                candidate = MyLine(self.vertices[i], self.vertices[j], False)
                intersectsObj = False
                
                polygonCount = 0
                while polygonCount < len(self.polygons) and not intersectsObj:
                    obj = self.polygons[polygonCount]
                    pts = obj.exterior.coords
                    numPts = len(pts)
                    edgeNum = 0
                    # pts, i.e. obj.exterior.coords, is organized where the 1st vertex is repeated at the end.
                    # Therefore, for the edge between the 1st and last vertices,
                    # we don't need to cycle around with a (n + 1)%numPts or anything.
                    # The line between the 1st and last vertices is created using the 2nd-last and last array items.
                    while edgeNum < numPts-1 and not intersectsObj:
                        # Get the two vertices on either side of the line.
                        # np.array is used as a vector.
                        v0 = np.array(pts[edgeNum])
                        v1 = np.array(pts[edgeNum+1])
                        edgeLine = MyLine(v0, v1, True)
                        intersection = candidate.intersection(edgeLine)
                        intersection.meetS 
                        if intersection.doMeet and intersection.meetS > -EQUAL_THRESHOLD and intersection.meetS < edgeLine.length + EQUAL_THRESHOLD:
                            # If the lines intersect, the line and edge/segment probably do...
                            intersectsThisTime = True
                            # ...but we should test and rule out non-transversal intersections
                            # Infinite intersections are already "discarded" by the intersection() function.
                            # But we need to rule out intersections with a vertex that do not pierce the shape,
                            # because these are fine (in fact, they are REQUIRED for the algorithm).
                            # We first deal with the line intersecting the vertex at the start of its edge.
                            if (abs(intersection.meetS) < EQUAL_THRESHOLD):
                                # Get the vertex before v0 and v1
                                # numPts-2 is justified by pts structure described above.
                                prevIndex = (edgeNum - 1) if edgeNum > 0 else numPts - 2
                                vPrev = np.array(pts[prevIndex])
                                # Test if candidate.dir is between both edge dirs going AWAY from v0
                                intersectsThisTime = self.isLineInsideEdgeAngle(vPrev - v0, edgeLine.dir, candidate.dir)
                            # Same idea, but for the case where the intersection is at
                            # the other side of the edge, closer to v1
                            elif (abs(intersection.meetS - edgeLine.length) < EQUAL_THRESHOLD):
                                # Get the vertex after v1
                                # Logic here is justified by pts structure described above.
                                nextIndex = (edgeNum + 2) if (edgeNum + 2) < numPts else 1
                                vNext = np.array(pts[nextIndex])
                                # Test if candidate.dir is between both edge dirs going AWAY from v1
                                intersectsThisTime = self.isLineInsideEdgeAngle(vNext - v1, -edgeLine.dir, candidate.dir)
                            
                            intersectsObj = (intersectsObj or intersectsThisTime)
                            
                        edgeNum += 1
                    polygonCount += 1
                if not intersectsObj:
                    self.addLine(self.vertices[i], self.vertices[j])
                    
                    
    def isLineInsideEdgeAngle(self, dir0, dir1, dirToTest):
        # Make sure both dirs are normalized
        length0 = np.linalg.norm(dir0)
        length1 = np.linalg.norm(dir1)
        dir0 = dir0 / length0
        dir1 = dir1 / length1
        
        
        # Get the line bisecting the vertex's angle
        unnormedBisect = dir0 + dir1
        bisector = unnormedBisect/np.linalg.norm(unnormedBisect)
        
        # If the dot product of candidate's dir with bisector is less than that
        # of one of the 2 edges of centre vert, then it doesn't go into the polygon
        dotThresh = abs(np.dot(bisector, dir0))
        testDot =  abs(np.dot(bisector, dirToTest))
        if testDot <= dotThresh:
            return False
        return True
    
    # Take in vertices v0, v1, v2 and whether mesh is counter-clockwise (ccw).
    # Output whether v1 is a concave vertex.
    def isVertexConcave(self, v0, v1, v2, cw):
        
        # Construct a local coordinate frame.
        # The "up" vector will be v2 - v1.
        # From this, the "right" vector will be [y, -x]
        # The matrix to bring vectors into this local coord frame will be:
        # | y  -x |
        # | x   y | 
        up = v2 - v1
        changeBasis = np.array([
            [up[1], -up[0]],
            [up[0], up[1]]
        ])

        # Convert v0 - v1 into this local coordinate frame via matrix mult.        
        backVecGlobal = v0 - v1
        backVec = changeBasis @ backVecGlobal
        
        # If the x-component of this is == 0, just treat it as convex I guess
        # If the x-component is negative and the polygon is clockwise, then concave
        # Alternatively, if the x-component is positive and the polygon is ccw, then concave.
        # Else, it is convex.
        if (backVec[0] > 0 and not cw) or (backVec[0] < 0 and cw):
            return True
        return False
        
        
        
        
    
    def drawScene(self):
        print("cwList:", self.cwList)
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
            borderX = 0.1*(self.maxX - self.minX)
            borderY = 0.1*(self.maxY - self.minY)
            forwardXHit = self.maxX + borderX
            forwardYHit = self.maxY + borderY
            backwardXHit = self.minX - borderX
            backwardYHit = self.minY -borderY
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
            
        convex = []
        concave = []
        for i in range(len(self.polygons)):
            cw = self.cwList[i]
            
            pVerts = np.array(self.polygons[i].exterior)[:-1, :]
            nextVerts = np.roll(pVerts, -1, axis=0)
            prevVerts = np.roll(pVerts, 1, axis=0)
            for j in range(pVerts.shape[0]):
                if self.isVertexConcave(prevVerts[j], pVerts[j], nextVerts[j], cw):
                    concave.append(pVerts[j])
                else:
                    convex.append(pVerts[j])
        npConvex = np.array(convex)
        npConcave = np.array(concave)
        if npConvex.shape[0] > 0:
            plt.plot(npConvex[:, 0], npConvex[:, 1], 'bo')
        if npConcave.shape[0] > 0:
            plt.plot(npConcave[:, 0], npConcave[:, 1], 'go')
        plt.show()
    
world0 = Scene()
world1 = Scene()
world2 = Scene()

# These are the tris from Petitjean's diagram
polygon1 = [(0, 0), (2.25, 0.5), (1.25, 2.3)] # [(0,3),(1,1),(3,0),(4,0),(3,4)]
polygon2 = [(1.15, 3.15), (4, 4), (0.9, 5.25)] # [(1,4),(2,5),(2,1),(1,3)]
polygon3 = [(3, 0.7), (4.85, 1.75), (4.85, 3.4)]

world0.addPolygon(polygon1)
world0.addPolygon(polygon2)
world0.addPolygon(polygon3)

polygon1 = [(0, 0), (5, 0), (5, 5), (4, 5), (4, 3), (1, 3), (1, 5), (0, 5)]
world1.addPolygon(polygon1)

polygon1 = [(0, 0), (5, 0), (5, 3), (4, 3), (4, 5), (1, 5), (1, 3), (0, 3)]
world2.addPolygon(polygon1)


#world.addLine((0, 2.5), (3, 2.5))

world0.calcFreeLines()
world0.drawScene()

world1.calcFreeLines()
world1.drawScene()

world2.calcFreeLines()
world2.drawScene()

reminders = [
     "isLineInsideEdgeAngle and parallelism and concavity!!!\n\nMake sure that discared if intersecting NEAR a concave vertex EVEN IF said vertex is NOT one of the TWO that made up the line!",
     "MULTIPLE LINES MAY BE IDENTICAL AT THE END! DOES THIS NEED TO BE DEALT WITH???",
     "Right now, 2x as many lines as necessary are being created, because all i and j are considered for vertices nested loop.\n\nShould instead to i from 0..n, j from (i+1)..n"
     ]

for reminder in reminders:
    sep = "==========="
    print("\n" + sep + "\n" + reminder + "\n" + sep + "\n")