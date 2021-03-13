from shapely.geometry import Polygon, LineString
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from enum import Enum


EQUAL_THRESHOLD = 0.0001 # Threshold for considering certain fp numbers equal below.




def findIntersection(segments):
    q = Something() # Event queue initialization
    
    startVertex = None
    
    # Insert segment endpoints into q.
        # Base it on y-coord. Tie break on x-coord.
    # Items in here should contain:
        # The endpoint coords or index
        # An enum TYPE that is either UPPER, LOWER, or INTERSECTION
        # An array for segments[] that connect to the intersection and/or endpoint.
        
    
    
    t = SomethingElse() # status structure T
    
    while not q.isEmpty():
        p = q.pop()
        
        
        if p.eventType == EventType.Upper:
            s = p.upperSegments[0]
            
            # Use p to insert s into t
            t.insert(p, s)
            
            # If s intersects left neighbour in t:
                # Insert the intersection pt into q
            # If s intersects right neighbour in t:
                # Insert the intersection into q
        
        elif p.eventType == EventType.Lower:
            s = p.lowerSegments[0]
            
            sLeft = t.left(s)
            sRight = t.right(s)
            
            # Use p to delete s from t
            
            # If sLeft != None and sRight!= None and sLeft intersects sRight:
                # If the intersection is below the sweep line (i.e., p's y pos):
                    # Insert the intersection into q
        
        else: # It's an intersection
            
            # Reverse order of p.segments in t
            # (Need to do something else if intersect @ an endpoint)
            
            
            # Out of these, get sLeftmost and sRightmost
            
            sLeftmostNeighbour = t.left(sLeftmost)
            sRightmostNeighbour = t.right(sRightmost)
            
            # if sLeftmostNeighbour != None and sLeftmostNeighbour intersects sLeftmost:
                # If the intersection is below the sweep line (i.e., p's y pos):
                    # Insert the intersection into q
                    
            # if sRightmostNeighbour != None and sRightmostNeighbour intersects sRightmost:
                # If the intersection is below the sweep line (i.e., p's y pos):
                    # Insert the intersection into q
            
            # DO SOMETHING WITH THE INTERSECTION POINT!
            # . . .
            # IT'S A VERTEX FOR SOME NEW GRAPH/MESH!
            
            if startVertex == None and p.

        

class SegmentType(Enum):
    A = 1
    B = 2
    C = 3
    
class NodeBST:
    __init__(self)

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

    
class MyActiveLine(MyLine):
    def __init__(self, p0, p1, isSegment, activeType):
        super().__init__(p0, p1, isSegment)
        self.activeType = activeType


class Scene:
        
    def __init__(self):
        self.polygons = []
        self.cwList = []
        
        # In addition to keeping track of individual polygons,
        # we also keep track of ALL vertices in the scene.
        self.vertices = np.empty((0, 2))
        # These can maybe be combined with self.vertices into a dataframe or something
        self.prevIndices = np.empty(0, dtype=np.int)
        self.nextIndices = np.empty(0, dtype=np.int)
        self.polygonIndices = np.empty(0, dtype=np.int)
        
        # May also combine these into a single thing
        self.lines = []
        self.lineTypes = []
        
        # Boundaries for the scene.
        self.minX = math.inf
        self.maxX = -math.inf
        
        self.minY = math.inf
        self.maxY = -math.inf
        
    def getLineType(self, index0, index1):
        # If the two vertices form an edge, then it's the first case.
        if self.prevIndices[index0] ==  index1 or self.nextIndices[index0] == index1:
            return SegmentType.A
        
        # Otherwise, need to determine which side of the line the two vertices "triangles" are on.
        # I'm going to use the inward-pointing bisector of each vertex's angle represent the direction pointing "inside" the triangle from the vertex.
        # The reason for using the bisector, rather than just one of the edges, is because
        # it is possible for one of the edges to lie on the line, but the bisector
        # never will.
        v00 = self.vertices[self.prevIndices[index0]]
        v01 = self.vertices[index0]
        v02 = self.vertices[self.nextIndices[index0]]
        
        v10 = self.vertices[self.prevIndices[index1]]
        v11 = self.vertices[index1]
        v12 = self.vertices[self.nextIndices[index1]]
        
        dir00 = v00 - v01
        dir01 = v02 - v01
        # Make sure both dirs are normalized
        length00 = np.linalg.norm(dir00)
        length01 = np.linalg.norm(dir01)
        dir00 = dir00 / length00
        dir01 = dir01 / length01
        
        dir10 = v10 - v11
        dir11 = v12 - v11
        # Make sure both dirs are normalized
        length10 = np.linalg.norm(dir10)
        length11 = np.linalg.norm(dir11)
        dir10 = dir10 / length10
        dir11 = dir11 / length11
        
        
        # Get the line bisecting the vertex's angle
        unnormedBisect0 = dir00 + dir01
        bisector0 = unnormedBisect0/np.linalg.norm(unnormedBisect0)
        
        
        # Get the line bisecting the vertex's angle
        unnormedBisect1 = dir10 + dir11
        bisector1 = unnormedBisect1/np.linalg.norm(unnormedBisect1)

        
        # Then, we will consider a local coordinate frame where the free line is the up vector
        # From this, the "right" vector will be [y, -x]
        # The matrix to bring vectors into this local coord frame will be:
        # | y  -x |
        # | x   y | 
        up = v11 - v01
        changeBasis = np.array([
            [up[1], -up[0]],
            [up[0], up[1]]
        ])
        
        # We convert the bisectors into this coord frame and see if their x values have the same sign.
        localBisector0 = changeBasis @ bisector0
        localBisector1 = changeBasis @ bisector1
        
        if (localBisector0[0] * localBisector1[0] > 0):
            return SegmentType.B
        return SegmentType.C
        
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
            
        # Update the prevIndices and newIndices lists
        startIndex = self.vertices.shape[0]
        newPrevIndices = np.roll(np.arange(startIndex, startIndex + newVertices.shape[0]), 1, axis=0)
        newNextIndices = np.roll(np.arange(startIndex, startIndex + newVertices.shape[0]), -1, axis=0)
        self.prevIndices = np.concatenate((self.prevIndices, newPrevIndices))
        self.nextIndices = np.concatenate((self.nextIndices, newNextIndices))
        
        # Update the polygonIndices list
        self.polygonIndices = np.concatenate((self.polygonIndices, np.full(newVertices.shape[0], len(self.polygons) - 1)))
        

        # Once that is done, update the vertices list.
        self.vertices = np.concatenate((self.vertices, newVertices))
        
    def addLine(self, p0, p1):
        # At this time, I'm assuming all lines added to the scene are 
        # full "lines", not "segments".
        self.lines.append(MyLine(p0, p1, False))
        
    def addActiveLine(self, p0, p1, lineType):
        # At this time, I'm assuming all lines added to the scene are 
        # full "lines", not "segments".
        self.lines.append(MyActiveLine(p0, p1, True, lineType))
        
    def calcFreeLines(self):
        for i in range(len(self.vertices)):
            if self.isVertexConcave(i):
                continue
            for j in range(i+1, len(self.vertices)):
                if self.isVertexConcave(j):
                    continue
                candidate = MyLine(self.vertices[i], self.vertices[j], False)
                intersectsObj = False
                
                polygonCount = 0
                vertCount = 0 # Vertex index of start of current edge analyzed
                while polygonCount < len(self.polygons) and not intersectsObj:
                    obj = self.polygons[polygonCount]
                    pts = obj.exterior.coords
                    numPts = len(pts)
                    edgeNum = 0 # Like vertCount, but just for this polygon rather than whole scene.
                    
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
                            # We first deal with the line intersecting the vertex at the start of its edge, at v0.
                            if (abs(intersection.meetS) < EQUAL_THRESHOLD):
                                # Test if candidate.dir is between both edge dirs going AWAY from v0
                                intersectsThisTime = self.isLineInsideEdgeAngle(vertCount, candidate.dir)
                            # Same idea, but for the case where the intersection is at
                            # the other side of the edge, closer to v1
                            elif (abs(intersection.meetS - edgeLine.length) < EQUAL_THRESHOLD):
                                # Test if candidate.dir is between both edge dirs going AWAY from v1
                                intersectsThisTime = self.isLineInsideEdgeAngle(self.nextIndices[vertCount], candidate.dir)
                            
                           
                            intersectsObj = (intersectsObj or intersectsThisTime)
                            
                        edgeNum += 1
                        vertCount += 1
                    polygonCount += 1
                if not intersectsObj:
                    lineType = self.getLineType(i, j)
                    self.addActiveLine(self.vertices[i], self.vertices[j], lineType)
                    
                    
    def isLineInsideEdgeAngle(self, vertIndex, dirToTest):
        if self.isVertexConcave(vertIndex):
            return True

        v0 = self.vertices[self.prevIndices[vertIndex]]
        v1 = self.vertices[vertIndex]
        v2 = self.vertices[self.nextIndices[vertIndex]]
        
        dir0 = v0 - v1
        dir1 = v2 - v1
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
    def isVertexConcave(self, vertIndex):
        v0 = self.vertices[self.prevIndices[vertIndex]]
        v1 = self.vertices[vertIndex]
        v2 = self.vertices[self.nextIndices[vertIndex]]
        cw = self.cwList[self.polygonIndices[vertIndex]]
        
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
            plt.fill(x,y, "#A0A0A0") # light grey fill
            plt.plot(x,y, "#505050") # dark grey edges/outline
        
        for ln in self.lines:
            newP0 = ln.p0
            newP1 = ln.p1
            if not ln.isSegment:
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
            
            colString = "g"
            if type(ln) is MyActiveLine:
                if ln.activeType == SegmentType.A:
                    colString = "r"
                elif ln.activeType == SegmentType.B:
                    colString = "b"
            
            # Line is drawn
            plt.plot([newP0[0], newP1[0]], [newP0[1], newP1[1]], colString)
            
        convex = []
        concave = []
        for i in range(self.vertices.shape[0]):
            if self.isVertexConcave(i):
                concave.append(self.vertices[i])
            else:
                convex.append(self.vertices[i])
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

#%%
reminders = [
     "isLineInsideEdgeAngle and parallelism and concavity!!!\n\nMake sure that discared if intersecting NEAR a concave vertex EVEN IF said vertex is NOT one of the TWO that made up the line!",
     "MULTIPLE LINES MAY BE IDENTICAL AT THE END! DOES THIS NEED TO BE DEALT WITH???",
     "Right now, 2x as many lines as necessary are being created, because all i and j are considered for vertices nested loop.\n\nShould instead to i from 0..n, j from (i+1)..n",
     "REMOVE SHAPELY? NOT REALLY USING IT THAT MUCH!",
     "Maybe something like sweep can be done for line-segment intersections?\n\nReplace line with segment touching bounding box borders?",
     "Pruning of lines that intersect obj at CONTACT verts.",
     "Pruning of segments outside convex hull."
     "To handle 'unions', make n x n mat of 0s, check for cancelling, etc."
     ]

for reminder in reminders:
    sep = "==========="
    print("\n" + sep + "\n" + reminder + "\n" + sep + "\n")
    
# temp1 = np.array(polygon1)
# temp2 = np.roll(np.arange(5, 5+temp1.shape[0]), -1, axis=0)
# temp3 = np.roll(np.arange(5, 5+temp1.shape[0]), 1, axis=0)
# ...
# testVertices = np.empty((temp1.shape[0]), dtype=[("x", np.float64), ("y", np.float64), ("prevIndex", np.int), ("nextIndex", np.int)])
# testVertices["x"] = temp1[:, 0]
# testVertices["y"] = temp1[:, 1]
# testVertices["prevIndex"] = temp2
# testVertices["nextIndex"] = temp3
# testVertices[3]["x"]
# testVertices[["x", "y"]]
# testVertices[0][["x", "y"]]
# Does not work: np.dot(testVertices[2][["x", "y"]], testVertices[3][["x", "y"]])
# The error msg: Can't cast from structure to non-structure, except if the structure only has a single field.
