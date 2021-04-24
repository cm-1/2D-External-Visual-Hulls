import numpy as np
import math
from enum import Enum
from collections import deque
import heapq

from rbt import RedBlackTree # REALLY NEED TO REWRITE THIS!

EQUAL_THRESHOLD = 0.0001 # Threshold for considering certain fp numbers equal below.
EQUAL_DECIMAL_PLACES = -round(math.log(EQUAL_THRESHOLD, 10))

class EventType(Enum):
    LEFT = 0
    INTERSECTION = 1
    RIGHT = 2
 

class SweepLine:
    def __init__(self, x, y, eventType):
        self.x = x
        self.y = y
        self.eventType = eventType
            
class Vertex:
    def __init__(self, position, outgoingHalfEdge, vertexID = -1):
        self.position = np.array(position)
        self.outgoingHalfEdge = outgoingHalfEdge
        self.vertexID = vertexID
    '''def __eq__(self, other):
        return np.linalg.norm(self.position - other.position) < EQUAL_THRESHOLD and self.vertexID == other.vertexID and self.outgoingHalfEdge == other.outgoingHalfEdge
    def __hash__(self):
        return hash((self.position[0], self.position[1], self.vertexID))'''
        
class Face:
    def __init__(self, halfEdge, index):
        self.halfEdge = halfEdge
        self.index = index
        self.visualNumber = -1
    def getCoords(self):
        v = self.halfEdge.headVertex.position
        vertices = [v]
        origHE = self.halfEdge
        he = self.halfEdge.next
        while he != origHE:
            #if he.headVertex is None:
            #    break
            v = he.headVertex.position
            vertices.append(v)
            he = he.next
        return np.array(vertices)
    def __eq__(self, other):
        return self.index == other.index
    def __hash__(self):
        return hash(self.index)

class HalfEdge:
    def __init__(self, index, increasesLeft):
        self.headVertex = None
        self.next = None
        self.prev = None
        self.pair = None
        self.leftFace = None
        self.index = index
        self.increasesLeft = increasesLeft

class HalfEdgeStructure:
    def __init__(self):
        self.verts = []
        self.halfEdges = {}
        self.faces = {}
        self.halfEdgeIndexCounter = 0
        self.faceIndexCounter = 0
        self._exteriorFaceIndex = -1
        self.vertexOnShape = None

    def assignExteriorFace(self, halfEdge):
        if self._exteriorFaceIndex < 0:
            self._exteriorFaceIndex = self.faceIndexCounter
            self.createNewFace(halfEdge)
        halfEdge.leftFace = self.faces[self._exteriorFaceIndex]
        
    def removeFace(self, face):
        if face.index in self.faces:
            del self.faces[face.index]

    def removeHalfEdgePair(self, halfEdge):
        index0 = halfEdge.index
        index1 = halfEdge.pair.index
        if index0 > index1:
            index0, index1 = index1, index0
            
        del self.halfEdges[index1]
        del self.halfEdges[index0]

    def createNewPairOfHalfEdges(self, vertex, increasesLeft):
        newEdge = HalfEdge(self.halfEdgeIndexCounter, increasesLeft)
        newPair = HalfEdge(self.halfEdgeIndexCounter + 1, not increasesLeft)
        newPair.headVertex = vertex
        
        newEdge.pair = newPair
        newPair.pair = newEdge
        self.halfEdges[self.halfEdgeIndexCounter] = newEdge
        self.halfEdges[self.halfEdgeIndexCounter + 1] = newPair
        self.halfEdgeIndexCounter += 2
        return newEdge
    
    def createNewFace(self, halfEdge):
        newFace = Face(halfEdge, self.faceIndexCounter)
        self.faces[self.faceIndexCounter] = newFace
        self.faceIndexCounter += 1
        return newFace
    
    def isExteriorFace(self, face):
        return face.index == self._exteriorFaceIndex
        
class MySweepEvent:
    def __init__(self, x, y, segments, eventType, debugID = -1):
        self.x = x
        self.y = y
        self.segments = segments
        self.eventType = eventType
        self.debugID = debugID
        
    def __repr__(self):
        retStr = ""
        if self.eventType == EventType.INTERSECTION:
            retStr = "({0}, {1}), segIDS: {2}, {3}. dbID: {4}".format(self.x, self.y, [s.index for s in self.segments], "INTERSECTION", self.debugID) 
        else:
            eventStr = "LEFT" if self.eventType == EventType.LEFT else "RIGHT"
            seg = next(iter(self.segments))
            retStr = "{0}, segID: {1}, {2}".format(str(seg), seg.index, eventStr)
        return retStr

    def __eq__(self, other):
        xEqual = abs(self.x - other.x) < EQUAL_THRESHOLD
        yEqual = abs(self.y - other.y) < EQUAL_THRESHOLD
        typesEqual = self.eventType == other.eventType
        return xEqual and yEqual and typesEqual

    def __lt__(self, other):
        if self.__eq__(other):
            return False
        
        retVal = False
        if self.x < other.x - EQUAL_THRESHOLD:
            retVal = True
        elif abs(self.x - other.x) < EQUAL_THRESHOLD:
            if self.y < other.y - EQUAL_THRESHOLD:
                retVal = True
            elif abs(self.y - other.y) < EQUAL_THRESHOLD:
                if self.eventType.value < other.eventType.value:
                    retVal = True
        return retVal
    
    def merge(self, other):
        self.segments = self.segments.union(other.segments)
        return self
    

            
        
class SegmentType(Enum):
    A = 1 # The case where the segment is an edge of the polygon
    B = 2 # The case where the two "tris" are on the same side.
    C = 3 # The case where the two "tris" are on opposite sides.
    D = 4 # A mixed case caused by the "union" scenario of verts on same line
    


# This is basically a struct for the line intersection algorithm to return.
class MyIntersection:
    # meetS is for the first intersection
    # meetT is for the second intersection
    def __init__(self, doMeet, meetS, meetT, meetPt):
        self.doMeet = doMeet
        self.meetS = meetS
        self.meetT = meetT
        self.meetPt = meetPt
        

class MyLine:
    # By taking x0 and x1 as arguments, can create both lines and segments with same object
    def __init__(self, p0, p1, isSegment):
        self.p0 = np.array(p0)
        self.p1 = np.array(p1)
        self.isSegment = isSegment
        self.length = np.linalg.norm(self.p1 - self.p0)
        
        self.isVertical = abs(p0[0] - p1[0]) < EQUAL_THRESHOLD
        self.m = None
        self.b = None
        if not self.isVertical:
            self.m = (p1[1] - p0[1])/(p1[0] - p0[0])
            self.b = p0[1] - self.m * p0[0]
        
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
        
        # If lines parallel, there is no (normal) intersection.
        # If they are both infinite lines, they MIGHT infinitely intersect.
        # But in this case, it'll be treated as "false".
        # If both lines are segments, they MIGHT intersect at just one point.
        # While it is technically possible for line segments to intersect
        # at an endpoint and also intersect infinitely, that will never
        # happen in the case this function is applied to, so I'm
        # only going to check if exactly one pair of endpoints are equal.
        if abs(dy*ex - ey*dx) < EQUAL_THRESHOLD:
            if self.isSegment and other.isSegment:
                sP0 = self.p0.round(EQUAL_DECIMAL_PLACES)
                sP1 = self.p1.round(EQUAL_DECIMAL_PLACES)
                oP0 = other.p0.round(EQUAL_DECIMAL_PLACES)
                oP1 = other.p1.round(EQUAL_DECIMAL_PLACES)
                endpointEqualities = np.array([np.all(sP0 == oP0), np.all(sP0 == oP1), np.all(sP1 == oP0), np.all(sP1 == oP1)])
                numEqual = np.sum(endpointEqualities)
                if numEqual == 1:
                    eqIndex = np.where(endpointEqualities)[0][0]
                    s = int(eqIndex > 1) * self.length
                    t = int(eqIndex % 2) * other.length
                    return MyIntersection(True, s, t, self.p0 + s*self.dir)
            # If none of the above hold, there's no or infinite intersection.
            return MyIntersection(False, 0, 0, (0,0)) 
        
        # If dx is 0, we need to switch x and y.
        # Otherwise, we'd be dividing by 0 later on.
        # This change in coordinates won't affect s or t.
        if (abs(dx) < EQUAL_THRESHOLD):
            deltaX, deltaY = deltaY, deltaX
            dx, dy = dy, dx
            ex, ey = ey, ex
            
        # Math checks out here when done on paper, solving augmented matrix.
        t = (dy*deltaX - dx*deltaY)/(dy*ex - ey*dx)
        s = -deltaX/dx + (ex/dx)*t
        
        # Return the struct-like object.
        return MyIntersection(True, s, t, self.p0 + s*self.dir)

    
class MyActiveLine(MyLine):
    def __init__(self, p0, p1, p0Index, p1Index, activeType, increasesToTheRight):
        super().__init__(p0, p1, True)
        self.p0Index = p0Index
        self.p1Index = p1Index
        self.activeType = activeType
        self.increasesToTheRight = increasesToTheRight
    def __repr__(self):
        return "{0}->{1}, right+ is {2}".format(self.p0, self.p1, self.increasesToTheRight)
    def swapDir(self):
        self.p0, self.p1 = self.p1, self.p0
        self.p0Index, self.p1Index = self.p1Index, self.p0Index
        self.increasesToTheRight = not self.increasesToTheRight

class MySortableSegment(MyActiveLine):
    def __init__(self, activeLine, sweepLine, index):
        super().__init__(
            activeLine.p0,
            activeLine.p1,
            activeLine.p0Index,
            activeLine.p1Index,
            activeLine.activeType,
            activeLine.increasesToTheRight
        )
        self.sweepLine = sweepLine
        self.index = index
        self.node = None
        self.lastIntersectionY = activeLine.p0[1]
        self.forwardHalfEdge = None
        
    def __repr__(self):
        return "[ ({0}, {1}) -> ({2}, {3}) ]".format(self.p0[0], self.p0[1], self.p1[0], self.p1[1])
        
    def currentY(self):
        if self.isVertical:
            return self.lastIntersectionY
        return self.m * self.sweepLine.x + self.b

    # Might rethink this at some point.
    # But be careful not to break == usage in tree!
    def __eq__(self, other): 
        if other:
            diff0 = np.linalg.norm(self.p0 - other.p0)
            diff1 = np.linalg.norm(self.p1 - other.p1)
            if diff0 < EQUAL_THRESHOLD and diff1 < EQUAL_THRESHOLD:
                return True
        return False
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __lt__(self, other):
        retVal = False
        selfY = self.currentY()
        otherY = other.currentY()
        diff = selfY - otherY
        if abs(diff) < EQUAL_THRESHOLD:
            surpassed = (self.sweepLine.y > selfY + EQUAL_THRESHOLD)
            hereButRemoving = abs(self.sweepLine.y - selfY) < EQUAL_THRESHOLD and self.sweepLine.eventType == EventType.RIGHT
            # Special case where two vertical line segments connect
            if self.isVertical and other.isVertical:
                retVal = self.p1[1] > other.p1[1]
            elif self.isVertical:
                retVal = True
            elif other.isVertical:
                retVal = False
            elif abs(self.m - other.m) < EQUAL_THRESHOLD:
                return self.p0[0] < other.p0[0]
            else:
                retVal = self.m > other.m
                    
            if surpassed or hereButRemoving:
                retVal = not retVal
        else:
            retVal = diff < 0
        return retVal
    
    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)
    def __gt__(self, other):
        return not self.__le__(other)
    def __ge__(self, other):
        return not self.__lt__(other)

    def __hash__(self):
        return hash(self.index) # Maybe return just self.index?
    
class MyPolygon:
    def __init__(self, pts):
        npts = np.array(pts)
        # "Close" the polygon s.t. the first and last vertex are identical.
        # First, check if the first and last points are already the same.
        if not np.all(npts[0] == npts[-1]):
            # If not, add the first pt to the end.
            npts = np.vstack((npts, npts[0, :]))
        self._coords = npts
    
    def getCoords(self):
        return np.copy(self._coords) # Look into the shallow/deep copy nature of this at some point!
    
    def getSeparateXYs(self):
        xs = self._coords[:, 0]
        ys = self._coords[:, 1]
        
        return (xs, ys)
    
class Scene:
        
    def __init__(self):
        # These can maybe be combined into a dataframe or list of structs at some point.
        self.polygons = []
        self.cwList = []
        
        # In addition to keeping track of individual polygons,
        # we also keep track of ALL vertices in the scene.
        self.vertices = np.empty((0, 2))
        # These can maybe be combined with self.vertices into a dataframe or something
        self.prevIndices = np.empty(0, dtype=np.int)
        self.nextIndices = np.empty(0, dtype=np.int)
        self.polygonIndices = np.empty(0, dtype=np.int)
        
        self.lines = []
        self.activeSegments = []
        
        # Boundaries for the scene.
        self.minX = math.inf
        self.maxX = -math.inf
        
        self.minY = math.inf
        self.maxY = -math.inf
        
        self.partitionMesh = None
        self.drawableFaces = []
        
    def createActiveSegments(self, index0, index1):
        v00 = self.vertices[self.prevIndices[index0]]
        v01 = self.vertices[index0]
        v02 = self.vertices[self.nextIndices[index0]]
        
        v10 = self.vertices[self.prevIndices[index1]]
        v11 = self.vertices[index1]
        v12 = self.vertices[self.nextIndices[index1]]
        
        cwV0 = self.cwList[self.polygonIndices[index0]]
        
        # If the two vertices form an edge, then it's the first case.
        if self.prevIndices[index0] ==  index1:
            return [MyActiveLine(v11, v01, index1, index0, SegmentType.A, not cwV0)]
        elif self.nextIndices[index0] == index1:
            return [MyActiveLine(v01, v11, index0, index1, SegmentType.A, not cwV0)]

        
        # Otherwise, need to determine which side of the line the two vertices "triangles" are on.
        # I'm going to use the inward-pointing bisector of each vertex's angle represent the direction pointing "inside" the triangle from the vertex.
        # The reason for using the bisector, rather than just one of the edges, is because
        # it is possible for one of the edges to lie on the line, but the bisector
        # never will.
       
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
        
        retVals = []
        if localBisector0[0] > 0 and localBisector1[0] > 0:
            retVals = [MyActiveLine(v11, v01, index1, index0, SegmentType.B, True)]
        elif localBisector0[0] < 0 and localBisector1[0] < 0:
            retVals = [MyActiveLine(v01, v11, index0, index1, SegmentType.B, True)]
        else:
            b0, b1 = self.sceneBorderHitPoints(MyLine(v01, v11, False))
            if np.dot((b1 - b0), up) < 0:
                b0, b1 = b1, b0
            
            incToRight = localBisector0[0] < 0
            seg1 = MyActiveLine(b0, v01, -1, index0, SegmentType.C, incToRight)
            seg2 = MyActiveLine(b1, v11, -1, index1, SegmentType.C, incToRight)
            retVals = [seg1, seg2]
            
        return retVals
        
    def addPolygon(self, pts):
        newVertices = np.array(pts, dtype=np.float64)
        #newVertices[:, 0] = newVertices[:, 0] - 2.33

        self.polygons.append(MyPolygon(pts)) #newVertices.tolist()))
        
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
        
    def calcFreeLines(self):
        vertLineDict = {}
        nonVertLineDict = {}
        for i in range(len(self.vertices) - 1):
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
                    pts = obj.getCoords()
                    numPts = len(pts)
                    edgeNum = 0 # Like vertCount, but just for this polygon rather than whole scene.
                    
                    # pts, i.e. obj.getCoords(), is organized where the 1st vertex is repeated at the end.
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
                        if intersection.doMeet and intersection.meetT > -EQUAL_THRESHOLD and intersection.meetT < edgeLine.length + EQUAL_THRESHOLD:
                            # If the lines intersect, the line and edge/segment probably do...
                            intersectsThisTime = True
                            # ...but we should test and rule out non-transversal intersections
                            # Infinite intersections are already "discarded" by the intersection() function.
                            # But we need to rule out intersections with a vertex that do not pierce the shape,
                            # because these are fine (in fact, they are REQUIRED for the algorithm).
                            # We first deal with the line intersecting the vertex at the start of its edge, at v0.
                            if (abs(intersection.meetT) < EQUAL_THRESHOLD):
                                # Test if candidate.dir is between both edge dirs going AWAY from v0
                                intersectsThisTime = self.isLineInsideEdgeAngle(vertCount, candidate.dir)
                            # Same idea, but for the case where the intersection is at
                            # the other side of the edge, closer to v1
                            elif (abs(intersection.meetT - edgeLine.length) < EQUAL_THRESHOLD):
                                # Test if candidate.dir is between both edge dirs going AWAY from v1
                                intersectsThisTime = self.isLineInsideEdgeAngle(self.nextIndices[vertCount], candidate.dir)
                            #if intersectsThisTime:
                            #    print(candidate.p0, "->", candidate.p1, "intersects", v0, "->", v1, "meetT:", intersection.meetT, "len:", edgeLine.length)
                            intersectsObj = (intersectsObj or intersectsThisTime)
                            
                        edgeNum += 1
                        vertCount += 1
                    polygonCount += 1
                if not intersectsObj:
                    newSegments = self.createActiveSegments(i, j)                    
                    
                    for newSeg in newSegments:

                        cKey = round(newSeg.p0[0], EQUAL_DECIMAL_PLACES)
                        dictOfInterest = vertLineDict
                        if not newSeg.isVertical:
                            cKey = (round(newSeg.m, EQUAL_DECIMAL_PLACES), round(newSeg.b, EQUAL_DECIMAL_PLACES))
                            dictOfInterest = nonVertLineDict

                        if cKey in dictOfInterest:
                            dictOfInterest[cKey].append(newSeg)
                        else:
                            dictOfInterest[cKey] = [newSeg]
        '''print("\n=====\n NON VERT LINES:")
        for nvldk in nonVertLineDict:
            print("Key:", nvldk)
            for nvl in nonVertLineDict[nvldk]:
                print(" -", nvl)
        print("\n=====\n VERT LINES:")
        for vldk in vertLineDict:
            print("Key:", vldk)
            for vl in vertLineDict[vldk]:
                print(" -", vl)
        print("\n=====\n")'''
                
        self.unifySegments(nonVertLineDict, False)
        self.unifySegments(vertLineDict, True)
        self.calculateVisualHull()
            
                
    def unifySegments(self, segmentDictionary, isVertical):
        axisNum = 0
        axisKey = "x"
        if isVertical:
            axisNum = 1
            axisKey = "y"
        for _, segsToUnify in segmentDictionary.items():
                # Skip over the complex unification process if only one segment.
                if len(segsToUnify) == 1:
                    self.activeSegments.append(segsToUnify[0])
                    continue
                # Also skip over if it's just two "type-C" segments.
                if len(segsToUnify) == 2 and segsToUnify[0].activeType == SegmentType.C and segsToUnify[1].activeType == SegmentType.C:
                    self.activeSegments.append(segsToUnify[0])
                    self.activeSegments.append(segsToUnify[1])
                    continue
                coordsOnLn = []
                for i in range(len(segsToUnify)):
                    s = segsToUnify[i]
                    if s.p0[axisNum] > s.p1[axisNum]:
                        s.swapDir()
                    coordsOnLn.append({"x": s.p0[0], "y": s.p0[1], "index": s.p0Index, "segsStartingHere": [i]})
                    coordsOnLn.append({"x": s.p1[0], "y": s.p1[1], "index": s.p1Index, "segsStartingHere": []})
                    
                coordsOnLn.sort(key = (lambda a: a[axisKey]))
                            
                prevCoord = coordsOnLn[0]
                uniqueCoords = [prevCoord]
                for i in range(1, len(coordsOnLn)):
                    coord = coordsOnLn[i]
                    if abs(coord[axisKey] - prevCoord[axisKey]) > EQUAL_THRESHOLD:
                        uniqueCoords.append(coord)
                    else:
                        uniqueCoords[-1]["segsStartingHere"] += coord["segsStartingHere"]
                    prevCoord = coord
                
                intervals = []
                for i in range(len(uniqueCoords) - 1):
                    intervals.append( {"right": 0, "left": 0} )
                
                # This next bit looks O(n^3) at a glance.
                # But keep in mind that each segment only is in "segsStartingHere"
                # for one coord, and that the number of possible intervals
                # a segment can span is also limited.
                for i in range(len(uniqueCoords) - 1):
                    coord = uniqueCoords[i]
                    for sIndex in coord["segsStartingHere"]:
                        s = segsToUnify[sIndex]
                        intervalIndex = i
                        while uniqueCoords[intervalIndex][axisKey] < s.p1[axisNum] - EQUAL_THRESHOLD and intervalIndex < len(intervals):
                            if s.increasesToTheRight:
                                intervals[intervalIndex]["right"] += 1
                            else:
                                intervals[intervalIndex]["left"] += 1
                            intervalIndex += 1
                for i in range(len(intervals)):
                    interval = intervals[i]
                    # The cancelling out effect over the interval. So no segment created.
                    if interval["right"] > 0 and interval["left"] > 0:
                        continue
                    # No segment at all over this line
                    elif interval["right"] == 0 and interval["left"]  == 0:
                        continue
                    else:
                        p0 = (uniqueCoords[i]["x"], uniqueCoords[i]["y"])
                        p1 = (uniqueCoords[i+1]["x"], uniqueCoords[i+1]["y"])
                        p0Index = (uniqueCoords[i]["index"])
                        p1Index = (uniqueCoords[i+1]["index"])
                        self.activeSegments.append(MyActiveLine(p0, p1, p0Index, p1Index, SegmentType.D, interval["right"] > 0))
                    
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
        if testDot <= dotThresh + EQUAL_THRESHOLD:
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
    
    def sceneBorderHitPoints(self, ln):
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
        return (newP0, newP1)
    
    def calculateVisualHull(self):      
        self.partitionMesh = self.findIntersections()
    
        self.drawableFaces = []
        for f in self.partitionMesh.faces.values():
            if self.partitionMesh.isExteriorFace(f):
                continue
            #he = f.halfEdge
            #if he.headVertex is None:
            #    continue
            '''v = he.headVertex.position
            origHE = he
            he = he.next
            while he != origHE:
                #if he.headVertex is None:
                #    break
                v = he.headVertex.position
                he = he.next'''
            self.drawableFaces.append(f)
        
        print("Num of drawable faces:", len(self.drawableFaces))

        # We know that the below vertex is on the shape.
        # Now we need to find out which of its faces has visual number 0.
        # First, we can assume the vertex is convex, else it wouldn't
        # be a part of an active segment processed above.
        # So, the "bisector" of its two edges points into the shape.
        # We just need to find two half edges that "enclose" it.
        # That would thus mean they also enclose that part of the shape.
        vertOnShape = self.partitionMesh.vertexOnShape
        vertIndex = vertOnShape.vertexID
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
        
        startingFace = None
        vertHalfEdge = vertOnShape.outgoingHalfEdge
        while startingFace is None:
            v0Edge = vertHalfEdge
            # Next edge in ccw direction.
            v2Edge = vertHalfEdge.prev.pair
            
            v0 = v0Edge.headVertex.position
            v2 = v2Edge.headVertex.position
            # v1 is the same as in the bisector calculation.
            
            v0Angle = math.atan2(v0[1] - v1[1], v0[0] - v1[0]) + math.pi
            v2Angle = math.atan2(v2[1] - v1[1], v2[0] - v2[0]) + math.pi
            bisectorAngle = math.atan2(bisector[1], bisector[0]) + math.pi
            # Case where the v0 to v2 range crosses over 0 radians axis.
            if v2Angle < v0Angle:
                v2Angle += (2.0 * math.pi) - v0Angle
                bisectorAngle += (2.0 * math.pi) - v0Angle 
            
            if bisectorAngle > v0Angle and bisectorAngle < v2Angle:
                startingFace = v0Edge.leftFace
            
            vertHalfEdge = vertHalfEdge.pair.next
            

        # Now, DFS to assign visual numbers to all faces.
        stack = [{"face": startingFace, "visualNumber": 0}]
        while len(stack) > 0:
            currFace = stack.pop()
            if currFace["face"].visualNumber < 0:
                vn = currFace["visualNumber"]
                currFace["face"].visualNumber = vn
                halfEdge = currFace["face"].halfEdge
                adjFace = halfEdge.pair.leftFace
                if (not self.partitionMesh.isExteriorFace(adjFace)) and (adjFace in self.drawableFaces):
                    vnChange = -1 if halfEdge.increasesLeft else 1
                    stack.append({"face": adjFace, "visualNumber": vn+vnChange})
                    
                origHalfEdge = halfEdge
                halfEdge = halfEdge.next
                while halfEdge != origHalfEdge:
                    #if halfEdge is None:
                    #    break
                    adjFace = halfEdge.pair.leftFace
                    if (not self.partitionMesh.isExteriorFace(adjFace)) and (adjFace in self.drawableFaces):
                        vnChange = -1 if halfEdge.increasesLeft else 1
                        stack.append({"face": adjFace, "visualNumber": vn+vnChange})
                    halfEdge = halfEdge.next    
    
        
    # Sweep line implementation!
    def findIntersections(self):
        t = RedBlackTree()
        
        q = []
        # I'm pretty sure this next line's not needed, but I'm not taking chances right now. In a hurry.
        heapq.heapify(q)
        
        sortableSegments = []
        sweepLine = SweepLine(0, 0, EventType.LEFT)
        
        for i in range(len(self.activeSegments)):
            s = self.activeSegments[i]
            pL = s.p0
            pR = s.p1
            shouldSwap = False
            if pR[0] < pL[0] - EQUAL_THRESHOLD:
                shouldSwap = True
            elif abs(pR[0] - pL[0]) < EQUAL_THRESHOLD:
                if pR[1] < pL[1] - EQUAL_THRESHOLD:
                    shouldSwap = True
            
            if shouldSwap:
                s.swapDir()
            
            sortableSegment = MySortableSegment(s, sweepLine, i)
            sortableSegments.append(sortableSegment)
            
            lEnd = MySweepEvent(s.p0[0], s.p0[1], {sortableSegment}, EventType.LEFT)
            rEnd = MySweepEvent(s.p1[0], s.p1[1], {sortableSegment}, EventType.RIGHT)
            
            heapq.heappush(q, lEnd)
            heapq.heappush(q, rEnd)
            
        intersections = []  
        partitionMesh = HalfEdgeStructure()
    
        eventCount = 0
        
            
        while len(q) > 0:
            '''if eventCount == 31:
                print("here!")
            print("\nEvents:", eventCount)'''
            eventCount += 1
            p = heapq.heappop(q)
            #print("Event: ", p)
            
            # print("Intersections({0}):".format(len(intersections)))
            # for isec in intersections:
            #     print(isec, end=", ")
            # print()
            
            if p.eventType == EventType.INTERSECTION:
                while q[0] == p:
                    # print("merging:", [s.index for s in p.segments], ",", [s.index for s in q[0].segments])
                    pToMerge = heapq.heappop(q)
                    p.merge(pToMerge)
                    # print("merged segSet:", [s.index for s in p.segments])
            sweepLine.x = p.x
            sweepLine.y = p.y
            sweepLine.eventType = p.eventType
                                
            if p.eventType == EventType.LEFT:
                s = next(iter(p.segments))
                
                sNode = t.add(s)
                s.node = sNode
                sNode.subscribers.add(s)
                
                succ = t.successor(sNode)
                pred = t.predecessor(sNode)
                if succ:
                    succSeg = succ.value
                    # print("succSeg:", succSeg)
                    succInt = s.intersection(succSeg)
                    onFirstSegment = succInt.meetS > -EQUAL_THRESHOLD and succInt.meetS < s.length + EQUAL_THRESHOLD
                    onSecondSegment = succInt.meetT > -EQUAL_THRESHOLD and succInt.meetT < succSeg.length + EQUAL_THRESHOLD
                    
                    if succInt.doMeet and onFirstSegment and onSecondSegment:
                        intEvent = MySweepEvent(succInt.meetPt[0], succInt.meetPt[1], {s, succSeg}, EventType.INTERSECTION, eventCount-1)
                        # print("\tintEvent:", intEvent)
                        heapq.heappush(q, intEvent)
    
                if pred:
                    predSeg = pred.value
                    # print("predSeg:", predSeg)
                    predInt = s.intersection(predSeg)
                    onFirstSegment = predInt.meetS > -EQUAL_THRESHOLD and predInt.meetS < s.length + EQUAL_THRESHOLD
                    onSecondSegment = predInt.meetT > -EQUAL_THRESHOLD and predInt.meetT < predSeg.length + EQUAL_THRESHOLD
                    
                    if predInt.doMeet and onFirstSegment and onSecondSegment:
                        intEvent = MySweepEvent(predInt.meetPt[0], predInt.meetPt[1], {s, predSeg}, EventType.INTERSECTION, eventCount-1)
                        # print("\tintEvent:", intEvent)
                        heapq.heappush(q, intEvent)
                        
            elif p.eventType == EventType.RIGHT:
                s = next(iter(p.segments))
                
                sNode = s.node
                
                pred = t.predecessor(sNode)
                succ = t.successor(sNode)
                
                t.removeGivenNode(sNode)
                
                if (s.forwardHalfEdge is not None) and (s.forwardHalfEdge.headVertex is None): #possibly need to check x coords of pair.headVertex against current x coords?
                    halfEdge = s.forwardHalfEdge    
                    halfEdge.prev.next = halfEdge.pair.next
                    halfEdge.pair.next.prev = halfEdge.prev
                    
    
                    # The only thing on the side of an edge terminating outside of the convex hull
                    # Is an exterior face, not an interior one. So, we delete any non-exterior
                    # faces on both sides of this half-edge.
                    if not partitionMesh.isExteriorFace(halfEdge.leftFace):
                        partitionMesh.removeFace(halfEdge.leftFace)
                    if not partitionMesh.isExteriorFace(halfEdge.pair.leftFace):
                        partitionMesh.removeFace(halfEdge.pair.leftFace)
                        
                    partitionMesh.assignExteriorFace(halfEdge.prev)
                    partitionMesh.assignExteriorFace(halfEdge.pair.next)
                    
                    
                    partitionMesh.removeHalfEdgePair(halfEdge)

                
                if pred and succ:
                    predSeg = pred.value
                    succSeg = succ.value
                    # print("predSeg:", predSeg)
                    # print("succSeg:", succSeg)
                    newInt = predSeg.intersection(succSeg)
                    onFirstSegment = newInt.meetS > -EQUAL_THRESHOLD and newInt.meetS < predSeg.length + EQUAL_THRESHOLD
                    onSecondSegment = newInt.meetT > -EQUAL_THRESHOLD and newInt.meetT < succSeg.length + EQUAL_THRESHOLD
                    toTheRight = newInt.meetPt[0] > sweepLine.x + EQUAL_THRESHOLD
                    onSweepLine = (abs(newInt.meetPt[0] - sweepLine.x) < EQUAL_THRESHOLD)
                    higherOnSweepLine = onSweepLine and (newInt.meetPt[1] > sweepLine.y + EQUAL_THRESHOLD)
                    if newInt.doMeet and onFirstSegment and onSecondSegment and (toTheRight or higherOnSweepLine):
                        intEvent = MySweepEvent(newInt.meetPt[0], newInt.meetPt[1], {predSeg, succSeg}, EventType.INTERSECTION, eventCount-1)
                        # print("\tintEvent:", intEvent)
                        heapq.heappush(q, intEvent)
                # for sThing in t.valueList():
                #     if not t.isMatchingNodeInTree(sThing.node):
                #         print("Problem for", sThing, "node!!!")
            
            else: # It's an intersection
                newElem = np.array((p.x, p.y))
                intersections.append(newElem)
                intSegments = deque(sorted(p.segments))
                
                # These segments will "become" min and max after the swaps.
                maxSeg = intSegments[0]
                minSeg = intSegments[-1]
    
                # For the face assignment, need to know which line segments
                # exist before and after this intersection.
                extendBeforeInt = []
                extendAfterInt = []
    
                for intSeg in intSegments:
                    x0Diff = p.x - intSeg.p0[0]
                    y0Diff = p.y - intSeg.p0[1]
                    x1Diff = intSeg.p1[0] - p.x
                    y1Diff = intSeg.p1[1] - p.y

                    x0Before = x0Diff > EQUAL_THRESHOLD
                    x0Equal = abs(x0Diff) < EQUAL_THRESHOLD
                    y0Before = y0Diff > EQUAL_THRESHOLD
    
                    x1After = x1Diff > EQUAL_THRESHOLD
                    x1Equal = abs(x1Diff) < EQUAL_THRESHOLD
                    y1After = y1Diff > EQUAL_THRESHOLD
                    
                    # Note: Before and After not mutually exclusive here.
                    before = x0Before or (x0Equal and y0Before)
                    after = x1After or (x1Equal and y1After)
    
                    if before and intSeg.forwardHalfEdge:
                        extendBeforeInt.append(intSeg)
                    if after:
                        extendAfterInt.insert(0, intSeg)  
                
                # The half-edge data structure will have a vertex
                # at this intersection.
                newVertex = Vertex((p.x, p.y), None)

                # We need one vertex that's also an original polygon vertex
                # in order to choose a starting face for the visual number assignments
                if partitionMesh.vertexOnShape is None:
                    segIndex = 0
                    while newVertex.vertexID < 0 and segIndex < len(extendBeforeInt):
                        vertSeg = extendBeforeInt[segIndex]
                        if vertSeg.p1Index >= 0 and abs(vertSeg.p1[0] - p.x) < EQUAL_THRESHOLD and abs(vertSeg.p1[1] - p.y) < EQUAL_THRESHOLD:
                            newVertex.vertexID = vertSeg.p1Index
                        segIndex += 1
                    segIndex = 0
                    while newVertex.vertexID < 0 and segIndex < len(extendAfterInt):
                        vertSeg = extendAfterInt[segIndex]
                        if vertSeg.p0Index >= 0 and abs(vertSeg.p0[0] - p.x) < EQUAL_THRESHOLD and abs(vertSeg.p0[1] - p.y) < EQUAL_THRESHOLD:
                            newVertex.vertexID = vertSeg.p0Index
                        segIndex += 1
                    partitionMesh.verts.append(newVertex)
                    if newVertex.vertexID >= 0:
                        partitionMesh.vertexOnShape = newVertex
    
                # Swap segment order in tree
                while len(intSegments) >= 2:
                    s0 = intSegments.popleft()
                    s1 = intSegments.pop()
                    tempNode = s0.node
                    s0.node = s1.node
                    s1.node = tempNode
                    s0.node.subscribers.remove(s1)
                    s0.node.subscribers.add(s0)
                    s1.node.subscribers.remove(s0)
                    s1.node.subscribers.add(s1)
                    s0.node.value = s0
                    s1.node.value = s1
                    
                    s0.lastIntersectionY = p.y
                    s1.lastIntersectionY = p.y
                
                # print("maxSeg:", maxSeg)
                # print("minSeg:", minSeg)
                
                pred = t.predecessor(minSeg.node)
                succ = t.successor(maxSeg.node)
    
                # For each half-edge that comes before the intersection:
                #  - Assign the new vertex as the half-edge's head.
                #  - "Close" the face created by each consecutive pair
                #    of half-edges by connecting said half-edges.
                for i in range(len(extendBeforeInt)):
                    preSeg = extendBeforeInt[i]
                    preSeg.forwardHalfEdge.headVertex = newVertex
                    if i < len(extendBeforeInt) - 1:
                        nextHalfEdge = extendBeforeInt[i+1].forwardHalfEdge.pair
                        preSeg.forwardHalfEdge.next = nextHalfEdge
                        nextHalfEdge.prev = preSeg.forwardHalfEdge
    
                # Create the new half-edges for segments extending
                # past the intersection point.
                newForwardHalfEdges = []
                for i in range(len(extendAfterInt)):
                    newForwardHalfEdges.append(partitionMesh.createNewPairOfHalfEdges(newVertex, not extendAfterInt[i].increasesToTheRight))
    
                # Handle the outermost half-edges in the "fans" before and/or after the intersection.
                # First two cases only have a fan on one side (before or after), creating a "corner".
                # Third case is when there are lines both before and after the intersection.
                if len(extendAfterInt) == 0:
                    topForwardHalfEdge = extendBeforeInt[-1].forwardHalfEdge
                    bottomBackHalfEdge = extendBeforeInt[0].forwardHalfEdge.pair
                    topForwardHalfEdge.next = bottomBackHalfEdge
                    bottomBackHalfEdge.prev = topForwardHalfEdge
                    newVertex.outgoingHalfEdge = bottomBackHalfEdge
                    # If this "corner" forms a concave "dent" in a region, then
                    # the two faces on either side of the corner are actually
                    # the same, but will have been created without "knowing"
                    # that, so they'll currently be two separate ones.
                    # Thus, this must be reconciled. We'll keep one and
                    # replace the other with it.
                    faceSetToRemove = set()
                    if not partitionMesh.isExteriorFace(topForwardHalfEdge.leftFace):
                        halfEdgeToReplaceFaceOn = topForwardHalfEdge
                        while halfEdgeToReplaceFaceOn is not None and halfEdgeToReplaceFaceOn.leftFace != bottomBackHalfEdge.leftFace:
                            faceSetToRemove.add(halfEdgeToReplaceFaceOn.leftFace)
                            halfEdgeToReplaceFaceOn.leftFace = bottomBackHalfEdge.leftFace
                            halfEdgeToReplaceFaceOn = halfEdgeToReplaceFaceOn.prev
                    else:
                        halfEdgeToReplaceFaceOn = bottomBackHalfEdge
                        while halfEdgeToReplaceFaceOn is not None and halfEdgeToReplaceFaceOn.leftFace != topForwardHalfEdge.leftFace:
                            faceSetToRemove.add(halfEdgeToReplaceFaceOn.leftFace)
                            halfEdgeToReplaceFaceOn.leftFace = topForwardHalfEdge.leftFace
                            halfEdgeToReplaceFaceOn = halfEdgeToReplaceFaceOn.next
                    for faceToRemove in faceSetToRemove:
                        partitionMesh.removeFace(faceToRemove)

                    
                    
                elif len(extendBeforeInt) == 0:
                    bottomBackHalfEdge = newForwardHalfEdges[0].pair
                    topForwardHalfEdge = newForwardHalfEdges[-1]
                    bottomBackHalfEdge.next = topForwardHalfEdge
                    topForwardHalfEdge.prev = bottomBackHalfEdge
                    # Since this is a "new" corner, we cannot set the outer
                    # half-edges' faces using ones that come from before
                    # So, we need to find if we're inside a convex face.
                    # To do this, we look at segments "outside" this "fan"
                    # in the tree until we find something or reach the exterior.
                    halfEdgePred = pred
                    halfEdgeSucc = succ
                    isOutsideFaceFound = False
                    while halfEdgePred is not None and not isOutsideFaceFound:
                        if halfEdgePred.value.forwardHalfEdge is not None:
                            sharedFace = halfEdgePred.value.forwardHalfEdge.leftFace
                            topForwardHalfEdge.leftFace = sharedFace
                            bottomBackHalfEdge.leftFace = sharedFace
                            isOutsideFaceFound = True
                        else:
                            halfEdgePred = t.predecessor(halfEdgePred)
                    while halfEdgeSucc is not None and not isOutsideFaceFound:
                        if halfEdgeSucc.value.forwardHalfEdge is not None:
                            sharedFace = halfEdgeSucc.value.forwardHalfEdge.pair.leftFace
                            topForwardHalfEdge.leftFace = sharedFace
                            bottomBackHalfEdge.leftFace = sharedFace
                            isOutsideFaceFound = True
                        else:
                            halfEdgeSucc = t.successor(halfEdgeSucc)
                            
                    if not isOutsideFaceFound:
                        partitionMesh.assignExteriorFace(topForwardHalfEdge)
                        partitionMesh.assignExteriorFace(bottomBackHalfEdge)
                    
                    newVertex.outgoingHalfEdge = newForwardHalfEdges[0]
                else:
                    bottomHalfEdgeBeforeInt = extendBeforeInt[0].forwardHalfEdge.pair
                    topHalfEdgeBeforeInt = extendBeforeInt[-1].forwardHalfEdge
                    bottomHalfEdgeAfterInt = newForwardHalfEdges[0].pair
                    topHalfEdgeAfterInt = newForwardHalfEdges[-1]

                    newVertex.outgoingHalfEdge = bottomHalfEdgeBeforeInt
    
                    topHalfEdgeBeforeInt.next = topHalfEdgeAfterInt
                    topHalfEdgeAfterInt.prev = topHalfEdgeBeforeInt
    
                    bottomHalfEdgeAfterInt.next = bottomHalfEdgeBeforeInt
                    bottomHalfEdgeBeforeInt.prev = bottomHalfEdgeAfterInt
    
                    topHalfEdgeAfterInt.leftFace = topHalfEdgeBeforeInt.leftFace
                    bottomHalfEdgeAfterInt.leftFace = bottomHalfEdgeBeforeInt.leftFace
    
                # For each half-edge that comes after the intersection:
                #  - Connect consecutive pairs of half-edges together
                #    at the intersection point.
                #  - Create a new face between each consecutive pair
                #    of half-edges.

                for i in range(len(extendAfterInt) - 1):
                    newForwardHalfEdges[i+1].pair.next = newForwardHalfEdges[i]
                    newForwardHalfEdges[i].prev = newForwardHalfEdges[i+1].pair 
    
                    newFace = partitionMesh.createNewFace(newForwardHalfEdges[i])
    
                    newForwardHalfEdges[i].leftFace = newFace
                    newForwardHalfEdges[i].prev.leftFace = newFace
                    
                # Assign the new forward half-edges to their respective segments.
                for i in range(len(extendAfterInt)):
                    extendAfterInt[i].forwardHalfEdge = newForwardHalfEdges[i]
                
                # All of the face handling is now complete.
                # Next, normal line sweep intersection testing continues.
                # We need to check the new max and min segments against their
                # "outside" neighbour line segments for new intersections.
                if pred:
                    predSeg = pred.value
                    # print("predSeg:", predSeg)
                    predInt = minSeg.intersection(predSeg)
                    onFirstSegment = predInt.meetS > -EQUAL_THRESHOLD and predInt.meetS < minSeg.length + EQUAL_THRESHOLD
                    onSecondSegment = predInt.meetT > -EQUAL_THRESHOLD and predInt.meetT < predSeg.length + EQUAL_THRESHOLD
                    toTheRight = predInt.meetPt[0] > sweepLine.x + EQUAL_THRESHOLD
                    onSweepLine = (abs(predInt.meetPt[0] - sweepLine.x) < EQUAL_THRESHOLD)
                    higherOnSweepLine = onSweepLine and (predInt.meetPt[1] > sweepLine.y + EQUAL_THRESHOLD)
                    if predInt.doMeet and onFirstSegment and onSecondSegment and (toTheRight or higherOnSweepLine):
                        intEvent = MySweepEvent(predInt.meetPt[0], predInt.meetPt[1], {minSeg, predSeg}, EventType.INTERSECTION, eventCount-1)
                        if intEvent != p and (intEvent.x - sweepLine.x) > -EQUAL_THRESHOLD:
                            # print("\tintEvent:", intEvent)
                            heapq.heappush(q, intEvent)
                if succ:
                    succSeg = succ.value
                    # print("succSeg:", succSeg)
                    succInt = maxSeg.intersection(succSeg)
                    onFirstSegment = succInt.meetS > -EQUAL_THRESHOLD and succInt.meetS < maxSeg.length + EQUAL_THRESHOLD
                    onSecondSegment = succInt.meetT > -EQUAL_THRESHOLD and succInt.meetT < succSeg.length + EQUAL_THRESHOLD
                    toTheRight = succInt.meetPt[0] > sweepLine.x + EQUAL_THRESHOLD
                    onSweepLine = (abs(succInt.meetPt[0] - sweepLine.x) < EQUAL_THRESHOLD)
                    higherOnSweepLine = onSweepLine and (succInt.meetPt[1] > sweepLine.y + EQUAL_THRESHOLD)
                    if succInt.doMeet and onFirstSegment and onSecondSegment and (toTheRight or higherOnSweepLine):
                        intEvent = MySweepEvent(succInt.meetPt[0], succInt.meetPt[1], {maxSeg, succSeg}, EventType.INTERSECTION, eventCount-1)
                        if intEvent != p and (intEvent.x - sweepLine.x) > -EQUAL_THRESHOLD:
                            # print("\tintEvent:", intEvent)
                            heapq.heappush(q, intEvent)
            # t.printTree()
            # print("---")
            # print(t.valueList())
            
            # for sThing in t.valueList():
            #     if not t.isMatchingNodeInTree(sThing.node):
            #         print("Problem for", sThing, "node!!!")
        return partitionMesh

