import matplotlib.pyplot as plt
import numpy as np
import math
from enum import Enum
from collections import deque
import heapq

import random
from shapely.geometry import Polygon, LineString
from rbt import RedBlackTree # REALLY NEED TO REWRITE THIS!

EQUAL_THRESHOLD = 0.0001 # Threshold for considering certain fp numbers equal below.
EQUAL_DECIMAL_PLACES = -round(math.log(EQUAL_THRESHOLD, 10))

class EventType(Enum):
    LEFT = 0
    INTERSECTION = 1
    RIGHT = 2

  
 
#%%

class SweepLine:
    def __init__(self, x, y, eventType):
        self.x = x
        self.y = y
        self.eventType = eventType
            
# Sweep line implementation!
def findIntersections(segments):
    t = RedBlackTree()
    
    q = []
    # I'm pretty sure this next line's not needed, but I'm not taking chances right now. In a hurry.
    heapq.heapify(q)
    
    sortableSegments = []
    sweepLine = SweepLine(0, 0, EventType.LEFT)
    
    for i in range(len(segments)):
        s = segments[i]
        pL = s.p0
        pR = s.p1
        if pR[0] < pL[0]:
            pL, pR = pR, pL
        elif pR[0] == pL[0]:
            if pR[1] < pL[1]:
                pL, pR = pR, pL
        
        sortableSegment = MySortableSegment(pL, pR, sweepLine, i)
        sortableSegments.append(sortableSegment)
        
        lEnd = MySweepEvent(pL[0], pL[1], {sortableSegment}, EventType.LEFT)
        rEnd = MySweepEvent(pR[0], pR[1], {sortableSegment}, EventType.RIGHT)
        
        heapq.heappush(q, lEnd)
        heapq.heappush(q, rEnd)
        
    intersections = []  
    
    eventCount = 0
    
        
    while len(q) > 0:
        if eventCount == 31:
            print("here!")
        print("\nEvents:", eventCount)
        eventCount += 1

        p = heapq.heappop(q)
        print("Event: ", p)
        
        

        
        print("Intersections({0}):".format(len(intersections)))
        for isec in intersections:
            print(isec, end=", ")
        print()
        
        if p.eventType == EventType.INTERSECTION:
            while q[0] == p:
                print("merging:", [s.index for s in p.segments], ",", [s.index for s in q[0].segments])
                pToMerge = heapq.heappop(q)
                p.merge(pToMerge)
                print("merged segSet:", [s.index for s in p.segments])
        sweepLine.x = p.x
        sweepLine.y = p.y
        sweepLine.eventType = p.eventType
        
        for sThing in t.valueList():
            if not t.isMatchingNodeInTree(sThing.node):
                print("Problem for", sThing, "node!!!")
                
        if p.eventType == EventType.LEFT:
            s = next(iter(p.segments))
            
            sNode = t.add(s) # RBTODO
            s.node = sNode
            sNode.subscribers.add(s)
            
            succ = t.successor(sNode) # RBTODO
            pred = t.predecessor(sNode) # RBTODO
            if succ:
                succSeg = succ.value # RBTODO
                print("succSeg:", succSeg)
                succInt = s.intersection(succSeg);
                onFirstSegment = succInt.meetS > -EQUAL_THRESHOLD and succInt.meetS < s.length + EQUAL_THRESHOLD
                onSecondSegment = succInt.meetT > -EQUAL_THRESHOLD and succInt.meetT < succSeg.length + EQUAL_THRESHOLD
                
                if succInt.doMeet and onFirstSegment and onSecondSegment:
                    intEvent = MySweepEvent(succInt.meetPt[0], succInt.meetPt[1], {s, succSeg}, EventType.INTERSECTION, eventCount-1)
                    print("\tintEvent:", intEvent)
                    heapq.heappush(q, intEvent) # RBTODO

            if pred:
                predSeg = pred.value
                print("predSeg:", predSeg)
                predInt = s.intersection(predSeg);
                onFirstSegment = predInt.meetS > -EQUAL_THRESHOLD and predInt.meetS < s.length + EQUAL_THRESHOLD
                onSecondSegment = predInt.meetT > -EQUAL_THRESHOLD and predInt.meetT < predSeg.length + EQUAL_THRESHOLD
                
                if predInt.doMeet and onFirstSegment and onSecondSegment:
                    intEvent = MySweepEvent(predInt.meetPt[0], predInt.meetPt[1], {s, predSeg}, EventType.INTERSECTION, eventCount-1)
                    print("\tintEvent:", intEvent)
                    heapq.heappush(q, intEvent) # RBTODO
                    
        elif p.eventType == EventType.RIGHT:
            s = next(iter(p.segments))
            
            sNode = s.node
            
            for sThing in t.valueList():
                if not t.isMatchingNodeInTree(sThing.node):
                    print("Problem for", sThing, "node!!!")
            
            pred = t.predecessor(sNode) # RBTODO
            succ = t.successor(sNode) # RBTODO
            
            for sThing in t.valueList():
                if not t.isMatchingNodeInTree(sThing.node):
                    print("Problem for", sThing, "node!!!")
            
            t.removeGivenNode(sNode) # RBTODO
            
            for sThing in t.valueList():
                if not t.isMatchingNodeInTree(sThing.node):
                    print("Problem for", sThing, "node!!!")
            
            if pred and succ:
                predSeg = pred.value
                succSeg = succ.value
                print("predSeg:", predSeg)
                print("succSeg:", succSeg)
                newInt = predSeg.intersection(succSeg);
                onFirstSegment = newInt.meetS > -EQUAL_THRESHOLD and newInt.meetS < predSeg.length + EQUAL_THRESHOLD
                onSecondSegment = newInt.meetT > -EQUAL_THRESHOLD and newInt.meetT < succSeg.length + EQUAL_THRESHOLD
                toTheRight = newInt.meetPt[0] > sweepLine.x + EQUAL_THRESHOLD
                onSweepLine = (abs(newInt.meetPt[0] - sweepLine.x) < EQUAL_THRESHOLD)
                higherOnSweepLine = onSweepLine and (newInt.meetPt[1] > sweepLine.y + EQUAL_THRESHOLD)
                if newInt.doMeet and onFirstSegment and onSecondSegment and (toTheRight or higherOnSweepLine):
                    intEvent = MySweepEvent(newInt.meetPt[0], newInt.meetPt[1], {predSeg, succSeg}, EventType.INTERSECTION, eventCount-1)
                    print("\tintEvent:", intEvent)
                    heapq.heappush(q, intEvent) # RBTODO
            for sThing in t.valueList():
                if not t.isMatchingNodeInTree(sThing.node):
                    print("Problem for", sThing, "node!!!")
        
        else: # It's an intersection
            newElem = np.array((p.x, p.y))
            intersections.append(newElem)
            intSegments = deque(sorted(p.segments))
            
            # These segments will "become" min and max after the swaps.
            maxSeg = intSegments[0]
            minSeg = intSegments[-1]
            
            # Swap order in tree
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
                
            print("maxSeg:", maxSeg)
            print("minSeg:", minSeg)
            
            pred = t.predecessor(minSeg.node)
            succ = t.successor(maxSeg.node)
            
            if pred:
                predSeg = pred.value
                print("predSeg:", predSeg)
                predInt = minSeg.intersection(predSeg);
                onFirstSegment = predInt.meetS > -EQUAL_THRESHOLD and predInt.meetS < minSeg.length + EQUAL_THRESHOLD
                onSecondSegment = predInt.meetT > -EQUAL_THRESHOLD and predInt.meetT < predSeg.length + EQUAL_THRESHOLD
                toTheRight = predInt.meetPt[0] > sweepLine.x + EQUAL_THRESHOLD
                onSweepLine = (abs(predInt.meetPt[0] - sweepLine.x) < EQUAL_THRESHOLD)
                higherOnSweepLine = onSweepLine and (predInt.meetPt[1] > sweepLine.y + EQUAL_THRESHOLD)
                if predInt.doMeet and onFirstSegment and onSecondSegment and (toTheRight or higherOnSweepLine):
                    intEvent = MySweepEvent(predInt.meetPt[0], predInt.meetPt[1], {minSeg, predSeg}, EventType.INTERSECTION, eventCount-1)
                    if intEvent != p and (intEvent.x - sweepLine.x) > -EQUAL_THRESHOLD:
                        print("\tintEvent:", intEvent)
                        heapq.heappush(q, intEvent) # RBTODO
            if succ:
                succSeg = succ.value # RBTODO
                print("succSeg:", succSeg)
                succInt = maxSeg.intersection(succSeg);
                onFirstSegment = succInt.meetS > -EQUAL_THRESHOLD and succInt.meetS < maxSeg.length + EQUAL_THRESHOLD
                onSecondSegment = succInt.meetT > -EQUAL_THRESHOLD and succInt.meetT < succSeg.length + EQUAL_THRESHOLD
                toTheRight = succInt.meetPt[0] > sweepLine.x + EQUAL_THRESHOLD
                onSweepLine = (abs(succInt.meetPt[0] - sweepLine.x) < EQUAL_THRESHOLD)
                higherOnSweepLine = onSweepLine and (succInt.meetPt[1] > sweepLine.y + EQUAL_THRESHOLD)
                if succInt.doMeet and onFirstSegment and onSecondSegment and (toTheRight or higherOnSweepLine):
                    intEvent = MySweepEvent(succInt.meetPt[0], succInt.meetPt[1], {maxSeg, succSeg}, EventType.INTERSECTION, eventCount-1)
                    if intEvent != p and (intEvent.x - sweepLine.x) > -EQUAL_THRESHOLD:
                        print("\tintEvent:", intEvent)
                        heapq.heappush(q, intEvent) # RBTODO
        t.printTree()
        print("---")
        print(t.valueList())
        
        for sThing in t.valueList():
            if not t.isMatchingNodeInTree(sThing.node):
                print("Problem for", sThing, "node!!!")
    return intersections
        
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
        
        self.isVertical = (p0[0] == p1[0])
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
    def __init__(self, p0, p1, activeType, increasesToTheRight):
        super().__init__(p0, p1, True)
        self.activeType = activeType
        self.increasesToTheRight = increasesToTheRight
    def __repr__(self):
        return "{0}->{1}, right+ is {2}".format(self.p0, self.p1, self.increasesToTheRight)
    def swapDir(self):
        self.p0, self.p1 = self.p1, self.p0
        self.increasesToTheRight = not self.increasesToTheRight

class MySortableSegment(MyLine):
    def __init__(self, p0, p1, sweepLine, index):
        super().__init__(p0, p1, True)
        self.sweepLine = sweepLine
        self.index = index
        self.node = None
        self.lastIntersectionY = p0[1]
        
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
            return [MyActiveLine(v11, v01, SegmentType.A, not cwV0)]
        elif self.nextIndices[index0] == index1:
            return [MyActiveLine(v01, v11, SegmentType.A, not cwV0)]

        
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
            retVals = [MyActiveLine(v11, v01, SegmentType.B, True)]
        elif localBisector0[0] < 0 and localBisector1[0] < 0:
            retVals = [MyActiveLine(v01, v11, SegmentType.B, True)]
        else:
            b0, b1 = self.sceneBorderHitPoints(MyLine(v01, v11, False))
            if np.dot((b1 - b0), up) < 0:
                b0, b1 = b1, b0
            
            incToRight = localBisector0[0] < 0
            seg1 = MyActiveLine(b0, v01, SegmentType.C, incToRight)
            seg2 = MyActiveLine(b1, v11, SegmentType.C, incToRight)
            retVals = [seg1, seg2]
            
        return retVals
        
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
        self.unifySegments(nonVertLineDict, False)
        self.unifySegments(vertLineDict, True)
            
                
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
                    coordsOnLn.append({"x": s.p0[0], "y": s.p0[1], "segsStartingHere": [i]})
                    coordsOnLn.append({"x": s.p1[0], "y": s.p1[1], "segsStartingHere": []})
                    
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
                        self.activeSegments.append(MyActiveLine(p0, p1, SegmentType.D, interval["right"] > 0))
                    
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
    
    def drawScene(self):
        print("cwList:", self.cwList)
        # Plot all polygons.
        for obj in self.polygons:
            x,y = obj.exterior.xy
            plt.fill(x,y, "#A0A0A0") # light grey fill
            plt.plot(x,y, "#505050") # dark grey edges/outline
        for ln in self.lines:
            p0, p1 = self.sceneBorderHitPoints(ln)
            plt.plot([p0[0], p1[0]], [p0[1], p1[1]], "k--")
        for ln in self.activeSegments:
            colString = "g"
            if ln.activeType == SegmentType.A:
                colString = "r"
            elif ln.activeType == SegmentType.B:
                colString = "b"
                
            # Magenta if vn increase to right (for vert lines) or down
            # Cyan otherwise
            colString2 = "c"
            if ln.isVertical:
                if (ln.p1[1] > ln.p0[1] and ln.increasesToTheRight) or (ln.p1[1] < ln.p0[1] and not ln.increasesToTheRight):
                    colString2 = "m"
            else:
                if (ln.p1[0] > ln.p0[0] and ln.increasesToTheRight) or (ln.p1[0] < ln.p0[0] and not ln.increasesToTheRight):
                    colString2 = "m"
            plt.plot([ln.p0[0], ln.p1[0]], [ln.p0[1], ln.p1[1]], colString2)
        
        # !!!! Left off at event 21.
        myPts = findIntersections(self.activeSegments)
        print("Found: " , len(myPts), " intersections!")
        plt.plot([myPt[0] for myPt in myPts], [myPt[1] for myPt in myPts], 'go')
        
        convex = []
        concave = []
        for i in range(self.vertices.shape[0]):
            if self.isVertexConcave(i):
                concave.append(self.vertices[i])
            else:
                convex.append(self.vertices[i])
        npConvex = np.array(convex)
        npConcave = np.array(concave)
        
        '''
        for maxSeg in self.activeSegments:
            for succSeg in self.activeSegments:
                succInt = maxSeg.intersection(succSeg);
                onFirstSegment = succInt.meetS > -EQUAL_THRESHOLD and succInt.meetS < maxSeg.length + EQUAL_THRESHOLD
                onSecondSegment = succInt.meetT > -EQUAL_THRESHOLD and succInt.meetT < succSeg.length + EQUAL_THRESHOLD
                if succInt.doMeet and onFirstSegment and onSecondSegment:
                    plt.plot([succInt.meetPt[0]], [succInt.meetPt[1]], 'ko')
        '''
        '''if npConvex.shape[0] > 0:
            plt.plot(npConvex[:, 0], npConvex[:, 1], 'bo')
        if npConcave.shape[0] > 0:
            plt.plot(npConcave[:, 0], npConcave[:, 1], 'go')'''
        plt.show()
    
world0 = Scene()
world1 = Scene()
world2 = Scene()
world3 = Scene()
world4 = Scene()


# These are the tris from Petitjean's diagram
polygon1 = [(0, 0), (2.25, 0.5), (1.25, 2.3)] # [(0,3),(1,1),(3,0),(4,0),(3,4)]
polygon2 = [(1.15, 3.15), (4, 4), (0.9, 5.25)] # [(1,4),(2,5),(2,1),(1,3)]
polygon3 = [(3, 0.7), (4.85, 1.75), (4.85, 3.4)]

world0.addPolygon(polygon1)
world0.addPolygon(polygon2)
world0.addPolygon(polygon3)
#world0.addPolygon(polygon4)

polygon1 = [(0, 0), (5, 0), (5, 5), (4, 5), (4, 3), (1, 3), (1, 5), (0, 5)]
world1.addPolygon(polygon1)

polygon1 = [(0, 0), (5, 0), (5, 3), (4, 3), (4, 5), (1, 5), (1, 3), (0, 3)]
polygon2 = [(1, 7), (3, 7), (5, 9), (4, 11), (4, 9), (1, 8), (2, 10), (0, 10)]
world2.addPolygon(polygon1)
world2.addPolygon(polygon2)

polygon1 = [(0, 2), (1,1), (2,2), (1,0)]
polygon2 = [(3,3), (4,2), (5,3)]
#polygon2 = [(p[0] - 3, p[1]) for p in polygon2]
world3.addPolygon(polygon1)
world3.addPolygon(polygon2)

polygon1 = [(0, 7), (2.25, 5), (1.25, 4), (5, 5)] # [(0, 0), (2.25, 0.5), (1.25, 2.3)] # [(0,3),(1,1),(3,0),(4,0),(3,4)]
polygon2 = [(1.15, -3.15), (4, -4), (2, -7), (0.9, -5.25)] #[(1.15, 3.15), (4, 4), (0.9, 5.25)] # [(1,4),(2,5),(2,1),(1,3)]
polygon3 = [(3, 1), (3, 0.0), (4.85, 0.75), (4.85, 2.4), (5,4)] #[(3, 0.7), (4.85, 1.75), (4.85, 3.4)]
polygon4 = [(-0.5, -1), (-0.5, 1.0), (0.5, 1), (0.5, -1)] #[(3, 0.7), (4.85, 1.75), (4.85, 3.4)]

world4.addPolygon(polygon1)
world4.addPolygon(polygon2)
world4.addPolygon(polygon3)
world4.addPolygon(polygon4)

#world.addLine((0, 2.5), (3, 2.5))

world0.calcFreeLines()
world0.drawScene()

world2.calcFreeLines()
world2.drawScene()


world1.calcFreeLines()
world1.drawScene()


world3.calcFreeLines()
world3.drawScene()

world4.calcFreeLines()
world4.drawScene()




#%%
reminders = [
     "Is there a better way (cos()) to handle parallelism in isLineInsideEdgeAngle()?",
     "OVERLAPPING LINE SEGMENTS!\n\n To handle 'unions', make n x n mat of 0s, check for cancelling, etc.",
     "REMOVE SHAPELY? NOT REALLY USING IT THAT MUCH!",
     "Pruning of lines that intersect obj at CONTACT verts.",
     "Pruning of segments outside convex hull.",
     "Replace RB Tree with my own, or one with better licensing!",
     "Checking sweepline.y in addition to sweepline.x when deciding whether an intersection happens in the past or future!"
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
