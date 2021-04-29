import matplotlib.pyplot as plt
import numpy as np
from visHullTwoD import Scene, SegmentType

#%%

def doubleFaceTest(f):
    doubleFace = False
    origHE = f.halfEdge
    he = f.halfEdge.next
    while he != origHE:
        if f.index != he.leftFace.index:
            doubleFace = True
            break
        he = he.next
    if doubleFace:
        print("Double face ({0}):".format(f.index))
        origHE = f.halfEdge
        he = f.halfEdge.next
        while he != origHE:
            fIndex = he.leftFace.index
            v0 = he.prev.headVertex.position
            v1 = he.headVertex.position
            print(" - F{0}, {1}->{2}".format(fIndex, v0, v1))
            he = he.next
        v0 = he.prev.headVertex.position
        v1 = he.headVertex.position
        print(" - F{0}, {1}->{2}".format(fIndex, v0, v1))
        print("-----")
        
        
def checkEventEquality(w0, w1):
    print("== Event check ==")
    numEvents0 = len(w0.eventsRecord)
    numEvents1 = len(w1.eventsRecord)
    if numEvents0 != numEvents1:
        print("NUMBER OF EVENT RECORDS DIFFERENT! w0: {0}, w1: {1}".format(numEvents0, numEvents1))
    minEvents = min(numEvents0, numEvents1)
    for i in range(minEvents):
        eventsEq = w0.eventsRecord[i].debugEq(w1.eventsRecord[i])
        if not np.all(list(eventsEq.values())):
            print(" - DIFF AT {0}: {1}".format(i, eventsEq))
    print("Done event check!\n")
#%%

def drawScene(scene):
    print("cwList:", scene.cwList)
    # Plot all polygons.
    for obj in scene.polygons:
        x,y = obj.getSeparateXYs()
        plt.fill(x,y, "#A0A0A0") # light grey fill
        plt.plot(x,y, "#505050") # dark grey edges/outline
        
    plt.show()
    return
    '''
    for ln in scene.lines:
        p0, p1 = scene.sceneBorderHitPoints(ln)
        plt.plot([p0[0], p1[0]], [p0[1], p1[1]], "k--")
    '''
    for ln in scene.activeSegments:
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
    
    '''for halfEdge in partitionMesh.halfEdges:
        if halfEdge.headVertex is not None and halfEdge.pair.headVertex is not None:
            v0 = halfEdge.headVertex.position
            v1 = halfEdge.pair.headVertex.position
            plt.plot([v0[0], v1[0]], [v0[1], v1[1]], "r--")
        else:
            print("Some problem")'''
            
    
    
    colours = ["k", "r", "g", "b", "y"]
    for f in scene.drawableFaces:
        #print("Visual number:", f.visualNumber)
        regionColour = colours[min(f.visualNumber, len(colours) - 1)]
        pts = f.getCoords()
        xs = pts[:, 0]
        ys = pts[:, 1]
        plt.fill(xs, ys, regionColour)
        
    convex = []
    concave = []
    for i in range(scene.vertices.shape[0]):
        if scene.isVertexConcave(i):
            concave.append(scene.vertices[i])
        else:
            convex.append(scene.vertices[i])
    npConvex = np.array(convex)
    npConcave = np.array(concave)
    
    '''
    for maxSeg in self.activeSegments:
        for succSeg in self.activeSegments:
            succInt = maxSeg.intersection(succSeg)
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
world5 = Scene()
world6 = Scene()
world7 = Scene()
world8 = Scene()
world9 = Scene()
world10 = Scene()
world11 = Scene()
world12 = Scene()

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
# polygon2 = [(p[0] - 3, p[1]) for p in polygon2]
# Horizontal flip for testing purposes.
polygon1 = [(-p[0], p[1]) for p in polygon1]
polygon2 = [(-p[0], p[1]) for p in polygon2]
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

polygon1 = [(0, 0.6), (1.5, 0), (2.5, 1.25), (1.25, 0.75), (1.125, 1.8)]
polygon2 = [(1.3, 2.25), (2.8, 2.8), (1.65, 3.125)]
polygon3 = [(2.8, 1.25), (4.125, 0.25), (3.5, 2.0)]

world5.addPolygon(polygon1)
world5.addPolygon(polygon2)
world5.addPolygon(polygon3)

polygon1 = [(0,0), (2.5, 0), (0, 1.5)]
polygon2 = [(0, 3.25), (5, 4.25), (0, 4.25)]
polygon3 = [(3.5, 0), (5, 0), (5, 2.75), (3.5, 2.75)]

world6.addPolygon(polygon1)
world6.addPolygon(polygon2)
world6.addPolygon(polygon3)

polygon1 = [(-1, 1), (-2, 1), (-2, -1), (-1, -1), (0, 0), (1, -1), (2, -1), (2, 1), (1, 1), (0, 2)]

world7.addPolygon(polygon1)

polygon1 = [(-1, 1), (-2, 1), (-2, -1), (-1, -1)]
polygon2 = [(-1, -1), (0, 0), (1, -1), (1, 1), (0, 2), (-1, 1)]
polygon3 = [(1, -1), (2, -1), (2, 1), (1, 1)]
# polygon1 = [(p[0], 0.9*p[1]) for p in polygon1]
# polygon3 = [(p[0], 0.9*p[1]) for p in polygon3]

world8.addPolygon(polygon1)
world8.addPolygon(polygon2)
world8.addPolygon(polygon3)

# 0.9999995231628418
polygon1 = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
polygon2 = [(1, 1), (2, -1), (3, 0), (2, 1)]

world9.addPolygon(polygon1)
world9.addPolygon(polygon2)

polygon1 = [(0.734870970249176, 0.26040399074554443), (-0.045375000685453415, 0.8651400208473206), (-0.8234530091285706, 0.4177840054035187), (-0.14182999730110168, 0.21450699865818024)]
polygon2 = [(-1.0, 1.0108875036239624), (1.0, 1.010890007019043), (1.0, 1.3735400438308716), (-1.0, 1.373543620109558)]

world10.addPolygon(polygon2)
world10.addPolygon(polygon1)

polygon0 = [(0.734870970249176, -1.1526894569396973), (-0.045375000685453415, 1.1651400327682495), (-0.8234530091285706, -0.9953095316886902), (-0.14182999730110168, -1.1985864639282227)]
polygon1 = [(2.1045942306518555, -2.0704498291015625), (2.1045916080474854, 1.9576737880706787), (1.7419415712356567, 1.9576740264892578), (1.7419381141662598, -2.0704498291015625)]
polygon2 = [(-1.7419382333755493, -2.0704498291015625), (-1.741940975189209, 1.9576740264892578), (-2.10459041595459, 1.9576740264892578), (-2.1045944690704346, -2.0704495906829834)]

world11.addPolygon(polygon0)
world11.addPolygon(polygon1)
world11.addPolygon(polygon2)

polygon0 = [(0.7000000476837158, -1.2000000476837158), (-0.10000000149011612, 1.2000000476837158), (-0.800000011920929, -1.0), (-0.10000000149011612, -1.25)]
polygon1 = [(2.0999999046325684, -2.0999999046325684), (2.0999999046325684, 1.899999976158142), (1.7000000476837158, 1.899999976158142), (1.7000000476837158, -2.0999999046325684)]
polygon2 = [(-1.7000000476837158, -2.0999999046325684), (-1.7000000476837158, 1.899999976158142), (-2.1000001430511475, 1.899999976158142), (-2.1000001430511475, -2.0999999046325684)]

world12.addPolygon(polygon0)
world12.addPolygon(polygon1)
world12.addPolygon(polygon2)
#world.addLine((0, 2.5), (3, 2.5))

worlds = [world0, world1, world2, world3, world4, world5, world6, world7, world8, world9, world10]

worldIndex = 0
for w in worlds:
    print("\nWorld:", worldIndex)
    worldIndex += 1
    w.calcFreeLines()
    drawScene(w)
    
    faceList = w.partitionMesh.faces
    for k in faceList:
        doubleFaceTest(faceList[k])

checkEventEquality(world12, world11)



#%%
reminders = [
     "Is there a better way, using cos(), to handle parallelism in isLineInsideEdgeAngle()?",
     "Pruning of lines that intersect obj at CONTACT verts. (I sort of forget what this self-reminder meant...)",
     "Pruning of segments outside convex hull.",
     "Replace RB Tree with my own, or one with better licensing!"
     "Right now, swapDir() side effect in findIntersections(). Should this be changed?",
]

for reminder in reminders:
    sep = "==========="
    print("\n" + sep + "\n" + reminder + "\n" + sep + "\n")