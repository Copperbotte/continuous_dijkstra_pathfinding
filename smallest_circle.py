
import numpy as np
from math_funcs import calc_circle

#https://en.wikipedia.org/wiki/Smallest-circle_problem#Welzl's_algorithm
#based on Welzl's algorithm to find the smallest circle containing
#a set of points. used to find a triangle larger than the set of points.

def welzl_recursive(points, indices=[], boundary=[]):
    #if the circle contains nothing, return the boundary set.
    #if the boundary is 0-2 points, return trivial circles.
    #if the boundary is 3, all other points lie within this circle.
    if len(indices) == 0 or len(boundary) == 3:
        p = list(map(lambda x: np.array(points[x]), boundary))
        if len(boundary) == 0:
            return (0,0), 0
        if len(boundary) == 1:
            return p[0], 0
        if len(boundary) == 2:
            #find a circle midway between the two points
            radius = np.linalg.norm(p[1]-p[0]) / 2
            center = (p[1]+p[0]) / 2
            return center, radius
        #3 point boundary
        return calc_circle(p)
    
    #choose a single point "randomly"
    p = indices[0]

    #find the optimal circle for everything besides this point
    center, radius = welzl_recursive(points, indices[1:], boundary)

    #if p is within this circle, return the circle
    delta = np.array(points[p]) - np.array(center)
    if delta.dot(delta) < radius**2:
        return center, radius

    #otherwise, p is somewhere on the boundary of the circle.
    #Re-traverse with it on the boundary.
    return welzl_recursive(points, indices[1:], boundary+[p])

#initializing function using points only
def welzl_smallest_circle(points):
    indices = [x for x in range(len(points))]
    return welzl_recursive(points, indices)


#Assuming points are circles with zero radius, then if a circle contains a set
#of other circles, anything containing the larger circle contains everything
#within it. By using divide and conquer, a circle that contains the entire set
#can be constructed.
#is this mergesort?
def containing_circle(points, radii=None):
    #radii for circles of size 0
    if radii == None:
        radii = [0 for i in points]
        
    #if there is one point, return a trivial solution.
    if len(points) == 1:
        return points[0], 0

    #if there are 2 circles, return a circle containing them.
    if len(points) == 2:
        p = np.array(points)
        #find a circle midway between the two points, plus their radii
        radius = (np.linalg.norm(p[1]-p[0]) + sum(radii)) / 2
        #find the center
        s = [radii[0]/(2*radius), 1-(radii[1]/(2*radius))]
        mid = (p[0]*(s[1]-1/2)+p[1]*(1/2-s[0])) / (s[1]-s[0])
        return mid, radius

    #if there are more than two, divide the set in half,
    #and find circles that contain those sets that are one or two in size.
    half = len(points) // 2
    c1, r1 = containing_circle(points[:half], radii[:half])
    c2, r2 = containing_circle(points[half:], radii[half:])
    return containing_circle([c1, c2], [r1, r2])


def test_welzl(points):
    global welzls
    welzls = []

    #this redefinition contains a global for debug visualization.
    #I'm not sure of a good way to animate this, without rewriting the function.
    def welzl_smallest_circle(points, indices=[], boundary=[]):
        #if the circle contains nothing, return the boundary set.
        #if the boundary is 0-2 points, return trivial circles.
        #if the boundary is 3, all other points lie within this circle.
        if len(indices) == 0 or len(boundary) == 3:
            p = list(map(lambda x: np.array(points[x]), boundary))
            if len(boundary) == 0:
                return (0,0), 0
            if len(boundary) == 1:
                return p[0], 0
            if len(boundary) == 2:
                #find a circle midway between the two points
                radius = np.linalg.norm(p[1]-p[0]) / 2
                center = (p[1]+p[0]) / 2
                return center, radius
            #3 point boundary
            return calc_circle(p)

        global welzls
        
        #choose a single point "randomly"
        p = indices[0]

        #find the optimal circle for everything besides this point
        center, radius = welzl_smallest_circle(points, indices[1:], boundary)

        #debug draw
        circ = plt.Circle(center, radius=radius, fill=False, color="gray")
        state = [indices, boundary]
        welzls.append([circ, state])

        #if p is within this circle, return the circle
        delta = np.array(points[p]) - np.array(center)
        if delta.dot(delta) < radius**2:
            return center, radius

        #otherwise, p is somewhere on the boundary of the circle.
        #Re-traverse with it on the boundary.
        return welzl_smallest_circle(points, indices[1:], boundary+[p])
    
    fig, ax = plt.subplots()

    import random
    #points = [(random.random(), random.random()) for i in range(1000)]

    #points = [(0,0), (0,1), (1,0), (0, -1)]
    points = sum([p[:-1] for p in paths], [])
    #points = paths[3][:-1]
    indices = [x for x in range(len(points))]
    
    center, radius = welzl_smallest_circle(points, indices=indices)
    ax.add_patch(plt.Circle(center, radius=radius, fill=False, color="green"))
    sc = ax.scatter(*tuple(zip(*points)))
    ax.set_aspect(1)
    for path in paths:
        ax.plot(*tuple(zip(*path)))

    def nextCircle(num):
        ax.patches = []
        #for i in range(num+1):
        ax.add_patch(welzls[num][0])
        colors = ['red' if w in welzls[num][1][1] else
                  'green' if w in welzls[num][1][0] else
                  'black' for w in indices]
        sc.set_color(colors)
    
    circles = ani.FuncAnimation(fig, nextCircle, len(welzls),
                                fargs=(), interval=1000//60)
    plt.show()

    #keep tests in global scope
    globals().update(locals())
    
def test_contain(points):
    #this redefinition contains a global for debug visualization.
    #I'm not sure of a good way to animate this, without rewriting the function.
    def containing_circle(points, radii=None):
        #radii for circles of size 0
        if radii == None:
            radii = [0 for i in points]
            
        #if there is one point, return a trivial solution.
        if len(points) == 1:
            return points[0], 0, None

        #if there are 2 circles, return a circle containing them.
        if len(points) == 2:
            p = np.array(points)
            #find a circle midway between the two points, plus their radii
            radius = (np.linalg.norm(p[1]-p[0]) + sum(radii)) / 2
            #find the center
            s = [radii[0]/(2*radius), 1-(radii[1]/(2*radius))]
            mid = (p[0]*(s[1]-1/2)+p[1]*(1/2-s[0])) / (s[1]-s[0])
            return mid, radius, None

        #if there are more than two, divide the set in half,
        #and find circles that contain those sets that are one or two in size.
        half = len(points) // 2
        
        c1, r1, ct1 = containing_circle(points[:half], radii[:half])
        circ1 = plt.Circle(c1, radius=r1, fill=False, color="gray")
        state = points[:half]
        #circs.append([circ1, state])
        
        c2, r2, ct2 = containing_circle(points[half:], radii[half:])
        circ2 = plt.Circle(c2, radius=r2, fill=False, color="gray")
        state = points[half:]
        #circs.append([circ2, state])

        c3, r3, ct3 = containing_circle([c1, c2], [r1, r2])
        circ3 = plt.Circle(c3, radius=r3, fill=False, color="gray")
        state = points
        #circs.append([circ3, state])

        circ_tree = [[circ1, ct1], [circ2, ct2], [circ3, None]]
        return c3, r3, circ_tree

    fig, ax = plt.subplots()
    
    center, radius, circ_tree = containing_circle(points)
    ax.add_patch(plt.Circle(center, radius=radius, fill=False, color="green"))
    sc = ax.scatter(*tuple(zip(*points)))
    ax.set_aspect(1)
    for path in paths:
        ax.plot(*tuple(zip(*path)))

    #convert circ_tree into a reverse breadth first list
    i = 0
    print(i)
    circ_tree.reverse()
    while True:    
        cur = circ_tree[i]
        #print(cur)
        circ_tree[i] = cur[0]
        if cur[1] != None:
            circ_tree += reversed(cur[1])
        i += 1
        if i == len(circ_tree):
            break

    circ_tree.reverse()

    def nextCircle(num):
        ax.patches = []
        for i in range(num+1):
            ax.add_patch(circ_tree[i])
    
    circles = ani.FuncAnimation(fig, nextCircle, len(circ_tree),
                                fargs=(), interval=1000)
    plt.show()

    #keep tests in global scope
    globals().update(locals())

#simpler smallest circle test
def test_smallest(paths):
    fig, ax = plt.subplots()
    for n in range(4):
        ax.add_patch(plt.Circle(*welzl_smallest_circle(paths[n][:-1]), fill=False))
        ax.plot(*tuple(zip(*paths[n])))
    ax.set_aspect(1)
    plt.show()
    globals().update(locals())

if __name__ == "__main__":
    from load_svg import loadSvgData, plotSvgData
    import matplotlib.pyplot as plt
    import matplotlib.animation as ani
    points, paths = loadSvgData()

    import random
    #points = [(random.random(), random.random()) for i in range(1000)]

    #points = [(0,0), (0,1), (1,0), (0, -1)]
    points = sum([p[:-1] for p in paths], [])
    #points = paths[3][:-1]
    
    #test_contain(points)
    #test_welzl(points)
    test_smallest(paths)
    
