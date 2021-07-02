

#this script triangulates a path.
#uses code i wrote for this demo:
#https://www.shadertoy.com/view/stjGRD

import numpy as np
from math_funcs import normalize, within_circle, calc_circle
from smallest_circle import welzl_smallest_circle

#given a constraint path, a tri, and a point, check if the point is a valid
#delaunay point, then check if it is visible through the paths.
#returns true if no visible paths are found.
def validate_delaunay_constraint(path, tri, point):
    #first, check if the point is within the given circle
    if within_circle(tri[:-1], point):
        return True

    P = np.array(point)
    T = np.array(tri[:-1])
    
    #check visibility for a given constraint segment
    #maybe this can be better done in a bsp format?
    for c in zip(path[:-1], path[1:]):
        #if the given circle doesn't contain either constraint, continue
        #if not any(map(lambda n: within_circle(tri, n), c)):
        #    continue

        C = np.array(c)

        #generate a pair of lines between the point, and the constraint
        dc = C[1] - C[0]
        
        #tangent basis
        tc = normalize(dc)

        #find constraint tangent coords
        c_coords = np.dot(tc, C.transpose())

        #normal basis
        #this method is not general in 3d
        nc = np.array([-tc[1], tc[0]])

        #normal offset coord
        oc = np.dot(nc, C[0])

        #iterate through all test points
        def test_point(t):
            #generate a line through the two points
            dt = P - t
            tt = normalize(dt)
            t_coords = np.dot(tt, np.array([t, P]).transpose())
            
            nt = np.array([-tt[1], tt[0]])
            ot = np.dot(nt, P)
            #using the matrix definition of lines
            #if the lines are parallel, continue
            M = np.array([nt, nc])
            if np.linalg.det(M) == 0.0:
                return True
                #continue
            #find their intersection point
            ip = np.dot(np.linalg.inv(M), np.array([ot, oc]))

            #determine how scaled the intersection is from
            #the constraint vertices coordinates via inverse lerp
            c_ip = np.dot(tc, ip)
            c_ip = (c_ip - c_coords[0]) / (c_coords[1] - c_coords[0])

            #if coords of ip are between 0 and 1, hit constraint
            if 0 < c_ip < 1:
                #check if the constraint is behind the test point
                #via inverse lerp
                o_ip = np.dot(tt, ip)
                o_ip = (o_ip - t_coords[0]) / (t_coords[1] - t_coords[0])
                
                if 0 < o_ip < 1:
                    return True
                    #continue

            #otherwise, this is an invalid point
            return False
        #if any of the points are blocked, this is valid
        if not any(map(test_point, T)):
            return False
    
    #if both loops pass completely, this is
    #the best constrained delaunay circle for the constraints.
    return True
    

#constrained delaunay using a simple n^2 search algorithm
def triangulate_delaunay_simple(path):

    vertices = path[:-1]
    indices = [(a, (a+1)%len(vertices)) for a, _ in enumerate(vertices)]

    #for each constrained side, find the best deluanay point
    tris = []
    tri_sets = []
    for i1, i2 in indices:
        curi3 = None
        
        for i3, _ in enumerate(vertices):
            #skip duplicate nodes
            if len(set([i1, i2, i3])) != 3:
                continue

            if curi3 == None:
                curi3 = i3
                continue
            
            tri = [curi3, i1, i2, curi3]
            v_tri = list(map(lambda x: vertices[x], tri))
            valid = validate_delaunay_constraint(path, v_tri, vertices[i3])
            if not valid:
                curi3 = i3

        #check if this tri was already found
        tri = [curi3, i1, i2, curi3]
        ts = set(tri)
        if ts not in tri_sets:
            tris.append(tri)
            tri_sets.append(ts)

    #convert from indices to vertices
    #tris = [[vertices[i] for i in tri] for tri in tris]

    return tris

#test methods
def test_delaunay(path, plot=False):
    p = path[:-1]
    tri = p[:3] + [p[0]]
    point = (1,1)
    
    constraints = [[(0.7, 1.1), (1.0, 0.5)],
                   [(0.7, 1.1), (1.0, 1.1)]]

    fig, axs = plt.subplots(1, 2)
    fig.suptitle("Constrained Delaunay triangle test")
    
    uv, r = calc_circle(tri[:-1])

    for ax, con in zip(axs, constraints):
        color = next(ax._get_lines.prop_cycler)
        val = validate_delaunay_constraint(con, tri, point)
        if not val:
            color = {'color':'red'}
        ax.add_patch(plt.Circle(uv, radius=r, fill=False, **color))
        ax.plot(*tuple(zip(*tri)))
        ax.plot(*tuple(zip(*con)))
        ax.scatter(*point)
        ax.scatter(*uv)
        ax.set_aspect(1)
    if plot:
        plt.show()

def test_triangulate(paths, plot=False):
    #build a square like region
    w = int(np.ceil(np.sqrt(len(paths))))
    h = int(np.ceil(len(paths) / w))
    
    fig, axs = plt.subplots(w, h)
    fig.suptitle("Triangulation of simple paths")

    for ax, path in zip(axs.reshape((1,len(paths)))[0], paths):
        i_tris = triangulate_delaunay_simple(path)
        tris = [[path[i] for i in tri] for tri in i_tris]
        ax.plot(*tuple(zip(*path)))
        for t in tris:
            ax.plot(*tuple(zip(*t)))
        ax.title.set_text("Triangle count: %d"%len(tris))
    if plot:
        plt.show()

if __name__ == "__main__":
    from load_svg import loadSvgData, plotSvgData
    import matplotlib.pyplot as plt
    import matplotlib.animation as ani
    points, paths = loadSvgData()

    test_delaunay(paths[3])
    test_triangulate(paths)
    plt.show()
