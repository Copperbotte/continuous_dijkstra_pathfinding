

#this script triangulates a path.
#uses code i wrote for this demo:
#https://www.shadertoy.com/view/stjGRD

import matplotlib.pyplot as plt
import numpy as np
import random
from load_svg import loadSvgData, plotSvgData

#makes a numpy vector a unit vector
def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm

#isolate a minor for a given n dimensional array
#skip is a tuple of which coords to skip
def minor(matrix, skip):
    M = matrix
    for ns, s in enumerate(skip):
        idx = [slice(None)] * M.ndim #arbitrarily sized matrices
        idx[ns] = [i for i in range(M.shape[ns]) if i is not s]
        M = M[tuple(idx)]
    return M
    #M = matrix
    #R = [i for i in range(M.shape[0]) if i is not skip[0]]
    #M = M[R, :]
    #R = [i for i in range(M.shape[1]) if i is not skip[1]]
    #M = M[:, R]
    #return M

#calculate a cofactor for a given 2d matrix
def cofactor(matrix, skip):
    cof = np.linalg.det(minor(matrix, skip))
    #multiply the cofactor by (-1)^(i+j)
    cof *= 1 if sum(skip)%2 == 0 else -1
    return cof

#find the radius and position of a circle from a tri
#using a variant of validate_delaunay's matrix method
def calc_circle(tri):
    def basis(x,y):
        return [x, y, x**2+y**2, 1]
    M = np.array([basis(*t) for t in tri[:-1] + [(0,0)]])
    #generate cofactors
    U = cofactor(M, (3, 0))
    V = cofactor(M, (3, 1))
    W = cofactor(M, (3, 2))
    T = cofactor(M, (3, 3))
    #by completing the square, the 4th point can be calculated
    #as a difference of coords and a radius.
    #0 = Ux + Vy + W(x^2 + y^2) + T
    #E + F - T/W = x^2 + y^2 + (U/W)x + (V/W)*y + E + F
    #(U/2W)^2 + (V/2W)^2 - T/W = (x+U/2W)^2 + (y+V/2W)^2
    #r^2 = (x-u)^2 + (y-v)^2
    
    #solve for coords
    u = -U/(2*W)
    v = -V/(2*W)
    r = np.sqrt(u**2 + v**2 - T/W)
    return (u, v), r

#given a tri, determine if the trial point is within the delaunay
#circle of the given tri.
def validate_delaunay(tri, point):
    #delaunay tris form a circumscribed circle of the smallest possible areas.
    #by comparing the distance from a center p,q as a radius r to a point x,y
    #r^2 = (x-p)^2 + (y-q)^2
    #it can be expanded out into the form:
    #0 = (x^2 - 2xp + p^2) + (y^2 - 2yq + q^2) - r^2
    #and by collecting like terms:
    #0 = -2p*(x) - 2q*(y) + 1*(x^2 + y^2) + 1*(p^2 + q^2 - r^2)
    #which forms a basis under [x, y, x^2 + y^2, 1].
    #if four vectors in this basis are coplanar, it has zero determinant.
    #if the determinant is positive, the 4th point lies within the circumcircle.

    def basis(x,y):
        return [x, y, x**2+y**2, 1]

    M = np.array([basis(*t) for t in tri[:-1] + [point]])

    #true if the point is outside the circle
    return np.linalg.det(M) < 0

#given a constraint path, a tri, and a point, check if the point is a valid
#delaunay point, then check if it is visible through the paths.
#returns true if no visible paths are found.
def validate_delaunay_constraint(path, tri, point):
    #first, check if the point is within the given circle
    if validate_delaunay(tri, point):
        return True

    P = np.array(point)
    T = np.array(tri[:-1])
    
    #check visibility for a given constraint segment
    #maybe this can be better done in a bsp format?
    for c in zip(path[:-1], path[1:]):
        #if the given circle doesn't contain either constraint, continue
        #if not any(map(lambda n: validate_delaunay(tri, n), c)):
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
    
    uv, r = calc_circle(tri)

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
    points, paths = loadSvgData()
    
    test_delaunay(paths[3])
    test_triangulate(paths)
    plt.show()
