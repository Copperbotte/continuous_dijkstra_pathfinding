

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
def triangulate_delaunay(path):

    #for each constrained side, find the best deluanay point
    tris = []
    tri_sets = []
    for p1, p2 in zip(path[:-1], path[1:]):
        curp3 = None
        
        for p3 in path[:-1]:
            #skip duplicate nodes
            if len(set([p1, p2, p3])) != 3:
                continue

            if curp3 == None:
                curp3 = p3
                continue
            
            tri = [curp3, p1, p2, curp3]
            valid = validate_delaunay_constraint(path, tri, p3)
            if not valid:
                curp3 = p3

        #check if this tri was already found
        tri = [curp3, p1, p2, curp3]
        ts = set(tri)
        if ts not in tri_sets:
            tris.append(tri)
            tri_sets.append(ts)

    return tris


#picks a random triangle, validates it.
#def triangulate(path):


if __name__ == "__main__":
    points, paths = loadSvgData()
    path = paths[3]
    
    def ezquiver(ax, x, y, color=None):
        if color == None:
            color = next(ax._get_lines.prop_cycler)
        ax.quiver(x[:-1], y[:-1], *tuple([e-s for s,e in zip(xy[:-1], xy[1:])] for xy in [x,y]),
           scale_units='xy', angles='xy', scale=1, **color)
    
    def eztri(tri, ax=None):
        if ax == None:
            fig, ax = plt.subplots()
        valid = validate(path, tri)
        ezquiver(ax, *tuple(zip(*path)))
        
        color = next(ax._get_lines.prop_cycler)
        if valid == "True":
            color = {'color': 'green'}
        ezquiver(ax, *tuple(zip(*tri)), color=color)
        ax.title.set_text(str(valid))
        #ax.title.set_text(str(n)+": "+t)

    def test_delaunay():
        p = path[:-1]
        tri = p[:3] + [p[0]]
        point = (1,1)

        constraint = [(0.7, 1.1), (1.0, 0.5)]
        
        uv, r = calc_circle(tri)
        fig, ax = plt.subplots()
        color = next(ax._get_lines.prop_cycler)
        val = validate_delaunay_constraint(constraint, tri, point)
        if not val:
            color = {'color':'red'}
        ax.add_patch(plt.Circle(uv, radius=r, fill=False, **color))
        ax.plot(*tuple(zip(*tri)))
        ax.plot(*tuple(zip(*constraint)))
        ax.scatter(*point)
        ax.scatter(*uv)
        plt.show()
    #test_delaunay()

    def test_triangulate():
        fig, axs = plt.subplots(2, 2)

        for ax, path in zip(axs.reshape((1,4))[0], paths):
            tris = triangulate_delaunay(path)
            ax.plot(*tuple(zip(*path)))
            for t in tris:
                ax.plot(*tuple(zip(*t)))
            ax.title.set_text("Triangle count: %d"%len(tris))
        plt.show()
    test_triangulate()