
import numpy as np

#####                  #####
##### Simple functions #####
#####                  #####

#forward linear interpolate
def lerp(x, low, high):
    return (high - low)*x + low

#inverse lerp
def inv_lerp(x, low, high):
    return (x - low) / (high - low) 

#makes a numpy vector a unit vector
def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm

#####                           #####
##### Matrix building functions #####
#####                           #####

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


#####                  #####
##### Matrix functions #####
#####                  #####

#point in triangle using inverse method
#This can be generalized in 3d using the moore-penrose psuedoinverse.
def pointInTri(point, tri):
    point = np.array(point)
    tri = np.array(tri)

    #convert the tri into a basis
    point -= tri[0]
    tri -= tri[0]

    #strip the other tri, check if the inverse is valid
    M = tri[1:].transpose()
    if np.linalg.det(M) == 0:
        return False

    #bring the point into the triangle's basis space
    #psuedoinverse if this was in 3d
    inv = np.linalg.inv(M)
    xy = inv @ point

    #the transformed triangle's edges are now
    #the coordinate axes, and y + x = 1.
    return np.all(0 < xy) and np.sum(xy) < 1

#find the radius and position of a circle from a tri
#using a variant of within_circle's matrix method
def calc_circle(tri):
    def basis(x,y):
        return [x, y, x**2+y**2, 1]
    M = np.array([basis(*t) for t in tri + [(0,0)]])
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

#given three points, determine if the fourth point
#is within a circle constructed from the other three.
def within_circle(tri, point):
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

    M = np.array([basis(*t) for t in tri + [point]])

    #true if the point is outside the circle
    #this has right handed chirality
    return 0 < np.linalg.det(M)

#####                  #####
#####  Test functions  #####
#####                  #####

#tests calc_circle and within_circle functions.
#Given a generic tri, find a circle that touches its vertices.
#Create a set of points, and test weather its in the circle.
def test_circle():
    tri = [(0,0),(1,1), (0,1), (0,0)]

    center, rad = calc_circle(tri[:-1])
    
    points = [[(x,y) for x in np.linspace(-1,2,num=32)]
              for y in np.linspace(-1,2,num=32)]
    
    points = sum(points, [])
    values = list(map(lambda p: within_circle(tri[:-1], p), points))
    intri = list(map(lambda v: pointInTri(v, tri[:-1]), points))
    
    fig, ax = plt.subplots()
    color = next(ax._get_lines.prop_cycler)
    ax.add_patch(plt.Circle(center, rad, fill=False, **color))
    ax.plot(*tuple(zip(*tri)))
    
    color_true = next(ax._get_lines.prop_cycler)['color']
    color_false = next(ax._get_lines.prop_cycler)['color']
    colors = list(map(lambda v: color_true if v else color_false, values))
    sizes = list(map(lambda v: 10 if v else 1, intri))
    
    ax.scatter(*tuple(zip(*points)), c=colors, s=sizes)

    ax.set_aspect(1)
    plt.show()

    #merge with previous scope to test with terminal
    return locals()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    locals().update(test_circle())
