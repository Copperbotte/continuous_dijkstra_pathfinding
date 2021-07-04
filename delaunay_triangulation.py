

#this script triangulates a path.
#uses code i wrote for this demo:
#https://www.shadertoy.com/view/stjGRD

import numpy as np
from math_funcs import normalize, within_circle, calc_circle, pointInTri, inv_lerp
from smallest_circle import welzl_smallest_circle


def dual_debug(duals):
    #test dual validity
    valid = True
    for i_d, d in enumerate(duals):
        for i_n, n in enumerate(d):
            if n == -1:
                continue
            neighbor = duals[n]
            if not i_d in neighbor:
                print("Invalid dual detected between", d, i_d, n, duals[n])
                valid = False
    if valid:
        print("Duals valid")

def del_debug(points, tris, duals):
    fig, ax = plt.subplots()

    ax.set_aspect(1)
    
    for ip, p in enumerate(points):
        ax.annotate(str(ip), p)
    
    for it, t in enumerate(tris):
        p = list(map(lambda x: np.array(points[x]), t))
        pavg = sum(p) / 3
        puv, prad = calc_circle(p)
        puv = np.array(puv)
        shrink = 0.1 * 0.0
        p = list(map(lambda x: x*(1-shrink) + shrink*pavg, p))
        ax.plot(*tuple(zip(*(p + [p[0]]))))
        ax.add_patch(plt.Circle(puv, prad, fill=False, color='gray'))
        ax.annotate("%s\n%s"%(str(it), str(duals[it])), pavg, ha='center', va='center')
        
        #quiver through duals
        for d in duals[it]:
            if d == -1:
                continue
            ntri = tris[d]
            npo = list(map(lambda x: np.array(points[x]), ntri))
            navg = sum(npo) / 3
            nuv, nrad = calc_circle(npo)
            nuv = np.array(nuv)
            ndelta = [(a-b)*0.99 for a,b in zip(navg, pavg)]
            ax.quiver(*pavg, *ndelta,
                      scale_units='xy', angles='xy', scale=1,
                      width=0.005)
    print("tris")
    print(tris)
    print()
    print("duals")
    print(duals)

    dual_debug(duals)
    
    plt.show()
    
        
#given a constraint path, a tri, and a point, check if the point is a valid
#delaunay point, then check if it is visible through the paths.
#returns true if no visible paths are found.
def validate_delaunay_constraint(path, tri, point):
    #first, check if the point is within the given circle
    if not within_circle(tri, point):
        return True

    P = np.array(point)
    T = np.array(tri)
    
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

#https://en.wikipedia.org/wiki/Bowyer%E2%80%93Watson_algorithm
#given a set of points, triangulates the points to a delaunay triangulation.
#Starting with a triangle that is larger than the set of all points,
#for each new point, it replaces the triangle it lies within with triangles
#constructed to each vertex. It then performs flips on each new triangle,
#if it is no longer delaunay. It then checks neighboring triangles and repeats.
def delaunay_new(points, paths):
    #find a circle larger than the set of points
    coords, radius = welzl_smallest_circle(points)

    #generate an equilateral triangle that is guaranteed to fit every point
    radius *= 2
    s_tri = np.array(coords) + radius * np.array([
        [0,2], [-np.sqrt(3),-1], [np.sqrt(3),-1]])
    s_tri = list(map(tuple, s_tri))

    #tris are a list of list of indices of points
    #build the intial triangle using the super triangle
    #print(coords)
    #points = points[:13]
    points = s_tri + points
    tris = [[0,1,2]]

    #build a looping index offset
    def tri_off(tri, i, off):
        return tri[(i+off)%3]

    #keep a dual graph for easy flipped edge lookup
    #dual index is the tri, followed by list of edge, tri pairs
    #edges are differences in indices
    #tri -1 is an outside edge
    duals = [[-1,-1,-1]]

    #for every point:
    for ip, p in enumerate(points[3:], 3):
        #find a triangle that the point lies within
        intri = 0
        for it, t in enumerate(tris):
            tri = [points[i] for i in t]
            if pointInTri(p, tri):
                intri = it
                break
        ###print("\n\nvertex", ip)
        ###print("intri", intri)
        #get the tri and dual of the found tri
        tri = tris[intri]
        dual = duals[intri]

        ###print("tri", tri)
        ###print("dual", dual)

        #if using a star-shaped hole algorithm here, find the set of tris and
        #their duals to delete, replacing with the edge - new vertex method
        #used below.

        #split this triangle into three, preserving winding order
        n_tris = [[tri[n], tri_off(tri, n, 1), ip] for n in range(3)]
        
        #insert new tris
        tris[intri] = n_tris[0]
        tris += n_tris[1:]
        
        #adjust duals
        #first index comes from old tri, new indices are from new tris
        i_new = [intri, len(tris)-2, len(tris)-1]
        n_duals = [[dual[n], tri_off(i_new, n, 1), tri_off(i_new, n, 2)]
                   for n in range(3)]

        ###for i_n, n in enumerate(n_tris):
        ###    print('n_tris', i_n, n, 'dual', n_duals[i_n])

        #insert new duals
        duals[intri] = n_duals[0]
        duals += n_duals[1:]

        #update neighbor duals
        ###print("i_new", i_new)
        for i in i_new:

            ###print("\nupdating dual", i, "from", intri)
            ###print("attached dual", duals[i])
            
            #if neighbor dual is a new tri or -1, ignore
            neighbor = duals[i][0]
            ###print("neighbor", neighbor)
            
            if neighbor in i_new + [-1]:
                ###print("Skipped!")
                continue

            ###print("neighbor data", duals[neighbor])
            
            #find the update index of the old triangle
            ind = duals[neighbor].index(intri)
            ###print("ind", ind)
            duals[neighbor][ind] = i
            ###print("new neighbor", duals[neighbor])

        #del_debug(points, tris, duals)

        #print(duals)
        
        #continue

        #terminating the loop here produces a valid triangulation of the points
        #and the super triangle, but not a valid delaunay triangulation.
        #If any of the new triangles are non delaunay to their neighbors, their
        #edges must be flipped. This process repeats for the new edges too.
        #This algorithm is used instead of a star-shaped algorithm, which may
        #not work for holes.

        #for every neighbor, the neighbor's non shared point lies within the
        #current tri's circle (and vice versa?) perform a flip, and recursively
        #apply to the new neighbors. otherwise, do nothing.

        ###print("\n---------> Flip zone")

        #performs a flip between two triangles, assuming they have a shared edge
        def flip(points, tris, duals, tri1, tri2):
            ###dual_debug(duals)
            ###print(tris)
            ###print(duals)

            ###print("flipping tris", tri1, tri2)

            #get the tri's point indices and duals
            tris_t1 = tris[tri1]
            tris_t2 = tris[tri2]

            duals_t1 = duals[tri1]
            duals_t2 = duals[tri2]

            ###print("tri", tri1, tris_t1, "dual", duals_t1)
            ###print("tri", tri2, tris_t2, "dual", duals_t2)

            #find the shared edge via dual index
            ind_t1 = duals_t1.index(tri2)
            ind_t2 = duals_t2.index(tri1)

            ###print("flipping inds", ind_t1, ind_t2)
            ###print()

            #find the flipping points via shared edge offset
            npt_t1 = tri_off(tris_t1, ind_t1, 2)
            npt_t2 = tri_off(tris_t2, ind_t2, 2)

            #build two new tris using points not in these edges.
            #assuming they have the same winding, they traverse
            #the shared edge in opposite order. The other edges
            #are used to build the new triangles.

            ###print("input tri", tri1, tris_t1)
            ###print("input tri", tri2, tris_t2)
            adj_i_t1 = [tri_off(tris_t1, ind_t1, n) for n in range(3)]
            adj_i_t2 = [tri_off(tris_t2, ind_t2, n) for n in range(3)]
            ###print("adjusted tri", tri1, adj_i_t1)
            ###print("adjusted tri", tri2, adj_i_t2)
            t1, t2 = adj_i_t1, adj_i_t2
            new_i_t1 = [t2[2], t1[2], t1[0]]
            new_i_t2 = [t1[2], t2[2], t2[0]]
            ###print("new tri", tri1, new_i_t1)
            ###print("new tri", tri2, new_i_t2)
            ###print()

            #adjust duals to new tris
            ###print("input dual", tri1, duals_t1)
            ###print("input dual", tri2, duals_t2)
            adj_d_t1 = [tri_off(duals_t1, ind_t1, n) for n in range(3)]
            adj_d_t2 = [tri_off(duals_t2, ind_t2, n) for n in range(3)]
            ###print("adjusted duals", tri1, adj_d_t1)
            ###print("adjusted duals", tri2, adj_d_t2)
            d1, d2 = adj_d_t1, adj_d_t2
            new_d_t1 = [d1[0], d1[2], d2[1]]
            new_d_t2 = [d2[0], d2[2], d1[1]]
            ###print("new dual", tri1, new_d_t1)
            ###print("new dual", tri2, new_d_t2)
            ###print()

            #adjust neighbor duals to new tris
            #only the two edges that swapped tris need to be updated
            #these are the final edges in the new duals
            #this is the tri from tri 2 that's now in tri 1
            neighbor_t1 = new_d_t1[2]
            if neighbor_t1 != -1:
                tri_neighbor_t1 = tris[neighbor_t1]
                duals_neighbor_t1 = duals[neighbor_t1]
                ###print("neighbor of old tri", tri1, "->", neighbor_t1, tri_neighbor_t1)
                ###print("duals of neighbor", tri1, "->", neighbor_t1, duals_neighbor_t1)
                
                #these indices look up from the other index
                ind_n1 = duals_neighbor_t1.index(tri2)
                ###print("neighbor dual index", tri1, ind_n1)

            neighbor_t2 = new_d_t2[2]
            if neighbor_t2 != -1:
                tri_neighbor_t2 = tris[neighbor_t2]
                duals_neighbor_t2 = duals[neighbor_t2]
                ###print("neighbor of old tri", tri2, "->", neighbor_t2, tri_neighbor_t2)
                ###print("duals of neighbor", tri2, "->", neighbor_t2, duals_neighbor_t2)
                
                #these indices look up from the other index
                ind_n2 = duals_neighbor_t2.index(tri1)
                ###print("neighbor dual index", tri2, ind_n2)
            
            #apply changes
            tris[tri1] = new_i_t1
            duals[tri1] = new_d_t1
            if neighbor_t1 != -1:
                duals[neighbor_t1][ind_n1] = tri1
            
            tris[tri2] = new_i_t2
            duals[tri2] = new_d_t2
            if neighbor_t2 != -1:
                duals[neighbor_t2][ind_n2] = tri2

            """
            print("tris[tri1]", tris[tri1])
            print("duals[tri1]", duals[tri1])
            if neighbor_t1 != -1:
                print("tris[neighbor_t1]", tris[neighbor_t1])
                print("duals[neighbor_t1]", duals[neighbor_t1])
            else:
                print("tris[neighbor_t1]", -1)
                print("duals[neighbor_t1]", -1)
            print()
            print("tris[tri2]", tris[tri2])
            print("duals[tri2]", duals[tri2])
            if neighbor_t2 != -1:
                print("tris[neighbor_t2]", tris[neighbor_t2])
                print("duals[neighbor_t2]", duals[neighbor_t2])
            else:
                print("tris[neighbor_t2]", -1)
                print("duals[neighbor_t2]", -1)
            dual_debug(duals)
            print(tris)
            print(duals)
            #"""


        #figure out a way to make this track changes after a flip, because edges rotate
        def recursiveFlip(points, tris, duals, tri):
            #print("flipping from tri", tri)
            init_d_tri = duals[tri]
            #print("initial attached dual", init_d_tri)
            
            for i_d, i in enumerate(init_d_tri):
                ###print("\nchecking", i, "for", tri)
                if i == -1:
                    ###print("Skipped!")
                    continue

                i_tri = tris[tri]
                d_tri = duals[tri]

                n_tri = tris[i]
                n_duals = duals[i]
                ###print("n_duals", n_duals)

                #find the update index of the old triangle
                ind = n_duals.index(tri)
                ###print("n_duals index", ind)
                
                #the third point is the shared edge plus two
                p = points[tri_off(n_tri, ind, 2)]

                #check if this point lies within the source triangle's circle
                #if not, do nothing with this neighbor
                #within_circle has right handed chirality
                p_tri = [points[x] for x in i_tri]
                
                if not any([validate_delaunay_constraint(path, p_tri, p) for path in paths]):
                #if within_circle(p_tri, p):

                    ###print('\n\n\n\n\n')
                    ###print("========> within!")
                    flip(points, tris, duals, tri, i)
                    ###print('\n\n\n\n\n')

                    duals_t1 = duals[tri]
                    duals_t2 = duals[i]
                    
                    for tri in list(set(duals_t1 + duals_t2)):
                        if tri != -1:
                            recursiveFlip(points, tris, duals, tri)
                    #by recursing to the source tri, debug headaches from the
                    #above for loop are ignored. Returning leaves the for.
                    return
                    
            #print("---------> end of flip for", tri)
            #print()

        
        for i in i_new:
            recursiveFlip(points, tris, duals, i)

    #return s_tri for now
    return tris, duals, s_tri

    """
    #delete tris that contain vertices in the super triangle
    #find the tris and duals that contain the vertices of the super triangle
    skips = list(map(lambda t: any(x in [0,1,2] for x in t), tris))
    target_tris, target_duals, target_ids, _ = tuple(zip(*list(filter(
        lambda x: x[3], zip(tris, duals, range(len(tris)), skips)))))

    print(tris)
    print(duals)
    print(skips)

    #replace all neighbor duals with -1
    for i in target_ids:
        for d in duals[i]:
            if d == -1:
                continue
            ind = duals[d].index(i)
            duals[d][ind] = -1

    print()
    print(tris)
    print(duals)
    print(skips)
    print()
    
    #decrement all tri and dual indices by the number of found triangles in skips
    delta = 0
    for i, skip in enumerate(skips):
        if skip:
            delta += 1
            continue
        print(tris[i])
        print(duals[i])
        for n in range(3):
            tris[i][n] -= delta
            if duals[i][n] == -1:
                continue
            duals[i][n] -= delta
        print(tris[i])
        print(duals[i])
        tris[i-delta] = tris[i]
        duals[i-delta] = duals[i]

    #cull extra data
    dec = sum(skips)
    tris, duals = tris[:-dec], duals[:-dec]    

    print(tris)
    print(duals)

    return tris, duals
    """

#test methods
def test_delaunay(points, paths, constraints=[]):

    tris, duals, s_tri = delaunay_new(points, paths)

    points = s_tri + points
    
    #get a set of vertices
    outputs = [[points[i] for i in tri] for tri in tris]

    fig, ax = plt.subplots()
    ax.scatter(*tuple(zip(*points)))
    #ax.add_patch(plt.Circle(coords, radius, fill=False))

    #draw circles on every tri
    if False:
        for tri in outputs:
            ax.add_patch(plt.Circle(*calc_circle(tri), fill=False, color='gray'))

    #draw the input paths
    for p in paths:
        ax.plot(*tuple(zip(*p)), color='black')

    #draw the output tris
    if True:
        for tri in outputs:
            p = list(map(lambda x: np.array(x), tri))
            pavg = sum(p) / 3
            shrink = 0.01 * 0.0
            p = list(map(lambda x: x*(1-shrink) + shrink*pavg, p))
            ax.plot(*tuple(zip(*(p + [p[0]]))))

    #annotate output points
    for ip, p in enumerate(points):
        ax.annotate(str(ip), p)

    #draw duals from average point in tris
    if False:
        for i_d, d in enumerate(duals):
            avg = sum(np.array(outputs[i_d])) / 3
            ax.annotate("%s\n%s"%(str(i_d), str(d)), avg, ha='center', va='center')
            #get dual tri
            for t in d:
                if t == -1:
                    continue
                avg2 = sum(np.array(outputs[t])) / 3
                ax.plot([avg[0], avg2[0]], [avg[1], avg2[1]], c='black')

    #draw voronoi
    if False:         
        for i_d, d in enumerate(duals):
            c1, r1 = calc_circle(outputs[i_d])
            #get dual tri
            for t in d:
                if t == -1:
                    continue
                c2, r2 = calc_circle(outputs[t])
                ax.plot([c1[0], c2[0]], [c1[1], c2[1]], c='black')

        
    ax.set_aspect(1)
    plt.show()
    globals().update(locals())

if __name__ == "__main__":
    from load_svg import loadSvgData, plotSvgData
    import matplotlib.pyplot as plt
    import matplotlib.animation as ani
    points, paths = loadSvgData()
    
    test_delaunay(sum([p[:-1] for p in paths], []), paths)
    plt.show()
