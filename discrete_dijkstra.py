
#This is a program that solves for generalized dijkstra for 2d meshes.
#It will eventually come with a visualizer.

#https://fribbels.github.io/shortestpath/writeup.html
#https://fribbels.github.io/shortestpath/wavepropagation.html
#https://cs.au.dk/~gerth/advising/thesis/nick-bakkegaard-peter-burgaard.pdf
#http://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Hershberger97.pdf
#https://ecommons.cornell.edu/bitstream/handle/1813/8715/TR000832.pdf

import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke

#sample data taken from
#https://programingmanual.blogspot.com/2018/02/implement-dijkstras-algorithm-to.html

#algorithm from 
#https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm

dijk_nodes = [
    ((0,   1/6), 'A'),
    ((1/3, 0  ), 'B'),
    ((1/3, 1/3), 'C'),
    ((2/3, 0  ), 'D'),
    ((2/3, 1/3), 'E'),
    ((1,   1/6), 'F'),
    ((1,   1/3), 'G'),
    ]

def indexFromLabel(nodes, i):
    return [x[1] for x in nodes].index(i)

dijk_weights = {
    'A': [('B', 2), ('C', 5)],
    'B': [('C', 8), ('D', 7)],
    'C': [('D', 2), ('E', 4)],
    'D': [('E', 6), ('F', 1)],
    'E': [('F', 3)],
    #'G': [('F', 1), ('C', 1)],
    }

#make dijkstra weights bidirectional
def make_bidirectional(weights):
    #make static list, in case more keys are added
    key = list(weights.keys())

    #loop through all keys
    for k in key:

        #loop through all items in the current weight list
        for t, w in weights[k]:

            #if target is not in the weight dict, add an empty one
            if t not in weights.keys():
                weights[t] = []

            #if the current (node, weight) pair isn't in the target list, append it
            v = (k, w)
            if v not in weights[t]:
                weights[t].append(v)

    #sort alphabetically
    for k in weights.keys():
        weights[k].sort(key=lambda x: x[0])
        
make_bidirectional(dijk_weights)

def dijkstra(nodes, weights, start='A', end='F'):
    #grab node labels
    labels = [x[1] for x in nodes]

    #set all distances to infinity, except the starting index.
    dists = [float('inf') for x in nodes]
    dists[labels.index(start)] = 0

    #set all previous nodes to nothing
    prevs = [None for x in nodes]

    #current index is the start index
    node = start

    #print(labels)
    #print(dists)

    #get a set of unvisited nodes, loop through their links
    unvisited = labels.copy()
    while len(unvisited):
        
        #remove the current node from the unvisited list
        unvisited.pop(unvisited.index(node))

        #return if the current node is the target node
        if node == end:
            break

        #check if node has valid weights, else return.
        if not node in weights.keys():
            print("No valid path from %s to %s found."%(start, end))
            return []

        #check all links connected to this node
        for t, w in weights[node]:
            #if t has already been visited, skip
            if t not in unvisited:
                continue
            
            #compute distance from this point
            curdist = dists[labels.index(node)] + w
            
            #compare against current distance, replace if smaller
            ti = indexFromLabel(nodes, t)
            if curdist < dists[ti]:
                dists[ti] = curdist

                #set the last prev to the current node
                prevs[ti] = node

        #if there are no unvisited nodes, the target can't be found.
        if not len(unvisited):
            print("Node %s not found."%end)
            return []

        #find the closest unvisited node
        unvisited_dists = [(dists[labels.index(u)], u) for u in unvisited]
        dist, node = min(unvisited_dists, key=lambda x: x[0])

        print(node, dist)

        #if the next node is invalid, there's no path.
        if dist == float('inf'):
            print("No valid path from %s to %s found."%(start, end))
            return []

    #found shortest path
    path = []

    #generate path by iterating backwards through prev list
    while node:
        path.append(node)
        node = prevs[labels.index(node)]

    path.reverse()
    
    return path

#https://stackoverflow.com/questions/14827650/pyplot-scatter-plot-marker-size
#https://stackoverflow.com/questions/48172928/scale-matplotlib-pyplot-axes-scatter-markersize-by-x-scale/48174228#48174228
#https://scipython.com/blog/visualizing-a-vector-field-with-matplotlib/
#https://stackoverflow.com/questions/7519467/line-plot-with-arrows-in-matplotlib
#https://www.pythonpool.com/matplotlib-circle/#:~:text=%206%20Ways%20to%20Plot%20a%20Circle%20in,scatter%20plot%20is%20a%20graphical%20representation...%20More%20

#nodes and weights are in the format of djik_nodes and weights above
#path is a list nodes.
def draw(nodes, weights, path, nodeSize=33):
    xy = list(zip(*[i[0] for i in nodes]))
    labels = [x[1] for x in nodes]

    #build dijkstra weight matrix
    #rows are from, columns are to
    mweights = [[float("inf") for i in nodes] for j in nodes]
    for k in weights.keys():
        ki = indexFromLabel(nodes, k)
        for t, w in weights[k]:
            ti = indexFromLabel(nodes, t)
            mweights[ki][ti] = w

    #generate path weight matrix
    pmweight = [[False for i in nodes] for j in nodes]
    for st, en in zip(path[:-1], path[1:]):
        sti = indexFromLabel(nodes, st)
        eni = indexFromLabel(nodes, en)
        pmweight[sti][eni] = True
        pmweight[eni][sti] = True

    fig, ax = plt.subplots()

    #draw links
    for isrc, src in enumerate(labels):
        for idst, dst in enumerate(labels[isrc:], isrc):
            link = mweights[isrc][idst]
            if link == float('inf'):
                continue
            x = [xy[0][isrc], xy[0][idst]]
            y = [xy[1][isrc], xy[1][idst]]

            color = 'blue'
            if pmweight[isrc][idst]:
                color = 'red'
            
            ax.plot(x, y, color=color, linewidth=link, zorder=1)

            #label link
            ax.text(sum(x)/2, sum(y)/2, str(link), fontsize=nodeSize*(1/3), zorder=2,
                    horizontalalignment='center',
                    verticalalignment='center').set_path_effects(
                        [withStroke(linewidth=5, foreground='white')])
    
    #draw nodes
    edgecolors = ['red' if l in path else 'blue' for l in labels]
    ax.scatter(xy[0], xy[1], s=nodeSize**2, facecolors='white', edgecolors=edgecolors, zorder=3)

    #label nodes
    for pos, label in nodes:
        ax.text(pos[0], pos[1], label, fontsize=nodeSize*(3/4), zorder=4,
                horizontalalignment='center',
                verticalalignment='center') #this isn't quite centered?
        
    return fig, ax


if __name__ == "__main__":
    #print dijk_nodes
    print([x[1] for x in dijk_nodes])

    #print dijk_weights
    for d in dijk_weights.keys():
        print("{%s:"%(d), str(dijk_weights[d]), "}")
    
    path = dijkstra(dijk_nodes, dijk_weights)
    print("dijkstra path:", path)
    
    fig, ax = draw(dijk_nodes, dijk_weights, path)
    ax.set_xbound((-1/6, 7/6))
    ax.set_ybound((-1/6, 3/6))
    ax.set_aspect(1)
    plt.show()
