

#this is the intial svg load script.
#The mesh is trianglated by hand afterwards.

#https://stackoverflow.com/questions/20808614/parsing-svg-file-paths-with-python/35781017

import xml.dom.minidom
import svg.path
import matplotlib.pyplot as plt

def parsePath(dom):
    path = svg.path.parse_path(dom.getAttribute('d'))
    actions = [e for e in path]
    
    c_vertices = [a.end for a in actions] #ignore the start
    vertices = [(v.real, v.imag) for v in c_vertices]
    
    return vertices

def parseCircle(dom):
    x = float(dom.getAttribute('cx'))
    y = float(dom.getAttribute('cy'))
    return (x,y)

def inv_lerp(x, low, high):
    return (x - low) / (high - low)

def loadSvgData(src="sample_src.svg"):
    #load svg from file
    doc = xml.dom.minidom.parse(src)

    #get a list of paths and points, extracted from circles
    def_paths = doc.getElementsByTagName("path")
    def_points = doc.getElementsByTagName("circle")

    #convert paths and points to cyclic lists of tuples
    points = list(map(parseCircle, def_points))
    paths =  list(map(parsePath, def_paths))

    #scale paths and points to [0, 1)
    #keeping aspect ratio to the larger dimension.
    #performs an inverse lerp.

    #find the min and max of the whole scene
    #perform inv lerps for them
    #scaled_points = list(map(list(map(inv_lerp
    po_x, po_y = tuple(zip(*points))
    pa_x, pa_y = tuple(zip(*map(lambda p: zip(*p), paths)))

    #max of paths and points
    max_x = max(list(map(max, pa_x)) + list(po_x))
    max_y = max(list(map(max, pa_y)) + list(po_y))

    #min of paths and points
    min_x = min(list(map(min, pa_x)) + list(po_x))
    min_y = min(list(map(min, pa_y)) + list(po_y))

    #find the ranges
    dx = max_x - min_x
    dy = max_y - min_y

    #set the target aspect ratio to keep the smallest size 1.0
    #expanding dy and dx before the transform keeps things simple
    #reversing the comparison here keeps the biggest size at 1.0
    if dy < dx:
        dx *= dy/dx
    else:
        dy *= dx/dy
    
    #mirror the y axis can be done by inverse lerping y
    #from the max point, rather than the min. This could probably be
    #done with a matrix, but this scipt doesn't use numpy, and writing
    #one just for here doesn't seem like its worth it.
    t_points = [((p[0]-min_x)/dx, (max_y-p[1])/dy) for p in points]
    t_paths = list(map(lambda path:
                       [((p[0]-min_x)/dx, (max_y-p[1])/dy) for p in path],
                       paths))
    
    return t_points, t_paths
    
def plotSvgData(points, paths):
    fig, ax = plt.subplots()
    for p in paths:
        ax.plot(*tuple(zip(*p)))

    #if len(points): #for the visa sample
    ax.scatter(*tuple(zip(*points)))
    ax.set_aspect(1)
    plt.show()

if __name__ == "__main__":
    plotSvgData(*loadSvgData("sample_src.svg"))
    #plotSvgData(*loadSvgData("Visa_Inc._logo.svg"))
