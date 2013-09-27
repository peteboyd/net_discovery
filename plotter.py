#!/usr/bin/env python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
import numpy as np

class GraphPlot(object):
    
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.cell = np.identity(3)

    def plot_cell(self, cell, origin=np.zeros(3), colour='b'):
        # add axes labels
        self.cell = cell.copy()
        xyz_a = (cell[0]+origin)/2.
        xyz_b = (cell[1]+origin)/2.
        xyz_c = (cell[2]+origin)/2.
        self.ax.text(xyz_a[0], xyz_a[1], xyz_a[2], 'a')
        self.ax.text(xyz_b[0], xyz_b[1], xyz_b[2], 'b')
        self.ax.text(xyz_c[0], xyz_c[1], xyz_c[2], 'c')

        all_points = [np.sum(a, axis=0)+origin
                      for a in list(self.powerset(self.cell)) if a]
        all_points.append(origin)
        for s, e in itertools.combinations(np.array(all_points), 2):
            if any([self.zero_cross(s-e, i) for i in self.cell]):
                self.ax.plot3D(*zip(s, e), color=colour)
                
    def add_point(self, point=np.zeros(3), label=None, colour='r'):
        p = point.copy()
        try:
            self.ax.scatter(*p, color=colour)
        except TypeError:
            p = p.tolist()
            self.ax.scatter(p, color=colour)
        if label:
            self.ax.text(*p, s=label)

    def add_edge(self, vector, origin=np.zeros(3), label=None, colour='y'):
        """Accounts for periodic boundaries by splitting an edge where
        it intersects with the plane of the boundary conditions.

        """
        point = origin + vector
        max = [0, 0]
        periodics = [(ind, mag) for ind, mag in enumerate(point)
                      if mag > 1. or mag < 0.]
        # the check should be if any of the values are greater than one or
        # less than zero
        #for ind, mag in enumerate(point):
        #    if abs(mag) > abs(max[1]):
        #        max = [ind, mag]
        # determine how many planes intersect the two points.
        if periodics:
            # temp fix
            max = periodics[0]
            # periodic boundary found
            # plane is defined by the other cell vectors
            plane_vec1, plane_vec2 = np.delete(self.cell, max[0], axis=0)
            # plane point is defined by the cell vector
            plane_pt = np.trunc(max[1]) * self.cell[max[0]]

            point1 = np.dot(origin, self.cell)
            vector1 = np.dot(vector, self.cell)
            point2 = self.point_of_intersection(point1, vector1, plane_pt,
                                           plane_vec1, plane_vec2)
            # periodic shift of point2
            point3 = point2 + np.floor(max[1])*-1 * self.cell[max[0]]
            # periodic shift of point
            point4 = np.dot(point - np.floor(point), self.cell)

            self.ax.plot3D(*zip(point2, point1), color=colour)
            self.ax.plot3D(*zip(point4, point3), color=colour)
        else:
            point1 = np.dot(origin, self.cell)
            point2 = np.dot(point, self.cell)
            self.ax.plot3D(*zip(point2, point1), color=colour)
        if label:
            p = origin + 0.5*vector
            p = p - np.floor(p)
            self.ax.text(*p, s=label)
            
    def plot(self):
        plt.show()

    def powerset(self, iterable):
        s = list(iterable)
        return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

    def zero_cross(self, vector1, vector2):
        vector1 = vector1/np.linalg.norm(vector1)
        vector2 = vector2/np.linalg.norm(vector2)
        return np.allclose(np.zeros(3), np.cross(vector1, vector2), atol=0.01)
    
    def point_of_intersection(self, p_edge, edge, p_plane, plane_vec1, plane_vec2):
        """
        Returns a point of intersection between an edge and a plane
        p_edge is a point on the edge vector
        edge is the vector direction
        p_plane is a point on the plane
        plane_vec1 represents one of the vector directions of the plane
        plane_vec2 represents the second vector of the plane

        """
        n = np.cross(plane_vec1, plane_vec2)
        n = n / np.linalg.norm(n)
        l = edge / np.linalg.norm(edge)
        
        ldotn = np.dot(l, n)
        pdotn = np.dot(p_plane - p_edge, n)
        if ldotn == 0.:
            return np.zeros(3) 
        if pdotn == 0.:
            return p_edge 
        return pdotn/ldotn*l + p_edge 

