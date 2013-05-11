#!/usr/bin/env python
import math
from scipy import weave
import scipy
from scipy.spatial import distance
import numpy as np
import sys
"""
Module to accompany genstruct.  Contains functions to do a diverse
range of things.
"""

# Constants

RAD2DEG = 180./math.pi
DEG2RAD = math.pi/180.

def rotation_matrix2(angle, direction, point=None):
    """
    returns a matrix to rotate about an axis defined by a point and
    direction.
    """
    sina = np.sin(angle)
    cosa = np.cos(angle)
    direction = np.array(direction[:3])/length(direction[:3])

    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array([[0., -direction[2], direction[1]],
                   [direction[2], 0., -direction[0]],
                   [-direction[1], direction[0], 0.]])
    M = np.identity(4)
    M[:3,:3] = R
    if point is not None:
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3,3] = point - np.dot(R, point)
    return M

def unit_vector(vector):
    return vector / length(vector)
    
def project(vector, base_vector):
    """Returns first vector entry projected onto the second."""
    vector = vector[:3].copy()
    base_vector = base_vector[:3].copy()
    return np.dot(base_vector, vector)/np.dot(base_vector, base_vector) * base_vector

def rotation_matrix(axis, angle, point=None):
    """
    returns a 3x3 rotation matrix based on the 
    provided axis and angle
    """
    axis = np.array(axis)
    axis = axis / length(axis)
    a = np.cos(angle / 2.)
    b, c, d = -axis*np.sin(angle / 2.)

    R = np.array([[a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
                  [2*(b*c + a*d), a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
                  [2*(b*d - a*c), 2*(c*d + a*b), a*a + d*d - b*b - c*c]])
    
    M = np.identity(4)
    M[:3,:3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3,3] = point - np.dot(R, point)

    return M

def parallel(vector1, vector2, tol=0.01):

    vector1 = vector1[:3]/length(vector1[:3])
    vector2 = vector2[:3]/length(vector2[:3])

    return np.allclose(np.dot(vector1, vector2), 1., atol=tol)

def antiparallel(vector1, vector2, tol=0.01):

    vector1 = vector1[:3]/length(vector1[:3])
    vector2 = vector2[:3]/length(vector2[:3])

    return np.allclose(np.dot(vector1, vector2), -1., atol=tol)

def rotation_matrix_weave(axis, angle):
    """
    uses c library to compute rotation matrix,
    apparently a 20-fold decrease in time
    NOTE: this fails with error on Wooki.  Keeping code for now...
    """
    R = np.identity(3)
    axis = np.array(axis)
    axis = axis / length(axis)

    support = '#include <math.h>'

    code = """
        double x = sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
        double a = cos(angle / 2.0);
        double b = -(axis[0] / x) * sin(angle / 2.0);
        double c = -(axis[1] / x) * sin(angle / 2.0);
        double d = -(axis[2] / x) * sin(angle / 2.0);

        mat[0] = a*a + b*b - c*c - d*d;
        mat[1] = 2 * (b*c - a*d)
        mat[2] = 2 * (b*d + a*c)

        mat[3*1 + 0] = 2*(b*c + a*d);
        mat[3*1 + 1] = a*a + c*c - b*b - d*d;
        mat[3*1 + 2] = 2*(c*d - a*b);

        mat[3*2 + 0] = 2*(b*d - a*c);
        mat[3*2 + 1] = 2*(c*d + a*b);
        mat[3*2 + 2] = a*a + d*d - b*b - c*c;
        """

    weave.inline(code, ['axis', 'angle', 'R'], support_code=support,
                     libraries=['m'])
    return R


def length(coord1, coord2=np.zeros(3)):
    """ 
    Returns the length between two vectors.
    If only one vector is specified, it returns
    the length of that vector from the origin
    """
    coord1 = np.array(coord1)
    diff = coord2[:3] - coord1[:3]
    return np.sqrt(np.dot(diff, diff.conj()))

def calc_angle(coord1, coord2):
    ic1 = coord1[:3] / length(coord1[:3])
    ic2 = coord2[:3] / length(coord2[:3])
    a = min(max(np.dot(ic1, ic2), -1.0), 1.0)
    return math.acos(a)

def tofrac(x, largest_denom = 32):

    negfrac = False
    if not x >= 0:
        negfrac = True
        x = abs(x)

    scaled = int(round(x * largest_denom))
    whole, leftover = divmod(scaled, largest_denom)
    if leftover:
        while leftover % 2 == 0:
            leftover >>= 1
            largest_denom >>= 1
    if negfrac:
        return -1*whole, leftover, largest_denom

    else:
        return whole, leftover, largest_denom

def to_x(val):
    """ assumes integer value returned to x """
    if val == 0:
        return ""
    elif val == 1:
        return "x"
    elif val == -1:
        return "-x"
    else:
        return "%ix"%val

def to_y(val):
    if val == 0:
        return ""
    elif val == 1:
        return "y"
    elif val == -1:
        return "-y"
    else:
        return "%iy"%val

def to_z(val):
    if val == 0:
        return ""
    elif val == 1:
        return "z"
    elif val == -1:
        return "-z"
    else:
        return "%iz"%val
