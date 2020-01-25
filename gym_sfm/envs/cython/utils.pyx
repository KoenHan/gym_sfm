# cython: language_level=3
import cython
import math
import numpy as np

cimport numpy as np

ctypedef np.float64_t DTYPE_t

EPS = 1e-6

ReLU = lambda x : x*(x>0)
check_EPS = lambda x : x if abs(x) > EPS else 0.0

cdef inline double clip(double x, double min, double max) :
    cdef double v = 0.0
    if x < min : v = min
    elif x > max : v = max
    else : v = check_EPS(x)
    return v

# Angle between two vectors (signed, positive if a -> b is counterclockwise)
cpdef inline double calc_theta(double ax, double ay, double bx, double by):
    cdef :
        double a_norm = math.sqrt(ax*ax + ay*ay)
        double b_norm = math.sqrt(bx*bx + by*by)
        double inner = ax*bx + ay*by
        double cross = ax*by - ay*bx
        double theta = 0.0
        double cos_theta = 0.0
    if abs(a_norm) < EPS or abs(b_norm) < EPS : return theta
    cos_theta = clip(inner/(a_norm*b_norm), -1.0+EPS, 1.0-EPS)
    theta = (2*(cross > 0.0)-1)*math.acos(cos_theta)
    theta = check_EPS(theta)
    return theta

cpdef inline double check_theta(double theta):
    cdef double t = theta
    if theta > np.pi : t -= 2*np.pi
    elif theta < -np.pi : t += 2*np.pi
    else : t = check_EPS(theta)
    return t

cpdef inline double check_bound(double value, double max, double min):
    cdef double v = 0.0
    if value > max: v = max
    elif value < min: v = min
    else: v = check_EPS(value)
    return v

cpdef inline np.ndarray check_EPS_vector(np.ndarray value):
    cdef list val = []
    for v in value.tolist():
        v = check_EPS(v)
        val.append(v)
    return np.array(val, dtype=np.float64)

# Unit vector from point a to point b and its distance
cpdef inline tuple calc_to_point_vec(double ax, double ay, double bx, double by):
    cdef:
        double nx = bx - ax
        double ny = by - ay
        double dis = math.sqrt(nx*nx + ny*ny)
    if abs(dis) < EPS : return [0, 0], 0
    nx /= dis
    ny /= dis
    return [nx, ny], dis

cpdef inline np.ndarray affine_trans(np.ndarray pose, np.ndarray origin, double yaw):
    cdef:
        np.ndarray delta = pose - origin
        np.ndarray R = np.array([
            [math.cos(yaw), -math.sin(yaw)],
            [math.sin(yaw),  math.cos(yaw)]],
            dtype=np.float64)
        np.ndarray global_pose = R@delta + origin
    return global_pose

cpdef inline bint judge_intersect(double ax, double ay, double bx, double by, double cx, double cy, double dx, double dy):
    cdef:
        double ta = 0.0
        double tb = 0.0
        double tc = 0.0
        double td = 0.0
    if abs(ax-cx) < EPS and abs(ay-cy) < EPS or abs(ax-dx) < EPS and abs(ay-dy) < EPS : return True
    if abs(bx-cx) < EPS and abs(by-cy) < EPS or abs(bx-dx) < EPS and abs(by-dy) < EPS : return True
    ta = (cx - dx)*(ay - cy) + (cy - dy)*(cx - ax)
    tb = (cx - dx)*(by - cy) + (cy - dy)*(cx - bx)
    if ta*tb >= 0 : return False
    tc = (ax - bx)*(cy - ay) + (ay - by)*(ax - cx)
    td = (ax - bx)*(dy - ay) + (ay - by)*(ax - dx)
    if tc*td >= 0 : return False
    else : return True

# Distance from point c to line segment ab and
# whether the perpendicular of point c to line ab intersects line segment ab
cpdef inline tuple calc_to_line_dis(double ax, double ay, double bx, double by, double cx, double cy):
    cdef:
        bint on_line = False
        double ux = cx - ax
        double uy = cy - ay
        double vx = bx - ax
        double vy = by - ay
        double u_norm = math.sqrt(ux*ux + uy*uy)
        double v_norm = math.sqrt(vx*vx + vy*vy)
        double theta = calc_theta(ux, uy, vx, vy)
        double origin = u_norm*math.cos(theta)
        double cross = ux*vy - uy*vx
        double dis = 0.0
    if abs(v_norm) > EPS : dis = cross/v_norm
    if origin < EPS or origin > v_norm - EPS: on_line = False
    else : on_line = True
    return dis, on_line
