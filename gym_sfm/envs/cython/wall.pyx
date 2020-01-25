# cython: language_level=3
import cython
import math
import numpy as np
from copy import copy, deepcopy

from gym_sfm.envs.cython.utils import affine_trans, judge_intersect, check_EPS_vector, check_EPS

cimport numpy as np

@cython.cclass
@cython.final
cdef class Wall():
    cdef public :
        str name
        double scale
        list sides, side_center
        np.ndarray pose, vertices, mean, normal
    cdef double EPS

    def __init__(self, name, vertices, scale=1):
        cdef :
            int len_vertices = 0
            list n = []
            list nn = []
        self.EPS = 1e-6
        self.name = name
        self.scale = scale
        self.pose = vertices[0]/self.scale
        self.vertices = vertices/self.scale
        self.mean = np.mean(self.vertices, axis=0)
        len_vertices = len(self.vertices)
        self.sides = [ [self.vertices[i], self.vertices[(i+1)%len_vertices]] for i in xrange(len_vertices)]
        self.side_center = [ (s[0] + s[1])/2 for s in self.sides] # Midpoint of each side
        # Normal vector of each side
        n = []
        n_append = n.append
        for s in self.sides:
            nn, _ = calc_to_point_vec(s[0][0], s[0][1], s[1][0], s[1][1])
            n_append([nn[1], -nn[0]])
        self.normal = np.array(n, dtype=np.float64)

    def __str__(self):
        print('name: '+self.name)
        print('pose: ', self.pose)
        print('side: ')
        print(self.sides)
        print('vertices: ')
        print(self.vertices)
        print('scale: ', self.scale)
        return ''

    # Calculate unit vector and distance to wall
    @cython.ccall
    cpdef inline calc_to_wall_vec(self, np.ndarray ac_pose, list walls, double ob_radius):
        cdef :
            bint on_line = False
            bint intersect = False
            double d = 0.0
            double dis = 0.0
            double min_d = float('inf')
            list pose = ac_pose.tolist()
            list n_list = []
            list positive_vertice = []
        # Select observable side
        for i, s in enumerate(self.sides) :
            d, on_line = calc_to_line_dis(s[0][0], s[0][1], s[1][0], s[1][1], pose[0], pose[1])
            if self.EPS < d < ob_radius and on_line :
                intersect = self.calc_intersect(pose, self.side_center[i].tolist(), walls)
                if not intersect : return self.normal[i], d, [self.side_center[i]]
        # If there is no observable side, select observable vertices
        for i, v in enumerate(self.vertices.tolist()) :
            n_list, d = calc_to_point_vec(v[0], v[1], pose[0], pose[1])
            if self.EPS < d < ob_radius and d < min_d :
                # intersect = self.calc_intersect(pose, v, walls)
                # if not intersect : positive_vertice = [np.array(n_list, dtype=np.float64), d, [self.vertices[i]]]
                positive_vertice = [np.array(n_list, dtype=np.float64), d, [self.vertices[i]]]
                min_d = d
        if len(positive_vertice) : return positive_vertice
        else : return False

    @cython.ccall
    cdef inline bint calc_intersect(self, list p1, list element, list walls):
        cdef :
            bint intersect = False
            double d = 0.0
        for w in walls :
            if w.name == self.name : continue
            intersect = False
            for ss in w.sides :
                intersect = judge_intersect(*p1, *element, *ss[0].tolist(), *ss[1].tolist())
                if intersect : break
            if intersect : break
        return intersect

cpdef list make_wall(dict wall, double scale):
    cdef :
        str name = ''
        int w_num = 0
        int len_frames_1 = 0
        int len_frames_2 = 0
        bint use_s_vertices = False
        double EPS = 1e-6
        double yaw = 0.0
        double theta = 0.0
        list w_v = []
        list sign = []
        list frames = []
        list w_list = []
        list vertices = []
        list s_vertices = []
        np.ndarray O = np.zeros(2)
        np.ndarray f = np.zeros(2)
        np.ndarray v = np.zeros(2)
        np.ndarray pose = np.array(wall['pose'], dtype=np.float64)
        np.ndarray shape = np.zeros(2)
        np.ndarray frames_lf1 = np.zeros(2)
        object w = None
        object vertices_append = vertices.append
    if isinstance(wall['shape'][0], list) : shape = np.array(wall['shape'], dtype=np.float64)
    else : shape = np.array([wall['shape']], dtype=np.float64)
    f = copy(pose)
    frames = [pose]
    for s in shape :
        f += s
        frames.append(copy(f))
    # Calculate rectangular vertices from frames
    len_frames_1 = len(frames) - 1
    if abs(np.linalg.norm(frames[0] - frames[len_frames_1])) < EPS :
        use_s_vertices = True
        len_frames_2 = len_frames_1 - 1
        yaw = np.arctan2(shape[len_frames_2][1], shape[len_frames_2][0])
        theta = calc_theta(-shape[len_frames_2][0], -shape[len_frames_2][1], shape[0][0], shape[0][1])
        sign = [1, -1, -1, 1]
        v = affine_trans(np.array([wall['width']/(2*math.tan(theta/2)), wall['width']/2]), O, yaw)
        v = check_EPS_vector(v)
        frames_lf1 = frames[len_frames_1]
        s_vertices = [ copy(check_EPS_vector(ii*v + frames_lf1)) for ii in sign ]
    for i, frame in enumerate(frames) :
        if i == 0 :
            if use_s_vertices :
                vertices.extend(s_vertices[0:2])
                continue
            else :
                yaw = np.arctan2(shape[i][1], shape[i][0])
                theta = np.pi
                sign = [1, -1]
        elif i == len_frames_1 :
            if use_s_vertices :
                vertices.extend(s_vertices[2:4])
                continue
            else :
                yaw = np.arctan2(shape[i-1][1], shape[i-1][0])
                theta = np.pi
                sign = [-1, 1]
        else :
            yaw = np.arctan2(shape[i-1][1], shape[i-1][0])
            theta = calc_theta(-shape[i-1][0], -shape[i-1][1], shape[i][0], shape[i][1])
            sign = [-1, 1, 1, -1]
        v = affine_trans(np.array([wall['width']/(2*math.tan(theta/2)), wall['width']/2]), O, yaw)
        v = check_EPS_vector(v)
        for ii in sign : vertices_append(copy(check_EPS_vector(ii*v + frame)))
    # Generate walls
    w_num = int(len(vertices)/4)
    for i in xrange(w_num) :
        w_v = vertices[4*i:4*(i+1)]
        name = wall['name'] + '-' + str(i)
        w = Wall(name, np.array(w_v, dtype=np.float64), scale)
        w_list.append(w)
    return w_list

# The following functions are copied directly from utils.pyx for faster execution speed
cdef inline tuple calc_to_point_vec(double ax, double ay, double bx, double by):
    cdef:
        double nx = bx - ax
        double ny = by - ay
        double dis = math.sqrt(nx*nx + ny*ny)
    nx /= dis
    ny /= dis
    return [nx, ny], dis

cdef inline tuple calc_to_line_dis(double ax, double ay, double bx, double by, double cx, double cy):
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
        double EPS = 1e-6
    if abs(v_norm) > EPS : dis = cross/v_norm
    if origin < EPS or origin > v_norm - EPS: on_line = False
    else : on_line = True
    return dis, on_line

cdef inline double calc_theta(double ax, double ay, double bx, double by):
    cdef :
        double a_norm = math.sqrt(ax*ax + ay*ay)
        double b_norm = math.sqrt(bx*bx + by*by)
        double inner = ax*bx + ay*by
        double cross = ax*by - ay*bx
        double theta = 0.0
        double cos_theta = 0.0
        double EPS = 1e-6
    if abs(a_norm) < EPS or abs(b_norm) < EPS : return theta
    cos_theta = clip(inner/(a_norm*b_norm), -1.0+EPS, 1.0-EPS)
    if cross < 0.0 : theta = -1*math.acos(cos_theta)
    else : theta = math.acos(cos_theta)
    theta = check_EPS(theta)
    return theta

cdef inline double clip(double x, double a, double b):
    cdef double v = 0.0
    if x < a : v = a
    elif x > b : v = b
    else : v = check_EPS(x)
    return v