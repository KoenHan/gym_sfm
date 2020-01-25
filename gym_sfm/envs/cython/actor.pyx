# cython: language_level=3
import cython
import numpy as np
import math
import random
from copy import copy, deepcopy
from collections import deque

from gym_sfm.envs.node import Node, make_node_net, get_node
from gym_sfm.envs.cython.utils import calc_theta, check_theta, calc_to_point_vec, check_EPS_vector, ReLU

cimport numpy as np

@cython.cclass
@cython.final
cdef class Actor():
    cdef public:
        str name
        double yaw, radius, consider_wall_radius, must_avoid_radius, consider_actor_radius
        list affected_walls, F_target, F_walls, F_actors
        np.ndarray color, pose, target, v, a, to_goal_vec
        object traj, box2d_obj
    cdef :
        int generated_step, unupdate_time, t_index, tp_index
        double consider_actor_theta, convergence_radius
        double max_v, max_a, optimal_v, dt, k, kappa, mass, scale, tau, Fac_A, Fac_B, F_avoid, Fw_A, Fw_B
        double DYAW, DTHETA, DV, DACCEL, EPS, UNUPDATE_LIMIT
        list checkpoints, checkpoints_pose
        np.ndarray DV_VEC, DACCEL_VEC
        object target_zone

    def __init__(self, dict conf, list init_state, list factor, list other, list nodes, list walls):
        self.DYAW = 15*np.pi/180
        self.DTHETA = 30*np.pi/180
        self.DV = 1e-3
        self.DACCEL = 1e-3
        self.DACCEL_VEC = np.array([self.DACCEL, self.DACCEL], dtype=np.float64)
        self.DV_VEC = np.array([self.DV, self.DV], dtype=np.float64)
        self.EPS = 1e-6
        self.UNUPDATE_LIMIT = 300

        self.max_a = conf['max_a']
        self.dt = factor[0]
        self.scale = factor[1]
        self.name = factor[2]
        self.mass = np.random.normal(conf['mass'], conf['sigma']['mass'])
        self.radius = conf['radius']
        self.Fac_A = np.random.normal(conf['Fac_A'], conf['sigma']['Fac_A'])
        self.Fac_B = np.random.normal(conf['Fac_B'], conf['sigma']['Fac_B'])
        self.F_avoid = conf['F_avoid']
        self.Fw_A = np.random.normal(conf['Fw_A'], conf['sigma']['Fw_A'])
        self.Fw_B = np.random.normal(conf['Fw_B'], conf['sigma']['Fw_B'])
        self.consider_actor_radius = conf['consider_actor_radius']
        self.consider_actor_theta = conf['consider_actor_theta']*np.pi/180
        self.must_avoid_radius = conf['must_avoid_radius']
        self.consider_wall_radius = conf['consider_wall_radius']
        self.tau = np.random.normal(conf['tau'], conf['sigma']['tau'])
        self.k = np.random.normal(conf['k'], conf['sigma']['k'])
        self.kappa = np.random.normal(conf['kappa'], conf['sigma']['kappa'])

        self.pose = np.array(init_state[0], dtype=np.float64)/self.scale
        self.target = np.array(init_state[1], dtype=np.float64)/self.scale
        self.yaw = conf['yaw']*np.pi/180
        self.checkpoints, self.checkpoints_pose = self.make_checkpoints(nodes, walls)
        self.v = np.zeros(2)
        self.a = np.zeros(2)
        # self.optimal_v = conf['optimal_v'] + random.uniform(-1, 1)
        self.optimal_v = conf['optimal_v']
        self.max_v = 2*self.optimal_v
        self.to_goal_vec = np.zeros(2)

        self.t_index = 0
        self.tp_index = 0
        self.color = np.array(other[0])/255
        self.convergence_radius = other[1]
        self.box2d_obj = other[2]
        self.box2d_obj.angle = self.yaw
        self.box2d_obj.linearVelocity = self.v
        self.box2d_obj.position = self.pose
        self.target_zone = init_state[2]
        self.F_target = []
        self.F_actors = []
        self.F_walls = []
        self.affected_walls = []
        self.unupdate_time = 0
        self.traj = deque(maxlen=100)
        self.generated_step = factor[3]

    cpdef bint update(self, np.ndarray f_w, np.ndarray f_a, int env_step):
        cdef :
            double delta = 0.0
            np.ndarray f_t = np.zeros(2)
        f_t = self.calc_F_target()
        if self.unupdate_time > self.UNUPDATE_LIMIT :
            if self.t_index > 0 :
                self.t_index -= 1
                self.tp_index -= 1
                self.unupdate_time = 0
            else : return True
        elif self.checkpoints[self.t_index] == 'target':
            self.to_goal_vec = self.target - self.pose
            delta = np.linalg.norm(self.to_goal_vec)
            self.to_goal_vec /= delta
            if self.target_zone.check_goal(self.pose) : return True
        else :
            self.to_goal_vec = self.checkpoints_pose[self.tp_index] - self.pose
            delta = np.linalg.norm(self.to_goal_vec)
            self.to_goal_vec /= delta
            if delta < self.convergence_radius :
                self.t_index += 1
                self.tp_index += 1
        self.a = (f_t + f_w + f_a)/self.mass
        self.a = check_EPS_vector(self.a)
        self.v += self.a*self.dt
        self.yaw = np.arctan2(self.v[1], self.v[0])
        self.box2d_obj.linearVelocity = self.v
        self.box2d_obj.angle = self.yaw
        self.pose = np.array(self.box2d_obj.position, dtype=np.float64)
        if delta < self.convergence_radius : self.tau = 0.3
        if np.all(self.v < 2*self.DV_VEC) and np.all(self.a < 2*self.DACCEL_VEC) : self.unupdate_time += 1
        else : self.unupdate_time = 0
        if env_step % 5 == 0 :
            r = (2 + 1.5*(env_step - self.generated_step)/25) if env_step % 25 == 0 else 2
            self.traj.append([self.pose, r])
        return False

    cdef inline np.ndarray calc_F_target(self):
        cdef :
            double dis = 0.0
            list e_list = []
            list sp_list = self.pose.tolist()
            list tgv_list = self.to_goal_vec.tolist()
            list tp_list = self.checkpoints_pose[self.tp_index].tolist()
            np.ndarray e = np.zeros(2)
            np.ndarray F = np.zeros(2)
        e_list, dis = calc_to_point_vec(sp_list[0], sp_list[1], tp_list[0], tp_list[1])
        if dis < self.radius : return np.zeros(2)
        e = np.array(e_list, dtype=np.float64)
        F = self.mass*(self.optimal_v*e - self.v) / self.tau
        F = check_EPS_vector(F)
        self.F_target.append(F)
        return F

    cpdef np.ndarray calc_F_actor(self, np.ndarray n, double dis, double aa_r, np.ndarray aa_v):
        cdef :
            double r_sum = 0.0
            np.ndarray t = np.zeros(2)
            np.ndarray F = np.zeros(2)
            np.ndarray F1 = np.zeros(2)
            np.ndarray F2 = np.zeros(2)
            np.ndarray delta_v = np.zeros(2)
        if dis >= self.radius : F = self.Fac_A*np.exp(-dis/self.Fac_B)*n
        else:
            r_sum = self.radius + aa_r - dis
            t = np.array([-n[1], -n[0]], dtype=np.float64)
            delta_v = self.v - aa_v
            F1 = (self.Fac_A*np.exp(r_sum/self.Fac_B) + self.k*ReLU(r_sum))*n
            F2 = self.kappa*ReLU(r_sum)*(delta_v@t)*t
            F1 = check_EPS_vector(F1)
            F2 = check_EPS_vector(F2)
            F = F1 + F2
        F = check_EPS_vector(F)
        self.F_actors.append(F)
        return F

    cpdef np.ndarray calc_F_avoid_actor(self, np.ndarray n, np.ndarray aa_v, double aa_yaw, np.ndarray aa_to_goal_vec):
        cdef :
            list v_list = self.v.tolist()
            list atgv_list = aa_to_goal_vec.tolist()
            list tgc_list = self.to_goal_vec.tolist()
            double a_norm = 0.0
            double yaw = abs(calc_theta(v_list[0], v_list[1], tgc_list[0], tgc_list[1]))
            double theta = calc_theta(atgv_list[0], atgv_list[1], tgc_list[0], tgc_list[1])
            double to_goal_yaw = np.arctan2(self.to_goal_vec[1], self.to_goal_vec[0])
            np.ndarray F = np.zeros(2)
            np.ndarray avoid_vec = np.zeros(2)
            np.ndarray l_n = affine_trans(-n, np.zeros(2), -to_goal_yaw)
        to_goal_yaw = check_theta(to_goal_yaw)
        if abs(np.pi/2 - abs(theta)) <= self.DTHETA :
            a_norm = (np.linalg.norm(aa_v) - np.linalg.norm(self.v))*0.01*self.dt
            avoid_vec = -a_norm*self.to_goal_vec
        elif abs(np.pi - abs(theta)) <= self.DTHETA and abs(l_n[1]) < 0.5 and yaw < self.DYAW:
            if l_n[1] > 0.0 : avoid_vec = np.array([self.to_goal_vec[1], -self.to_goal_vec[0]], dtype=np.float64)
            else : avoid_vec = np.array([-self.to_goal_vec[1], self.to_goal_vec[0]], dtype=np.float64)
        else : return F
        F = self.mass*self.F_avoid*avoid_vec
        F = check_EPS_vector(F)
        self.F_actors.append(F)
        return F

    cpdef np.ndarray calc_F_wall(self, np.ndarray n, double dis):
        cdef :
            double r_sum = 0.0
            np.ndarray t = np.zeros(2)
            np.ndarray F = np.zeros(2)
            np.ndarray F1 = np.zeros(2)
            np.ndarray F2 = np.zeros(2)
        if dis >= self.radius : F = self.Fw_A*np.exp(-dis/self.Fw_B)*n
        else:
            r_sum = self.radius - dis
            t = np.array([n[1], -n[0]], dtype=np.float64)
            F1 = (self.Fw_A*np.exp(r_sum/self.Fw_B) + self.k*ReLU(r_sum))*n
            F2 = self.kappa*ReLU(r_sum)*(self.v@t)*t
            F = F1 + F2
        F = check_EPS_vector(F)
        self.F_walls.append(F)
        return F

    cpdef can_consider_actor(self, np.ndarray aa_pose, list walls):
        cdef :
            bint intersect = False
            double aa_theta = 0.0
            double aa_dis = np.linalg.norm(aa_pose - self.pose)
            list aa_n_list = []
            list ap_list = aa_pose.tolist()
            list sp_list = self.pose.tolist()
            list tgv_list = self.to_goal_vec.tolist()
        if aa_dis > self.consider_actor_radius : return False
        else :
            aa_n_list, aa_dis = calc_to_point_vec(ap_list[0], ap_list[1], sp_list[0], sp_list[1])
            aa_theta = abs(calc_theta(-aa_n_list[0], -aa_n_list[1], tgv_list[0], tgv_list[1]))
            if aa_theta > self.consider_actor_theta : return False
            else :
                intersect = False
                for w in walls:
                    for s in w.sides:
                        intersect = judge_intersect(ap_list[0], ap_list[1], sp_list[0], sp_list[1], s[0][0], s[0][1], s[1][0], s[1][1])
                        if intersect : break
                    if intersect : break
                if intersect : return False
                else : return np.array(aa_n_list, dtype=np.float64), aa_dis

    @cython.ccall
    cdef inline make_checkpoints(self, list base_nodes, list walls):
        cdef :
            double cost = 0.0
            list checkpoints = ['target']
            list checkpoints_pose = [ self.target ]
            list nodes = []
            list base_node_plus = []
            object s = None
            object g = None
            object s_node = None
            object g_node = None
            object to_node = None
            object now_node = None
        if not len(base_nodes) : return checkpoints, checkpoints_pose
        s = Node('start', self.pose, [])
        g = Node('target', self.target, [])
        base_nodes_plus = deepcopy(base_nodes)
        base_nodes_plus.extend([s, g])
        nodes = make_node_net(base_nodes_plus, walls)
        s_node = get_node(nodes, 'start')
        g_node = get_node(nodes, 'target')
        # Select a passing point by the Dijkstra method
        now_node = s_node
        now_node.cost = 0
        now_node.from_name = 'start'
        while True:
            now_node.done = True
            for n in now_node.connect :
                cost = now_node.cost + n['dis']
                to_node = get_node(nodes, n['name'])
                if cost < to_node.cost :
                    to_node.cost = cost
                    to_node.from_name = now_node.name
            now_node = None
            for n in nodes :
                if n.done : continue
                elif now_node is None or n.cost < now_node.cost : now_node = n
            if now_node is None or now_node.name == g_node.name : break
        now_node = g_node
        while now_node.from_name is not 'start' :
            checkpoints.append(now_node.from_name)
            now_node = get_node(nodes, now_node.from_name)
            checkpoints_pose.append(now_node.pose)
        checkpoints.reverse()
        checkpoints_pose.reverse()
        return checkpoints, checkpoints_pose

    cpdef void reset_Force(self):
        self.F_target = []
        self.F_actors = []
        self.F_walls = []
        self.affected_walls = []

cpdef object make_actor_random(dict actor, list init_state, list factor, double convergence_radius, list nodes, list walls, object world):
    cdef :
        list color = [random.randint(0, 200) for _ in xrange(3)]
        object box2d_ac = world.make_actor(init_state[0], actor['radius'])
    return Actor(actor, init_state, factor, [color, convergence_radius, box2d_ac], nodes, walls)

# The following functions are copied directly from utils.pyx for faster execution speed
cdef inline np.ndarray affine_trans(np.ndarray pose, np.ndarray origin, double yaw):
    cdef:
        np.ndarray delta = pose - origin
        np.ndarray R = np.array([
            [math.cos(yaw), -math.sin(yaw)],
            [math.sin(yaw),  math.cos(yaw)]],
            dtype=np.float64)
        np.ndarray global_pose = R@delta + origin
    return global_pose

cdef inline bint judge_intersect(double ax, double ay, double bx, double by, double cx, double cy, double dx, double dy):
    cdef:
        double ta = 0.0
        double tb = 0.0
        double tc = 0.0
        double td = 0.0
        double EPS = 1e-6
    if abs(ax-cx) < EPS and abs(ay-cy) < EPS or abs(ax-dx) < EPS and abs(ay-dy) < EPS : return True
    if abs(bx-cx) < EPS and abs(by-cy) < EPS or abs(bx-dx) < EPS and abs(by-dy) < EPS : return True
    ta = (cx - dx)*(ay - cy) + (cy - dy)*(cx - ax)
    tb = (cx - dx)*(by - cy) + (cy - dy)*(cx - bx)
    if ta*tb >= 0 : return False
    tc = (ax - bx)*(cy - ay) + (ay - by)*(ax - cx)
    td = (ax - bx)*(dy - ay) + (ay - by)*(ax - dx)
    if tc*td >= 0 : return False
    else : return True