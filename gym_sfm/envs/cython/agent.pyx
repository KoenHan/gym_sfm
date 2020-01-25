# cython: language_level=3
import cython
from gym import spaces
import Box2D
import math
import random
import numpy as np
from collections import deque

from gym_sfm.envs.cython.utils import check_bound, check_theta

cimport numpy as np

ctypedef np.float64_t DTYPE_t

@cython.cclass
@cython.final
cdef class Agent():
    cdef public:
        double lidar_linear_range, lidar_rad_range, lidar_rad_step
        double yaw, v, omega, dis, angle_to_goal, radius
        list lidar, lidar_point
        np.ndarray pose, target, color
        object action_space, box2d_obj, traj
    cdef :
        str name
        double mass, scale, dt, max_v, min_v, max_omega, was_dis

    def __init__(self, dict conf, list init_pose, list factor, list other):
        self.max_v = conf['max_v']
        self.min_v = conf['min_v']
        self.max_omega = conf['max_omega']
        self.dt = factor[0]
        self.scale = factor[1]
        self.name = factor[2]
        self.color = np.array(conf['color'])
        self.mass = conf['mass']
        self.lidar_linear_range = conf['lidar_linear_range'] # [m]
        self.lidar_rad_range = conf['lidar_rad_range']*math.pi/180  # deg -> rad
        self.lidar_rad_step = conf['lidar_rad_step']*math.pi/180    # deg -> rad
        self.radius = conf['radius'] # [m]
        self.lidar_point = []

        self.pose = np.array(init_pose[0], dtype=np.float64)/self.scale
        # self.pose = np.array([2.0, 1.0], dtype=np.float64)/self.scale
        self.target = np.array(init_pose[1], dtype=np.float64)/self.scale
        # self.target = np.array([2.0, 4.0], dtype=np.float64)/self.scale
        self.yaw = init_pose[2] + 0.25*random.uniform(-math.pi, math.pi)
        # self.yaw = init_pose[2]
        self.v = check_bound(conf['v'], self.max_v, self.min_v) # linear v
        self.omega = check_bound(conf['omega'], self.max_omega, -self.max_omega) # angle v
        self.action_space = spaces.Box(np.array([self.min_v, -self.max_omega]), np.array([self.max_v, self.max_omega]), dtype=np.float64)
        self.lidar = [ LidarCallback() for _ in xrange(int(self.lidar_rad_range/self.lidar_rad_step)+1) ]
        delta = self.target - self.pose
        self.dis = np.linalg.norm(delta)
        self.was_dis = np.nan
        self.angle_to_goal = self.calc_angle_to_goal(delta)

        self.traj = deque(maxlen=100)
        self.box2d_obj = other[0]
        self.box2d_obj.angle = self.yaw
        self.box2d_obj.linearVelocity = [self.v*math.cos(self.yaw), self.v*math.sin(self.yaw)]
        self.box2d_obj.angularVelocity = self.omega
        self.box2d_obj.position = self.pose

    cpdef list update(self, np.ndarray action, int env_step):
        cdef :
            bint is_goal = False
            double r = 0
            np.ndarray delta = np.zeros(2)
        self.v = check_bound(action[0], self.max_v, self.min_v)
        self.omega = check_bound(action[1], self.max_omega, -self.max_omega)
        self.box2d_obj.linearVelocity = (self.v*math.cos(self.yaw), self.v*math.sin(self.yaw))
        self.box2d_obj.angularVelocity = self.omega
        self.pose = np.array(self.box2d_obj.position)
        self.yaw = check_theta(self.box2d_obj.angle)
        if env_step % 5 == 0 :
            r = (2 + 1.5*env_step/25) if env_step % 25 == 0 else 2
            self.traj.append([self.pose, r])
        self.was_dis = self.dis
        delta = self.target - self.pose
        self.dis = np.linalg.norm(delta)
        is_goal = True if self.dis < self.radius else False
        self.angle_to_goal = self.calc_angle_to_goal(delta, is_goal)
        return [is_goal, self.dis, self.angle_to_goal, self.was_dis - self.dis]

    cpdef tuple observation(self, object world, list to_goal_info):
        cdef :
            bint is_collision = False
            list obs = []
        obs = self.raycast(world)
        is_collision = True if min(obs) < (self.radius+0.05)/self.lidar_linear_range else False
        obs.extend(to_goal_info)
        return np.array(obs, dtype=np.float64), is_collision

    # Observation by lidar
    cdef list raycast(self, object world):
        cdef :
            double yaw = 0.0
            list pos = [ float(p) for p in self.pose ]
            list lidar_point = []
        for i, l in enumerate(self.lidar):
            yaw = self.yaw + i*self.lidar_rad_step - 0.5*self.lidar_rad_range
            l.fraction = 1.0
            l.p1 = pos
            l.p2 = (float(pos[0] + self.lidar_linear_range*math.cos(yaw)),
                    float(pos[1] + self.lidar_linear_range*math.sin(yaw)))
            world.RayCast(l, l.p1, l.p2)
            # lidar_point.append(l.fraction*self.lidar_linear_range + np.random.normal(0, 0.01))
            lidar_point.append(l.fraction + np.random.normal(0, 0.01)/self.lidar_linear_range)
        return lidar_point

    cpdef double calc_angle_to_goal(self, delta, is_goal=False):
        return abs(check_theta(np.arctan2(delta[1], delta[0]) - self.yaw))*(not is_goal)

cpdef object make_agent_random(dict agent, list init_pose, list factor, object world):
    cdef object box2d_ag = world.make_agent(init_pose[0], agent['radius'])
    return Agent(agent, init_pose, factor, [box2d_ag])

class LidarCallback(Box2D.b2.rayCastCallback):
    def ReportFixture(self, fixture, point, normal, fraction):
        if (fixture.filterData.categoryBits & 1) == 0 : return 1
        self.p2 = point
        self.fraction = fraction
        return fraction
