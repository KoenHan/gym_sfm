# cython: language_level=3
import cython
import numpy as np
import math
import random
from copy import copy

from gym_sfm.envs.cython.utils import affine_trans, check_EPS_vector, calc_theta

cimport numpy as np

# Actor creation zone (square only)
@cython.cclass
@cython.final
cdef class Zone():
    cdef public :
        str name, target_zone_name
        bint agent_only, actor_only, can_generate_actor, can_generate_agent
        int generated_step, suspend_limit
        list gac_vertices, ng_vertices, v_vectors, ngv_vectors
        np.ndarray vertices
        double base_yaw
    cdef double scale

    def __init__(self,str name, str tzn, list agac, np.ndarray vertices, double scale, double base_yaw, int suspend_limit=3):
        cdef :
            np.ndarray d1 = np.zeros(2)
            np.ndarray d2 = np.zeros(2)
            np.ndarray ngd1 = np.zeros(2)
            np.ndarray ngd2 = np.zeros(2)
        self.name = name
        self.agent_only = agac[0]
        self.actor_only = agac[1]
        self.target_zone_name = tzn
        self.vertices = vertices
        self.v_vectors = [ self.vertices[(i+1)%4] - self.vertices[i] for i in range(len(self.vertices)) ]
        # Vertex of the actor or agent generation
        d1 = (self.vertices[2] - self.vertices[0])/5
        d2 = (self.vertices[3] - self.vertices[1])/5
        self.gac_vertices = [
            scale*(self.vertices[0] + d1),
            scale*(self.vertices[1] + d2),
            scale*(self.vertices[2] - d1),
            scale*(self.vertices[3] - d2)]
        # Actor cannot be created if agent enters this area
        ngd1 = -d1/np.abs(d1)
        ngd2 = -d2/np.abs(d2)
        self.ng_vertices = [
            self.vertices[0] + ngd1,
            self.vertices[1] + ngd2,
            self.vertices[2] - ngd1,
            self.vertices[3] - ngd2]
        self.ngv_vectors = [ self.ng_vertices[(i+1)%4] - self.ng_vertices[i] for i in range(len(self.ng_vertices)) ]
        self.can_generate_actor = True
        self.can_generate_agent = True
        self.generated_step = 0
        self.suspend_limit = suspend_limit
        self.base_yaw = base_yaw

    cpdef np.ndarray generate_actor_pose(self):
        cdef :
            double x = random.uniform(self.gac_vertices[0][0], self.gac_vertices[2][0])
            double y = random.uniform(self.gac_vertices[0][1], self.gac_vertices[2][1])
        return np.array([x, y], dtype=np.float64)

    cpdef bint check_goal(self, np.ndarray ac_pose):
        cdef :
            double theta = 0
            np.ndarray vp = np.zeros(2)
            np.ndarray vv = np.zeros(2)
            np.ndarray delta = np.zeros(2)
        for vp, vv in zip(self.vertices, self.v_vectors) :
            delta = ac_pose - vp
            theta = calc_theta(*delta.tolist(), *vv.tolist())
            if theta > 0 : return False
        self.can_generate_actor = True
        return True

    cpdef inline bint safety_check(self, np.ndarray ag_pose):
        cdef :
            double theta = 0
            np.ndarray ngvp, ngvv, delta
        for ngvp, ngvv in zip(self.ng_vertices, self.ngv_vectors) :
            delta = ag_pose - ngvp
            theta = calc_theta(*delta.tolist(), *ngvv.tolist())
            if theta > 0 : return True
        self.can_generate_actor = False
        return False

@cython.ccall
cdef inline object get_zone(list zones, str zone_name):
    cdef object z = None
    for z in zones :
        if z.name == zone_name : return z

cpdef object make_zone(object zone, list map_info, int suspend_limit):
    cdef :
        str tzn = 'any'
        bint is_ag = False
        bint is_ac = False
        np.ndarray pose = np.array(zone['pose'], dtype=np.float64)
        np.ndarray shape = np.array(zone['shape'], dtype=np.float64)
        np.ndarray vertices = np.empty((0, 2), dtype=np.float64)
    tzn, is_ag, is_ac = check_tzn(zone)
    vertices = calc_vertices(pose, shape, zone['width'], map_info[2])
    if 'base_yaw' in zone : base_yaw = zone['base_yaw']
    else : base_yaw = calc_base_yaw(vertices, map_info)
    return Zone(zone['name'], tzn, [is_ag, is_ac], vertices, map_info[2], base_yaw, suspend_limit)

@cython.ccall
cdef inline tuple check_tzn(object zone):
    cdef :
        str tzn = 'any'
        bint is_ag = 'ag' == zone['name'][:2]
        bint is_ac = 'ac' == zone['name'][:2]
        bint t_is_ag = False
        bint t_is_ac = False
    if 'target_zone_name' in zone :
        t_is_ag = 'ag' == zone['target_zone_name'][:2]
        t_is_ac = 'ac' == zone['target_zone_name'][:2]
        if zone['target_zone_name'] is zone['name'] :
            raise RuntimeError('--- Cannot set itself as a target.('+zone['name']+') ---')
        elif is_ag and t_is_ac :
            raise RuntimeError('\n--- Agent only Zone\'s target cannot be set to Actor only Zone. ---\n'\
                            '(zone name: '+zone['name']+', target zone name: '+zone['target_zone_name']+')')
        elif is_ac and t_is_ag :
            raise RuntimeError('\n--- Actor only Zone\'s target cannot be set to Agent only Zone. ---\n'\
                            '(zone name: '+zone['name']+', target zone name: '+zone['target_zone_name']+')')
        tzn = zone['target_zone_name']
    return tzn, is_ag, is_ac

@cython.ccall
cdef inline np.ndarray calc_vertices(np.ndarray pose, np.ndarray shape, double width, double scale):
    cdef :
        int i = 0
        int ii = 0
        double yaw = 0.0
        np.ndarray f = copy(pose) + shape
        np.ndarray v = np.array([], dtype=np.float64)
        np.ndarray vv = np.zeros(2)
        np.ndarray frame = np.zeros(2)
        np.ndarray vertices = np.empty((0, 2), dtype=np.float64)
        list sign = []
        list frames = [pose, f]
    # Calculate Zone vertices from frame
    for i, frame in enumerate(frames) :
        yaw = np.arctan2(shape[1], shape[0])
        if i == 0 : sign = [1, -1]
        else : sign = [-1, 1]
        v = np.array([0.0, width/2])
        v = affine_trans(v, np.zeros(2), yaw)
        v = check_EPS_vector(v)
        for ii in sign :
            vv = frame + ii*v
            vv = check_EPS_vector(vv)
            vertices = np.append(vertices, [copy(vv)], axis=0)
    vertices /= scale
    return vertices

@cython.ccall
cdef inline double calc_base_yaw(np.ndarray vertices, list map_info):
    cdef list center = (0.25*vertices.sum(axis=0)).tolist()
    return np.arctan2(map_info[1]/2 - center[1], map_info[0]/2 - center[0])

cpdef list select_generate_ac_zone(list zones, int total_step, np.ndarray ag_pose):
    cdef :
        int i = 0
        int sz_i = 0
        list az_index = []  # available zone index
        list atz_index = [] # available target zone index
        object z = None
        object start_zone = None
        object target_zone = None
    for i, z in enumerate(zones) :
        if not z.agent_only :
            atz_index.append(i)
            if z.can_generate_actor :
                if z.safety_check(ag_pose) : az_index.append(i)
            elif (total_step - z.generated_step) > z.suspend_limit :
                z.can_generate_actor = True
    if not az_index : return False
    sz_i = random.choice(az_index)
    atz_index.remove(sz_i)
    start_zone = zones[sz_i]
    start_zone.can_generate_actor = False
    start_zone.can_generate_agent = False
    start_zone.generated_step = total_step
    if start_zone.target_zone_name == 'any' : target_zone = zones[random.choice(atz_index)]
    else : target_zone = get_zone(zones, start_zone.target_zone_name)
    return [start_zone.generate_actor_pose(), target_zone.generate_actor_pose(), target_zone]

cpdef list select_generate_ag_zone(list zones, int total_step):
    cdef :
        int i = 0
        int sz_i = 0
        list az_index = []  # available zone index
        list atz_index = [] # available target zone index
        object z = None
        object start_zone = None
        object target_zone = None
    for i,z in enumerate(zones) :
        atz_index.append(i)
        if not z.actor_only :
            if z.can_generate_agent : az_index.append(i)
            elif (total_step - z.generated_step) > z.suspend_limit/2 : z.can_generate_actor = True
    if not az_index : return False
    sz_i = random.choice(az_index)
    atz_index.remove(sz_i)
    start_zone = zones[sz_i]
    start_zone.can_generate_actor = False
    start_zone.can_generate_agent = False
    start_zone.generated_step = total_step
    if start_zone.target_zone_name == 'any' : target_zone = zones[random.choice(atz_index)]
    else : target_zone = get_zone(zones, start_zone.target_zone_name)
    return [start_zone.generate_actor_pose(), target_zone.generate_actor_pose(), start_zone.base_yaw]

cpdef bint check_zone_target_existence(list zones):
    cdef :
        list name_list = [ z.name for z in zones ]
        object zone = None
    for zone in zones :
        if zone.target_zone_name != 'any' and zone.target_zone_name not in name_list :
            raise RuntimeError('--- No such target zone.(zone name: '+zone.name+', target zone name: '+zone.target_zone_name+') ---')
    return True