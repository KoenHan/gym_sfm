import numpy as np

from gym_sfm.envs.cython.utils import judge_intersect

get_node = lambda nodes, node_name : [f for f in nodes if f.name == node_name][0]

class Node():
    def __init__(self, name, pose, connect, scale=1):
        self.name = name
        self.scale = scale
        self.pose = np.array(pose, dtype=np.float64)/self.scale
        self.connect = connect
        # for Dijkstra method
        self.cost = float('inf')
        self.done = False
        self.from_name = ''

    def __str__(self):
        print('name: ', self.name)
        print('pose: ', self.pose)
        print('connect', self.connect)
        print('cost: ', self.cost)
        print('done: ', self.done)
        print('from_name: ', self.from_name)
        return ''

def make_node_net(nodes, walls):
    for i in range(0, len(nodes)):
        for j in range(i+1, len(nodes)) :
            search_connect_node(nodes[i], nodes[j], walls)
    return nodes

def search_connect_node(f_a, f_b, walls):
    intersect = False
    for w in walls:
        for s in w.sides:
            intersect = judge_intersect(*f_a.pose.tolist(), *f_b.pose.tolist(), *s[0].tolist(), *s[1].tolist())
            if intersect : break
        if intersect : break
    if not intersect :
        dis = np.linalg.norm(f_a.pose - f_b.pose)
        f_a.connect.append({ 'name' : f_b.name, 'pose' : f_b.pose, 'dis' : dis })
        f_b.connect.append({ 'name' : f_a.name, 'pose' : f_a.pose, 'dis' : dis })

def get_nearest_node(nodes, pose):
    delta = 0.0
    min_delta  = float("inf")
    closest_node = nodes[0]
    for node in nodes:
        delta = np.linalg.norm(node.pose - pose)
        if min_delta > delta:
            min_delta = delta
            closest_node = node
    return closest_node
