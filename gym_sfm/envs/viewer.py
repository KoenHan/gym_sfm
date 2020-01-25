import pyglet
import math
import numpy as np
from gym.envs.classic_control import rendering

class Viewer(rendering.Viewer):
    def __init__(self, screen, map_view_conf, actor_view_conf, agent_view_conf):
        super(Viewer, self).__init__(screen[0], screen[1])
        self.POINT_COLOR = [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)]
        self.POINT_RADIUS = [3.0, 6.0, 9.0, 12.0]
        self.scale = screen[2]
        self.show_nodes = map_view_conf['show_nodes'] if 'show_nodes' in map_view_conf else False
        self.show_zones = map_view_conf['show_zones'] if 'show_zones' in map_view_conf else False
        self.show_ng_zones = map_view_conf['show_ng_zones'] if 'show_ng_zones' in map_view_conf else False
        self.show_walls_mean = map_view_conf['show_walls_mean'] if 'show_walls_mean' in map_view_conf else False
        self.show_walls_vertices = map_view_conf['show_walls_vertices'] if 'show_walls_vertices' in map_view_conf else False
        self.show_v = actor_view_conf['show_v'] if 'show_v' in actor_view_conf else False
        self.show_goal = actor_view_conf['show_goal'] if 'show_goal' in actor_view_conf else False
        self.show_F_walls = actor_view_conf['show_F_walls'] if 'show_F_walls' in actor_view_conf else False
        self.show_F_actors = actor_view_conf['show_F_actors'] if 'show_F_actors' in actor_view_conf else False
        self.show_F_target = actor_view_conf['show_F_target'] if 'show_F_target' in actor_view_conf else False
        self.show_actor_traj = actor_view_conf['show_traj'] if 'show_traj' in actor_view_conf else False
        self.show_to_goal_vec = actor_view_conf['show_to_goal_vec'] if 'show_to_goal_vec' in actor_view_conf else False
        self.show_actor_color = actor_view_conf['show_actor_color'] if 'show_actor_color' in actor_view_conf else False
        self.show_affected_walls = actor_view_conf['show_affected_walls'] if 'show_affected_walls' in actor_view_conf else False
        self.show_consider_radius = actor_view_conf['show_consider_radius'] if 'show_consider_radius' in actor_view_conf else False

    def make_actor(self, env_actors):
        for actor in env_actors:
            ac_trans = rendering.Transform(actor.pose*self.scale)
            if self.show_consider_radius :
                self.draw_circle(actor.consider_actor_radius*self.scale, 20, color=(0.0, 0.749, 1.0), filled=False, linewidth=2).add_attr(ac_trans)
                self.draw_circle(actor.must_avoid_radius*self.scale, 20, color=(0.0, 0.749, 1.0), filled=False, linewidth=2).add_attr(ac_trans)
                self.draw_circle(actor.consider_wall_radius*self.scale, 20, color=(0.0, 0.0, 0.804), filled=False, linewidth=2).add_attr(ac_trans)
            if self.show_goal :
                t_trans = rendering.Transform(actor.target*self.scale)
                self.draw_circle(0.1*self.scale, 20, color=actor.color).add_attr(t_trans)
                self.draw_circle(0.1*self.scale*1.5, 20, color=actor.color, filled=False, linewidth=1.5).add_attr(t_trans)
            actor_color = actor.color if self.show_actor_color else (0.0, 0.0, 0.0)
            self.draw_circle(actor.radius*self.scale, 20, color=actor_color).add_attr(ac_trans)
            self.draw_circle(actor.radius*self.scale, 20, color=(0.0, 0.0, 0.0), filled=False, linewidth=2).add_attr(ac_trans)
            if self.show_v :
                self.draw_line((0, 0), actor.v*self.scale, color=(1.0, 0.0, 0.0)).add_attr(ac_trans)
            if self.show_to_goal_vec :
                self.draw_line((0, 0), actor.to_goal_vec*self.scale, color=(0.0, 0.0, 1.0)).add_attr(ac_trans)
            if self.show_F_target :
                for f_t in actor.F_target :
                    self.draw_line((0, 0), f_t, color=(0.0, 0.392, 0.0)).add_attr(ac_trans)
            if self.show_F_actors :
                for f_c in actor.F_actors :
                    self.draw_line((0, 0), f_c, color=(0.698, 0.133, 0.133)).add_attr(ac_trans)
            if self.show_F_walls :
                for f_w in actor.F_walls :
                    self.draw_line((0, 0), f_w, color=(1.0, 0.0, 1.0)).add_attr(ac_trans)
            if self.show_affected_walls :
                for a_w in actor.affected_walls :
                    aw_trans = rendering.Transform(a_w*self.scale)
                    self.draw_circle(self.POINT_RADIUS[1], 20, color=actor.color).add_attr(aw_trans)
            if self.show_actor_traj :
                for p in actor.traj :
                    p_trans = rendering.Transform(p[0]*self.scale)
                    self.draw_circle(p[1], 20, color=actor.color).add_attr(p_trans)

    def make_agent(self, agent):
        if agent is None : return
        for i in range(len(agent.lidar)):
            p1 = [ float(p)*self.scale for p in agent.lidar[i].p1 ]
            p2 = [ float(p)*self.scale for p in agent.lidar[i].p2 ]
            self.draw_polyline([p1, p2], color=(1.0 ,0.0 ,0.0), linewidth=1)
        for p in agent.traj :
            p_trans = rendering.Transform(p[0]*self.scale)
            self.draw_circle(p[1], 20, color=(1.0, 0.0, 0.0)).add_attr(p_trans)
        ag_trans = rendering.Transform(agent.pose*self.scale)
        t_trans = rendering.Transform(agent.target*self.scale)
        self.draw_circle(0.2*self.scale, 20, color=(1.0, 0.0, 0.0)).add_attr(t_trans)
        self.draw_circle(0.2*self.scale*1.5, 20, color=(1.0, 0.0, 0.0), filled=False, linewidth=1.5).add_attr(t_trans)
        # self.draw_circle(0.2*self.scale, 20, color=agent.color).add_attr(t_trans)
        # self.draw_circle(0.2*self.scale*2, 20, color=agent.color, filled=False, linewidth=2).add_attr(t_trans)
        self.draw_circle(agent.radius*self.scale, 20, color=agent.color).add_attr(ag_trans)
        v = [agent.v*math.cos(agent.yaw)*self.scale, agent.v*math.sin(agent.yaw)*self.scale]
        self.draw_polyline([[0,0], v], color=(0.0, 0.800, 0.0), linewidth=3).add_attr(ag_trans)

    def make_map(self, env_walls, env_zones, env_forks):
        for wall in env_walls:
            vertices = wall.vertices*self.scale
            self.draw_polygon(vertices, color=(0.0, 0.0, 0.0))
            if self.show_walls_mean :
                w_trans = rendering.Transform(wall.mean*self.scale)
                self.draw_circle(0.1*self.scale, 20, color=(1.0, 0.0, 0.0)).add_attr(w_trans)
            if self.show_walls_vertices :
                for v, c, p_r in zip(vertices, self.POINT_COLOR, self.POINT_RADIUS):
                    v_trans = rendering.Transform(v)
                    self.draw_circle(p_r, 20, color=c).add_attr(v_trans)
        if self.show_zones :
            for zone in env_zones:
                if self.show_ng_zones :
                    ng_vertices = [ ngv*self.scale for ngv in zone.ng_vertices ]
                    self.draw_polygon(ng_vertices, color=(0.114, 0.568, 1.0))
                    for v, c, p_r in zip(ng_vertices, self.POINT_COLOR, self.POINT_RADIUS):
                        v_trans = rendering.Transform(v)
                        self.draw_circle(p_r, 20, color=c).add_attr(v_trans)
                vertices = zone.vertices*self.scale
                z_c = (0.5, 0.623, 1.0) if zone.name[0:2] == 'ag' else (0.6, 0.898, 1.0)
                self.draw_polygon(vertices, color=z_c)
        if self.show_nodes :
            for fork in env_forks :
                f_trans = rendering.Transform(fork.pose*self.scale)
                self.draw_circle(0.1*self.scale, 20, color=(0.0, 0.0, 1.0)).add_attr(f_trans)
