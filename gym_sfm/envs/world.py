import Box2D
from Box2D.b2 import circleShape, fixtureDef, polygonShape, contactListener

class World(Box2D.b2World):
    def __init__(self):
        super(World, self).__init__(gravity=[0, 0], doSleep=True, contactListener=Collision())
        self.actors = []
        self.agents = []
        self.walls = []

    def _destroy(self):
        for actor in self.actors :
            self.DestroyBody(actor)
        for agent in self.agents :
            self.DestroyBody(agent)
        for wall in self.walls :
            self.DestroyBody(wall)
        self.actors = []
        self.agents = []
        self.walls = []

    def make_actor(self, sim_actor_pose, sim_actor_radius):
        world_actor = self.make_circle(sim_actor_pose, sim_actor_radius)
        self.actors.append(world_actor)
        return world_actor

    def make_agent(self, sim_agent_pose, sim_agent_radius):
        world_agent = self.make_circle(sim_agent_pose, sim_agent_radius)
        self.agents.append(world_agent)
        return world_agent

    def make_walls(self, sim_walls):
        for wall in sim_walls:
            w = self.CreateStaticBody( fixtures=fixtureDef( shape=polygonShape( vertices=wall.vertices.tolist() ) ) )
            self.walls.append(w)

    def make_circle(self, obj_pose, obj_radius):
        circle = self.CreateDynamicBody(
                position=obj_pose.tolist(),
                fixtures=fixtureDef( shape=circleShape( radius=float(obj_radius), pos=(0,0) ) ) )
        return circle

class Collision(Box2D.b2ContactListener):
    def __init__(self):
        super(Collision, self).__init__()
        self.position = (-1, -1)
    def BeginContact(self, contact):
        self.position = contact.worldManifold.points[0]
    def EndContact(self, contact):
        self.position = (-1, -1)
