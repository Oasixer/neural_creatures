import pygame as pg
from pygame import Rect
import random
import math
import os
import numpy as np
import copy
import enum

class Facing(enum.Enum):
    UP = 0
    RIGHT = 1.0/3
    DOWN = 2.0/3
    LEFT = 1


BOARD_SIZE_X = 100
BOARD_SIZE_Y = 100
PIXEL_RATIO = 3
SCREEN_SIZE = (BOARD_SIZE_X * PIXEL_RATIO, BOARD_SIZE_Y * PIXEL_RATIO)
LONGEST_DIST = math.sqrt(BOARD_SIZE_X**2+BOARD_SIZE_Y**2)

MUTATION_PROBABILITY = 5.0/1000

EVOLUTION_METHOD = 'right_stays'

n_evolutions = 0

GENOME_LENGTH = 5

NUM_INNER_NEURONS = 2

FPS = 100
TICKS_PER_GENERATION = 100

POPULATION = 25

CAPTION = "Neural creatures"


# Inputs produce 0 to 1
# Connection weights -4 to 4
# Neuron output = tanh(sum(inputs)) = -1 to 1


class Neuron:
    def __init__(self, c):
        self.c = c
        self.value = 0

    def __str__(self):
        return f"Neuron(val: {self.value})"

class SensoryNeuron(Neuron):
    def __init__(self, c):
        super().__init__(c)
        self.sinks = []
    
    def update(self):
        raise NotImplementedError()
    
    def add_sink(self, sink):
        self.sinks.append(sink)
    
    def remove_sink(self, sink):
        self.sinks.remove(sink)
    
    def __str__(self):
        return f"SensoryNeuron(val: {self.value} sinks: {self.sinks})"

class FacingNeuron(SensoryNeuron):
    def __init__(self, c):
        super().__init__(c)

    def update(self):
        return self.c.facing.value

class SinkNeuron(Neuron):
    def __init__(self, c):
        super().__init__(c)
        self.sources = []
    
    def update(self):
        raise NotImplementedError()
    
    def add_source(self, source):
        self.sources.append(source)
    
    def remove_source(self, source):
        self.sources.remove(source)

class ActionNeuron(SinkNeuron):
    def __init__(self, c):
        self.reactThreshhold = 0.25
        self.sources = []
        super().__init__(c)
    
    def add_source(self, source):
        self.sources.append(source)
    
    def remove_source(self, source):
        self.sources.remove(source)

    def update(self):
        self.value = np.tanh(sum([i.value() for i in self.sources]))

    def activate(self):
        raise NotImplementedError()

    def shouldFire(self):
        return abs(self.value) > self.reactThreshhold
    
    def __str__(self):
        return f"ActionNeuron(val: {self.value} sources: {[str(s) for s in self.sources]})"


class NearestCreatureDistanceNeuron(SensoryNeuron):
    def __init__(self, c):
        super().__init__(c)

    def update(self):
        min_dist = 10000000000000
        for c in self.c.app.creatures:
            if c is self or (self.c.posX == c.posX and self.c.posY == c.posY):
                pass
            else:
                new_dist = math.sqrt((self.c.posX-c.posX)**2+(self.c.posY-c.posY)**2)
                min_dist = min(new_dist, min_dist)

        self.value = normalize(min_dist, 0, LONGEST_DIST, 0, 1)

class PosXNeuron(SensoryNeuron):
    def __init__(self, c):
        super().__init__(c)

    def update(self):
        self.value = normalize(self.c.posX, 0, BOARD_SIZE_X, 0, 1)

class PosYNeuron(SensoryNeuron):
    def __init__(self, c):
        super().__init__(c)

    def update(self):
        #  print(self.c)
        self.value = normalize(self.c.posY, 0, BOARD_SIZE_Y, 0, 1)

class RngNeuron(SensoryNeuron):
    def __init__(self, c):
        super().__init__(c)
        self.update()

    def update(self):
        self.value = random.uniform(0,1)

class AlwaysOnNeuron(SensoryNeuron):
    def __init__(self, c):
        super().__init__(c)
        self.value = 1

    def update(self):
        pass

class MoveXNeuron(ActionNeuron):
    def __init__(self, c):
        super().__init__(c)

    def activate(self):
        if super().shouldFire():
            sign = int(self.value / abs(self.value))
            print(f"shouldFire true w/ sign {sign}")
            newPosX = self.c.posX + sign
            if self.c.app.position_available(newPosX, self.c.posY):
                self.c.posX = newPosX
                self.c.rect.centerx = newPosX * PIXEL_RATIO

class MoveYNeuron(ActionNeuron):
    def __init__(self, c):
        super().__init__(c)

    def activate(self):
        if super().shouldFire():
            sign = int(self.value / abs(self.value))
            print(f"shouldFire true w/ sign {sign}")
            newPosY = self.c.posY + sign
            if self.c.app.position_available(self.c.posX, newPosY):
                self.c.posY = newPosY
                self.c.rect.centerY = newPosY * PIXEL_RATIO

class MoveForwardNeuron(ActionNeuron):
    def __init__(self, c):
        super().__init__(c)

    def activate(self):
        if super().shouldFire():
            print(f"shouldFire true w/ sign {sign}")
            sign = int(self.value / abs(self.value))
            if self.c.facing is Facing.DOWN:
                newPosY = self.c.posY + sign
                newPosX = self.c.posX
            elif self.c.facing is Facing.UP:
                newPosY = self.c.posY - sign
                newPosX = self.c.posX
            elif self.c.facing is Facing.RIGHT:
                newPosY = self.c.posY
                newPosX = self.c.posX + sign
            elif self.c.facing is Facing.LEFT:
                newPosY = self.c.posY
                newPosX = self.c.posX - sign
            if self.c.app.position_available(newPosX, newPosY):
                self.c.posX = newPosX
                self.c.posY = newPosY
                self.c.rect.centerx = newPosX * PIXEL_RATIO
                self.c.rect.centery = newPosY * PIXEL_RATIO

class TurnNeuron(ActionNeuron):
    def __init__(self, c):
        super().__init__(c)

    def activate(self):
        if self.value == 0:
            return

        if self.value < 0:
            if self.c.facing is Facing.UP:
                self.c.facing = Facing.LEFT

            elif self.c.facing is Facing.LEFT:
                self.c.facing = Facing.DOWN

            elif self.c.facing is Facing.DOWN:
                self.c.facing = Facing.RIGHT

            else:
                self.c.facing = Facing.UP
        else:
            if self.c.facing is Facing.UP:
                self.c.facing = Facing.RIGHT

            elif self.c.facing is Facing.LEFT:
                self.c.facing = Facing.UP

            elif self.c.facing is Facing.DOWN:
                self.c.facing = Facing.LEFT

            else:
                self.c.facing = Facing.DOWN


class InnerNeuron(SinkNeuron):
    def __init__(self, c):
        super().__init__(c)
        self.sinks = []
        self.value = 0

    def add_sink(self, sink):
        self.sinks.append(sink)
    
    def remove_sink(self, sink):
        self.sinks.remove(sink)

    def update(self):
        self.value = np.tanh(sum(i.value() for i in self.sources))

    def __str__(self):
        return f"InnerNeuron(sources: {[str(s) for s in self.sources]})"

class Connection:
    def __init__(self, weight, source, sink, c):
        self.weight = weight
        self.source = source
        self.sink = sink
        self.c = c

    def __str__(self):
        return f"Connection(val: {self.value()} sourceVal: {self.source.value} weight {self.weight}, source: {self.source}, sink: REDACT)"

    @classmethod
    def random(cls, c):
        if random.randint(0,1):
            source = random.choice(c.sensory_neurons)
        else:
            source = random.choice(c.inner_neurons)
        
        if random.randint(0,1):
            sink = random.choice(c.action_neurons)
        else:
            sink = random.choice(c.inner_neurons)

        weight = random.uniform(-4,4)

        new_connection = cls(weight, source, sink, c)
        sink.add_source(new_connection)
        source.add_sink(new_connection)
        return new_connection
    
    def value(self):
        return self.weight * self.source.value

    @classmethod
    def copy(cls, new_c, parent):
        if issubclass(type(parent.source), SensoryNeuron):
            idx = parent.c.sensory_neurons.index(parent.source)
            source = new_c.sensory_neurons[idx]
        else:
            idx = parent.c.inner_neurons.index(parent.source)
            source = new_c.inner_neurons[idx]

        if issubclass(type(parent.sink), ActionNeuron):
            idx = parent.c.action_neurons.index(parent.sink)
            sink = new_c.action_neurons[idx]
        else:
            idx = parent.c.inner_neurons.index(parent.sink)
            sink = new_c.inner_neurons[idx]

        new_inst = cls(parent.weight, source, sink, new_c)
        sink.add_source(new_inst)
        source.add_sink(new_inst)

        return new_inst
    
    
    def mutate_sink(self, change_type):
        self.sink.remove_source(self)

        if change_type:
            if issubclass(type(self.sink), ActionNeuron):
                self.sink = random.choice(self.c.inner_neurons)
            else:
                self.sink = random.choice(self.c.action_neurons)
        else:
            if issubclass(type(self.sink), ActionNeuron):
                self.sink = random.choice(self.c.action_neurons)
            else:
                self.sink = random.choice(self.c.inner_neurons)

        self.sink.add_source(self)
    
    def mutate_source(self, change_type):
        self.source.remove_sink(self)
        if change_type:
            if issubclass(type(self.source), SensoryNeuron):
                self.source = random.choice(self.c.inner_neurons)
            else:
                self.source = random.choice(self.c.sensory_neurons)
        else:
            if issubclass(type(self.source), SensoryNeuron):
                self.source = random.choice(self.c.sensory_neurons)
            else:
                self.source = random.choice(self.c.inner_neurons)

        self.source.add_sink(self)

            
    def mutate_weight(self):
        self.weight = random.uniform(-4,4)

    def mutate(self):
        seed = random.randint(0,9)
        # 0-1 = mutate source
        # 2 = mutate source, change source type
        # 3-4 = mutate sink
        # 5 = mutate sink, change sink type
        # 6-10 = mutate weight
        if seed < 2:
            self.mutate_source(False)
        elif seed == 2:
            self.mutate_source(True)
        elif seed < 5:
            self.mutate_sink(False)
        elif seed == 5:
            self.mutate_sink(True)
        else:
            self.mutate_weight()


class Creature:
    def __init__(self, app, posX, posY, colour, connections=[]):
        self.app = app
        self.sensory_neurons = [PosXNeuron(self), PosYNeuron(self), RngNeuron(self), AlwaysOnNeuron(self), NearestCreatureDistanceNeuron(self), FacingNeuron(self)]
        self.action_neurons = [MoveXNeuron(self), MoveYNeuron(self), MoveForwardNeuron(self), TurnNeuron(self)]
        self.inner_neurons = [InnerNeuron(self) for i in range(NUM_INNER_NEURONS)]
        self.posX = posX
        self.posY = posY
        self.connections = connections
        self.colour = colour
        self.rect = Rect(0,0,PIXEL_RATIO,PIXEL_RATIO)
        self.rect.centerx = self.posX*PIXEL_RATIO
        self.rect.centery = self.posY*PIXEL_RATIO
        self.facing = Facing(random.randint(1,4))

    def __str__(self):
        return f"Creature(posX {self.posX} posY {self.posY})"

    def randomize_connections(self):
        self.connections = [Connection.random(self) for _ in range(GENOME_LENGTH)]

    @classmethod
    def random(cls, app):
        colour = [random.randint(0,255) for _ in range(3)]
        posX, posY = app.get_random_valid_position()
        inst = cls(app, posX, posY, colour, [])
        inst.randomize_connections()
        return inst
    
    def move_to(self, posX, posY):
        self.posX = posX
        self.posY = posY
        self.rect.centerx = posX * PIXEL_RATIO
        self.rect.centery = posY * PIXEL_RATIO

    def update(self):
        #  print("updating creature.")
        for n in self.sensory_neurons:
            if n.sinks:
                n.update()
            #  print(f"SensoryNeuron value: {n.value}")
        for n in self.inner_neurons:
            n.update()
            #  print(n)
        for n in self.action_neurons:
            n.update()
            #  print(n)
            n.activate()

    def render(self, screen):
        pg.draw.rect(screen, self.colour, self.rect)


    def create_child(self):
        posX, posY = self.app.get_random_valid_position()
        child = Creature(self.app, posX, posY, self.colour, [])
        for connection in self.connections:
            child.connections.append(Connection.copy(child, parent=connection))

        if random.uniform(0,1) < MUTATION_PROBABILITY:
            random.choice(child.connections).mutate()
        return child

def normalize(value, actual_lower, actual_upper, desired_lower, desired_upper):
    return desired_lower + (value - actual_lower) * (desired_upper - desired_lower) / (actual_upper - actual_lower)

class App:
    """
    A class to manage our event, game loop, and overall program flow.
    """
    def __init__(self):
        """
        Get a reference to the display surface; set up required attributes;
        and create a Player instance.
        """
        self.clock = pg.time.Clock()
        self.SCREEN=pg.display.get_surface()
        self.SCREEN_RECT=self.SCREEN.get_rect()
        self.fps = FPS
        self.done = False
        self.keys = pg.key.get_pressed()
        self.creatures = []
        self.make_creatures()
        pg.font.init()
        self.font = pg.font.SysFont('Arial', 30)

    def get_random_valid_position(self):
        while True:
            posX = random.randint(0,BOARD_SIZE_X)
            posY = random.randint(0,BOARD_SIZE_Y)
            if self.position_available(posX, posY):
                return (posX, posY)


    def make_creatures(self):
        for i in range(POPULATION):
            self.creatures.append(C.random(self))
        for creature in self.creatures: # update them to initialize values which may rely on other creatures

    def evolve(self):
        if EVOLUTION_METHOD == "right_stays":
            self.evolve_right_stays()

    def evolve_right_stays(self):
        cutoff_x = BOARD_SIZE_X/2
        new_creatures = []
        i = 0
        while len(new_creatures) < POPULATION:
            c = self.creatures[i]
            if c.posX < cutoff_x:
                self.creatures.remove(c)
            else:
                new_creatures.append(c.create_child())
            i = (i+1)%len(self.creatures)
        self.creatures = new_creatures


    def position_available(self, posX, posY):
        if posX < 0 or posY < 0 or posX > BOARD_SIZE_X or posY > BOARD_SIZE_Y:
            return False
        for c in self.creatures:
            if c.posX == posX and c.posY == posY:
                return False
        return True


    def event_loop(self):
        for event in pg.event.get():
            if event.type == pg.QUIT or self.keys[pg.K_ESCAPE]:
                self.done = True
            elif event.type in (pg.KEYUP, pg.KEYDOWN):
                self.keys = pg.key.get_pressed()

    def render(self):
        self.SCREEN.fill(pg.Color("white"))
        for creature in self.creatures:
            creature.render(self.SCREEN)

        #  textsurface_x = self.font.render('blah', False, (0, 0, 0))
        #  self.SCREEN.blit(textsurface_x,(0,0))
        pg.display.update()

    def main_loop(self):
        self.clock.tick(self.fps)/1000.0
        dt = 0.0
        ticks_since_last_evolution = 0
        while not self.done:
            current_time=pg.time.get_ticks()
            self.event_loop()

            if ticks_since_last_evolution < TICKS_PER_GENERATION:
                for creature in self.creatures:
                    creature.update()
            else:
                self.evolve()
                ticks_since_last_evolution = 0


            self.render()
            ticks_since_last_evolution += 1
            dt = self.clock.tick(self.fps)/1000.0

def main():
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    pg.init()
    pg.display.set_caption(CAPTION)
    pg.display.set_mode(SCREEN_SIZE)

    App().main_loop()
    pg.quit()
    sys.exit()

if __name__ == "__main__":
    main()
