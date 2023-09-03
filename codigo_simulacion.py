import scipy as sp
import math
import scipy.constants
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation


class TreeNode:
    def __init__(self, x, y, w):
        self.x = x
        self.y = y
        self.w = w
        self.children = [None, None, None, None]  # NW, NE, SW, SE
        self.leaf = True
        self.particle = None
        self.total_center = Vector(0, 0)
        self.center = None
        self.total_mass = 0
        self.count = 0

    def split(self):
        new_width = self.w * 0.5
        self.children[0] = TreeNode(self.x, self.y, new_width)  # NW
        self.children[1] = TreeNode(self.x + new_width, self.y, new_width)  # NE
        self.children[2] = TreeNode(self.x, self.y + new_width, new_width)  # SW
        self.children[3] = TreeNode(self.x + new_width, self.y + new_width, new_width)  # SE
        self.leaf = False

    def which(self, pos):
      half_width = self.w * 0.5
      if pos.y < self.y + half_width:
          return 0 if pos.x < self.x + half_width else 1
      return 2 if pos.x < self.x + half_width else 3


    def insert(self, new_particle):
      # Actualiza el centro de masa y la masa total del nodo
      new_total_mass = self.total_mass + new_particle.mass
      new_total_center = Vector(
          (self.total_center.x * self.total_mass + new_particle.pos.x * new_particle.mass) / new_total_mass,
          (self.total_center.y * self.total_mass + new_particle.pos.y * new_particle.mass) / new_total_mass
      )  
      self.total_mass = new_total_mass
      self.total_center = new_total_center

    # Si el nodo es una hoja y no tiene ninguna partícula
      if self.leaf and self.particle is None:
          self.particle = new_particle
          return

    # Si el nodo es una hoja y ya tiene una partícula
      if self.leaf and self.particle is not None:
          old_particle = self.particle
          self.split()  # Divide el nodo en cuatro hijos
          self.insert(old_particle)  # Inserta la partícula existente en el nodo hijo correspondiente
          self.insert(new_particle)  # Inserta la nueva partícula en el nodo hijo correspondiente
          return

    # Si el nodo no es una hoja
      if not self.leaf:
          quadrant = self.which(new_particle.pos)  # Determina en qué cuadrante insertar la partícula
          self.children[quadrant].insert(new_particle)  # Inserta la partícula en el nodo hijo correspondiente
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return '<{:.6f}, {:.6f}>'.format(self.x, self.y)

    def copy(self):
        return Vector(self.x, self.y)

    def equals(self, other):
        return self.x == other.x and self.y == other.y

    def add(self, b):
        self.x += b.x
        self.y += b.y

    def sub(self, b):
        self.x -= b.x
        self.y -= b.y

    def mult(self, scalar):
        self.x *= scalar
        self.y *= scalar

    def mag_squared(self):
        return self.x * self.x + self.y * self.y

    def mag(self):
        return (self.mag_squared())**0.5

    def normalize(self):
        magnitude = self.mag()
        if magnitude != 0:
            self.mult(1 / magnitude)



########################################################################################################################################################
# holds phase space information of a particle
class Particle:
    def __init__(self, x, y, vx, vy,mass, name):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.mass = mass
        self.name = name
        self.next_vel = Vector(0, 0)
        # Asumiré que quieres inicializar la aceleración a cero, pero puedes cambiar esto si lo necesitas.
        self.ax, self.ay = 0, 0

    def actualizar_posicion(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt

    def actualizar_velocidad(self, dt):
        self.vx += self.ax * dt
        self.vy += self.ay * dt

    def kick(self, node, theta):
      # Si el nodo es una hoja y contiene una partícula
      if node.leaf and node.particle is not None and node.particle != self:
          r = Vector(node.particle.x - self.x, node.particle.y - self.y)
          dist_sq = r.mag_squared()
          force_mag = G * node.particle.mass / (dist_sq * math.sqrt(dist_sq))
          r.normalize()
          r.mult(force_mag)
          self.ax += r.x
          self.ay += r.y
          return

      # Si el nodo no es una hoja
      if not node.leaf:
          r = Vector(node.center.x - self.x, node.center.y - self.y)
          dist = r.mag()

          # Si el nodo cumple con el criterio de Barnes-Hut
          if node.w / dist < theta:
              force_mag = G * node.total_mass / (dist * dist * dist)
              r.normalize()
              r.mult(force_mag)
              self.ax += r.x
              self.ay += r.y
          else:
              # Repite el proceso recursivamente para cada hijo del nodo
              for child in node.children:
                  self.kick(child, theta)


    # drift step where position is changed by velocity
    def drift(self, TIMESTEP):
        self.x += self.vx * TIMESTEP
        self.y += self.vy * TIMESTEP



########################################################################################################################################################
# seed the RNG to give the same Plummer initial conditions
def seed():
  random.seed("this is a seed for timing purposes")

def plummersphere_2D(N, a, NEWTON_G):
    x, y = [], []
    vx, vy = [], []
    masses = []  # Lista para almacenar las masas de las estrellas
    nstars = 0
    while nstars < N:
        radius = a / np.sqrt(np.random.random()**(-2/3) - 1)
        xx = np.random.random()
        yy = np.random.random()*0.1
        if yy < xx**2 * (1-xx**2)**3.5:
            nstars += 1
            vmag = xx * np.sqrt(2*NEWTON_G*N)*(radius**2+a**2)**(-0.25)
            phi = np.random.random()*2*np.pi
            x.append(radius*np.cos(phi))
            y.append(radius*np.sin(phi))
            phi = np.random.random()*2*np.pi
            vx.append(vmag*np.cos(phi))
            vy.append(vmag*np.sin(phi))
            
            # Generar una masa aleatoria para la estrella dentro de un rango [min_mass, max_mass]
            min_mass = 0.8  # Puedes ajustar este valor
            max_mass = 1.2  # Puedes ajustar este valor
            star_mass = np.random.uniform(min_mass, max_mass)
            masses.append(star_mass)
            
    return [Particle(x[i], y[i], vx[i], vy[i], masses[i], str(i)) for i in range(N)]

#########################################################################################################################################################################

# Constantes
G = 6.67430e-11  # Constante gravitacional
r = 1.0  # Un valor de referencia, puedes ajustarlo según tus necesidades
dt = 0.1  # Paso de tiempo, ya lo tienes definido en tu código
theta = 0.5  # Un valor común para el criterio de Barnes-Hut, pero es ajustable

# Funciones auxiliares
def dist_squared(a, b):
    return (a.x - b.x)**2 + (a.y - b.y)**2

def mult(vector, scalar):
    return Vector(vector.x * scalar, vector.y * scalar)

def sub(a, b):
    return Vector(a.x - b.x, a.y - b.y)

def gravityAcc(a, b, m_b):
    dSq = dist_squared(a, b)
    if dSq <= 4 * r * r:
        return Vector(0, 0)
    return mult(sub(a, b), G * m_b / (dSq * math.sqrt(dSq)))

def dist(a, b):
    return math.sqrt(dist_squared(a, b))

def gravitate(p, tn):
    if tn.leaf:
        if tn.particle is None or p == tn.particle:
            return
        p.vel.add(mult(gravityAcc(tn.particle.pos, p.pos, tn.particle.mass), dt))  # Multiplicamos por dt aquí
        return

    if tn.center is None:
        tn.center = mult(tn.total_center, 1.0 / tn.count)
    if tn.w / dist(p.pos, tn.center) < theta:
        p.vel.add(mult(gravityAcc(tn.center, p.pos, tn.total_mass), dt))  # Multiplicamos por dt aquí
        return

    for child in tn.children:
        gravitate(p, child)

# Inicialización de condiciones iniciales
dt = 60. * 60. * 24  # Un día en segundos
N = 1000  # Número de estrellas por galaxia
a = 5  # Radio de escala en parsecs
NEWTON_G = 6.67430e-11  # Constante gravitacional

galaxia1 = plummersphere_2D(N, a, NEWTON_G)
galaxia2 = plummersphere_2D(N, a, NEWTON_G)


def build_tree(particles):
    # Definir los límites de tu espacio de simulación. Ajusta según sea necesario.
    root = TreeNode(0, 0, 1000)  # Asume un espacio de 1000x1000 para este ejemplo

    for particle in particles:
        root.insert(particle)

    return root
def gravity(particles, root):
    for p in particles:
        gravitate(p, root)

def inicializar_galaxias(N, a, NEWTON_G):
    galaxia1 = plummersphere_2D(N, a, NEWTON_G)
    galaxia2 = plummersphere_2D(N, a, NEWTON_G)
    return galaxia1, galaxia2

# Actualizar posiciones y velocidades de las partículas
def update_positions(particles, dt, NEWTON_G, SOFTENING):
    root = build_tree(particles)  # Reconstruir el árbol en cada paso de tiempo
    gravity(particles, root)  # Actualizar las velocidades
    for particle in particles:
        particle.kick(root, NEWTON_G, dt, SOFTENING)
        particle.drift(dt)

# Función para animar
def animate(N, a, NEWTON_G, dt, SOFTENING):
    galaxia1, galaxia2 = inicializar_galaxias(N, a, NEWTON_G)

    fig, ax = plt.subplots()
    sc = ax.scatter([], [])
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)

    def init():
        data = [(p.x, p.y) for p in galaxia1 + galaxia2]
        sc.set_offsets(data)
        return sc,

    def update(frame):
        update_positions(galaxia1 + galaxia2, dt, NEWTON_G, SOFTENING)
        data = [(p.x, p.y) for p in galaxia1 + galaxia2]
        sc.set_offsets(data)
        return sc,

    ani = FuncAnimation(fig, update, frames=range(100), init_func=init, blit=True)
    plt.show()

# Parámetros
N = 1000
a = 5
NEWTON_G = 6.67430e-11
dt = 60. * 60. * 24
SOFTENING = 0.1  # Ajusta según sea necesario

# Llamar a la función para crear la animación
animate(N, a, NEWTON_G, dt, SOFTENING)
