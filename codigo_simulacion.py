import scipy as sp
import scipy.constants
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "<{:.6f}, {:.6f}>".format(self.x, self.y)

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
  def __init__(self, x, y, vx, vy, name):
    self.x, self.y = x, y
    self.vx, self.vy, = vx, vy
    self.name = name

  # kick step where the velocity is changed by force
  def kick(self, cell, NEWTON_G, TIMESTEP, SOFTENING):
    # distance
    rx = cell.xcen - self.x
    ry = cell.ycen - self.y
    
    r2 = rx*rx + ry*ry 

    # if outside softening length, don't need softening
    if r2 > SOFTENING*SOFTENING:
      self.vx += NEWTON_G * TIMESTEP * rx * cell.n / r2**1.5
      self.vy += NEWTON_G * TIMESTEP * ry * cell.n / r2**1.5
      
    # else use solid sphere softening
    else:
      r = r2**0.5
      x = r / SOFTENING
      f = x * (8 - 9 * x + 2 * x * x * x) # Dyer and Ip 1993, ApJ 409(1)
      self.vx += NEWTON_G * TIMESTEP * f * rx * cell.n / (SOFTENING*SOFTENING*r)
      self.vy += NEWTON_G * TIMESTEP * f * ry * cell.n / (SOFTENING*SOFTENING*r)
      

  # drift step where position is changed by velocity
  def drift(self, TIMESTEP):
    self.x += self.vx * TIMESTEP
    self.y += self.vy * TIMESTEP
    

# trees are composed of cells referring to their daughter cells
class Cell:
  def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax, name):
    self.xmin, self.xmax = xmin, xmax
    self.ymin, self.ymax = ymin, ymax
    self.zmin, self.zmax = zmin, zmax
    self.name = name
    # start with no particles
    self.n = 0 
    self.xcen, self.ycen, self.zcen = 0, 0, 0
    self.daughters = []
    self.particle = None

  # test if a (x,y,z) coordinate is in this cell's bounds
  def incell(self, x, y, z):
    if x > self.xmin and x <= self.xmax and y > self.ymin and y <= self.ymax \
      and z > self.zmin and z <= self.zmax:
      return True
    else:
      return False

  def add(self, particle):
     # if particle not in bounds of cell, do nothing
     if not self.incell(particle.x, particle.y, particle.z):
       return

     # if this cell already has some particles, also add this particle
     # to our descendents
     if self.n > 0:
       # if this is the second particle this cell has encountered,
       # need to make descendents, and try adding the first particle
       # to our descendents
       if self.n == 1:
         self.makedaughters()
         for daughter in self.daughters:
           daughter.add(self.particle)
         self.particle = None # this cell no longer holds just 1 particle
       # add incoming particle to descendents
       for daughter in self.daughters:
         daughter.add(particle)
     # if this is the first particle this cell has encountered
     else:
       self.particle = particle

     # change center of mass
     self.xcen = (self.n * self.xcen + particle.x) / float(self.n + 1)
     self.ycen = (self.n * self.ycen + particle.y) / float(self.n + 1)
     self.zcen = (self.n * self.zcen + particle.z) / float(self.n + 1)
     # increment particle counter
     self.n += 1

  # create this cell's eight daughters
  def makedaughters(self):
    xhalf = (self.xmin + self.xmax) / 2.
    yhalf = (self.ymin + self.ymax) / 2.
    zhalf = (self.zmin + self.zmax) / 2.
    daughter1 = Cell(self.xmin, xhalf, self.ymin, yhalf, self.zmin, zhalf, self.name+".0")
    daughter2 = Cell(xhalf, self.xmax, self.ymin, yhalf, self.zmin, zhalf, self.name+".1")
    daughter3 = Cell(self.xmin, xhalf, yhalf, self.ymax, self.zmin, zhalf, self.name+".2")
    daughter4 = Cell(xhalf, self.xmax, yhalf, self.ymax, self.zmin, zhalf, self.name+".3")
    daughter5 = Cell(self.xmin, xhalf, self.ymin, yhalf, zhalf, self.zmax, self.name+".4")
    daughter6 = Cell(xhalf, self.xmax, self.ymin, yhalf, zhalf, self.zmax, self.name+".5")
    daughter7 = Cell(self.xmin, xhalf, yhalf, self.ymax, zhalf, self.zmax, self.name+".6")
    daughter8 = Cell(xhalf, self.xmax, yhalf, self.ymax, zhalf, self.zmax, self.name+".7")
    self.daughters = [daughter1, daughter2, daughter3, daughter4, daughter5, daughter6, daughter7, daughter8]

  # makes cell forget current daughters and take in new daughters,
  # recalculating mass and center of mass
  def assigndaughters(self, daughters):
    self.daughters = []
    self.daughters = daughters
    self.n = sum([daughter.n for daughter in daughters])
    # only need to calculate center of mass if cell is not empty
    if self.n > 0:
      self.xcen = sum([daughter.n * daughter.xcen for daughter in daughters]) / float(self.n)
      self.ycen = sum([daughter.n * daughter.ycen for daughter in daughters]) / float(self.n)
      self.zcen = sum([daughter.n * daughter.zcen for daughter in daughters]) / float(self.n)

  # traverse the tree to get a list of particles in this cell and below
  def particles(self):
    # if this is a bottom level cell with a particle
    if self.particle:
      return [self.particle]
    # else, if this cell has daughters, forward request to its daughters and
    # accumulate their answers
    elif self.daughters:
      l = []
      for daughter in self.daughters:
        l.extend(daughter.particles())
      return l
    # else, this is a bottom level cell with no particle
    else:
      return []

  # test if this cell is far enough from the specified particle for
  # force calculation
  def meetscriterion(self, particle, TREE_THRES, SOFTENING):
     # if this cell has many particles,
     # cell center of mass must be farther from particle than size of cell
     if self.daughters:
       s = self.xmax - self.xmin
       dx = particle.x - self.xcen
       dy = particle.y - self.ycen
       dz = particle.z - self.zcen
       d  = (dx*dx + dy*dy + dz*dz)**0.5
       # test d/s > tree_thres (equivalent to s/d < theta = tree_thres^-1)
       # also, we want to calculate individual particle forces within the
       # softening length
       return (d/s) > TREE_THRES and d > SOFTENING
     # else, just check that particle isn't trying to interact with itself
     # using object identity
     else:
       return self.particle != particle
########################################################################################################################################################
# seed the RNG to give the same Plummer initial conditions
def seed():
  random.seed("this is a seed for timing purposes")

# Plummer sphere with N solar mass stars, scale radius a parsecs
def plummersphere(N, a, NEWTON_G):
  x, y, z = [], [], []
  vx, vy, vz = [], [], []
  nstars = 0
  while nstars < N:
    # Sample radius from cumulative mass distribution
    radius = a / np.sqrt(random.random()**(-2/3) - 1)
    # Sample velocity magnitude through inversion sampling from velocity
    # distribution
    xx = random.random()
    yy = random.random()*0.1
    if yy < xx**2 * (1-xx**2)**3.5: # the star is added to the sample
      nstars += 1
      vmag = xx * np.sqrt(2*NEWTON_G*N)*(radius**2+a**2)**(-0.25)
        # N = total mass by construction
      # Calculate location coordinates
      phi = random.random()*2*np.pi
      theta = np.arccos(random.random() * 2 - 1)
      x.append(radius*np.sin(theta)*np.cos(phi))
      y.append(radius*np.sin(theta)*np.sin(phi))
      z.append(radius*np.cos(theta))
      # Calculate velocity coordinates
      phi = random.random()*2*np.pi
      theta = np.arccos(random.random() * 2 - 1)
      vx.append(vmag*np.sin(theta)*np.cos(phi))
      vy.append(vmag*np.sin(theta)*np.sin(phi))
      vz.append(vmag*np.cos(theta))
  return [Particle(x[i], y[i],  vx[i], vy[i],  str(i)) \
    for i in range(N)]

# two body problem with orbit semimajor axis a, eccentricity e
def kepler(a, e, NEWTON_G):
  rmin = a * (1-e) #periastron
  h = (rmin * (1+e) * NEWTON_G * 2)**0.5 #angular momentum
  v = h/rmin
  return [Particle(rmin/2., 0, 0, 0, 0, v/2., "A"), \
    Particle(-rmin/2., 0, 0, 0, 0, -v/2., "B")]

# four body problem with inner semimajor axis a1, eccentricity e1
#                        outer semimajor axis a2, eccentricity e2
def doublekepler(a1, e1, a2, e2, NEWTON_G):
  particles = kepler(a1, e1, NEWTON_G) #inner binary
  rmin = a2 * (1-e2) #outer periastron
  h = (rmin * (1+e2) * 9 * NEWTON_G * 2)**0.5 #outer angular momentum
    # 9 because +1 from other outer binary, and inner binary is twice as close
    # (x4) and is composed of two stars (x2), giving +8
  v = h/rmin
  particles.extend([Particle(rmin/2., 0, 0, 0, 0, v/2., "C"), \
    Particle(-rmin/2., 0, 0, 0, 0, -v/2., "D")]) #outer binary
  return particles


#########################################################################################################################################################################




# Ecuacion de movimiento


# Aceleracion sobre particula i, ejercida por particula j
# La magnitud del vector se calcula con np.linalg.norm


def aceleracion_gravitacional(ri, rj, mj):
    diff = rj - ri
    return sp.constants.G * mj * (diff / (np.linalg.norm(diff) ** 3))

# Calculo de energia cinética


def energia_cinetica(m, v):
    return 0.5 * m * (np.linalg.norm(v) ** 2)

# Energía potencial gravitacional para 2 cuerpos
# r es separacion entre ambas masas


def energia_potencial(mi, mj, ri, rj):
    diff = rj - ri
    return (-sp.constants.G * mi * mj) / np.linalg.norm(diff)

#Calculo de la energia total: cinetica + potencial

def calcula_energia_total(cuerpos):
    ekin = 0
    epot = 0
    for cuerpo in cuerpos:
        ekin += energia_cinetica(cuerpo['masa'], cuerpo['velocidad'])
        for c in cuerpos:
            if np.any(cuerpo['posicion'] != c['posicion']):
                epot += energia_potencial(cuerpo['masa'], c['masa'],
                                          cuerpo['posicion'], c['posicion'])

    #Se divide entre 2 porque el aporte de cada particula se calculó 2 veces
    epot /= 2

    return ekin + epot

######################################################################################################################################################

# Metodo Leapfrog
# dt en segundos

def leapfrog_step(cuerpos, aceleraciones, dt):
    i = 0

    velocidades = []

    for cuerpo in cuerpos:
        velocity_half = cuerpo['velocidad'] + (aceleraciones[i] * (dt / 2.))
        velocidades.insert(i, velocity_half)
        cuerpo['posicion'] += (velocity_half * dt)
        i += 1

    i = 0
    for cuerpo in cuerpos:
        # Calculo las aceleraciones para el i + 1 basado en la nueva posicion
        aceleraciones[i] = calcular_aceleracion(cuerpo, cuerpos)
        cuerpo['velocidad'] = velocidades[i] + (aceleraciones[i] * (dt / 2.))
        i += 1


# inicializacion de condiciones iniciales

# Un día en segundos
dt = 60. * 60. * 24

# Condiciones iniciales
N = 1000  # Número de estrellas por galaxia
a = 5  # Radio de escala en parsecs
NEWTON_G = 6.67430e-11  # Constante gravitacional

galaxia1 = plummersphere(N, a, NEWTON_G)
galaxia2 = plummersphere(N, a, NEWTON_G)

# Desplaza la segunda galaxia para ponerla en una trayectoria de colisión
for estrella in galaxia2:
    estrella.x += 10  # Desplazamiento en x
    estrella.vx = -0.01  # Velocidad inicial en x para que se mueva hacia la galaxia1



def main():
    # Inicialización de las galaxias y condiciones iniciales
    galaxia_A, galaxia_B = inicializar_galaxias()
    
    # Duración de la simulación y paso de tiempo
    tiempo_total = 100  # Por ejemplo
    dt = 0.1  # Paso de tiempo
    
    # Bucle principal de la simulación
    for t in range(0, tiempo_total, dt):
        # Calcular aceleraciones para todas las partículas
        for particula in galaxia_A + galaxia_B:
            # Aquí debemos calcular la aceleración total en la partícula
            # usando todas las demás partículas y la función de aceleración gravitacional
            calcular_aceleracion(particula, galaxia_A + galaxia_B)
        
        # Actualizar posiciones y velocidades usando el método Leapfrog
        for particula in galaxia_A + galaxia_B:
            particula.actualizar_posicion(dt)
            particula.actualizar_velocidad(dt)
        
        # Opcional: Guardar o visualizar el estado actual de la simulación



