
import scipy as sp
import scipy.constants
import numpy as np


import matplotlib.pyplot as plt

import matplotlib.animation as animation

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


    ## constantes

MASA_SOL = 1.9891E30        # kg

MASA_JUPITER = 1.898E27     # kg

MASA_TIERRA = 5.972E24      # kg

MASA_LUNA = 7.3476E22       # kg

DISTANCIA_LUNA = 3.844E8    # metros



# Método de Euler
# dt en segundos


def euler_step(cuerpos, dt):
    for cuerpo in cuerpos:
        cuerpo['posicion'] += cuerpo['velocidad'] * dt
        aceleracion = calcular_aceleracion(cuerpo, cuerpos)
        cuerpo['velocidad'] += aceleracion * dt

def calcular_aceleracion(cuerpo, cuerpos):
    aceleracion = 0.0
    for c in cuerpos:
        if np.any(cuerpo['posicion'] != c['posicion']):
            aceleracion += aceleracion_gravitacional(cuerpo['posicion'], c['posicion'], c['masa'])
    return aceleracion

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
# El Sol estará en el origen de coordenadas y en reposo
sol = {'masa': MASA_SOL,
       'posicion': np.array([0., 0., 0.]),
       'velocidad': np.array([0., 0., 0.])}

# La Tierra estará a 1 UA y la velocidad inicial será el promedio de su velocidad orbital: ~29 km/s
tierra = {'masa': MASA_TIERRA,
          'posicion': np.array([sp.constants.astronomical_unit, 0., 0.]),
          'velocidad': np.array([0., 2.9E4, 0.])}

# Júpiter estará a 5 UA y la velocidad inicial será el promedio de su velocidad orbital: ~13 km/s
jupiter = {'masa': MASA_JUPITER,
           'posicion': np.array([-5.0 * sp.constants.astronomical_unit, 0., 0.]),
           'velocidad': np.array([0., -1.3E4, 0.])}

# Integraremos la ecuación de movimiento por 11 años
# (aproximadamente, el periodo orbital de Jupiter
steps = 365 * 11

# Lista de cuerpos que componen el sistema
cuerpos = [sol, tierra, jupiter]

# Listas en memoria para guardar todos los datos de la evolución para luego graficarlos.

historia_x1 = []
historia_y1 = []
historia_z1 = []

historia_x2 = []
historia_y2 = []
historia_z2 = []

historia_x3 = []
historia_y3 = []
historia_z3 = []

# Guardamos la energia total al inicio de la simulación
# para verificar que al final el sistema conserve la energía
etot_inicial = calcula_energia_total(cuerpos)

guarde_cada = 10

aceleraciones_i0 = []

i = 0

# for cuerpo in cuerpos:
#     aceleraciones_i0.insert(i, calcular_aceleracion(cuerpo, cuerpos))
#     i += 1

while steps >= 0:
    # leapfrog_step(cuerpos, aceleraciones_i0, dt)
    euler_step(cuerpos, dt)
    # Mensaje para ir viendo el avance del proceso
    if steps % 1000 == 0:
        print ("Faltan %d steps " % (steps))

    # En cada paso, guardamos los valores de posición y velocidad para graficarlos al final

    if steps % guarde_cada == 0:

        historia_x1.append(sol['posicion'][0])
        historia_y1.append(sol['posicion'][1])
        historia_z1.append(sol['posicion'][2])

        historia_x2.append(tierra['posicion'][0])
        historia_y2.append(tierra['posicion'][1])
        historia_z2.append(tierra['posicion'][2])

        historia_x3.append(jupiter['posicion'][0])
        historia_y3.append(jupiter['posicion'][1])
        historia_z3.append(jupiter['posicion'][2])

    steps -= 1

etot_final = calcula_energia_total(cuerpos)
print ("Energia total inicial: %s" % (str(etot_inicial)))
print ("Energia total final: %s" % (str(etot_final)))
print ("Ration: %s" % (abs(etot_inicial) / abs(etot_final)))
# ---------------------------

fig = plt.figure("Sistema Sol - Tierra - Jupiter")
ax = fig.add_subplot(111, title='xx')

sp, = ax.plot([], [], 'ro')

# Limites de visualizacion del canvas para que se base en las distancias
# reales de las particulas
ax.set_ylim(min([min(historia_y1), min(historia_y2), min(historia_y3)]),
            max([max(historia_y1), max(historia_y2), max(historia_y3)]))

ax.set_xlim(min([min(historia_x1), min(historia_x2), min(historia_x3)]),
            max([max(historia_x1), max(historia_x2), max(historia_x3)]))

# Funcion que se llama para generar cada frame de la animación
# Recibe un parámetro i que puede ser usado como indice
# Lo que hace es actualizar los datos del plot generado arriba

def update(i):
    # Posiciones x de las 3 particulas
    x = [historia_x1[i], historia_x2[i], historia_x3[i]]
    # Posiciones y de las 3 particulas
    y = [historia_y1[i], historia_y2[i], historia_y3[i]]
    sp.set_data(x, y)
    ax.set_title('Frame dibujado: %d' % (i))
    return sp,

# Creacion de la animación, 100 frames signifca que cada 100 ms
# llama a update con un valor entre 0 y 99
# Con repeat = False hace que solo se dibujen los frames una vez
#ani = animation.FuncAnimation(fig, update, frames=len(historia_x1), interval=50, repeat=False)
#ani.save('ejemplo.avi')
#plt.show()
ani = animation.FuncAnimation(fig, update, frames=len(historia_x1), interval=20, repeat=False)
display(HTML(ani.to_jshtml()))

