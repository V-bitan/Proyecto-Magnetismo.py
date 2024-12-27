# Proyecto-Magnetismo.py
# Código Python para calcular el rendimiento de un sistema con resistencias pasivas
class SistemaMagnetico:
    def __init__(self, potencia_aplicada, resistencia_pasiva):
        self.potencia_aplicada = potencia_aplicada
        self.resistencia_pasiva = resistencia_pasiva

    def calcular_trabajo_util(self):
        return self.potencia_aplicada - self.resistencia_pasiva

    def calcular_rendimiento(self):
        trabajo_util = self.calcular_trabajo_util()
        rendimiento = trabajo_util / self.potencia_aplicada
        return rendimiento

# Ejemplo de uso
potencia_aplicada = 50  # en kilogramos
resistencia_pasiva = 5  # en kilogramos

sistema = SistemaMagnetico(potencia_aplicada, resistencia_pasiva)
trabajo_util = sistema.calcular_trabajo_util()
rendimiento = sistema.calcular_rendimiento()

print(f"Potencia aplicada: {potencia_aplicada} kg")
print(f"Resistencia pasiva: {resistencia_pasiva} kg")
print(f"Trabajo útil: {trabajo_util} kg")
print(f"Rendimiento: {rendimiento * 100:.2f}%")

import numpy as np
import matplotlib.pyplot as plt

# Parámetros del vehículo
vehiculo_pos = np.array([0, 0])  # Posición inicial del vehículo (x, y)
velocidad = np.array([1, 0])     # Velocidad inicial del vehículo (m/s)
magnetismo_repulsion = 1.0       # Fuerza de repulsión magnética

# Parámetros de la carretera (con polos magnéticos)
carretera_longitud = 100
magnetico_pos = np.array([50, 0])  # Posición del polo magnético

# Simulación
tiempo_total = 100
dt = 0.1
posiciones = [vehiculo_pos]
desactivado = False
tiempo_desactivacion = 10  # segundos
contador_desactivacion = 0

for t in np.arange(0, tiempo_total, dt):
    # Simular desactivación temporal del vehículo
    if desactivado:
        contador_desactivacion += dt
        if contador_desactivacion >= tiempo_desactivacion:
            desactivado = False
            contador_desactivacion = 0
        else:
            posiciones.append(vehiculo_pos)
            continue

    # Calcular la distancia al polo magnético
    distancia = vehiculo_pos - magnetico_pos
    distancia_magnitud = np.linalg.norm(distancia)
    
    # Aplicar fuerza de repulsión magnética si el vehículo está cerca del polo magnético
    if distancia_magnitud < 10:
        repulsion = magnetismo_repulsion * distancia / distancia_magnitud**2
        desactivado = True  # Desactivar el vehículo en situación de emergencia
    else:
        repulsion = np.array([0, 0])
    
    # Actualizar la posición del vehículo
    vehiculo_pos = vehiculo_pos + velocidad * dt + repulsion * dt
    posiciones.append(vehiculo_pos)

# Convertir las posiciones en un array para graficar
posiciones = np.array(posiciones)

# Graficar la trayectoria del vehículo
plt.plot(posiciones[:, 0], posiciones[:, 1], label='Trayectoria del vehículo')
plt.scatter(magnetico_pos[0], magnetico_pos[1], color='red', label='Polo magnético')
plt.xlabel('Posición X')
plt.ylabel('Posición Y')
plt.legend()
plt.title('Simulación de la Interacción del Vehículo con el Sistema Magnético y Desactivación Temporal')
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Parámetros del vehículo
vehiculo_pos = np.array([0, 0])  # Posición inicial del vehículo (x, y)
velocidad = np.array([1, 0])     # Velocidad inicial del vehículo (m/s)
magnetismo_repulsion = 1.0       # Fuerza de repulsión magnética
friccion = 0.05                  # Coeficiente de fricción

# Parámetros de la carretera (con polos magnéticos)
carretera_longitud = 100
magnetico_pos = np.array([50, 0])  # Posición del polo magnético

# Simulación
tiempo_total = 100
dt = 0.1
posiciones = [vehiculo_pos]
desactivado = False
tiempo_desactivacion = 10  # segundos
contador_desactivacion = 0

for t in np.arange(0, tiempo_total, dt):
    # Simular desactivación temporal del vehículo
    if desactivado:
        contador_desactivacion += dt
        if contador_desactivacion >= tiempo_desactivacion:
            desactivado = False
            contador_desactivacion = 0
        else:
            posiciones.append(vehiculo_pos)
            continue

    # Calcular la distancia al polo magnético
    distancia = vehiculo_pos - magnetico_pos
    distancia_magnitud = np.linalg.norm(distancia)
    
    # Aplicar fuerza de repulsión magnética si el vehículo está cerca del polo magnético
    if distancia_magnitud < 10:
        repulsion = magnetismo_repulsion * distancia / distancia_magnitud**2
        desactivado = True  # Desactivar el vehículo en situación de emergencia
    else:
        repulsion = np.array([0, 0])
    
    # Calcular la fuerza de fricción
    fuerza_friccion = -friccion * velocidad

    # Actualizar la posición del vehículo
    vehiculo_pos = vehiculo_pos + velocidad * dt + repulsion * dt + fuerza_friccion * dt
    posiciones.append(vehiculo_pos)

# Convertir las posiciones en un array para graficar
posiciones = np.array(posiciones)

# Graficar la trayectoria del vehículo
plt.plot(posiciones[:, 0], posiciones[:, 1], label='Trayectoria del vehículo')
plt.scatter(magnetico_pos[0], magnetico_pos[1], color='red', label='Polo magnético')
plt.xlabel('Posición X')
plt.ylabel('Posición Y')
plt.legend()
plt.title('Simulación Avanzada: Interacción del Vehículo con el Sistema Magnético y Fricción')
plt.show()
