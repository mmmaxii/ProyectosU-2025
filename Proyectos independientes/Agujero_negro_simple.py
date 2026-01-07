"""
Simulación de Sombra de Agujero Negro + Lente Gravitacional (Métrica de Schwarzschild)
--------------------------------------------------------------------------------------
Este script genera una visualización de cómo un agujero negro de Schwarzschild distorsiona
un campo de estrellas de fondo.

Física implementada:
1. Fondo: Campo estelar generado procedimentalmente (PSFs gaussianas).
2. Sombra: Captura de fotones para parámetros de impacto b < b_crit.
3. Lente: Aproximación de deflexión débil alpha(b) = 2 r_s / b.
4. Anillo de Fotones: Resalte artificial cerca de la órbita de fotones inestable.

Dependencias: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 1. CONFIGURACIÓN Y PARÁMETROS FÍSICOS
# ==============================
np.random.seed(12) # Semilla para reproducibilidad del campo estelar

# --- Parámetros del Lienzo (Rendering) ---
npix = 900               # Resolución de la imagen (píxeles^2)
fov_rs = 20.0            # Campo de visión (FOV) en unidades de radios de Schwarzschild (r_s)
                         # El ancho total de la imagen será [-fov_rs, +fov_rs]

# --- Configuración del Anillo de Fotones ---
show_photon_ring = True  # ¿Mostrar el brillo del anillo de fotones?
ring_width = 0.06        # Ancho fraccional alrededor de b_crit donde se concentra el brillo
ring_gain = 2.2          # Factor de ganancia (boost) para simular la acumulación de luz

# --- Física del Agujero Negro (Unidades naturales) ---
r_s = 1.0                # Definimos el radio de Schwarzschild como la unidad de longitud (2GM/c^2)

# Cálculo del Parámetro de Impacto Crítico (b_crit)
# Para un BH de Schwarzschild, la órbita inestable de fotones está en r = 1.5 r_s.
# El parámetro de impacto visto desde el infinito es b_crit = sqrt(27)/2 * r_s ≈ 2.598 r_s
b_crit = (3.0 * np.sqrt(3.0) / 2.0) * r_s 

# --- Generación del Fondo (Campo Estelar) ---
n_stars = 6000           # Cantidad de estrellas en el fondo
star_sigma = 0.015       # Dispersión (sigma) de la PSF de cada estrella (fracción del FOV)
bg_sky_level = 0.02      # Nivel base del fondo de cielo (ruido de fondo)
vignetting = 0.15        # Oscurecimiento radial (vignetting) para realismo óptico (0 = desactivado)

# --- Ruido Instrumental ---
add_read_noise = True    # Simular ruido de lectura del detector
read_sigma = 0.01        # Desviación estándar del ruido gaussiano

# --- Salida ---
save_path = "bh_shadow_schwarzschild.png"
cmap_name = "magma"      # Paleta de colores ('magma' es ideal para intensidad radiativa)


# ==============================
# 2. SISTEMA DE COORDENADAS (Plano del Observador)
# ==============================
# Creamos una malla cuadrada que representa el plano del cielo (Plano de la Imagen)
# Las coordenadas están normalizadas en unidades de r_s
lin = np.linspace(-fov_rs, fov_rs, npix)
X, Y = np.meshgrid(lin, lin)

# Coordenadas polares en el plano de la imagen
# rho: Distancia radial desde el centro óptico (parámetro de impacto b)
rho = np.sqrt(X**2 + Y**2) + 1e-12 
# Vectores unitarios radiales (para la dirección de la deflexión)
ux, uy = X/rho, Y/rho 

# NOTA: En la aproximación de campo lejano (observador en infinito), asimilamos 
# la coordenada radial en pantalla 'rho' directamente al parámetro de impacto 'b'.

# ==============================
# 3. GENERACIÓN DEL FONDO (Source Plane)
# ==============================
def make_star_field(npix, fov_rs, n_stars, star_sigma, bg_sky_level, vignetting):
    """
    Genera una imagen sintética de un campo de estrellas con distribución aleatoria.
    Utiliza funciones de dispersión de punto (PSF) gaussianas para cada estrella.
    """
    # Inicializamos el lienzo con el nivel de fondo
    H = np.full((npix, npix), bg_sky_level, dtype=np.float32)
    
    # Posiciones aleatorias de las estrellas en el plano fuente
    xs = np.random.uniform(-fov_rs, fov_rs, n_stars)
    ys = np.random.uniform(-fov_rs, fov_rs, n_stars)
    
    # Magnitudes (brillo): sesgadas hacia estrellas tenues (ley de potencias simple)
    mags = np.random.random(n_stars)**2 
    
    # Conversión de sigma relativo a unidades de r_s
    sigma_rs = star_sigma * 2.0 * fov_rs
    # Pre-cálculo del inverso de 2*sigma^2 para optimizar el bucle
    inv2sig2 = 1.0 / (2.0 * sigma_rs**2 + 1e-20)

    # Superposición aditiva de las estrellas
    for x0, y0, m in zip(xs, ys, mags):
        dx2 = (X - x0)**2
        dy2 = (Y - y0)**2
        # Perfil Gaussiano: I = I0 * exp(-r^2 / 2sigma^2)
        H += m * np.exp(-(dx2 + dy2) * inv2sig2)

    # Aplicación de Vignetting (caída de luz en los bordes)
    if vignetting > 0:
        R = np.sqrt((X/(fov_rs))**2 + (Y/(fov_rs))**2)
        fall = 1.0 - vignetting * (R**2)
        fall = np.clip(fall, 0.0, 1.0)
        H *= fall
        
    return H

# Generamos el fondo "verdadero" (sin distorsión)
bg = make_star_field(npix, fov_rs, n_stars, star_sigma, bg_sky_level, vignetting)

# ==============================
# 4. MAPEO DE LENTE (Ray-Tracing Inverso)
# ==============================


# Calculamos el ángulo de deflexión alpha(b).
# Aproximación de campo débil: alpha = 4GM / (c^2 * b)
# En unidades donde r_s = 2GM/c^2, esto se simplifica a: alpha = 2 * r_s / b
alpha = 2.0 * r_s / rho

# Limitamos alpha para evitar divergencias numéricas en el centro (rho -> 0)
# (De todas formas, esa zona será ocultada por la sombra del agujero negro)
alpha = np.clip(alpha, 0.0, 2.0)

# Ecuación de la Lente: beta = theta - alpha
# (Xs, Ys): Coordenadas en el plano fuente (donde el rayo *realmente* se originó)
# (X, Y): Coordenadas en el plano imagen (donde *vemos* el rayo)
# El signo menos indica que la gravedad atrae la luz hacia el centro (lente convergente)
Xs = X - alpha * ux
Ys = Y - alpha * uy

# --- Interpolación ---
# Mapeamos las coordenadas calculadas (Xs, Ys) a índices de píxeles del array de fondo 'bg'.
# Usamos interpolación de vecino más cercano (Nearest-Neighbor) por eficiencia.

def to_index(coord):
    """Convierte coordenadas físicas r_s a índices de matriz [0, npix-1]"""
    # Mapeo lineal: [-fov, +fov] -> [0, npix]
    idx = ((coord + fov_rs) / (2 * fov_rs) * (npix - 1)).round().astype(int)
    return np.clip(idx, 0, npix - 1)

i = to_index(Ys)  # Índice de fila (eje Y)
j = to_index(Xs)  # Índice de columna (eje X)

# Creamos la imagen lenseada tomando los píxeles del fondo según el mapeo
lensed = bg[i, j].astype(np.float32)

# ==============================
# 5. SOMBRA Y ANILLO DE FOTONES
# ==============================

# --- La Sombra del Agujero Negro ---
# Cualquier rayo con parámetro de impacto b < b_crit caerá en espiral hacia el horizonte.
# No hay emisión desde el agujero negro, así que asignamos intensidad 0 (negro).
shadow_mask = (rho < b_crit)
lensed[shadow_mask] = 0.0

# --- El Anillo de Fotones ---
# Cerca de b_crit, los fotones orbitan múltiples veces el agujero negro.
# Esto acumula luz de muchas direcciones, creando un anillo brillante.
# Aquí lo simulamos con un "boost" de ganancia en una banda estrecha alrededor de b_crit.
if show_photon_ring:
    # Definimos una banda anular de ancho 'ring_width'
    photon_ring_band = np.abs(rho - b_crit) < (ring_width * b_crit)
    # Aplicamos ganancia solo fuera de la sombra
    lensed[photon_ring_band & (~shadow_mask)] *= ring_gain

# --- Post-Procesamiento ---
# Agregamos ruido gaussiano para simular una cámara CCD real
if add_read_noise:
    noise = np.random.normal(0, read_sigma, size=lensed.shape)
    lensed = lensed + noise
    lensed = np.clip(lensed, 0.0, None) # Evitamos valores negativos físicos

# Normalización final para visualización [0, 1]
lensed = lensed / (lensed.max() + 1e-12)

# ==============================
# 6. VISUALIZACIÓN
# ==============================
plt.figure(figsize=(8.0, 8.0))

# Mostramos la imagen procesada
plt.imshow(lensed, 
           extent=[-fov_rs, fov_rs, -fov_rs, fov_rs], 
           origin='lower', 
           cmap=cmap_name, 
           vmin=0, vmax=1)

# Etiquetas y decoración
plt.xlabel(r"$\theta_x$  [unidades de $r_s$]")
plt.ylabel(r"$\theta_y$  [unidades de $r_s$]")
plt.title("Sombra de Agujero Negro Schwarzschild + Lente Débil\n(Fondo estelar procedimental)")

# Barra de color
cbar = plt.colorbar(fraction=0.046, pad=0.04)
cbar.set_label("Brillo Normalizado")

# Superposición visual del radio crítico teórico (b_crit)
ang = np.linspace(0, 2*np.pi, 720)
plt.plot(b_crit*np.cos(ang), b_crit*np.sin(ang), 
         color='cyan', lw=1.5, ls='--', alpha=0.8, 
         label=r"Radio Crítico $b_{crit} \approx 2.6 r_s$")

plt.legend(loc="lower left", framealpha=0.9)
plt.tight_layout()

# Guardado
plt.savefig(save_path, dpi=220, bbox_inches='tight')
plt.show()

print(f"Renderizado completado. Imagen guardada en: {save_path}")