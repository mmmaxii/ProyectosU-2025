import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, Button

# -----------------------------
# 1. Definición de la Física y la Fuente
# -----------------------------

def sersic_intensity(x, y, xc, yc, Re, n, q, phi, I0=1.0):
    """
    Calcula el perfil de brillo superficial de la galaxia fuente usando un perfil de Sersic.
    Permite rotación y elipticidad para darle realismo a la galaxia de fondo.
    """
    # Transformación de coordenadas para rotación (phi)
    c, s = np.cos(phi), np.sin(phi)
    xr =  (x - xc)*c + (y - yc)*s
    yr = -(x - xc)*s + (y - yc)*c
    
    # Radio elíptico considerando la relación axial q = b/a
    R_ell = np.sqrt(xr**2 + (yr/q)**2 + 1e-12)
    
    # Constante bn aproximada para el perfil de Sersic
    bn = 2*n - 1/3 + 0.009876/n 
    
    # Retornamos la intensidad I(R)
    return I0 * np.exp(-bn * ((R_ell/Re)**(1/n) - 1.0))

def deflection_SIS(x, y, xc, yc, theta_E):
    """
    Calcula los ángulos de deflexión (alpha) para una Esfera Isoterma Singular (SIS).
    La deflexión es constante en magnitud y apunta radialmente hacia el centro.
    """
    dx = x - xc
    dy = y - yc
    # Evitamos división por cero sumando un epsilon pequeño
    r = np.sqrt(dx*dx + dy*dy) + 1e-9
    return theta_E * dx / r, theta_E * dy / r

def deflection_shear(x, y, gamma1, gamma2):
    """
    Calcula la contribución del 'Shear' externo (cizallamiento).
    Esto simula el efecto gravitacional de estructuras masivas cercanas a la lente principal.
    """
    ax = gamma1 * x + gamma2 * y
    ay = gamma2 * x - gamma1 * y
    return ax, ay

def render(theta_E, gamma, phi_gamma_deg,
           src_x, src_y, Re, n_sersic, q, phi_src_deg,
           noise_sigma, draw_crit):
    """
    Función principal de 'Inverse Ray-Shooting'.
    Mapeamos cada píxel del plano de la imagen hacia atrás al plano de la fuente
    usando la ecuación de la lente: beta = theta - alpha.
    """
    # 1. Preparamos componentes del Shear
    phi_g = np.deg2rad(phi_gamma_deg)
    g1 = gamma*np.cos(2*phi_g)
    g2 = gamma*np.sin(2*phi_g)
    
    # 2. Calculamos el ángulo de deflexión total (alpha) en cada punto de la malla
    # Deflexión por la lente principal (SIS)
    ax_s, ay_s = deflection_SIS(X, Y, lens_center[0], lens_center[1], theta_E)
    # Deflexión por el shear externo
    ax_g, ay_g = deflection_shear(X, Y, g1, g2)
    
    # Suma de deflexiones
    ax = ax_s + ax_g
    ay = ay_s + ay_g
    
    # 3. Aplicamos la Ecuación de la Lente: beta (fuente) = theta (imagen) - alpha (deflexión)
    # (X, Y) son coordenadas en el plano imagen (theta)
    # (Xs, Ys) serán las coordenadas en el plano fuente (beta)
    Xs = X - ax
    Ys = Y - ay
    
    # 4. Evaluamos el brillo de la fuente en las coordenadas distorsionadas (Xs, Ys)
    phi_s = np.deg2rad(phi_src_deg)
    Img = sersic_intensity(Xs, Ys, src_x, src_y, Re, n_sersic, q, phi_s, I0=1.0)
    
    # 5. Agregamos ruido instrumental gaussiano si se solicita
    if noise_sigma > 0:
        Img = Img + np.random.normal(0, noise_sigma, Img.shape)
        Img = np.clip(Img, 0, None) # Evitamos valores negativos de flujo
    
    # Normalizamos para visualizar mejor (max = 1)
    Img = Img / (Img.max() + 1e-12)
    
    # Actualizamos los datos de la imagen en el plot
    im_artist.set_data(Img)
    
    # Actualizamos la visibilidad de la curva crítica
    crit_line.set_visible(bool(draw_crit))
    
    # Redibujamos la figura de manera eficiente
    fig.canvas.draw_idle()

# -----------------------------
# 2. Configuración de la Malla (Grid) y Parámetros
# -----------------------------
np.random.seed(7) # Semilla para reproducibilidad del ruido

# Definimos el Campo de Visión (FOV) en segundos de arco
fov = 6.0
npix = 700
half = fov/2
lin = np.linspace(-half, half, npix)

# Creamos la malla de coordenadas (Plano de la Imagen / Plano del Cielo)
X, Y = np.meshgrid(lin, lin)

# Posición del centro de la lente (fija en el origen)
lens_center = (0.0, 0.0)

# Diccionario con los parámetros iniciales del modelo
p = dict(
    theta_E=1.2,           # Radio de Einstein
    gamma=0.07,            # Magnitud del Shear
    phi_gamma_deg=20.0,    # Ángulo del Shear
    src_x=-0.25, src_y=0.15, # Posición de la fuente
    Re=0.25,               # Radio efectivo de la fuente
    n_sersic=2.0,          # Índice de Sersic
    q=0.6,                 # Relación axial (b/a)
    phi_src_deg=-30.0,     # Ángulo de posición de la fuente
    noise_sigma=0.015,     # Nivel de ruido
    draw_crit=True         # Mostrar curva crítica
)

# -----------------------------
# 3. Configuración Visual (Layout)
# -----------------------------
plt.close('all')
fig = plt.figure(figsize=(8.6, 8.6))

# Ejes principales para la imagen del lente
ax_img = plt.axes([0.08, 0.28, 0.84, 0.68])
ax_img.set_title("Simulación Interactiva de Lente Gravitacional (SIS + Shear)")
ax_img.set_xlabel(r"$\theta_x$ [arcsec]")
ax_img.set_ylabel(r"$\theta_y$ [arcsec]")

# Imagen inicial (vacía, se llenará en el primer update)
Img0 = np.zeros_like(X)
im_artist = ax_img.imshow(Img0, extent=[-half, half, -half, half],
                          origin='lower', cmap='viridis', vmin=0, vmax=1)

# Cálculo inicial de la Curva Crítica para el modelo SIS (es un círculo de radio theta_E)
th = np.linspace(0, 2*np.pi, 720)
crit_x = lens_center[0] + p["theta_E"]*np.cos(th)
crit_y = lens_center[1] + p["theta_E"]*np.sin(th)
crit_line, = ax_img.plot(crit_x, crit_y, color='orange', lw=2, alpha=0.9, label='Radio de Einstein')

# -----------------------------
# 4. Interfaz de Usuario (Sliders)
# -----------------------------
# Definición de las áreas (axes) donde irán los controles
# Columna Izquierda: Propiedades de la Lente (Masa y Shear)
ax_thetaE   = plt.axes([0.08, 0.22, 0.26, 0.025])
ax_gamma    = plt.axes([0.08, 0.19, 0.26, 0.025])
ax_phiG     = plt.axes([0.08, 0.16, 0.26, 0.025])

# Columna Derecha: Geometría de la Fuente (Posición y Tamaño)
ax_srcx     = plt.axes([0.42, 0.22, 0.26, 0.025])
ax_srcy     = plt.axes([0.42, 0.19, 0.26, 0.025])
ax_Re       = plt.axes([0.42, 0.16, 0.26, 0.025])

# Abajo Izquierda: Perfil de la Fuente (Morfología)
ax_n        = plt.axes([0.08, 0.11, 0.26, 0.025])
ax_q        = plt.axes([0.08, 0.08, 0.26, 0.025])
ax_phiS     = plt.axes([0.08, 0.05, 0.26, 0.025])

# Abajo Derecha: Ruido
ax_noise    = plt.axes([0.42, 0.11, 0.26, 0.025])

# Creación de los objetos Slider
s_thetaE = Slider(ax_thetaE, r'$\theta_E$ ["]', 0.2, 2.5, valinit=p["theta_E"])
s_gamma  = Slider(ax_gamma,  r'$\gamma$',      0.0, 0.25, valinit=p["gamma"])
s_phiG   = Slider(ax_phiG,   r'PA$_\gamma$ [°]', -90.0, 90.0, valinit=p["phi_gamma_deg"])

s_srcx   = Slider(ax_srcx, r'$\beta_x$ ["]', -2.5, 2.5, valinit=p["src_x"])
s_srcy   = Slider(ax_srcy, r'$\beta_y$ ["]', -2.5, 2.5, valinit=p["src_y"])
s_Re     = Slider(ax_Re,   r'$R_e$ ["]', 0.05, 0.8, valinit=p["Re"])

s_n      = Slider(ax_n,    r'$n_{\rm Sersic}$', 0.5, 6.0, valinit=p["n_sersic"])
s_q      = Slider(ax_q,    r'$q=b/a$', 0.2, 1.0, valinit=p["q"])
s_phiS   = Slider(ax_phiS, r'PA$_{\rm src}$ [°]', -90.0, 90.0, valinit=p["phi_src_deg"])

s_noise  = Slider(ax_noise, 'Ruido σ', 0.0, 0.08, valinit=p["noise_sigma"])

# Checkbox para activar/desactivar la visualización de la curva crítica
rax = plt.axes([0.72, 0.06, 0.20, 0.10])
checks = CheckButtons(rax, ['Ver Curva Crítica'], [p["draw_crit"]])

# Botón de Reset para volver a los valores iniciales
bax = plt.axes([0.72, 0.18, 0.20, 0.05])
btn_reset = Button(bax, 'Resetear', hovercolor='0.85')

# -----------------------------
# 5. Callbacks y Actualización
# -----------------------------
def update(_=None):
    """
    Función que se llama cada vez que se mueve un slider.
    Lee los valores actuales y llama a render().
    """
    # Recalculamos la geometría de la curva crítica si cambia el radio de Einstein
    thE = s_thetaE.val
    crit_line.set_xdata(lens_center[0] + thE*np.cos(th))
    crit_line.set_ydata(lens_center[1] + thE*np.sin(th))
    
    # Llamamos al renderizador con todos los valores actuales de la UI
    render(
        theta_E=thE,
        gamma=s_gamma.val,
        phi_gamma_deg=s_phiG.val,
        src_x=s_srcx.val, src_y=s_srcy.val,
        Re=s_Re.val, n_sersic=s_n.val, q=s_q.val, phi_src_deg=s_phiS.val,
        noise_sigma=s_noise.val,
        draw_crit=checks.get_status()[0]
    )

# Vinculamos la función update a los eventos de cambio en los sliders
for s in [s_thetaE, s_gamma, s_phiG, s_srcx, s_srcy, s_Re, s_n, s_q, s_phiS, s_noise]:
    s.on_changed(update)

def on_checks(label):
    update()
checks.on_clicked(on_checks)

def on_reset(event):
    """Restablece todos los parámetros a sus valores por defecto"""
    s_thetaE.reset(); s_gamma.reset(); s_phiG.reset()
    s_srcx.reset(); s_srcy.reset(); s_Re.reset()
    s_n.reset(); s_q.reset(); s_phiS.reset(); s_noise.reset()
    
    # Manejo del estado del checkbox
    if checks.get_status()[0] != p["draw_crit"]:
        checks.set_active(0)
        
btn_reset.on_clicked(on_reset)

# -----------------------------
# 6. Ejecución Inicial
# -----------------------------
update() # Primera renderización
plt.show()