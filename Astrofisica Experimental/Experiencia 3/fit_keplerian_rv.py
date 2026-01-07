import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Constantes físicas
G_SI = 6.67430e-11            # m^3 kg^-1 s^-2
M_SUN = 1.98847e30            # kg
DAY = 86400.0                 # s
AU = 1.495978707e11           # m
C_KMS = 299792.458            # km/s

# ==========================
#  PARSEO DE TIEMPOS ROBUSTO
# ==========================
def _to_days_since_ref(dates):
    """
    Convierte 'dates' a tiempos relativos en días desde la primera observación.
    - Si 'dates' son numéricos (MJD/JD): usa diferencias directamente.
    - Si son strings: intenta pandas (ISO), si falla intenta astropy (isot/utc).
    Devuelve: (t_rel_days, ref_date or None)
    """
    arr = np.asarray(dates)

    # Caso 1: ya son números (MJD/JD/días)
    if np.issubdtype(arr.dtype, np.number):
        t_days = arr.astype(float)
        t0 = np.min(t_days)
        return t_days - t0, None

    # Caso 2: strings -> sanitizar y parsear
    s = pd.Series(arr.astype(str))

    # A veces vienen HH_MM_SS en lugar de HH:MM:SS
    # Reemplazamos '_' por ':' solo en el segmento horario probable.
    # (Esto es conservador y no toca las rutas de archivos)
    s_clean = s.str.replace('_', ':', regex=False)

    # Intento 1: pandas
    dt = pd.to_datetime(s_clean, utc=True, errors='coerce')
    if not dt.isna().any():
        # Asegurarnos de que sea Serie
        dt = pd.Series(dt)

        # ns → s → días
        t_sec = dt.astype('int64') / 1e9
        t_days = t_sec / DAY
        t0 = np.min(t_days)
        # Fecha de referencia = la de t0 (no imprescindible para el ajuste)
        ref_idx = int(np.argmin(t_days))
        ref_date = dt.utcfromtimestamp(float(t_sec[ref_idx]))
        return t_days - t0, ref_date

    # Intento 2: astropy (si está disponible)
    try:
        from astropy.time import Time
        # Asumimos formato isot (YYYY-MM-DDThh:mm:ss(.sss)), escala UTC
        T = Time(list(s_clean), format='isot', scale='utc')
        t_days = T.jd  # días julianos
        t0 = np.min(t_days)
        return t_days - t0, None
    except Exception:
        pass

    raise ValueError(
        "No pude parsear 'Fecha'. Si ya tienes MJD, pásalo como float en df['Fecha']."
    )

# ======================
#  KEPLER: ECUACIONES RV
# ======================
def _solve_kepler(M, e, tol=1e-12, maxiter=100):
    M = np.asarray(M)
    E = M + e*np.sin(M)
    for _ in range(maxiter):
        f = E - e*np.sin(E) - M
        fp = 1.0 - e*np.cos(E)
        dE = -f/fp
        E = E + dE
        if np.max(np.abs(dE)) < tol:
            break
    return E

def _true_anomaly(t, P, e, T0):
    n = 2.0*np.pi / P
    M = n*(t - T0)
    # Envolver a [-pi, pi] para estabilidad numérica
    M = (M + np.pi) % (2.0*np.pi) - np.pi
    E = _solve_kepler(M, e)
    cos_nu = (np.cos(E) - e) / (1.0 - e*np.cos(E))
    sin_nu = (np.sqrt(1.0 - e**2) * np.sin(E)) / (1.0 - e*np.cos(E))
    nu = np.arctan2(sin_nu, cos_nu)
    return nu

def keplerian_rv(t, P, K, e, omega, T0, gamma):
    nu = _true_anomaly(t, P, e, T0)
    return gamma + K*(np.cos(nu + omega) + e*np.cos(omega))

# =========================
#  INICIALIZACIÓN DEL AJUSTE
# =========================
def _initial_guess(t, rv):
    # Amplitud y gamma iniciales
    K0 = 0.5*(np.nanmax(rv) - np.nanmin(rv))
    if not np.isfinite(K0) or K0 <= 0:
        K0 = max(0.1, np.nanstd(rv))
    gamma0 = np.nanmedian(rv)

    # Búsqueda log-espaciada de periodo (cruda pero útil)
    periods = np.logspace(np.log10(0.5), np.log10(3000.0), 400)
    scores = []
    for P in periods:
        w = 2*np.pi / P
        s, c = np.sin(w*t), np.cos(w*t)
        A = np.vstack([np.ones_like(t), s, c]).T
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, rv, rcond=None)
            model = A @ coeffs
            rss = np.sum((rv - model)**2)
        except Exception:
            rss = np.inf
        scores.append(rss)
    P0 = periods[int(np.nanargmin(scores))]

    e0 = 0.1
    omega0 = 0.0
    T00 = t[0]
    return [P0, K0, e0, omega0, T00, gamma0]

# ============
#  AJUSTE MAIN
# ============
def fit_keplerian(df, M_star_solar=None, period_guess=None, make_plots=True):
    """
    Ajusta un modelo Kepleriano (1 planeta) a un DataFrame con columnas:
      ['Archivo', 'VR_corr', 'Error', 'Fecha']
    - VR_corr en km/s
    - Error en km/s
    - Fecha: ISO str (o MJD/JD numérico)

    M_star_solar: masa estelar (M_sun) para derivar m2*sin(i) [Mjup]
    period_guess: si conoces un periodo aproximado (días)
    """
    df2 = df.dropna(subset=['VR_corr', 'Error', 'Fecha']).copy()
    if len(df2) < 5:
        raise ValueError("Se requieren al menos 5 puntos para un ajuste Kepleriano razonable.")

    # Construir eje temporal robusto
    t_days, t0_datetime = _to_days_since_ref(df2['Fecha'].values)
    rv = df2['VR_corr'].astype(float).values
    err = df2['Error'].astype(float).values

    # Semillas
    if period_guess is None:
        p0 = _initial_guess(t_days, rv)
    else:
        P0 = float(period_guess)
        K0 = 0.5*(np.nanmax(rv) - np.nanmin(rv))
        if not np.isfinite(K0) or K0 <= 0:
            K0 = max(0.1, np.nanstd(rv))
        p0 = [P0, K0, 0.1, 0.0, t_days[0], np.nanmedian(rv)]

    # Límites
    lower = [0.2, 0.0, 0.0, -np.pi, np.min(t_days) - 1e5, -1e6]
    upper = [1.0e5, 1.0e3, 0.95,  np.pi, np.max(t_days) + 1e5,  1e6]

    # Ajuste ponderado por errores
    popt, pcov = curve_fit(
        lambda t, P, K, e, omega, T0, gamma: keplerian_rv(t, P, K, e, omega, T0, gamma),
        t_days, rv, p0=p0, sigma=err, absolute_sigma=True, bounds=(lower, upper), maxfev=200000
    )
    perr = np.sqrt(np.diag(pcov))

    P, K, e, omega, T0, gamma = popt
    P_err, K_err, e_err, omega_err, T0_err, gamma_err = perr

    # Derivados
    derived = {}
    a1_sin_i_m = (K*1e3) * (P*DAY) * np.sqrt(1.0 - e**2) / (2.0*np.pi)
    derived['a1_sin_i_AU'] = a1_sin_i_m / AU

    if M_star_solar is not None and M_star_solar > 0:
        M_star = M_star_solar * M_SUN
        fM = ((K*1e3)**3) * (P*DAY) * (1.0 - e**2)**1.5 / (2.0*np.pi * G_SI)
        m2_sini_kg = (fM * (M_star**2))**(1.0/3.0)
        M_JUP = 1.89813e27
        derived['m2_sin_i_Mjup'] = m2_sini_kg / M_JUP

        result = {
            't0_datetime': t0_datetime,
            'params': {
                'P_days': P, 'P_err': P_err,
                'K_kms': K, 'K_err': K_err,
                'e': e, 'e_err': e_err,
                'omega_rad': omega, 'omega_err': omega_err,
                'T0_days_since_ref': T0, 'T0_err': T0_err,
                'gamma_kms': gamma, 'gamma_err': gamma_err
            },
            'derived': derived,
            'covariance': pcov
        }

    if make_plots:
        # 1) RV vs tiempo
        t_plot = np.linspace(np.min(t_days), np.max(t_days), 1000)
        rv_model = keplerian_rv(t_plot, *popt)

        plt.figure(figsize=(7,4))
        plt.errorbar(t_days, rv, yerr=err, fmt='o', ms=4, capsize=2)
        plt.plot(t_plot, rv_model)
        plt.xlabel("Tiempo desde referencia [d]")
        plt.ylabel("RV [km/s]")
        plt.title("RV vs tiempo (ajuste Kepleriano)")
        plt.tight_layout()
        plt.show()

        # 2) Fase
        phase = ((t_days - T0) / P) % 1.0
        phase_model = ((t_plot - T0) / P) % 1.0
        order = np.argsort(phase_model)
        plt.figure(figsize=(7,4))
        plt.errorbar(phase, rv, yerr=err, fmt='o', ms=4, capsize=2)
        plt.plot(phase_model[order], rv_model[order])
        plt.xlabel("Fase orbital")
        plt.ylabel("RV [km/s]")
        plt.title("RV plegada en fase")
        plt.tight_layout()
        plt.show()

        # 3) Residuos
        rv_fit = keplerian_rv(t_days, *popt)
        resid = rv - rv_fit
        plt.figure(figsize=(7,3.2))
        plt.axhline(0.0, linestyle='--')
        plt.errorbar(t_days, resid, yerr=err, fmt='o', ms=4, capsize=2)
        plt.xlabel("Tiempo desde referencia [d]")
        plt.ylabel("Residuos [km/s]")
        plt.title("Residuos")
        plt.tight_layout()
        plt.show()

    return result
