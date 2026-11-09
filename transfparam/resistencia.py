"""Resistencia al avance: Holtrop-Mennen (1982/1984) y van Oortmerssen (1971).

Referencias
-----------
[HM82]  Holtrop, J. y Mennen, G.G.J. (1982). An approximate power prediction method.
        International Shipbuilding Progress, 29(335), 166-170.
[H84]   Holtrop, J. (1984). A statistical re-analysis of resistance and propulsion data.
        International Shipbuilding Progress, 31(363), 272-276.
[VO71]  van Oortmerssen, G. (1971). A power prediction method and its application to
        small ships. International Shipbuilding Progress, 18(207), 397-415.
[MTH11] Molland, A.F., Turnock, S.R. y Hudson, D.A. (2011). Ship Resistance and
        Propulsion. Cambridge University Press. Reproduce las ecuaciones de [HM82]/[H84]
        (ec. 4.23 y 10.24-10.34) y de [VO71] (ec. 10.49, tabla A3.23, con la errata
        d3,5 = +9.86873 corregida), y se ha usado para comprobar los coeficientes.
[MXS]   Bentley Maxsurf Resistance, Manual, Appendix B "Applicability" (rangos de validez).

Todas las magnitudes en SI (m, m/s, N, W); los parámetros de forma se miden en el casco
(ver hidrostatica.py). LCB en % de L desde la mitad de L, positivo hacia proa.
"""

from __future__ import annotations

import numpy as np

G = 9.81
NUDO = 1852.0 / 3600.0
RHO = 1025.0               # kg/m3, agua de mar a 15 °C
NU = 1.18831e-6            # m2/s, agua de mar a 15 °C (ITTC 7.5-02-01-03)


def cf_ittc57(re):
    """Línea de fricción ITTC-1957: CF = 0.075 / (log10 Re - 2)^2."""
    return 0.075 / (np.log10(re) - 2.0) ** 2


def parametros(H, lpp: float | None = None) -> dict:
    """Parámetros de entrada a partir de unas Hidrostaticas (casco completo)."""
    return {
        "L": H.lwl, "Lpp": lpp or H.lwl, "B": H.bwl, "T": H.calado, "TF": H.calado_proa,
        "vol": H.volumen, "S": H.sup_mojada, "Cb": H.cb, "Cp": H.cp, "Cm": H.cm,
        "Cwp": H.cwp, "lcb": H.lcb_pct, "iE": H.ie, "AT": H.a_espejo, "S_skeg": H.sup_skeg,
    }


# ---------------------------------------------------------------------------
# Rangos de validez
# ---------------------------------------------------------------------------

# (nombre, función que da el valor, mínimo, máximo, fuente)
def _valores(p: dict, fn_max: float) -> dict:
    return {"Fn": fn_max, "L/B": p["L"] / p["B"], "B/T": p["B"] / p["T"], "Cp": p["Cp"],
            "Cm": p["Cm"], "LCB [%L]": p["lcb"], "iE [°]": p["iE"], "L [m]": p["L"],
            "Volumen [m3]": p["vol"]}


RANGOS = {
    "Holtrop-Mennen": [
        # [HM82] tabla de rangos por tipo de buque, fila "fishing vessels, tugs"
        # (reproducida en [MTH11] tabla 10.2)
        ("Fn", None, 0.38, "HM82 (pesqueros/remolcadores)"),
        ("Cp", 0.55, 0.65, "HM82 (pesqueros/remolcadores)"),
        ("L/B", 3.9, 6.3, "HM82 (pesqueros/remolcadores)"),
        # Rango general del método en Maxsurf
        ("B/T", 2.1, 4.0, "MXS App. B"),
        ("Cp", 0.55, 0.85, "MXS App. B"),
        ("L/B", 3.9, 15.0, "MXS App. B"),
        ("Fn", None, 0.80, "MXS App. B"),
    ],
    "van Oortmerssen": [
        # Límites del método según [MXS] App. B
        ("L [m]", 8.0, 80.0, "MXS App. B"),
        ("Fn", None, 0.50, "MXS App. B"),
        ("L/B", 3.0, 6.2, "MXS App. B"),
        ("B/T", 1.9, 4.0, "MXS App. B"),
        ("Cp", 0.50, 0.73, "MXS App. B"),
        ("Cm", 0.70, 0.97, "MXS App. B"),
        ("LCB [%L]", -8.0, 2.8, "MXS App. B"),
        ("iE [°]", 10.0, 46.0, "MXS App. B"),
        ("Volumen [m3]", 5.0, 3000.0, "MXS App. B"),
        # Zona con datos de los ensayos de [VO71], según [MTH11] §10.3.4.1
        ("L/B", 3.4, 6.2, "MTH11 §10.3.4.1 (datos VO71)"),
        ("B/T", 1.9, 3.2, "MTH11 §10.3.4.1 (datos VO71)"),
        ("Cp", 0.55, 0.70, "MTH11 §10.3.4.1 (datos VO71)"),
        ("Cm", 0.76, 0.94, "MTH11 §10.3.4.1 (datos VO71)"),
        ("LCB [%L]", -4.4, 1.6, "MTH11 §10.3.4.1 (datos VO71)"),
        ("iE [°]", 15.0, 35.0, "MTH11 §10.3.4.1 (datos VO71)"),
    ],
}


def avisos_validez(metodo: str, p: dict, v_kn) -> list:
    """Textos de los parámetros fuera de rango (Fn: se comprueba la velocidad máxima)."""
    v = np.atleast_1d(v_kn) * NUDO
    fn = v / np.sqrt(G * p["L"])
    val = _valores(p, fn.max())
    avisos = []
    for nombre, mn, mx, fuente in RANGOS[metodo]:
        x = val[nombre]
        if (mn is not None and x < mn) or (mx is not None and x > mx):
            rango = f"[{'-' if mn is None else mn}, {'-' if mx is None else mx}]"
            extra = ""
            if nombre == "Fn":
                lim = mx
                v_lim = lim * np.sqrt(G * p["L"]) / NUDO
                extra = f" (por encima de {v_lim:.1f} kn)"
            avisos.append(f"{metodo}: {nombre} = {x:.3f} fuera de {rango}{extra} [{fuente}]")
    return avisos


# ---------------------------------------------------------------------------
# Holtrop-Mennen
# ---------------------------------------------------------------------------

def holtrop_mennen(p: dict, v_kn, cstern: float = 0.0, abt: float = 0.0, hb: float = 0.0,
                   s_app: float = 0.0, k2_app: float = 1.75, rho: float = RHO, nu: float = NU,
                   version: str = "1984") -> dict:
    """Resistencia total RT = RF(1+k1) + RAPP + RW + RB + RTR + RA  ([H84]; [MTH11] ec. 10.24).

    version "1984": factor de forma y RW de la reanálisis [H84] (por defecto).
            "1982": factor de forma y RW (término m2) del artículo original [HM82].
    cstern  parámetro de forma de popa (0 normal, -10 secciones en V, +10 en U) [HM82]
    abt, hb área transversal del bulbo y altura de su centro (sin bulbo: 0)
    s_app   superficie mojada de apéndices no incluidos en S (p. ej. el skeg si se
            excluyó de las hidrostáticas); (1+k2) = 1.5-2.0 para un skeg [HM82]
    """
    if version not in ("1982", "1984"):
        raise ValueError("version debe ser '1982' o '1984'")
    v = np.atleast_1d(np.asarray(v_kn, float)) * NUDO
    L, B, T, TF = p["L"], p["B"], p["T"], p["TF"]
    vol, S, Cp, Cm, Cwp, Cb = p["vol"], p["S"], p["Cp"], p["Cm"], p["Cwp"], p["Cb"]
    lcb, iE, AT = p["lcb"], p["iE"], p["AT"]
    fn = v / np.sqrt(G * L)
    q = 0.5 * rho * v ** 2

    # Fricción ITTC-57 y factor de forma [H84]; [MTH11] ec. 4.23-4.24
    cf = cf_ittc57(v * L / nu)
    LR = L * (1 - Cp + 0.06 * Cp * lcb / (4 * Cp - 1))
    if version == "1984":
        c14 = 1 + 0.011 * cstern
        k1 = (0.93 + 0.487118 * c14 * (B / L) ** 1.06806 * (T / L) ** 0.46106
              * (L / LR) ** 0.121563 * (L ** 3 / vol) ** 0.36486 * (1 - Cp) ** -0.604247)
    else:
        # [HM82]: 1+k1 = c13 {0.93 + c12 (B/LR)^0.92497 (0.95-Cp)^-0.521448 (1-Cp+0.0225 lcb)^0.6906}
        tl = T / L
        if tl > 0.05:
            c12 = tl ** 0.2228446
        elif tl > 0.02:
            c12 = 48.20 * (tl - 0.02) ** 2.078 + 0.479948
        else:
            c12 = 0.479948
        c13 = 1 + 0.003 * cstern
        k1 = c13 * (0.93 + c12 * (B / LR) ** 0.92497 * (0.95 - Cp) ** -0.521448
                    * (1 - Cp + 0.0225 * lcb) ** 0.6906)
    rf = q * S * cf
    rapp = q * s_app * k2_app * cf

    # Resistencia por formación de olas [H84]; [MTH11] ec. 10.25-10.27
    if B / L < 0.11:
        c7 = 0.229577 * (B / L) ** 0.33333
    elif B / L < 0.25:
        c7 = B / L
    else:
        c7 = 0.5 - 0.0625 * L / B
    c1 = 2223105 * c7 ** 3.78613 * (T / B) ** 1.07961 * (90 - iE) ** -1.37565
    c3 = 0.56 * abt ** 1.5 / (B * T * (0.31 * np.sqrt(abt) + TF - hb)) if abt > 0 else 0.0
    c2 = np.exp(-1.89 * np.sqrt(c3))
    c5 = 1 - 0.8 * AT / (B * T * Cm)
    lam = 1.446 * Cp - 0.03 * L / B if L / B < 12 else 1.446 * Cp - 0.36
    c16 = (8.07981 * Cp - 13.8673 * Cp ** 2 + 6.984388 * Cp ** 3) if Cp < 0.8 else 1.73014 - 0.7067 * Cp
    m1 = 0.0140407 * L / T - 1.75254 * vol ** (1 / 3) / L - 4.79323 * B / L - c16
    l3v = L ** 3 / vol
    if l3v < 512:
        c15 = -1.69385
    elif l3v < 1726.91:
        c15 = -1.69385 + (L / vol ** (1 / 3) - 8) / 2.36
    else:
        c15 = 0.0
    d = -0.9
    c17 = 6919.3 * Cm ** -1.3346 * (vol / L ** 3) ** 2.00977 * (L / B - 2) ** 1.40692
    m3 = -7.2035 * (B / L) ** 0.326869 * (T / B) ** 0.605375
    peso = vol * rho * G

    def m4(f):
        return c15 * 0.4 * np.exp(-0.034 * f ** -3.29)

    def rw_a(f):   # Fn < 0.40
        return c1 * c2 * c5 * peso * np.exp(m1 * f ** d + m4(f) * np.cos(lam * f ** -2))

    def rw_b(f):   # Fn > 0.55
        return c17 * c2 * c5 * peso * np.exp(m3 * f ** d + m4(f) * np.cos(lam * f ** -2))

    if version == "1984":
        rw = np.where(fn <= 0.40, rw_a(fn),
                      np.where(fn >= 0.55, rw_b(fn),
                               rw_a(0.40) + (10 * fn - 4) * (rw_b(0.55) - rw_a(0.40)) / 1.5))
    else:
        # [HM82]: una sola expresión con m2 = c15 Cp² exp(-0.1 Fn^-2)
        m2 = c15 * Cp ** 2 * np.exp(-0.1 * fn ** -2)
        rw = c1 * c2 * c5 * peso * np.exp(m1 * fn ** d + m2 * np.cos(lam * fn ** -2))

    # Bulbo [H84]; [MTH11] ec. 10.30-10.31
    if abt > 0:
        pb = 0.56 * np.sqrt(abt) / (TF - 1.5 * hb)
        fni = v / np.sqrt(G * (TF - hb - 0.25 * np.sqrt(abt)) + 0.15 * v ** 2)
        rb = 0.11 * np.exp(-3 * pb ** -2) * fni ** 3 * abt ** 1.5 * rho * G / (1 + fni ** 2)
    else:
        rb = np.zeros_like(v)

    # Espejo sumergido [HM82]; [MTH11] ec. 10.32-10.33
    if AT > 0:
        fnt = v / np.sqrt(2 * G * AT / (B + B * Cwp))
        c6 = np.where(fnt < 5, 0.2 * (1 - 0.2 * fnt), 0.0)
        rtr = q * AT * c6
    else:
        rtr = np.zeros_like(v)

    # Correlación modelo-buque [HM82]; [MTH11] ec. 10.34
    c4 = min(TF / L, 0.04)
    ca = (0.006 * (L + 100) ** -0.16 - 0.00205
          + 0.003 * np.sqrt(L / 7.5) * Cb ** 4 * c2 * (0.04 - c4))
    ra = q * S * ca

    rt = rf * k1 + rapp + rw + rb + rtr + ra
    return {"V_kn": v / NUDO, "Fn": fn, "Rn": v * L / nu, "CF": cf, "1+k1": np.full_like(v, k1),
            "RF": rf, "RAPP": rapp, "RW": rw, "RB": rb, "RTR": rtr, "RA": ra, "RT": rt,
            "EHP": rt * v, "CA": np.full_like(v, ca)}


# ---------------------------------------------------------------------------
# van Oortmerssen
# ---------------------------------------------------------------------------

# Coeficientes d_i,j de [VO71], tal como los da [MTH11] tabla A3.23 (d3,5 corregido a
# +9.86873). Filas i = 1..4; columnas j = 0..11 multiplican a
# [1, LCB, LCB², Cp, Cp², L/B, (L/B)², Cwl, Cwl², B/T, (B/T)², Cm]
D_VO = np.array([
    [79.32134, -0.09287, -0.00209, -246.45896, 187.13664, -1.42893, 0.11898,
     0.15727, -0.00064, -2.52862, 0.50619, 1.62851],
    [6714.88397, 19.83000, 2.66997, -19662.02400, 14099.90400, 137.33613, -13.36938,
     -4.49852, 0.02100, 216.44923, -35.07602, -128.72535],
    [-908.44371, 2.52704, -0.35794, 755.18660, -48.93952, 9.86873, -0.77652,
     3.79020, -0.01879, -9.24399, 1.28571, 250.64910],
    [3012.14549, 2.71437, 0.25521, -9198.80840, 6886.60416, -159.92694, 16.23621,
     -0.82014, 0.00225, 236.37970, -44.17820, 207.25580],
])

# Suma de los suplementos de prueba de [MTH11] tabla 10.7 (rugosidad 0.00035,
# timón 0.00004, quillas de balance 0.00004, aire 0.00008)
DCF_VO = 0.00051


def rr_sobre_peso_vo(fn, L, B, T, Cp, Cm, lcb, iE):
    """R_R/Δ de [VO71]; [MTH11] ec. 10.49. Δ = peso de desplazamiento (N)."""
    fn = np.atleast_1d(np.asarray(fn, float))
    cwl = iE * L / B                       # parámetro de entrada Cwl = iE·L/B
    x = np.array([1, lcb, lcb ** 2, Cp, Cp ** 2, L / B, (L / B) ** 2, cwl, cwl ** 2,
                  B / T, (B / T) ** 2, Cm])
    c = D_VO @ x * 1e-3
    m = 0.14347 * Cp ** -2.1976
    f2 = fn ** -2.0
    e = np.exp(-m * f2)
    return c[0] * np.exp(-m / 9 * f2) + e * (c[1] + c[2] * np.sin(f2) + c[3] * np.cos(f2))


def van_oortmerssen(p: dict, v_kn, dcf: float = DCF_VO, s_app: float = 0.0,
                    rho: float = RHO, nu: float = NU) -> dict:
    """RT = R_R + (CF + ΔCF)·½ρSV²  ([VO71]; [MTH11] §10.3.4.1).

    Se usa como eslora L_D = (Lpp + Lwl)/2 si se da Lpp; si no, Lwl (Molland indica que
    el error es pequeño). s_app: superficie de apéndices no incluida en S.
    """
    v = np.atleast_1d(np.asarray(v_kn, float)) * NUDO
    L = 0.5 * (p["Lpp"] + p["L"])
    fn = v / np.sqrt(G * L)
    peso = p["vol"] * rho * G
    rr = rr_sobre_peso_vo(fn, L, p["B"], p["T"], p["Cp"], p["Cm"], p["lcb"], p["iE"]) * peso
    cf = cf_ittc57(v * L / nu)
    q = 0.5 * rho * v ** 2
    rf = (cf + dcf) * q * (p["S"] + s_app)
    rt = rr + rf
    return {"V_kn": v / NUDO, "Fn": fn, "Rn": v * L / nu, "CF": cf, "RR": rr, "RF": rf,
            "RT": rt, "EHP": rt * v}


def curvas(H, v_kn, con_skeg: bool = True, lpp: float | None = None, cstern: float = 0.0,
           version_holtrop: str = "1984") -> dict:
    """Curvas de los dos métodos y avisos de validez para unas hidrostáticas."""
    p = parametros(H, lpp)
    s_app = 0.0 if con_skeg else p["S_skeg"]
    return {
        "Holtrop-Mennen": holtrop_mennen(p, v_kn, cstern=cstern, s_app=s_app, version=version_holtrop),
        "van Oortmerssen": van_oortmerssen(p, v_kn, s_app=s_app),
        "avisos": avisos_validez("Holtrop-Mennen", p, v_kn) + avisos_validez("van Oortmerssen", p, v_kn),
        "parametros": p,
    }
