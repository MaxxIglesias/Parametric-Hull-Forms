"""Transformación paramétrica: escalado afín + desplazamiento de secciones de Lackenby.

Referencia: Lackenby, H. (1950). On the systematic geometrical variation of ship forms.
Transactions of the Institution of Naval Architects (Trans. INA), 92, 289-316.

Paso 1. Escalado afín (respecto a x = 0 espejo, y = 0 crujía, z = 0 quilla):
        x' = sx·x,  y' = sy·y,  z' = sz·z,  con sx = L*/Lwl, sy = B*/Bwl, sz = T*/T.
        Es lineal, así que aplicarlo a los puntos de control es EXACTO: el volumen pasa
        a ser V·sx·sy·sz y los coeficientes de forma no cambian.

Paso 2. Desplazamiento longitudinal de las secciones (Lackenby). Cada sección se mueve
        en x sin cambiar de forma:   x' = x + δx(x)
        Se separa el casco en cuerpo de popa [x_AP, x_M] y de proa [x_M, x_FP] (x_M:
        sección de área máxima; x_AP, x_FP: extremos de la flotación). Con la abscisa
        adimensional ξ ∈ [0, 1] medida desde la maestra hacia el extremo:
            δx = ± c · L_h · ξ (1 − ξ)          (+ hacia proa en el cuerpo de proa)
        que es nulo en la maestra y en los extremos (la eslora, la manga, el calado y el
        área maestra no cambian). Es la ley de Lackenby sin cuerpo cilíndrico (p = 0).

        Formulación de Lackenby (primer orden). Para cada medio cuerpo, con la SAC
        adimensional a(ξ) = A/Am, su coeficiente prismático Cp_h = ∫ a dξ, la posición
        de su centro x̄ = ∫ a ξ dξ / Cp_h y su radio de giro r² = ∫ a ξ² dξ / Cp_h:
            δCp_h        = c · Cp_h (1 − 2 x̄)
            δ(Cp_h · x̄)  = c · Cp_h (2 x̄ − 3 r²)
        Con los cambios de volumen y de momento que piden el Cb y el LCB objetivo se
        resuelve un sistema lineal 2x2 en (c_popa, c_proa).

        Ajuste exacto. Como las secciones se mueven sin cambiar de forma, la SAC nueva
        es exactamente A'(x + δx) = A(x), de donde
            V'  = ∫ A (1 + δx') dx                    (lineal en c)
            M'  = ∫ A (x + δx)(1 + δx') dx            (cuadrático en c)
        Este sistema se resuelve exactamente (de V' se despeja c_proa en función de
        c_popa y M' queda una ecuación de segundo grado) y, después, se corrige
        con las hidrostáticas MEDIDAS en el casco transformado hasta llegar al Cb y al
        LCB objetivo con tolerancia 1e-4.

        Aplicación. δx se aplica a los puntos de control de todas las superficies a la
        vez. Como δx no es lineal, antes se insertan nudos (sin cambiar la geometría)
        hasta que ningún tramo de superficie abarque más de h_refino en x; así el error
        de mover los puntos de control en lugar de los de la superficie es O(h²) y las
        juntas entre piezas no se abren (se comprueba en los tests).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import RHO_AGUA_MAR
from . import hidrostatica as hidro
from .modelo import Casco, separacion_juntas


# ---------------------------------------------------------------------------
# Escalado
# ---------------------------------------------------------------------------

def escalar(superficies: dict, sx: float, sy: float, sz: float) -> dict:
    f = np.array([sx, sy, sz])
    return {k: s.copia(s.P * f) for k, s in superficies.items()}


def refinar(superficies: dict, h: float, estaciones=(), h_fino: float | None = None) -> dict:
    """Refina todas las superficies (tramos <= h en x; <= h_fino junto a las estaciones)."""
    return {k: s.refinar_en_x(h, estaciones=estaciones, h_fino=h_fino) for k, s in superficies.items()}


def estaciones_lackenby(H) -> list:
    """Secciones donde la ley de Lackenby cambia de pendiente (en los ejes del casco H):
    extremos de la flotación, sección de área máxima y mitad de la flotación."""
    return [H.x_popa_wl, H.x_am, 0.5 * (H.x_popa_wl + H.x_proa_wl), H.x_proa_wl]


def refinar_para_lackenby(superficies: dict, H0, L_objetivo: float, h: float = 0.10,
                          h_fino: float = 0.02) -> dict:
    """Refinado del casco de partida antes de transformarlo. h y h_fino son tamaños en el
    barco transformado (se pasan a los ejes del casco de partida con la escala en x).
    El escalado lleva estas estaciones a las del casco escalado, así que sirven para
    cualquier variante con la misma flotación de partida."""
    f = H0.lwl / L_objetivo
    return refinar(superficies, h * f, estaciones_lackenby(H0), h_fino * f)


# ---------------------------------------------------------------------------
# Ley de desplazamiento de Lackenby
# ---------------------------------------------------------------------------

@dataclass
class Lackenby:
    x_popa: float     # extremo de popa de la flotación (AP)
    x_maestra: float  # sección que separa los cuerpos (área máxima)
    x_proa: float     # extremo de proa de la flotación (FP)
    c_popa: float = 0.0
    c_proa: float = 0.0

    @property
    def l_popa(self):
        return self.x_maestra - self.x_popa

    @property
    def l_proa(self):
        return self.x_proa - self.x_maestra

    def phi(self, x, cuerpo: str) -> np.ndarray:
        """Forma de la ley con amplitud unidad: δx = c·phi (m)."""
        x = np.asarray(x, float)
        if cuerpo == "proa":
            xi = (x - self.x_maestra) / self.l_proa
            return np.where((xi > 0) & (xi < 1), self.l_proa * xi * (1 - xi), 0.0)
        eta = (self.x_maestra - x) / self.l_popa
        return np.where((eta > 0) & (eta < 1), -self.l_popa * eta * (1 - eta), 0.0)

    def dphi(self, x, cuerpo: str) -> np.ndarray:
        x = np.asarray(x, float)
        if cuerpo == "proa":
            xi = (x - self.x_maestra) / self.l_proa
            return np.where((xi > 0) & (xi < 1), 1 - 2 * xi, 0.0)
        eta = (self.x_maestra - x) / self.l_popa
        return np.where((eta > 0) & (eta < 1), 1 - 2 * eta, 0.0)

    def dx(self, x) -> np.ndarray:
        return self.c_popa * self.phi(x, "popa") + self.c_proa * self.phi(x, "proa")

    def ddx(self, x) -> np.ndarray:
        """d(δx)/dx. La transformación no cruza secciones si 1 + ddx > 0, es decir,
        si |c_popa| < 1 y |c_proa| < 1."""
        return self.c_popa * self.dphi(x, "popa") + self.c_proa * self.dphi(x, "proa")

    def es_monotona(self) -> bool:
        return abs(self.c_popa) < 1 and abs(self.c_proa) < 1

    def aplicar(self, superficies: dict) -> dict:
        out = {}
        for k, s in superficies.items():
            P = s.P.copy()
            P[..., 0] = P[..., 0] + self.dx(P[..., 0])
            out[k] = s.copia(P)
        return out


def _integrales(x, A, ley: Lackenby):
    """Integrales de la SAC que dan V'(c) y M'(c) exactos (ver cabecera)."""
    I = {}
    for h in ("popa", "proa"):
        f, df = ley.phi(x, h), ley.dphi(x, h)
        I["K_" + h] = np.trapezoid(A * df, x)                  # dV/dc
        I["M_" + h] = np.trapezoid(A * (f + x * df), x)        # dM/dc (lineal)
        I["Q_" + h] = np.trapezoid(A * f * df, x)              # término cuadrático de M
    I["V0"] = np.trapezoid(A, x)
    I["M0"] = np.trapezoid(A * x, x)
    return I


def volumen_momento(I: dict, c_popa: float, c_proa: float):
    V = I["V0"] + c_popa * I["K_popa"] + c_proa * I["K_proa"]
    M = (I["M0"] + c_popa * I["M_popa"] + c_proa * I["M_proa"]
         + c_popa ** 2 * I["Q_popa"] + c_proa ** 2 * I["Q_proa"])
    return V, M


def lackenby_primer_orden(x, A, ley: Lackenby, V_obj: float, M_obj: float) -> dict:
    """Coeficientes (c_popa, c_proa) de la formulación lineal de Lackenby y los cambios
    de Cp de cada medio cuerpo que implican."""
    I = _integrales(x, A, ley)
    Mat = np.array([[I["K_popa"], I["K_proa"]], [I["M_popa"], I["M_proa"]]])
    c = np.linalg.solve(Mat, [V_obj - I["V0"], M_obj - I["M0"]])
    Am = A.max()
    res = {"c_popa": c[0], "c_proa": c[1]}
    for h, ci, L in (("popa", c[0], ley.l_popa), ("proa", c[1], ley.l_proa)):
        # Cp, x̄ y r² adimensionales del medio cuerpo (ver cabecera)
        m = (x >= ley.x_maestra) if h == "proa" else (x <= ley.x_maestra)
        xi = np.abs(x[m] - ley.x_maestra) / L
        a = A[m] / Am
        o = np.argsort(xi)
        cp = np.trapezoid(a[o], xi[o])
        xb = np.trapezoid(a[o] * xi[o], xi[o]) / cp
        r2 = np.trapezoid(a[o] * xi[o] ** 2, xi[o]) / cp
        res[f"Cp_{h}"] = cp
        res[f"xbar_{h}"] = xb
        res[f"r2_{h}"] = r2
        res[f"dCp_{h}"] = ci * cp * (1 - 2 * xb)
    return res


def lackenby_exacto(x, A, ley: Lackenby, V_obj: float, M_obj: float, c_ini=(0.0, 0.0)):
    """Resuelve exactamente V'(c) = V_obj y M'(c) = M_obj sobre la SAC transportada.

    V' es lineal en (c_popa, c_proa): se despeja c_proa = α + β·c_popa. Sustituido en
    M' (cuadrático) queda una ecuación de segundo grado en c_popa; se toma la raíz más
    cercana a c_ini (la solución de primer orden de Lackenby).
    """
    I = _integrales(x, A, ley)
    alfa = (V_obj - I["V0"]) / I["K_proa"]
    beta = -I["K_popa"] / I["K_proa"]
    a2 = I["Q_popa"] + I["Q_proa"] * beta ** 2
    a1 = 2 * I["Q_proa"] * alfa * beta + I["M_popa"] + I["M_proa"] * beta
    a0 = I["Q_proa"] * alfa ** 2 + I["M_proa"] * alfa + I["M0"] - M_obj
    if abs(a2) < 1e-14 * max(abs(a1), 1e-30):
        raices = np.array([-a0 / a1])
    else:
        disc = a1 ** 2 - 4 * a2 * a0
        if disc < 0:
            raise ValueError("Lackenby: no hay amplitudes que den ese Cb y ese LCB "
                             "(objetivo fuera del alcance de la transformación)")
        raices = np.array([(-a1 + np.sqrt(disc)) / (2 * a2), (-a1 - np.sqrt(disc)) / (2 * a2)])
    c_popa = raices[np.argmin(np.abs(raices - c_ini[0]))]
    c = np.array([c_popa, alfa + beta * c_popa])
    V, M = volumen_momento(I, *c)
    if abs(V - V_obj) > 1e-9 * abs(V_obj) or abs(M - M_obj) > 1e-9 * abs(M_obj):
        raise RuntimeError("Lackenby: la solución exacta no cumple las ecuaciones")
    return c


# ---------------------------------------------------------------------------
# Objetivos
# ---------------------------------------------------------------------------

def dimensiones_objetivo(H0: hidro.Hidrostaticas, L=None, B=None, T=None, cb=None, delta=None,
                         libre: str | None = None, rho: float = RHO_AGUA_MAR) -> dict:
    """Completa (L, B, T, Cb, Δ) como Maxsurf: como mucho 3 de {Δ, L, B, T} fijos.

    Los valores no dados toman los del casco de partida. Si se da Δ, una variable queda
    libre (por defecto T si no se dio, si no B, L y por último Cb) y se despeja de
    Δ = ρ·Cb·L·B·T.
    """
    dados = {"L": L, "B": B, "T": T, "Cb": cb}
    v = {"L": L or H0.lwl, "B": B or H0.bwl, "T": T or H0.calado, "Cb": cb or H0.cb}
    if delta is not None:
        if libre is None:
            no_dados = [k for k in ("T", "B", "L") if dados[k] is None]
            libre = no_dados[0] if no_dados else ("Cb" if cb is None else None)
        if libre is None:
            delta_calc = rho * v["Cb"] * v["L"] * v["B"] * v["T"]
            if abs(delta_calc - delta) > 5e-3 * delta:
                raise ValueError(f"Δ, L, B, T y Cb fijados a la vez no cuadran: ρ·Cb·L·B·T = "
                                 f"{delta_calc:.2f} t frente a Δ = {delta:.2f} t. Deja una libre.")
        else:
            otros = np.prod([v[k] for k in ("L", "B", "T", "Cb") if k != libre])
            v[libre] = delta / (rho * otros)
    v["delta"] = rho * v["Cb"] * v["L"] * v["B"] * v["T"]
    v["libre"] = libre
    return v


# ---------------------------------------------------------------------------
# Transformación completa
# ---------------------------------------------------------------------------

@dataclass
class Resultado:
    superficies: dict
    H_original: hidro.Hidrostaticas
    H_escalado: hidro.Hidrostaticas
    H_final: hidro.Hidrostaticas
    escalas: tuple
    dwl: float
    ley: Lackenby | None
    lackenby_lineal: dict = field(default_factory=dict)
    iteraciones: list = field(default_factory=list)
    juntas_mm: dict = field(default_factory=dict)
    objetivos: dict = field(default_factory=dict)


def transformar(casco: Casco, dwl0: float, L=None, B=None, T=None, cb=None, lcb_pct=None,
                delta=None, libre=None, con_skeg: bool = True, h_refino: float = 0.10,
                n_secciones: int = 200, n_z: int = 200, tol: float = 1e-4, iter_max: int = 8,
                H0: hidro.Hidrostaticas | None = None, sup_refinadas: dict | None = None,
                maestra: str = "area_maxima", verbose: bool = False) -> Resultado:
    """Escala el casco a (L, B, T) y desplaza las secciones (Lackenby) hasta el Cb y el
    LCB (% Lwl desde la mitad de la flotación, + a proa) objetivo.

    sup_refinadas: superficies del casco original ya refinadas (para no repetirlo en
    cada variante de una serie).
    """
    calc = lambda sups, dwl: hidro.calcular(casco, dwl, sups, con_skeg, n_secciones, n_z)
    if H0 is None:
        H0 = calc(None, dwl0)
    obj = dimensiones_objetivo(H0, L, B, T, cb, delta, libre)
    obj["lcb_pct"] = H0.lcb_pct if lcb_pct is None else lcb_pct

    # 1) Escalado afín sobre las superficies refinadas
    if sup_refinadas is None:
        sup_refinadas = refinar_para_lackenby(casco.superficies(), H0, obj["L"], h_refino)
    sx, sy, sz = obj["L"] / H0.lwl, obj["B"] / H0.bwl, obj["T"] / H0.calado
    S1 = escalar(sup_refinadas, sx, sy, sz)
    dwl = dwl0 * sz
    H1 = calc(S1, dwl)

    # 2) Lackenby
    lmed = 0.5 * (H1.x_popa_wl + H1.x_proa_wl)
    V_obj = obj["Cb"] * H1.lwl * H1.bwl * H1.calado
    x_lcb_obj = lmed + obj["lcb_pct"] / 100 * H1.lwl
    M_obj = V_obj * x_lcb_obj
    necesita = (abs(H1.cb - obj["Cb"]) > tol) or (abs(H1.lcb_pct - obj["lcb_pct"]) / 100 > tol)
    if not necesita:
        return Resultado(S1, H0, H1, H1, (sx, sy, sz), dwl, None, objetivos=obj,
                         juntas_mm=_juntas_mm(casco, S1))

    x_m = H1.x_am if maestra == "area_maxima" else lmed
    ley = Lackenby(H1.x_popa_wl, x_m, H1.x_proa_wl)
    lineal = lackenby_primer_orden(H1.x, H1.sac, ley, V_obj, M_obj)
    c = lackenby_exacto(H1.x, H1.sac, ley, V_obj, M_obj, (lineal["c_popa"], lineal["c_proa"]))

    # Corrección con las hidrostáticas medidas: se ajustan los objetivos "virtuales" de
    # la SAC hasta que el casco medido da el Cb y el LCB pedidos
    V_v, M_v = V_obj, M_obj
    historia = []
    mejor = None
    for it in range(iter_max):
        ley.c_popa, ley.c_proa = float(c[0]), float(c[1])
        if not ley.es_monotona():
            raise ValueError(f"La transformación cruzaría secciones (c_popa = {c[0]:.3f}, "
                             f"c_proa = {c[1]:.3f}; debe ser |c| < 1). Objetivo demasiado lejano.")
        S2 = ley.aplicar(S1)
        H2 = calc(S2, dwl)
        e_cb = H2.cb - obj["Cb"]
        e_lcb = (H2.lcb_pct - obj["lcb_pct"]) / 100
        historia.append({"iter": it, "c_popa": ley.c_popa, "c_proa": ley.c_proa,
                         "Cb": H2.cb, "LCB_pct": H2.lcb_pct, "error_Cb": e_cb, "error_LCB_frac": e_lcb})
        if verbose:
            print(f"    it {it}: c_popa={ley.c_popa:+.6f} c_proa={ley.c_proa:+.6f} "
                  f"Cb={H2.cb:.6f} LCB={H2.lcb_pct:+.5f}%")
        err = max(abs(e_cb), abs(e_lcb))
        if mejor is None or err < mejor[0]:
            mejor = (err, S2, H2, ley.c_popa, ley.c_proa)
        if err <= tol:
            break
        # Objetivos con las dimensiones MEDIDAS en esta iteración (Lwl puede variar unos mm)
        V_med = obj["Cb"] * H2.lwl * H2.bwl * H2.calado
        x_med = 0.5 * (H2.x_popa_wl + H2.x_proa_wl) + obj["lcb_pct"] / 100 * H2.lwl
        V_v += V_med - H2.volumen
        M_v += V_med * x_med - H2.volumen * H2.lcb
        try:
            c = lackenby_exacto(H1.x, H1.sac, ley, V_v, M_v, c)
        except RuntimeError:
            break
    err, S2, H2, ley.c_popa, ley.c_proa = mejor
    if err > tol:
        print(f"  aviso: Lackenby se queda a {err:.1e} del objetivo (tolerancia {tol})")

    return Resultado(S2, H0, H1, H2, (sx, sy, sz), dwl, ley, lineal, historia,
                     _juntas_mm(casco, S2), obj)


def _juntas_mm(casco: Casco, superficies: dict) -> dict:
    if not casco.juntas:
        return {}
    return {k: 1000 * v for k, v in separacion_juntas(casco, superficies).items()}
