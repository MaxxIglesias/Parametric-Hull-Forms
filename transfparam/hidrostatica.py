"""Hidrostáticas a una flotación z = DWL por secciones transversales.

Método
------
1. Se reúnen los triángulos de todas las piezas (teselado denso de las NURBS recortadas)
   y se cortan con planos x = cte (n secciones equiespaciadas entre los extremos).
2. El casco de partida es un mosaico de superficies que se solapan y tiene caras
   interiores (p. ej. el fondo del casco por encima del skeg). Por eso, en cada sección
   se toma el CONTORNO EXTERIOR: para cada cota z la semimanga máxima y_e(z) de todos
   los segmentos cortados (envolvente). Para una sección cuyo contorno corta una sola
   vez cada horizontal (caso de un monocasco), esto equivale a la integral de contorno
   A = ∮ y dz del enunciado:
        A(x) = 2 ∫_0^DWL y_e(z) dz         (casco completo, x2 por simetría)
3. Volumen, LCB, área de flotación y LCF se integran en x con la regla de los trapecios.
4. Superficie mojada: suma del área de los triángulos EXTERIORES por debajo de la DWL
   (recortados exactamente por el plano z = DWL). Un triángulo es exterior si su
   centroide está sobre la envolvente lateral y_e(x, z) o sobre la envolvente de fondo
   z_f(x, y) (cota mínima en cada (x, y)), con una tolerancia de 1,5 cm. El espejo no
   cuenta como superficie mojada (se da aparte su área sumergida A_T).
5. Eslora y manga en la flotación, del corte del casco con el plano z = DWL.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from . import RHO_AGUA_MAR
from .modelo import ROL_ESPEJO, ROL_SKEG, Casco, teselar_casco, triangulos_casco


# ---------------------------------------------------------------------------
# Geometría auxiliar
# ---------------------------------------------------------------------------

def cortar(T: np.ndarray, eje: int, valor: float) -> np.ndarray:
    """Segmentos (m, 2, 3) del corte de los triángulos T (n, 3, 3) con el plano eje = valor."""
    d = T[:, :, eje] - valor
    neg = d < 0
    pts, mascaras = [], []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        m = neg[:, a] != neg[:, b]
        den = np.where(m, d[:, a] - d[:, b], 1.0)
        t = np.where(m, d[:, a] / den, 0.0)
        pts.append(T[:, a] + t[:, None] * (T[:, b] - T[:, a]))
        mascaras.append(m)
    M = np.stack(mascaras, axis=1)
    sel = M.sum(axis=1) == 2
    P = np.stack(pts, axis=1)[sel]
    idx = np.argsort(~M[sel], axis=1, kind="stable")[:, :2]
    return np.take_along_axis(P, idx[:, :, None], axis=1)


def envolvente(seg: np.ndarray, eje_nivel: int, eje_valor: int, niveles: np.ndarray,
               modo: str = "max") -> np.ndarray:
    """Valor extremo (máx. o mín.) de la coordenada eje_valor de los segmentos en cada nivel
    de la coordenada eje_nivel. Niveles equiespaciados. Sin segmento: NaN."""
    out = np.full(len(niveles), np.nan)
    if len(seg) == 0:
        return out
    a0, a1 = seg[:, 0, eje_nivel], seg[:, 1, eje_nivel]
    b0, b1 = seg[:, 0, eje_valor], seg[:, 1, eje_valor]
    l0, dl = niveles[0], niveles[1] - niveles[0]
    lo, hi = np.minimum(a0, a1), np.maximum(a0, a1)
    i0 = np.ceil((lo - l0) / dl - 1e-9).astype(int)
    i1 = np.floor((hi - l0) / dl + 1e-9).astype(int)
    i0 = np.maximum(i0, 0)
    i1 = np.minimum(i1, len(niveles) - 1)
    n = i1 - i0 + 1
    ok = n > 0
    if not ok.any():
        return out
    s = np.repeat(np.nonzero(ok)[0], n[ok])
    inicio = np.repeat(np.cumsum(n[ok]) - n[ok], n[ok])
    k = np.repeat(i0[ok], n[ok]) + (np.arange(len(s)) - inicio)
    a = niveles[k]
    den = a1[s] - a0[s]
    plano = np.abs(den) < 1e-14
    t = np.where(plano, 0.0, (a - a0[s]) / np.where(plano, 1.0, den))
    b = b0[s] + t * (b1[s] - b0[s])
    # Segmento contenido en el nivel: cuenta su extremo más alejado
    if modo == "max":
        b = np.where(plano, np.maximum(b0[s], b1[s]), b)
        tmp = np.full(len(niveles), -np.inf)
        np.maximum.at(tmp, k, b)
        out = np.where(np.isfinite(tmp), tmp, np.nan)
    else:
        b = np.where(plano, np.minimum(b0[s], b1[s]), b)
        tmp = np.full(len(niveles), np.inf)
        np.minimum.at(tmp, k, b)
        out = np.where(np.isfinite(tmp), tmp, np.nan)
    return out


def _area_bajo_plano(T: np.ndarray, z: float) -> np.ndarray:
    """Área de la parte de cada triángulo por debajo de z (recorte exacto)."""
    A = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    bajo = T[:, :, 2] <= z
    nb = bajo.sum(axis=1)
    res = np.where(nb == 3, A, 0.0)
    for caso in (1, 2):
        idx = np.nonzero(nb == caso)[0]
        if len(idx) == 0:
            continue
        Ti, bi = T[idx], bajo[idx]
        # vértice solitario: el único bajo (caso 1) o el único sobre (caso 2)
        solo = np.argmax(bi if caso == 1 else ~bi, axis=1)
        o1, o2 = (solo + 1) % 3, (solo + 2) % 3
        r = np.arange(len(idx))
        zs, z1, z2 = Ti[r, solo, 2], Ti[r, o1, 2], Ti[r, o2, 2]
        t1 = (z - zs) / (z1 - zs)
        t2 = (z - zs) / (z2 - zs)
        f = np.clip(t1 * t2, 0, 1)  # fracción del triángulo pequeño junto al vértice solitario
        res[idx] = A[idx] * (f if caso == 1 else 1 - f)
    return res


# ---------------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------------

@dataclass
class Hidrostaticas:
    dwl: float
    volumen: float
    desplazamiento: float
    lwl: float
    bwl: float
    calado: float
    am: float
    cb: float
    cp: float
    cm: float
    cwp: float
    awp: float
    lcb: float               # m desde x = 0 (espejo)
    lcb_pct: float           # % Lwl desde la mitad de Lwl, + a proa
    lcf: float
    lcf_pct: float
    sup_mojada: float        # sin espejo
    x_popa_wl: float         # extremos de la flotación
    x_proa_wl: float
    x_am: float              # posición de la sección de área máxima
    a_espejo: float          # área sumergida del espejo, casco completo (A_T)
    ie: float                # semiángulo de entrada en la flotación (grados)
    calado_proa: float       # DWL - quilla en la perpendicular de proa (Holtrop T_F)
    sup_skeg: float          # superficie mojada de las piezas SKEG (casco completo)
    # Curvas
    x: np.ndarray = field(repr=False, default=None)
    sac: np.ndarray = field(repr=False, default=None)        # A(x), casco completo
    semimanga_wl: np.ndarray = field(repr=False, default=None)
    quilla: np.ndarray = field(repr=False, default=None)     # z mínima de cada sección

    def resumen(self) -> dict:
        d = asdict(self)
        for k in ("x", "sac", "semimanga_wl", "quilla"):
            d.pop(k)
        return d


ETIQUETAS = {
    "dwl": "DWL [m]", "volumen": "Volumen [m3]", "desplazamiento": "Desplazamiento [t]",
    "lwl": "Lwl [m]", "bwl": "Bwl [m]", "calado": "T [m]", "am": "Am [m2]",
    "cb": "Cb", "cp": "Cp", "cm": "Cm", "cwp": "Cwp", "awp": "Awp [m2]",
    "lcb": "LCB desde espejo [m]", "lcb_pct": "LCB [% Lwl desde mitad Lwl, + proa]",
    "lcf": "LCF desde espejo [m]", "lcf_pct": "LCF [% Lwl desde mitad Lwl, + proa]",
    "sup_mojada": "Superficie mojada S [m2]", "x_popa_wl": "x popa flotación [m]",
    "x_proa_wl": "x proa flotación [m]", "x_am": "x sección máxima [m]",
    "a_espejo": "Área espejo sumergida AT [m2]", "ie": "Semiángulo de entrada iE [°]",
    "calado_proa": "Calado en Ppr TF [m]", "sup_skeg": "Superficie mojada skeg [m2]",
}


# ---------------------------------------------------------------------------
# Cálculo
# ---------------------------------------------------------------------------

def calcular(casco: Casco, dwl: float, superficies: dict | None = None, con_skeg: bool = True,
             n_secciones: int = 200, n_z: int = 200, rho: float = RHO_AGUA_MAR,
             tol_exterior: float = 0.015) -> Hidrostaticas:
    """Hidrostáticas del casco completo (x2) a la flotación z = dwl.

    superficies  {id: Superficie} transformadas; None = las del casco leído.
    con_skeg     incluye las piezas con rol SKEG.
    """
    excl = () if con_skeg else (ROL_SKEG,)
    T_todo, ids = triangulos_casco(casco, superficies, roles_excluidos=excl)
    roles = {p.id: p.rol for p in casco.piezas}
    bajo = T_todo[:, :, 2].min(axis=1) <= dwl
    T, ids = T_todo[bajo], ids[bajo]
    es_espejo = np.array([roles[i] == ROL_ESPEJO for i in ids])
    es_skeg = np.array([roles[i] == ROL_SKEG for i in ids])

    zmin = T[:, :, 2].min()
    calado = dwl - zmin
    x0, x1 = T[:, :, 0].min(), T[:, :, 0].max()
    eps = 1e-7 * (x1 - x0)
    xs = np.linspace(x0 + eps, x1 - eps, n_secciones + 1)
    zs = np.linspace(zmin, dwl, n_z + 1)

    # Manga máxima (para los niveles de la envolvente de fondo)
    ymax = max(T[:, :, 1].max(), 1e-3)
    ys = np.linspace(0.0, ymax, n_z + 1)

    txmin, txmax = T[:, :, 0].min(axis=1), T[:, :, 0].max(axis=1)
    orden = np.argsort(txmin)
    txmin_o = txmin[orden]

    sac = np.zeros(len(xs))
    ywl = np.zeros(len(xs))
    quilla = np.full(len(xs), np.nan)
    Ye = np.zeros((len(xs), len(zs)))        # envolvente lateral y_e(x, z)
    Zf = np.full((len(xs), len(ys)), np.inf)  # envolvente de fondo z_f(x, y)
    for i, x in enumerate(xs):
        cand = orden[: np.searchsorted(txmin_o, x, side="right")]
        cand = cand[txmax[cand] >= x]
        seg = cortar(T[cand], 0, x)
        if len(seg) == 0:
            continue
        ye = envolvente(seg, 2, 1, zs, "max")
        ye = np.where(np.isnan(ye), 0.0, np.maximum(ye, 0.0))
        Ye[i] = ye
        sac[i] = 2.0 * np.trapezoid(ye, zs)
        ywl[i] = ye[-1]
        quilla[i] = seg[:, :, 2].min()
        zf = envolvente(seg, 1, 2, ys, "min")
        Zf[i] = np.where(np.isnan(zf), np.inf, zf)

    volumen = np.trapezoid(sac, xs)
    lcb = np.trapezoid(sac * xs, xs) / volumen
    awp = 2.0 * np.trapezoid(ywl, xs)
    lcf = np.trapezoid(2.0 * ywl * xs, xs) / awp if awp > 0 else np.nan
    i_am = int(np.argmax(sac))
    am, x_am = sac[i_am], xs[i_am]

    # Flotación: corte con z = dwl
    seg_wl = cortar(T_todo, 2, dwl)
    seg_wl = seg_wl[seg_wl[:, :, 1].max(axis=1) > 1e-6]
    xa, xf = seg_wl[:, :, 0].min(), seg_wl[:, :, 0].max()
    lwl = xf - xa
    bwl = 2.0 * seg_wl[:, :, 1].max()

    # Semiángulo de entrada: pendiente media de la flotación entre el 0,5 % y el 5 % de
    # Lwl desde la proa (se desprecia la forma local de la roda, como pide Holtrop)
    xn = np.linspace(xf - 0.05 * lwl, xf - 0.005 * lwl, 40)
    yn = envolvente(seg_wl, 0, 1, np.append(xn, xn[-1] + (xn[1] - xn[0])), "max")[:-1]
    ok = ~np.isnan(yn)
    pend = -np.polyfit(xn[ok], yn[ok], 1)[0] if ok.sum() > 3 else np.nan
    ie = float(np.degrees(np.arctan(pend)))

    # Calado a proa (T_F de Holtrop): línea de quilla entre la maestra y el 85 % de la
    # flotación, prolongada en línea recta hasta la perpendicular de proa (así no se
    # mide en la subida de la roda)
    ok = ~np.isnan(quilla) & (xs >= x_am) & (xs <= xa + 0.85 * lwl)
    if ok.sum() > 2:
        calado_proa = float(dwl - np.polyval(np.polyfit(xs[ok], quilla[ok], 1), xf))
    else:
        calado_proa = calado

    # Área sumergida del espejo (A_T): área de las caras ESPEJO por debajo de la DWL
    # (son planos x = cte, así que su área es la de la proyección transversal)
    cent_e = T[es_espejo].mean(axis=1) if es_espejo.any() else np.zeros((0, 3))
    a_espejo = 2.0 * float(_area_bajo_plano(T[es_espejo], dwl)[cent_e[:, 1] >= 0].sum()) \
        if es_espejo.any() else 0.0

    # Superficie mojada (solo triángulos exteriores, sin espejo)
    S, S_skeg = _superficie_mojada(T, xs, zs, ys, Ye, Zf, dwl, es_espejo, es_skeg, tol_exterior)

    lmed = 0.5 * (xa + xf)
    return Hidrostaticas(
        dwl=dwl, volumen=volumen, desplazamiento=rho * volumen, lwl=lwl, bwl=bwl,
        calado=calado, am=am, cb=volumen / (lwl * bwl * calado), cp=volumen / (am * lwl),
        cm=am / (bwl * calado), cwp=awp / (lwl * bwl), awp=awp,
        lcb=lcb, lcb_pct=100 * (lcb - lmed) / lwl, lcf=lcf, lcf_pct=100 * (lcf - lmed) / lwl,
        sup_mojada=S, x_popa_wl=xa, x_proa_wl=xf, x_am=x_am, a_espejo=a_espejo, ie=ie,
        calado_proa=calado_proa, sup_skeg=S_skeg,
        x=xs, sac=sac, semimanga_wl=ywl, quilla=quilla)


def _superficie_mojada(T, xs, zs, ys, Ye, Zf, dwl, es_espejo, es_skeg, tol):
    cent = T.mean(axis=1)
    lado = RegularGridInterpolator((xs, zs), Ye, bounds_error=False, fill_value=0.0)
    Zf_ = np.where(np.isfinite(Zf), Zf, 1e3)
    fondo = RegularGridInterpolator((xs, ys), Zf_, bounds_error=False, fill_value=1e3)
    q = np.column_stack([np.clip(cent[:, 0], xs[0], xs[-1]), np.clip(cent[:, 2], zs[0], zs[-1])])
    exterior_lado = cent[:, 1] >= lado(q) - tol
    q = np.column_stack([np.clip(cent[:, 0], xs[0], xs[-1]), np.clip(cent[:, 1], ys[0], ys[-1])])
    exterior_fondo = cent[:, 2] <= fondo(q) + tol
    exterior = (exterior_lado | exterior_fondo) & (cent[:, 1] >= 0) & ~es_espejo
    area = _area_bajo_plano(T, dwl)
    return 2.0 * float(area[exterior].sum()), 2.0 * float(area[es_skeg & (cent[:, 1] >= 0)].sum())


# ---------------------------------------------------------------------------
# Convergencia
# ---------------------------------------------------------------------------

CLAVES_CONVERGENCIA = ("volumen", "lcb", "awp", "lcf", "am", "sup_mojada", "lwl", "bwl", "cb", "cp")


def convergencia(casco: Casco, dwl: float, h: float, n_secciones: int, n_z: int,
                 superficies: dict | None = None, con_skeg: bool = True) -> dict:
    """Repite el cálculo doblando la resolución (teselado h/2, 2n secciones, 2n_z cotas)
    y devuelve la diferencia relativa (%) de cada magnitud. El teselado original se
    restaura al terminar."""
    campos = ("us", "vs", "tri", "borde_idx", "borde_uv")
    guardado = [{c: getattr(p, c) for c in campos} for p in casco.piezas]
    try:
        teselar_casco(casco, h)
        H1 = calcular(casco, dwl, superficies, con_skeg, n_secciones, n_z)
        teselar_casco(casco, h / 2)
        H2 = calcular(casco, dwl, superficies, con_skeg, 2 * n_secciones, 2 * n_z)
    finally:
        for p, g in zip(casco.piezas, guardado):
            for c, v in g.items():
                setattr(p, c, v)
    dif = {}
    for k in CLAVES_CONVERGENCIA:
        a, b = getattr(H1, k), getattr(H2, k)
        ref = H2.lwl if k in ("lcb", "lcf") else abs(b)   # posiciones: respecto a Lwl
        dif[k] = 100 * abs(a - b) / ref
    return {"base": H1, "fina": H2, "diferencias_pct": dif}


def secciones(casco: Casco, xs, superficies: dict | None = None, z_max: float | None = None,
              roles_excluidos=()) -> list:
    """Segmentos (y, z) de cada sección x (para dibujar la caja de cuadernas)."""
    T, _ = triangulos_casco(casco, superficies, roles_excluidos=roles_excluidos)
    if z_max is not None:
        T = T[T[:, :, 2].min(axis=1) <= z_max]
    return [cortar(T, 0, x)[:, :, 1:] for x in xs]
