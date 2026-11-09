"""Lectura del .3dm, piezas del casco, teselado con recortes y escritura.

El archivo de partida (exportado de Maxsurf) es un mosaico de superficies NURBS
RECORTADAS. rhino3dm no da las curvas de recorte 2D, pero sí las aristas 3D de cada
cara: se proyectan sobre la superficie (inversión de puntos, ver nurbs.py) para
rehacer el contorno de recorte en el espacio paramétrico (u, v). Con ese contorno se
teselan solo las zonas visibles de cada superficie.

Ejes del casco una vez cargado (como pide el enunciado):
    x hacia proa, con el espejo en x = 0
    y hacia estribor (solo se guardan las caras de estribor)
    z hacia arriba, con el punto más bajo del forro en z = 0
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rhino3dm as r3
from matplotlib.path import Path as Trayecto

from .nurbs import Superficie, nudos_a_rhino, nudos_desde_rhino

# Factor a metros de cada sistema de unidades de Rhino
_A_METROS = {
    "Millimeters": 1e-3, "Centimeters": 1e-2, "Meters": 1.0, "Inches": 0.0254,
    "Feet": 0.3048, "Decimeters": 0.1, "Kilometers": 1000.0,
}

ROL_CASCO = "CASCO"
ROL_SKEG = "SKEG"
ROL_ESPEJO = "ESPEJO"


# ---------------------------------------------------------------------------
# Pieza: una cara de una brep
# ---------------------------------------------------------------------------

@dataclass
class Pieza:
    id: str                       # "objeto.cara", p. ej. "13.0"
    nombre: str
    capa: str
    rol: str
    sup: Superficie               # superficie subyacente (sin recortar), en m
    lazos_uv: list                # [(es_exterior, polígono (n, 2))]
    bordes: list                  # aristas: [(uv (n, 2) sobre esta pieza)]
    # Teselado (se rellena con teselar)
    us: np.ndarray | None = None
    vs: np.ndarray | None = None
    tri: np.ndarray | None = None  # triángulos (m, 3) con índices de la rejilla aplanada
    borde_idx: np.ndarray | None = None  # vértices de la rejilla llevados al contorno
    borde_uv: np.ndarray | None = None   # sus (u, v) sobre el contorno de recorte

    def puntos_teselado(self, sup: Superficie | None = None) -> np.ndarray:
        """Vértices de la rejilla (aplanada) evaluados en sup (o en la propia).
        Los vértices exteriores de los triángulos del borde van sobre el recorte."""
        s = self.sup if sup is None else sup
        X = s.rejilla(self.us, self.vs).reshape(-1, 3)
        if self.borde_idx is not None and len(self.borde_idx):
            X[self.borde_idx] = s.puntos(self.borde_uv[:, 0], self.borde_uv[:, 1])
        return X

    def triangulos(self, sup: Superficie | None = None) -> np.ndarray:
        return self.puntos_teselado(sup)[self.tri]


@dataclass
class Casco:
    piezas: list
    origen: np.ndarray            # desplazamiento aplicado al leer (m)
    escala_unidades: float        # factor a metros del archivo original
    ruta: str = ""
    juntas: list = field(default_factory=list)  # [(pieza_a, uv_a, pieza_b, uv_b)]

    def pieza(self, id_: str) -> Pieza:
        return next(p for p in self.piezas if p.id == id_)

    def superficies(self) -> dict:
        return {p.id: p.sup for p in self.piezas}


# ---------------------------------------------------------------------------
# Lectura
# ---------------------------------------------------------------------------

def _superficie_desde_rhino(ns: r3.NurbsSurface, escala: float) -> Superficie:
    nu, nv = ns.Points.CountU, ns.Points.CountV
    Q = np.array([[[ns.Points[i, j].X, ns.Points[i, j].Y, ns.Points[i, j].Z, ns.Points[i, j].W]
                   for j in range(nv)] for i in range(nu)])
    W = Q[..., 3]
    if ns.IsRational:
        # rhino3dm devuelve los puntos de control racionales en coordenadas homogéneas
        P = Q[..., :3] / W[..., None]
    else:
        P, W = Q[..., :3], None
    return Superficie(ns.OrderU - 1, ns.OrderV - 1,
                      nudos_desde_rhino(ns.KnotsU), nudos_desde_rhino(ns.KnotsV),
                      P * escala, W)


def _muestrear_arista(arista, escala: float, paso: float) -> np.ndarray:
    """Puntos 3D a lo largo de una arista (más densos en aristas largas)."""
    d = arista.Domain
    p0, p1 = arista.PointAtStart, arista.PointAtEnd
    # longitud aproximada con 20 puntos
    t = np.linspace(d.T0, d.T1, 21)
    X = np.array([[q.X, q.Y, q.Z] for q in (arista.PointAt(ti) for ti in t)]) * escala
    long_ = np.linalg.norm(np.diff(X, axis=0), axis=1).sum()
    n = int(np.clip(np.ceil(long_ / paso), 8, 400))
    t = np.linspace(d.T0, d.T1, n + 1)
    return np.array([[q.X, q.Y, q.Z] for q in (arista.PointAt(ti) for ti in t)]) * escala


def _encadenar(tramos: list) -> np.ndarray:
    """Une tramos (X3d, uv) en un contorno cerrado eligiendo siempre el extremo 3D
    más cercano. Devuelve el contorno en (u, v)."""
    tramos = [t for t in tramos if len(t[0]) > 1]
    if not tramos:
        return np.zeros((0, 2))
    cadena = [tramos.pop(0)]
    while tramos:
        fin = cadena[-1][0][-1]
        d_ini = [np.linalg.norm(t[0][0] - fin) for t in tramos]
        d_fin = [np.linalg.norm(t[0][-1] - fin) for t in tramos]
        i, j = int(np.argmin(d_ini)), int(np.argmin(d_fin))
        if d_ini[i] <= d_fin[j]:
            cadena.append(tramos.pop(i))
        else:
            X, uv = tramos.pop(j)
            cadena.append((X[::-1], uv[::-1]))
    return np.vstack([uv for _, uv in cadena])


def puntos_reales(p: "Pieza", sup: Superficie | None = None) -> np.ndarray:
    """Puntos que pertenecen a la cara recortada (aristas y vértices del teselado,
    que ya está ajustado al contorno): sirven para buscar extremos."""
    s = p.sup if sup is None else sup
    X = [s.puntos(uv[:, 0], uv[:, 1]) for uv in p.bordes]
    if p.tri is not None and len(p.tri):
        X.append(p.puntos_teselado(s)[np.unique(p.tri)])
    return np.vstack(X)


def leer_3dm(ruta: str | Path, solo_estribor: bool = True, roles: dict | None = None,
             nombres: dict | None = None, paso_aristas: float = 0.02) -> Casco:
    """Lee el .3dm y devuelve el casco en metros, con el origen del enunciado.

    roles   {id_pieza: rol}; las caras no listadas son CASCO, salvo los espejos
            (planos x = cte en el extremo de popa), que se detectan solos.
    nombres {id_pieza: nombre} para las caras sin nombre en el archivo.
    """
    roles = roles or {}
    nombres = nombres or {}
    modelo = r3.File3dm.Read(str(ruta))
    if modelo is None:
        raise FileNotFoundError(f"No se puede leer {ruta}")
    unidad = str(modelo.Settings.ModelUnitSystem).split(".")[-1]
    if unidad not in _A_METROS:
        raise ValueError(f"Unidades del modelo no soportadas: {unidad}")
    esc = _A_METROS[unidad]
    capas = [c.Name for c in modelo.Layers]

    piezas = []
    for k, obj in enumerate(modelo.Objects):
        g = obj.Geometry
        if isinstance(g, (r3.Surface, r3.NurbsSurface)) and not isinstance(g, r3.Brep):
            g = r3.Brep.CreateFromSurface(g)
        if not isinstance(g, r3.Brep):
            continue
        aristas = {}
        for fi in range(len(g.Faces)):
            cara = g.Faces[fi]
            pid = f"{k}.{fi}"
            ns = cara.UnderlyingSurface().ToNurbsSurface()
            sup = _superficie_desde_rhino(ns, esc)

            lazos_uv, bordes = [], []
            for li in range(len(cara.Loops)):
                lazo = cara.Loops[li]
                tramos = []
                for ti in range(lazo.TrimCount):
                    ei = lazo.Trims[ti].EdgeIndex
                    if ei < 0:      # recorte singular (sin arista 3D)
                        continue
                    if ei not in aristas:
                        aristas[ei] = _muestrear_arista(g.Edges[ei], esc, paso_aristas)
                    X = aristas[ei]
                    uv, dist = sup.invertir(X)
                    if np.max(dist) > 5e-3:
                        print(f"  aviso: pieza {pid}, una arista se separa {1000 * dist.max():.1f} mm "
                              "de la superficie al proyectarla")
                    tramos.append((X, uv))
                    bordes.append(uv)
                uv = _encadenar(list(tramos))
                if len(uv) < 3:
                    continue
                exterior = str(lazo.LoopType).endswith("Outer")
                lazos_uv.append((exterior, uv))

            # Si la cara no tiene contorno (superficie sin recortar) se usa el dominio
            if not any(e for e, _ in lazos_uv):
                (u0, u1), (v0, v1) = sup.dominio_u, sup.dominio_v
                lazos_uv = [(True, np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1], [u0, v0]]))]

            nombre = obj.Attributes.Name or nombres.get(pid, "")
            capa = capas[obj.Attributes.LayerIndex] if 0 <= obj.Attributes.LayerIndex < len(capas) else ""
            piezas.append(Pieza(pid, nombre, capa, roles.get(pid, ROL_CASCO), sup, lazos_uv, bordes))

    # Teselado grueso para decidir lado, origen y espejos
    reales = {}
    for p in piezas:
        teselar(p, 0.10)
        reales[p.id] = puntos_reales(p)
    if solo_estribor:
        # Cara de estribor: llega más lejos hacia estribor que hacia babor (las
        # simétricas, que cruzan crujía, se quedan)
        piezas = [p for p in piezas
                  if reales[p.id][:, 1].max() > 1e-3
                  and reales[p.id][:, 1].max() >= -reales[p.id][:, 1].min() - 1e-3]

    # Origen: espejo (x mínima del forro) en x = 0 y punto más bajo en z = 0
    todos = np.vstack([reales[p.id] for p in piezas])
    origen = np.array([todos[:, 0].min(), 0.0, todos[:, 2].min()])
    for p in piezas:
        p.sup = p.sup.copia(p.sup.P - origen)

    # Espejos: caras planas en x = cte situadas en el extremo de popa
    for p in piezas:
        X = reales[p.id] - origen
        if p.id not in roles and np.ptp(X[:, 0]) < 1e-3 and X[:, 0].mean() < 0.02:
            p.rol = ROL_ESPEJO
    return Casco(piezas, origen, esc, str(ruta))


def cargar_casco(ruta: str | Path, config: str | Path | None = None,
                 carpeta_cache: str | Path | None = None, h_teselado: float = 0.05) -> Casco:
    """leer_3dm + teselado, guardando el resultado en una caché (pickle) para no
    repetir la lectura (unos 20 s) mientras no cambien el archivo ni la configuración."""
    import hashlib
    import pickle

    cfg = cargar_configuracion(config)
    ruta = Path(ruta)
    clave = hashlib.md5(
        f"{ruta.resolve()}|{ruta.stat().st_mtime}|{json.dumps(cfg, sort_keys=True)}|{h_teselado}|v4".encode()
    ).hexdigest()[:16]
    archivo = None
    if carpeta_cache is not None:
        Path(carpeta_cache).mkdir(parents=True, exist_ok=True)
        archivo = Path(carpeta_cache) / f"casco_{clave}.pkl"
        if archivo.is_file():
            with open(archivo, "rb") as f:
                return pickle.load(f)
    casco = leer_3dm(ruta, roles=cfg["roles"], nombres=cfg["nombres"])
    teselar_casco(casco, h_teselado)
    detectar_juntas(casco)
    if archivo is not None:
        for p in casco.piezas:
            p.sup._arbol = None
        with open(archivo, "wb") as f:
            pickle.dump(casco, f)
    return casco


def cargar_configuracion(ruta: str | Path | None) -> dict:
    """JSON con {"roles": {id: rol}, "nombres": {id: nombre}} (opcional)."""
    if ruta is None or not Path(ruta).is_file():
        return {"roles": {}, "nombres": {}}
    with open(ruta, encoding="utf-8") as f:
        d = json.load(f)
    return {"roles": d.get("roles", {}), "nombres": d.get("nombres", {})}


# ---------------------------------------------------------------------------
# Teselado
# ---------------------------------------------------------------------------

def teselar(p: Pieza, h: float) -> None:
    """Rejilla (u, v) con celdas de unos h metros ajustada al recorte.

    Se quedan los triángulos con algún vértice (o el centroide) dentro del contorno, y
    sus vértices que caen fuera se llevan al punto más cercano del contorno. Así el
    teselado acaba justo en la arista: sin huecos ni solapes con la pieza vecina.
    """
    (u0, u1), (v0, v1) = p.sup.dominio_u, p.sup.dominio_v
    # Tamaño físico aproximado en cada dirección
    G = p.sup.rejilla(np.linspace(u0, u1, 9), np.linspace(v0, v1, 9))
    lu = np.max(np.linalg.norm(np.diff(G, axis=0), axis=2).sum(axis=0))
    lv = np.max(np.linalg.norm(np.diff(G, axis=1), axis=2).sum(axis=1))
    nu = int(np.clip(np.ceil(lu / h), 2, 2000)) + 1
    nv = int(np.clip(np.ceil(lv / h), 2, 2000)) + 1
    # Incluye los nudos para no saltarse aristas vivas (p. ej. codillos)
    us = np.unique(np.concatenate([np.linspace(u0, u1, nu), np.unique(p.sup.U[p.sup.pu:-p.sup.pu])]))
    vs = np.unique(np.concatenate([np.linspace(v0, v1, nv), np.unique(p.sup.V[p.sup.pv:-p.sup.pv])]))
    nu, nv = len(us), len(vs)

    I, J = np.meshgrid(np.arange(nu - 1), np.arange(nv - 1), indexing="ij")
    a = (I * nv + J).ravel()
    b = ((I + 1) * nv + J).ravel()
    c = ((I + 1) * nv + J + 1).ravel()
    d = (I * nv + J + 1).ravel()
    tri = np.vstack([np.column_stack([a, b, c]), np.column_stack([a, c, d])])

    uu, vv = np.meshgrid(us, vs, indexing="ij")
    UV = np.column_stack([uu.ravel(), vv.ravel()])
    dentro_v = _dentro_contorno(p, UV)
    sigue = dentro_v[tri].any(axis=1) | _dentro_contorno(p, UV[tri].mean(axis=1))
    tri = tri[sigue]
    usados = np.unique(tri)
    fuera = usados[~dentro_v[usados]]
    p.us, p.vs, p.tri = us, vs, tri
    p.borde_idx = fuera
    p.borde_uv = _proyectar_contorno(p, UV[fuera])


def _proyectar_contorno(p: Pieza, UV: np.ndarray, bloque: int = 400) -> np.ndarray:
    """Punto más cercano (en (u, v) normalizado) de los contornos de recorte."""
    if len(UV) == 0:
        return np.zeros((0, 2))
    (u0, u1), (v0, v1) = p.sup.dominio_u, p.sup.dominio_v
    esc = np.array([u1 - u0, v1 - v0])
    org = np.array([u0, v0])
    A = np.vstack([pol[:-1] for _, pol in p.lazos_uv])
    B = np.vstack([pol[1:] for _, pol in p.lazos_uv])
    A, B = (A - org) / esc, (B - org) / esc
    AB = B - A
    L2 = np.maximum(np.einsum("sd,sd->s", AB, AB), 1e-30)
    Q = (UV - org) / esc
    out = np.empty_like(Q)
    for i in range(0, len(Q), bloque):
        q = Q[i:i + bloque]
        t = np.clip(np.einsum("ksd,sd->ks", q[:, None, :] - A[None], AB) / L2, 0, 1)
        P = A[None] + t[..., None] * AB[None]
        d = np.einsum("ksd,ksd->ks", P - q[:, None, :], P - q[:, None, :])
        out[i:i + bloque] = P[np.arange(len(q)), np.argmin(d, axis=1)]
    return out * esc + org


def _dentro_contorno(p: Pieza, UV: np.ndarray) -> np.ndarray:
    """¿Cae cada (u, v) dentro de la cara recortada? (lazos exteriores menos interiores)"""
    (u0, u1), (v0, v1) = p.sup.dominio_u, p.sup.dominio_v
    # Normaliza (u, v) al cuadrado unidad para que el test punto-en-polígono sea robusto
    esc = np.array([u1 - u0, v1 - v0])
    org = np.array([u0, v0])
    dentro = np.zeros(len(UV), bool)
    for exterior, pol in sorted(p.lazos_uv, key=lambda l: not l[0]):
        m = Trayecto((pol - org) / esc).contains_points((UV - org) / esc)
        dentro = (dentro | m) if exterior else (dentro & ~m)
    return dentro


def teselar_casco(casco: Casco, h: float) -> None:
    for p in casco.piezas:
        teselar(p, h)


def triangulos_casco(casco: Casco, superficies: dict | None = None, roles_excluidos=()) -> tuple:
    """Todos los triángulos del casco (n, 3, 3) y el id de pieza de cada uno."""
    T, ids = [], []
    for p in casco.piezas:
        if p.rol in roles_excluidos or p.tri is None or len(p.tri) == 0:
            continue
        sup = None if superficies is None else superficies[p.id]
        t = p.triangulos(sup)
        T.append(t)
        ids.extend([p.id] * len(t))
    return np.concatenate(T), np.array(ids)


# ---------------------------------------------------------------------------
# Juntas entre piezas
# ---------------------------------------------------------------------------

def detectar_juntas(casco: Casco, tol: float = 2e-3) -> list:
    """Puntos de las aristas de cada pieza que están sobre otra pieza.

    Para cada pareja (A, B) se guardan los parámetros del mismo punto en las dos
    superficies; tras transformar, la separación |A'(uv_a) - B'(uv_b)| mide si la
    junta se abre.
    """
    juntas = []
    cajas = {}
    for p in casco.piezas:
        X = p.puntos_teselado()
        cajas[p.id] = (X.min(axis=0) - tol, X.max(axis=0) + tol)
    for a in casco.piezas:
        for uv_a in a.bordes:
            Xa = a.sup.puntos(uv_a[:, 0], uv_a[:, 1])
            for b in casco.piezas:
                if b.id == a.id:
                    continue
                lo, hi = cajas[b.id]
                cerca = np.all((Xa >= lo) & (Xa <= hi), axis=1)
                if cerca.sum() < 3:
                    continue
                uv_b, d = b.sup.invertir(Xa[cerca])
                sobre = d < tol
                if sobre.sum() >= 3:
                    juntas.append((a.id, uv_a[cerca][sobre], b.id, uv_b[sobre]))
    casco.juntas = juntas
    return juntas


def separacion_juntas(casco: Casco, superficies: dict) -> dict:
    """Separación máxima (m) de cada junta {(a, b): s} con las superficies dadas."""
    res = {}
    for a, uv_a, b, uv_b in casco.juntas:
        Xa = superficies[a].puntos(uv_a[:, 0], uv_a[:, 1])
        Xb = superficies[b].puntos(uv_b[:, 0], uv_b[:, 1])
        s = float(np.linalg.norm(Xa - Xb, axis=1).max())
        clave = tuple(sorted((a, b)))
        res[clave] = max(res.get(clave, 0.0), s)
    return res


# ---------------------------------------------------------------------------
# Escritura
# ---------------------------------------------------------------------------

def _a_rhino(s: Superficie) -> r3.NurbsSurface:
    nu, nv = s.P.shape[:2]
    ns = r3.NurbsSurface.Create(3, s.racional, s.pu + 1, s.pv + 1, nu, nv)
    W = np.ones((nu, nv)) if s.W is None else s.W
    for i in range(nu):
        for j in range(nv):
            x, y, z = s.P[i, j]
            w = W[i, j]
            if s.racional:
                ns.Points[i, j] = r3.Point4d(x * w, y * w, z * w, w)
            else:
                ns.Points[i, j] = r3.Point4d(x, y, z, 1.0)
    for i, k in enumerate(nudos_a_rhino(s.U)):
        ns.KnotsU[i] = float(k)
    for i, k in enumerate(nudos_a_rhino(s.V)):
        ns.KnotsV[i] = float(k)
    return ns


def superficie_rhino_valida(s: Superficie) -> bool:
    return bool(_a_rhino(s).IsValid)


def _malla_rhino(X: np.ndarray, tri: np.ndarray) -> r3.Mesh:
    usados, inv = np.unique(tri.ravel(), return_inverse=True)
    m = r3.Mesh()
    for x, y, z in X[usados]:
        m.Vertices.Add(float(x), float(y), float(z))
    for a, b, c in inv.reshape(-1, 3):
        m.Faces.AddFace(int(a), int(b), int(c))
    try:
        m.Normals.ComputeNormals()
    except AttributeError:
        pass  # Rhino recalcula las normales al abrir el archivo
    return m


def malla_estribor(p: Pieza, sup: Superficie) -> tuple:
    """Vértices y triángulos de la pieza sin la parte de babor (y < 0)."""
    X = p.puntos_teselado(sup)
    tri = p.tri[X[p.tri][:, :, 1].mean(axis=1) >= 0.0]
    X = X.copy()
    X[:, 1] = np.maximum(X[:, 1], 0.0)   # vértices que cruzan crujía, a crujía
    return X, tri


def guardar_3dm(casco: Casco, superficies: dict, ruta: str | Path, deshacer_origen: bool = False,
                con_malla: bool = True) -> None:
    """Guarda el casco transformado en metros.

    Capas:
      <capa original>                       malla recortada de cada pieza (igual que el casco)
      <capa original>::NURBS_sin_recortar   superficies NURBS transformadas (ocultas)
      <capa original>::Bordes_recorte       aristas transformadas, para volver a recortar
                                            en Rhino (_Split) si se quiere la brep
    """
    out = r3.File3dm()
    out.Settings.ModelUnitSystem = r3.UnitSystem.Meters
    indices = {}

    def capa(nombre, padre=None, visible=True):
        clave = (nombre, padre)
        if clave in indices:
            return indices[clave]
        L = r3.Layer()
        L.Name = nombre
        L.Visible = visible
        if padre is not None:
            L.ParentLayerId = out.Layers[padre].Id
        indices[clave] = out.Layers.Add(L)
        return indices[clave]

    desplaz = casco.origen if deshacer_origen else np.zeros(3)
    for p in casco.piezas:
        s = superficies[p.id]
        if deshacer_origen:
            s = s.copia(s.P + desplaz)
        base = p.capa or "HULL"
        i_base = capa(base)
        i_nurbs = capa("NURBS_sin_recortar", i_base, visible=False)
        i_bordes = capa("Bordes_recorte", i_base, visible=False)
        nombre = p.nombre or f"pieza_{p.id}"

        a = r3.ObjectAttributes()
        a.Name = f"{nombre} ({p.rol})"
        a.SetUserString("pieza", p.id)
        a.SetUserString("rol", p.rol)
        if con_malla and p.tri is not None and len(p.tri):
            X, tri = malla_estribor(p, s)
            if len(tri):
                a.LayerIndex = i_base
                out.Objects.AddMesh(_malla_rhino(X, tri), a)

        a2 = r3.ObjectAttributes()
        a2.Name = nombre
        a2.LayerIndex = i_nurbs
        a2.SetUserString("pieza", p.id)
        out.Objects.AddSurface(_a_rhino(s), a2)

        for uv in p.bordes:
            X = s.puntos(uv[:, 0], uv[:, 1])
            a3 = r3.ObjectAttributes()
            a3.Name = f"borde {p.id}"
            a3.LayerIndex = i_bordes
            out.Objects.AddCurve(r3.PolylineCurve([r3.Point3d(*map(float, q)) for q in X]), a3)
    if not out.Write(str(ruta), 7):
        raise IOError(f"No se ha podido escribir {ruta}")
