"""Superficies B-spline/NURBS evaluadas con numpy.

Las funciones base se calculan con la recurrencia de Cox-de Boor (Piegl y Tiller,
"The NURBS Book", 2.ª ed., 1997: ec. 2.5 para N_i,p y ec. 2.7 para su derivada).
La inserción de nudos sigue el algoritmo de Boehm (Piegl y Tiller, ec. 5.15).

Convenios:
    U, V      vectores de nudos COMPLETOS (n + p + 1 valores). Rhino guarda dos nudos
              menos (sin el primero y el último repetidos); ver `nudos_desde_rhino`.
    P         puntos de control (nu, nv, 3) en coordenadas euclídeas.
    W         pesos (nu, nv) o None si la superficie no es racional.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Nudos y funciones base
# ---------------------------------------------------------------------------

def nudos_desde_rhino(knots) -> np.ndarray:
    """Rhino omite el primer y el último nudo: se añaden repitiendo los extremos."""
    k = np.asarray(list(knots), dtype=float)
    return np.concatenate([[k[0]], k, [k[-1]]])


def nudos_a_rhino(U: np.ndarray) -> np.ndarray:
    return np.asarray(U[1:-1], dtype=float)


def funciones_base(U: np.ndarray, p: int, u, derivada: bool = False):
    """Matriz de funciones base N[k, i] = N_i,p(u_k) (y su derivada si se pide).

    Recurrencia de Cox-de Boor sobre todos los parámetros a la vez (vectorizada).
    El extremo final u = U[-1] se asigna al último tramo no degenerado.
    """
    U = np.asarray(U, dtype=float)
    u = np.clip(np.atleast_1d(np.asarray(u, dtype=float)), U[p], U[-p - 1])
    m = len(U) - 1
    n = m - p  # número de funciones base

    # Grado 0: indicador del tramo [U_i, U_i+1)
    N = ((U[:-1][None, :] <= u[:, None]) & (u[:, None] < U[1:][None, :])).astype(float)
    ultimo = np.nonzero(U[:-1] < U[1:])[0].max()
    fin = u >= U[-p - 1]
    N[fin, :] = 0.0
    N[fin, ultimo] = 1.0

    N_prev = N
    for k in range(1, p + 1):
        i = np.arange(m - k)
        d1 = U[i + k] - U[i]
        d2 = U[i + k + 1] - U[i + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(d1 > 0, (u[:, None] - U[i][None, :]) / np.where(d1 > 0, d1, 1.0), 0.0)
            b = np.where(d2 > 0, (U[i + k + 1][None, :] - u[:, None]) / np.where(d2 > 0, d2, 1.0), 0.0)
        N_prev = N
        N = a * N[:, :-1] + b * N[:, 1:]

    if not derivada:
        return N
    if p == 0:
        return N, np.zeros_like(N)
    # Derivada (Piegl y Tiller, ec. 2.7): N'_i,p = p/(U_i+p - U_i) N_i,p-1 - p/(U_i+p+1 - U_i+1) N_i+1,p-1
    i = np.arange(n)
    d1 = U[i + p] - U[i]
    d2 = U[i + p + 1] - U[i + 1]
    c1 = np.where(d1 > 0, p / np.where(d1 > 0, d1, 1.0), 0.0)
    c2 = np.where(d2 > 0, p / np.where(d2 > 0, d2, 1.0), 0.0)
    dN = c1[None, :] * N_prev[:, :-1] - c2[None, :] * N_prev[:, 1:]
    return N, dN


# ---------------------------------------------------------------------------
# Superficie
# ---------------------------------------------------------------------------

@dataclass
class Superficie:
    pu: int
    pv: int
    U: np.ndarray
    V: np.ndarray
    P: np.ndarray                  # (nu, nv, 3)
    W: np.ndarray | None = None    # (nu, nv) o None
    _arbol: tuple | None = field(default=None, repr=False, compare=False)

    # --- utilidades -------------------------------------------------------
    @property
    def dominio_u(self):
        return self.U[self.pu], self.U[-self.pu - 1]

    @property
    def dominio_v(self):
        return self.V[self.pv], self.V[-self.pv - 1]

    @property
    def racional(self) -> bool:
        return self.W is not None and not np.allclose(self.W, 1.0)

    def copia(self, P: np.ndarray | None = None) -> "Superficie":
        return Superficie(self.pu, self.pv, self.U.copy(), self.V.copy(),
                          (self.P if P is None else P).copy(),
                          None if self.W is None else self.W.copy())

    def _homogeneos(self):
        if self.W is None:
            return self.P, None
        return self.P * self.W[..., None], self.W

    # --- evaluación -------------------------------------------------------
    def rejilla(self, us, vs) -> np.ndarray:
        """Puntos S(u_i, v_j) en rejilla: (len(us), len(vs), 3)."""
        Nu = funciones_base(self.U, self.pu, us)
        Nv = funciones_base(self.V, self.pv, vs)
        Pw, W = self._homogeneos()
        # Dos productos matriciales (BLAS): (Nu · P) · Nv^T
        A = np.tensordot(Nu, Pw, axes=(1, 0))            # (nus, nv, 3)
        S = np.tensordot(A, Nv, axes=(1, 1)).transpose(0, 2, 1)   # (nus, nvs, 3)
        if W is not None:
            S = S / (Nu @ W @ Nv.T)[..., None]
        return S

    def puntos(self, u, v, derivadas: bool = False):
        """S(u_k, v_k) para pares sueltos; con derivadas devuelve también Su y Sv."""
        u = np.atleast_1d(u)
        v = np.atleast_1d(v)
        Pw, W = self._homogeneos()
        nu_, nv_ = Pw.shape[:2]
        P2 = Pw.reshape(nu_, nv_ * 3)

        def ev(Na, Nb):   # Σ_ab Na[k,a] P[a,b] Nb[k,b] con un producto matricial (BLAS)
            return np.einsum("kbd,kb->kd", (Na @ P2).reshape(-1, nv_, 3), Nb)

        def ew(Na, Nb):
            return np.einsum("kb,kb->k", Na @ W, Nb)[:, None]

        if not derivadas:
            Nu = funciones_base(self.U, self.pu, u)
            Nv = funciones_base(self.V, self.pv, v)
            A = ev(Nu, Nv)
            if W is None:
                return A
            return A / ew(Nu, Nv)
        Nu, dNu = funciones_base(self.U, self.pu, u, derivada=True)
        Nv, dNv = funciones_base(self.V, self.pv, v, derivada=True)
        A = ev(Nu, Nv)
        Au = ev(dNu, Nv)
        Av = ev(Nu, dNv)
        if W is None:
            return A, Au, Av
        w = ew(Nu, Nv)
        wu = ew(dNu, Nv)
        wv = ew(Nu, dNv)
        S = A / w
        return S, (Au - wu * S) / w, (Av - wv * S) / w

    # --- inversión de puntos ---------------------------------------------
    def invertir(self, X: np.ndarray, iter_max: int = 40, n_ini: int = 60):
        """Parámetros (u, v) del punto de la superficie más cercano a cada X (Newton).

        Arranque desde el punto más cercano de una rejilla n_ini x n_ini; luego
        Gauss-Newton sobre la distancia con los parámetros limitados al dominio.
        Devuelve (uv, distancia).
        """
        X = np.atleast_2d(X)
        u0, u1 = self.dominio_u
        v0, v1 = self.dominio_v
        if self._arbol is None:
            us = np.linspace(u0, u1, n_ini)
            vs = np.linspace(v0, v1, n_ini)
            G = self.rejilla(us, vs).reshape(-1, 3)
            uu, vv = np.meshgrid(us, vs, indexing="ij")
            self._arbol = (cKDTree(G), np.column_stack([uu.ravel(), vv.ravel()]))
        arbol, uv_rejilla = self._arbol
        _, idx = arbol.query(X)
        uv = uv_rejilla[idx].copy()
        escala = np.array([u1 - u0, v1 - v0])
        for _ in range(iter_max):
            S, Su, Sv = self.puntos(uv[:, 0], uv[:, 1], derivadas=True)
            r = S - X
            a11 = np.einsum("kd,kd->k", Su, Su)
            a12 = np.einsum("kd,kd->k", Su, Sv)
            a22 = np.einsum("kd,kd->k", Sv, Sv)
            b1 = -np.einsum("kd,kd->k", Su, r)
            b2 = -np.einsum("kd,kd->k", Sv, r)
            det = a11 * a22 - a12 ** 2
            det = np.where(np.abs(det) < 1e-30, 1e-30, det)
            du = (a22 * b1 - a12 * b2) / det
            dv = (a11 * b2 - a12 * b1) / det
            paso = np.column_stack([du, dv])
            # Limita el paso a un cuarto del dominio para evitar saltos
            paso = np.clip(paso, -0.25 * escala, 0.25 * escala)
            uv = uv + paso
            uv[:, 0] = np.clip(uv[:, 0], u0, u1)
            uv[:, 1] = np.clip(uv[:, 1], v0, v1)
            if np.all(np.abs(paso) <= 1e-10 * escala):
                break
        S = self.puntos(uv[:, 0], uv[:, 1])
        return uv, np.linalg.norm(S - X, axis=1)

    # --- refinado ---------------------------------------------------------
    def insertar_nudo(self, t: float, direccion: str) -> "Superficie":
        """Inserta una vez el nudo t en la dirección 'u' o 'v' (algoritmo de Boehm).

        La geometría no cambia: solo se añade una fila/columna de puntos de control.
        """
        Pw, W = self._homogeneos()
        Q = Pw if W is None else np.concatenate([Pw, W[..., None]], axis=-1)
        if direccion == "v":
            Q = np.swapaxes(Q, 0, 1)
            U, p = self.V, self.pv
        else:
            U, p = self.U, self.pu
        k = int(np.searchsorted(U, t, side="right") - 1)
        n = Q.shape[0]
        Qn = np.empty((n + 1,) + Q.shape[1:])
        Qn[: k - p + 1] = Q[: k - p + 1]
        for i in range(k - p + 1, k + 1):
            a = (t - U[i]) / (U[i + p] - U[i])
            Qn[i] = a * Q[i] + (1 - a) * Q[i - 1]
        Qn[k + 1:] = Q[k:]
        Un = np.insert(U, k + 1, t)
        if direccion == "v":
            Qn = np.swapaxes(Qn, 0, 1)
        if W is None:
            P, Wn = Qn, None
        else:
            Wn = Qn[..., 3]
            P = Qn[..., :3] / Wn[..., None]
        if direccion == "v":
            return Superficie(self.pu, self.pv, self.U.copy(), Un, P, Wn)
        return Superficie(self.pu, self.pv, Un, self.V.copy(), P, Wn)

    def refinar_en_x(self, h: float, n_muestras: int = 12, estaciones=(), h_fino: float | None = None,
                     ancho: float | None = None) -> "Superficie":
        """Inserta nudos hasta que ningún tramo (en u o en v) abarque más de h en x, ni
        más de h_fino en x si el tramo está a menos de 'ancho' de alguna estación.

        La transformación de Lackenby solo depende de x; si cada tramo de la superficie
        es corto en x, mover los puntos de control equivale (con error O(h²)) a mover
        los puntos de la superficie. En las estaciones donde la ley cambia de pendiente
        (perpendiculares y maestra) el error es O(h·salto de pendiente), por eso ahí se
        refina mucho más.
        """
        estaciones = np.asarray(estaciones, float)
        h_fino = h if h_fino is None else h_fino
        ancho = 2 * h if ancho is None else ancho
        s = self
        for direccion in ("u", "v"):
            for _ in range(20):  # cada pasada divide por la mitad los tramos largos
                U, p = (s.U, s.pu) if direccion == "u" else (s.V, s.pv)
                nudos = np.unique(U[p: len(U) - p])
                otros = np.linspace(*(s.dominio_v if direccion == "u" else s.dominio_u), n_muestras)
                nuevos = []
                for a, b in zip(nudos[:-1], nudos[1:]):
                    ts = np.linspace(a, b, 5)
                    X = s.rejilla(ts, otros) if direccion == "u" else s.rejilla(otros, ts).swapaxes(0, 1)
                    extension_x = np.max(X[..., 0].max(axis=0) - X[..., 0].min(axis=0))
                    limite = h
                    # En direcciones de grado 1 la superficie es lineal entre filas de
                    # puntos de control, que se transforman exactamente: no hace falta
                    # el refinado fino
                    if len(estaciones) and p >= 2:
                        x0, x1 = X[..., 0].min(), X[..., 0].max()
                        if np.any((estaciones >= x0 - ancho) & (estaciones <= x1 + ancho)):
                            limite = h_fino
                    if extension_x > limite:
                        nuevos.append(0.5 * (a + b))
                if not nuevos:
                    break
                for t in nuevos:
                    s = s.insertar_nudo(t, direccion)
        return s
