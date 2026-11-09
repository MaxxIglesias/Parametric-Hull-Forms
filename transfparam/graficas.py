"""Figuras para la memoria (PNG, matplotlib).

Colores (paleta de referencia validada para daltonismo): azul = casco de partida o
base, naranja = casco transformado o mejor variante, gris = referencias. Los métodos
de resistencia se distinguen además por el tipo de línea (continua Holtrop-Mennen,
discontinua van Oortmerssen), para que el color nunca sea la única pista.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

AZUL = "#2a78d6"
NARANJA = "#eb6834"
GRIS = "#8a8984"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
REJILLA = "#e4e3df"
SECUENCIAL = LinearSegmentedColormap.from_list(
    "azul", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
ESTILO = {"Holtrop-Mennen": "-", "van Oortmerssen": "--"}


def _ejes(ax):
    ax.grid(True, color=REJILLA, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRIS)
    ax.tick_params(colors=TINTA_2, labelsize=9)
    ax.xaxis.label.set_color(TINTA_2)
    ax.yaxis.label.set_color(TINTA_2)
    ax.title.set_color(TINTA)


def sac(H_a, H_b, etiqueta_a: str, etiqueta_b: str, ruta, titulo: str = "Curva de áreas seccionales"):
    """SAC de dos cascos en ejes dimensionales (m, m²) y adimensionales."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for H, c, lab in ((H_a, AZUL, etiqueta_a), (H_b, NARANJA, etiqueta_b)):
        axs[0].plot(H.x, H.sac, color=c, lw=2, label=lab)
        xi = (H.x - H.x_popa_wl) / H.lwl
        axs[1].plot(xi, H.sac / H.am, color=c, lw=2, label=lab)
        axs[0].axvline(H.lcb, color=c, lw=1, ls=":")
    axs[0].set_xlabel("x desde el espejo [m]")
    axs[0].set_ylabel("Área seccional A(x) [m²]")
    axs[0].set_title("SAC (líneas de puntos: LCB)", fontsize=10)
    axs[1].set_xlabel("x / Lwl desde el extremo de popa de la flotación")
    axs[1].set_ylabel("A / Am")
    axs[1].set_title("SAC adimensional", fontsize=10)
    for ax in axs:
        _ejes(ax)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(titulo, color=TINTA)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160)
    plt.close(fig)


def caja_cuadernas(secs_a, secs_b, xs_a, xs_b, x_maestra_a, x_maestra_b, dwl_a, dwl_b,
                   etiqueta_a: str, etiqueta_b: str, ruta):
    """Caja de cuadernas superpuesta: popa a la izquierda (y < 0), proa a la derecha.

    secs_*: lista de segmentos (m, 2, 2) en (y, z) por sección; xs_*: abscisas.
    """
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for secs, xs, xm, c, lw, lab in ((secs_a, xs_a, x_maestra_a, GRIS, 1.2, etiqueta_a),
                                     (secs_b, xs_b, x_maestra_b, NARANJA, 1.2, etiqueta_b)):
        lineas = []
        for s, x in zip(secs, xs):
            s = s.copy()
            if x < xm:
                s[:, :, 0] *= -1           # popa a babor del dibujo
            lineas.append(s)
        lc = LineCollection(np.concatenate(lineas) if lineas else [], colors=c, linewidths=lw, label=lab)
        ax.add_collection(lc)
    for dwl, c in ((dwl_a, GRIS), (dwl_b, NARANJA)):
        ax.axhline(dwl, color=c, lw=0.8, ls="--")
    ax.axvline(0, color=TINTA_2, lw=0.6)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xlabel("y [m]   (izquierda: secciones de popa · derecha: secciones de proa)")
    ax.set_ylabel("z [m]")
    _ejes(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("Caja de cuadernas (11 secciones equiespaciadas en Lwl; trazos: DWL)", fontsize=10)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160)
    plt.close(fig)


def mapa_resistencia(filas: list, clave: str, ruta, v_kn: float, titulo_valor: str):
    """Mapa de clave (p. ej. Rt a v_kn) frente a Cb y LCB: un panel por L/B.

    filas: dicts con 'Cb_obj', 'LCB_obj', 'LB_obj', 'Cb', 'LCB_pct' (medidos) y clave.
    """
    lbs = sorted({round(f["LB_obj"], 6) for f in filas})
    vals = np.array([f[clave] for f in filas], float)
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    fig, axs = plt.subplots(1, len(lbs), figsize=(4.6 * len(lbs) + 1.2, 4.4), squeeze=False)
    axs = axs[0]
    mejor = min(filas, key=lambda f: f[clave])
    im = None
    for ax, lb in zip(axs, lbs):
        sub = [f for f in filas if round(f["LB_obj"], 6) == lb]
        cbs = sorted({f["Cb_obj"] for f in sub})
        lcbs = sorted({f["LCB_obj"] for f in sub})
        Z = np.full((len(lcbs), len(cbs)), np.nan)
        for f in sub:
            Z[lcbs.index(f["LCB_obj"]), cbs.index(f["Cb_obj"])] = f[clave]
        # Rejilla de los valores pedidos (los medidos coinciden con ellos a menos de 1e-3);
        # las variantes que no se han podido generar quedan en blanco
        im = ax.pcolormesh(cbs, lcbs, np.ma.masked_invalid(Z), cmap=SECUENCIAL, vmin=vmin, vmax=vmax,
                           shading="nearest", edgecolors="white", linewidth=2)
        for f in sub:
            claro = (f[clave] - vmin) / max(vmax - vmin, 1e-12) > 0.55
            ax.text(f["Cb_obj"], f["LCB_obj"], f"{f[clave]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if claro else TINTA)
        if mejor in sub:
            ax.plot(mejor["Cb_obj"], mejor["LCB_obj"], marker="s", ms=26, mfc="none", mec=NARANJA, mew=2.5)
        ax.set_title(f"L/B = {lb:.3f}", fontsize=10)
        ax.set_xlabel("Cb")
        ax.set_xticks(cbs)
        ax.set_xticklabels([f"{c:.3f}" for c in cbs])
        ax.set_yticks(lcbs)
        _ejes(ax)
        ax.grid(False)
    axs[0].set_ylabel("LCB [% Lwl desde la mitad de Lwl, + a proa]")
    cb = fig.colorbar(im, ax=list(axs), shrink=0.9)
    cb.set_label(titulo_valor, color=TINTA_2)
    fig.suptitle(f"{titulo_valor} a {v_kn:g} kn, desplazamiento constante "
                 f"(recuadro naranja: mínimo)", color=TINTA, fontsize=11)
    fig.savefig(ruta, dpi=160, bbox_inches="tight")
    plt.close(fig)


def curvas_rt(curvas: dict, ruta, v_servicio: float | None = None, magnitud: str = "RT"):
    """curvas: {etiqueta_casco: {metodo: dict de resistencia.py}}. Color = casco, línea = método."""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    colores = [AZUL, NARANJA, GRIS]
    escala = 1e-3
    for (casco, por_metodo), c in zip(curvas.items(), colores):
        for metodo, r in por_metodo.items():
            y = r[magnitud] * escala
            ax.plot(r["V_kn"], y, color=c, lw=2, ls=ESTILO[metodo], label=f"{casco} · {metodo}")
    if v_servicio is not None:
        ax.axvline(v_servicio, color=TINTA_2, lw=0.8, ls=":")
        ax.text(v_servicio, ax.get_ylim()[1], f" V servicio {v_servicio:g} kn", fontsize=8,
                color=TINTA_2, va="top")
    ax.set_xlabel("Velocidad [kn]")
    ax.set_ylabel("Resistencia total RT [kN]" if magnitud == "RT" else "EHP [kW]")
    _ejes(ax)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Resistencia al avance (continua: Holtrop-Mennen · discontinua: van Oortmerssen)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160)
    plt.close(fig)
