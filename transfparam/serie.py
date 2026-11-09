"""Matriz de variantes a desplazamiento constante.

Cada variante fija Cb, LCB y L/B; con Δ y T constantes, L·B = Δ / (ρ·Cb·T) y
L = sqrt(L·B · L/B), B = L / (L/B). Después se transforma el casco (escalado +
Lackenby) y se MIDEN sus hidrostáticas, que son las que se usan para la resistencia.
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np

from . import RHO_AGUA_MAR, graficas, hidrostatica as hidro, resistencia as res, salidas
from .transformacion import refinar_para_lackenby, transformar


def pasos(rango: float, n: int) -> np.ndarray:
    """n valores equiespaciados en [-rango, +rango] (n = 1 -> [0])."""
    return np.array([0.0]) if n <= 1 else np.linspace(-rango, rango, n)


def matriz(casco, dwl0: float, base: dict, carpeta: Path, delta: float | None = None,
           cb_rango=0.05, cb_n=5, lcb_rango=2.0, lcb_n=5, lb_rango=0.03, lb_n=3,
           con_skeg=True, v_kn=None, v_servicio=9.5, lpp=None, cstern=0.0, h_refino=0.10,
           n_secciones=200, n_z=200, guardar_3dm=True, con_malla=True,
           version_holtrop="1984") -> list:
    """Genera y mide todas las combinaciones. base: {'L','B','T','cb','lcb_pct'}.

    cb_rango y lb_rango son fracciones (0.05 = ±5 %); lcb_rango en puntos de % Lwl.
    """
    carpeta = Path(carpeta)
    (carpeta / "variantes").mkdir(parents=True, exist_ok=True)
    v_kn = np.union1d(np.arange(6.0, 11.0001, 0.5), [v_servicio]) if v_kn is None else v_kn

    H0 = hidro.calcular(casco, dwl0, None, con_skeg, n_secciones, n_z)
    L0, B0, T0 = base["L"] or H0.lwl, base["B"] or H0.bwl, base["T"] or H0.calado
    cb0 = base["cb"] or H0.cb
    lcb0 = H0.lcb_pct if base["lcb_pct"] is None else base["lcb_pct"]
    delta = delta or RHO_AGUA_MAR * cb0 * L0 * B0 * T0
    print(f"Matriz: Δ = {delta:.2f} t constante, T = {T0:.3f} m constante; "
          f"base Cb = {cb0:.4f}, LCB = {lcb0:+.3f} % Lwl, L/B = {L0 / B0:.4f}")

    t = time.time()
    ref = refinar_para_lackenby(casco.superficies(), H0, L0, h_refino)
    print(f"  superficies refinadas para Lackenby ({time.time() - t:.0f} s)")

    combinaciones = list(itertools.product(pasos(cb_rango, cb_n), pasos(lcb_rango, lcb_n),
                                           pasos(lb_rango, lb_n)))
    filas, curvas_todas, resultados = [], [], {}
    for k, (fcb, dlcb, flb) in enumerate(combinaciones):
        nombre = f"V{k:03d}"
        cb = cb0 * (1 + fcb)
        lb = L0 / B0 * (1 + flb)
        LB = delta / (RHO_AGUA_MAR * cb * T0)
        L = np.sqrt(LB * lb)
        B = L / lb
        lcb = lcb0 + dlcb
        fila = {"Variante": nombre, "dCb [%]": round(100 * fcb, 3), "dLCB [% Lwl]": round(dlcb, 3),
                "dL/B [%]": round(100 * flb, 3), "Cb_obj": cb, "LCB_obj": lcb, "LB_obj": lb,
                "L_obj [m]": L, "B_obj [m]": B, "T_obj [m]": T0}
        t = time.time()
        try:
            R = transformar(casco, dwl0, L=L, B=B, T=T0, cb=cb, lcb_pct=lcb, con_skeg=con_skeg,
                            h_refino=h_refino, n_secciones=n_secciones, n_z=n_z, H0=H0, sup_refinadas=ref)
        except (ValueError, RuntimeError) as e:
            fila["Error"] = str(e)
            filas.append(fila)
            print(f"  {nombre}: ERROR {e}")
            continue
        H = R.H_final
        cur = res.curvas(H, v_kn, con_skeg=con_skeg, lpp=lpp, cstern=cstern,
                         version_holtrop=version_holtrop)
        for clave, valor in H.resumen().items():
            fila[hidro.ETIQUETAS[clave]] = valor
        fila["Cb"] = H.cb
        fila["LCB_pct"] = H.lcb_pct
        fila["L/B medido"] = H.lwl / H.bwl
        fila["c_popa"] = R.ley.c_popa if R.ley else 0.0
        fila["c_proa"] = R.ley.c_proa if R.ley else 0.0
        fila["Iteraciones Lackenby"] = len(R.iteraciones)
        fila["Junta máx. [mm]"] = max(R.juntas_mm.values()) if R.juntas_mm else np.nan
        for metodo, corto in (("Holtrop-Mennen", "HM"), ("van Oortmerssen", "VO")):
            fila[f"RT_{corto} [kN] a {v_servicio:g} kn"] = salidas.valor_a(cur[metodo], v_servicio) / 1000
            fila[f"EHP_{corto} [kW] a {v_servicio:g} kn"] = salidas.valor_a(cur[metodo], v_servicio, "EHP") / 1000
        fila["Avisos de validez"] = " | ".join(cur["avisos"])
        filas.append(fila)
        curvas_todas += salidas.filas_resistencia(nombre, cur)
        resultados[nombre] = (R, cur)
        if guardar_3dm:
            salidas.guardar_variante(casco, R.superficies, carpeta / "variantes" / f"{nombre}.3dm", con_malla)
        print(f"  {nombre}: Cb {H.cb:.4f} (obj {cb:.4f})  LCB {H.lcb_pct:+.3f} (obj {lcb:+.3f})  "
              f"L {H.lwl:.3f}  B {H.bwl:.3f}  Δ {H.desplazamiento:.2f} t  "
              f"RT_HM {fila[f'RT_HM [kN] a {v_servicio:g} kn']:.2f} kN  ({time.time() - t:.0f} s)")

    salidas.escribir_tabla(filas, carpeta / "matriz_resumen.csv", "matriz")
    salidas.escribir_tabla(curvas_todas, carpeta / "matriz_curvas_resistencia.csv", "curvas")

    # Figuras
    validas = [f for f in filas if "Error" not in f]
    if not validas:
        return filas
    for corto, metodo in (("HM", "Holtrop-Mennen"), ("VO", "van Oortmerssen")):
        clave = f"RT_{corto} [kN] a {v_servicio:g} kn"
        graficas.mapa_resistencia(validas, clave, carpeta / f"mapa_RT_{corto}.png", v_servicio,
                                  f"RT {metodo} [kN]")
    clave = f"RT_HM [kN] a {v_servicio:g} kn"
    mejor = min(validas, key=lambda f: f[clave])["Variante"]
    centro = min(validas, key=lambda f: abs(f["dCb [%]"]) + abs(f["dLCB [% Lwl]"]) + abs(f["dL/B [%]"]))["Variante"]
    Rb, cb_ = resultados[centro]
    Rm, cm_ = resultados[mejor]
    graficas.curvas_rt({f"Base ({centro})": {m: cb_[m] for m in ("Holtrop-Mennen", "van Oortmerssen")},
                        f"Mejor ({mejor})": {m: cm_[m] for m in ("Holtrop-Mennen", "van Oortmerssen")}},
                       carpeta / "rt_v_base_vs_mejor.png", v_servicio)
    salidas.figuras_comparacion(casco, Rb.superficies, Rb.H_final, Rm.superficies, Rm.H_final,
                                f"Base ({centro})", f"Mejor ({mejor})", carpeta, "base_vs_mejor_")
    print(f"\nMejor variante (mínima RT Holtrop-Mennen a {v_servicio:g} kn): {mejor}; base: {centro}")
    return filas
