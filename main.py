"""Transformación paramétrica de cascos (Lackenby, 1950) sin Maxsurf ni Rhino abierto.

Ejemplos
--------
Hidrostáticas del casco de partida (y comprobación de convergencia):
    python main.py casco.3dm --dwl 2.36 --solo-hidrostaticas --convergencia

Llevar el casco al diseño (Lwl, B, T, Cb; LCB del casco de partida):
    python main.py casco.3dm --dwl 2.36 --L 20.45 --B 6.16 --T 2.36 --cb 0.545

Igual, con desplazamiento fijo y la manga libre (como Maxsurf: 3 de {Δ, L, B, T}):
    python main.py casco.3dm --dwl 2.36 --L 20.45 --T 2.36 --cb 0.545 --delta 166 --libre B

Matriz de variantes a Δ constante (Cb ±5 % en 5, LCB ±2 % Lwl en 5, L/B ±3 % en 3):
    python main.py casco.3dm --dwl 2.36 --L 20.45 --B 6.16 --T 2.36 --cb 0.545 --matriz

Ejes del casco cargado: x a proa con el espejo en x = 0, y a estribor, z arriba con el
punto más bajo del forro en z = 0 (la DWL se da en estos ejes).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from transfparam import hidrostatica as hidro
from transfparam import resistencia as res
from transfparam import salidas, serie
from transfparam.modelo import cargar_casco
from transfparam.transformacion import transformar

RAIZ = Path(__file__).resolve().parent


def argumentos(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("entrada", help="archivo .3dm del casco")
    p.add_argument("--config", default=str(RAIZ / "piezas_trawler.json"),
                   help="JSON con roles (SKEG...) y nombres de las caras")
    p.add_argument("--salida", default=str(RAIZ / "resultados"), help="carpeta de salida")
    p.add_argument("--dwl", type=float, default=2.36, help="flotación z = DWL del casco de partida [m]")
    g = p.add_argument_group("objetivos (si no se dan, los del casco de partida)")
    g.add_argument("--L", type=float, help="eslora en la flotación Lwl [m]")
    g.add_argument("--B", type=float, help="manga en la flotación Bwl [m]")
    g.add_argument("--T", type=float, help="calado [m]")
    g.add_argument("--cb", type=float, help="coeficiente de bloque")
    g.add_argument("--lcb", type=float, help="LCB en %% Lwl desde la mitad de Lwl (+ a proa)")
    g.add_argument("--delta", type=float, help="desplazamiento fijo [t]")
    g.add_argument("--libre", choices=["L", "B", "T", "Cb"],
                   help="variable que se despeja cuando se fija --delta")
    p.add_argument("--sin-skeg", action="store_true", help="excluye el skeg de las hidrostáticas "
                   "(se añade como apéndice en la resistencia)")
    m = p.add_argument_group("matriz de variantes")
    m.add_argument("--matriz", action="store_true", help="genera la matriz de variantes a Δ constante")
    m.add_argument("--cb-rango", type=float, default=5.0, help="± %% de Cb (defecto 5)")
    m.add_argument("--cb-pasos", type=int, default=5)
    m.add_argument("--lcb-rango", type=float, default=2.0, help="± puntos de %% Lwl de LCB (defecto 2)")
    m.add_argument("--lcb-pasos", type=int, default=5)
    m.add_argument("--lb-rango", type=float, default=3.0, help="± %% de L/B (defecto 3)")
    m.add_argument("--lb-pasos", type=int, default=3)
    m.add_argument("--sin-3dm", action="store_true", help="no guarda el .3dm de cada variante")
    r = p.add_argument_group("resistencia")
    r.add_argument("--v-min", type=float, default=6.0)
    r.add_argument("--v-max", type=float, default=11.0)
    r.add_argument("--v-paso", type=float, default=0.5)
    r.add_argument("--v-servicio", type=float, default=9.5)
    r.add_argument("--lpp", type=float, help="Lpp para la eslora de van Oortmerssen L_D = (Lpp+Lwl)/2")
    r.add_argument("--cstern", type=float, default=0.0, help="parámetro de popa de Holtrop (defecto 0)")
    r.add_argument("--holtrop", choices=["1982", "1984"], default="1984",
                   help="versión de Holtrop-Mennen (defecto 1984, reanálisis de Holtrop)")
    n = p.add_argument_group("precisión")
    n.add_argument("--secciones", type=int, default=200, help="secciones transversales (>= 200)")
    n.add_argument("--h-teselado", type=float, default=0.05, help="tamaño de celda del teselado [m]")
    n.add_argument("--h-refino", type=float, default=0.10, help="tramo máximo en x de las NURBS [m]")
    n.add_argument("--sin-malla", action="store_true", help="no guarda la malla recortada en el .3dm")
    n.add_argument("--convergencia", action="store_true", help="comprueba la convergencia doblando la resolución")
    p.add_argument("--solo-hidrostaticas", action="store_true", help="solo hidrostáticas del casco de partida")
    return p.parse_args(argv)


def main(argv=None):
    a = argumentos(argv)
    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    con_skeg = not a.sin_skeg
    v_kn = np.union1d(np.arange(a.v_min, a.v_max + 1e-9, a.v_paso), [a.v_servicio])

    t = time.time()
    print(f"Leyendo {a.entrada} ...")
    casco = cargar_casco(a.entrada, a.config, RAIZ / ".cache", a.h_teselado)
    print(f"  {len(casco.piezas)} caras de estribor, origen desplazado {np.round(casco.origen, 4)} m "
          f"({time.time() - t:.0f} s)")

    H0 = hidro.calcular(casco, a.dwl, None, con_skeg, a.secciones, a.secciones)
    if a.convergencia:
        cv = hidro.convergencia(casco, a.dwl, a.h_teselado, a.secciones, a.secciones, con_skeg=con_skeg)
        print("\nConvergencia (diferencia al doblar teselado, secciones y cotas):")
        for k, v in cv["diferencias_pct"].items():
            print(f"  {k:12s} {v:8.4f} %  {'OK' if v < 0.2 else 'NO CONVERGE'}")
        salidas.escribir_tabla([{"Magnitud": k, "Diferencia [%]": v} for k, v in cv["diferencias_pct"].items()],
                               salida / "convergencia.csv", "convergencia")

    if a.solo_hidrostaticas:
        filas = salidas.tabla_hidrostaticas({"Original": H0})
        print()
        salidas.imprimir_tabla(filas)
        salidas.escribir_tabla(filas, salida / "hidrostaticas_original.csv", "hidrostaticas")
        return 0

    if a.matriz:
        base = {"L": a.L, "B": a.B, "T": a.T, "cb": a.cb, "lcb_pct": a.lcb}
        serie.matriz(casco, a.dwl, base, salida, a.delta, a.cb_rango / 100, a.cb_pasos, a.lcb_rango,
                     a.lcb_pasos, a.lb_rango / 100, a.lb_pasos, con_skeg, v_kn, a.v_servicio, a.lpp,
                     a.cstern, a.h_refino, a.secciones, a.secciones, not a.sin_3dm, not a.sin_malla,
                     a.holtrop)
        return 0

    # Transformación única
    t = time.time()
    R = transformar(casco, a.dwl, L=a.L, B=a.B, T=a.T, cb=a.cb, lcb_pct=a.lcb, delta=a.delta,
                    libre=a.libre, con_skeg=con_skeg, h_refino=a.h_refino, n_secciones=a.secciones,
                    n_z=a.secciones, H0=H0, verbose=True)
    print(f"  transformación en {time.time() - t:.0f} s; escalas sx={R.escalas[0]:.5f} "
          f"sy={R.escalas[1]:.5f} sz={R.escalas[2]:.5f}; DWL nueva {R.dwl:.4f} m")
    if R.ley is not None:
        print(f"  Lackenby: c_popa = {R.ley.c_popa:+.5f}, c_proa = {R.ley.c_proa:+.5f} "
              f"(dCp_popa = {R.lackenby_lineal['dCp_popa']:+.5f}, dCp_proa = {R.lackenby_lineal['dCp_proa']:+.5f}"
              " en primer orden)")
    if R.juntas_mm:
        peor = max(R.juntas_mm.items(), key=lambda kv: kv[1])
        print(f"  separación máxima entre piezas: {peor[1]:.3f} mm (juntas {peor[0][0]}-{peor[0][1]})")

    nombre = Path(a.entrada).stem + "_transformado"
    salidas.guardar_variante(casco, R.superficies, salida / f"{nombre}.3dm", not a.sin_malla)
    filas = salidas.tabla_hidrostaticas({"Original": R.H_original, "Escalado": R.H_escalado,
                                         "Transformado": R.H_final}, R.objetivos)
    print()
    salidas.imprimir_tabla(filas)
    salidas.escribir_tabla(filas, salida / "hidrostaticas.csv", "hidrostaticas")

    cur_esc = res.curvas(R.H_escalado, v_kn, con_skeg, a.lpp, a.cstern, a.holtrop)
    cur_fin = res.curvas(R.H_final, v_kn, con_skeg, a.lpp, a.cstern, a.holtrop)
    salidas.escribir_tabla(salidas.filas_resistencia("Escalado", cur_esc)
                           + salidas.filas_resistencia("Transformado", cur_fin),
                           salida / "resistencia.csv", "resistencia")
    for aviso in cur_fin["avisos"]:
        print("  AVISO", aviso)
    for metodo in ("Holtrop-Mennen", "van Oortmerssen"):
        print(f"  {metodo}: RT a {a.v_servicio:g} kn = {salidas.valor_a(cur_fin[metodo], a.v_servicio) / 1000:.2f} kN, "
              f"EHP = {salidas.valor_a(cur_fin[metodo], a.v_servicio, 'EHP') / 1000:.1f} kW")

    salidas.figuras_comparacion(casco, casco.superficies(), R.H_original, R.superficies, R.H_final,
                                "Original", "Transformado", salida)
    from transfparam import graficas
    graficas.curvas_rt({"Escalado sin Lackenby": {m: cur_esc[m] for m in ("Holtrop-Mennen", "van Oortmerssen")},
                        "Transformado": {m: cur_fin[m] for m in ("Holtrop-Mennen", "van Oortmerssen")}},
                       salida / "rt_v.png", a.v_servicio)
    print(f"\nResultados en {salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
