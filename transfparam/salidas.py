"""Tablas (CSV y Excel), resumen en consola e informe de una transformación."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from . import graficas, hidrostatica as hidro, resistencia as res
from .modelo import ROL_ESPEJO, guardar_3dm


def escribir_tabla(filas: list, ruta_csv: Path, hoja: str = "datos") -> None:
    """Lista de dicts -> CSV (separador ',', punto decimal) y la misma tabla en .xlsx."""
    if not filas:
        return
    columnas = list(dict.fromkeys(k for f in filas for k in f))
    with open(ruta_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=columnas)
        w.writeheader()
        for fila in filas:
            w.writerow({k: _limpio(v) for k, v in fila.items()})
    try:
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = hoja[:31]
        ws.append(columnas)
        for fila in filas:
            ws.append([_limpio(fila.get(k)) for k in columnas])
        for i, c in enumerate(columnas, 1):
            ws.column_dimensions[openpyxl.utils.get_column_letter(i)].width = max(10, min(45, len(str(c)) + 2))
        wb.save(Path(ruta_csv).with_suffix(".xlsx"))
    except ImportError:
        pass


def _limpio(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def tabla_hidrostaticas(columnas: dict, objetivos: dict | None = None) -> list:
    """Filas magnitud x casco: columnas = {'Original': H, 'Escalado': H, ...}."""
    filas = []
    resumenes = {k: H.resumen() for k, H in columnas.items()}
    mapa_obj = {"lwl": "L", "bwl": "B", "calado": "T", "cb": "Cb", "lcb_pct": "lcb_pct",
                "desplazamiento": "delta"}
    for clave, etiqueta in hidro.ETIQUETAS.items():
        fila = {"Magnitud": etiqueta}
        for nombre, r in resumenes.items():
            fila[nombre] = round(float(r[clave]), 6)
        if objetivos is not None:
            o = mapa_obj.get(clave)
            fila["Objetivo"] = round(float(objetivos[o]), 6) if o and objetivos.get(o) is not None else ""
        filas.append(fila)
    return filas


def imprimir_tabla(filas: list, columnas: list | None = None, ancho: int = 12) -> None:
    if not filas:
        return
    columnas = columnas or list(filas[0])
    w0 = max(len(str(f[columnas[0]])) for f in filas) + 2
    print(f"{columnas[0]:<{w0}}" + "".join(f"{c:>{ancho}}" for c in columnas[1:]))
    for f in filas:
        celdas = []
        for c in columnas[1:]:
            v = f.get(c, "")
            celdas.append(f"{v:>{ancho}.4f}" if isinstance(v, (float, int)) else f"{str(v):>{ancho}}")
        print(f"{str(f[columnas[0]]):<{w0}}" + "".join(celdas))


def filas_resistencia(etiqueta: str, curvas: dict) -> list:
    filas = []
    for metodo in ("Holtrop-Mennen", "van Oortmerssen"):
        r = curvas[metodo]
        for i, v in enumerate(r["V_kn"]):
            fila = {"Casco": etiqueta, "Método": metodo, "V [kn]": round(float(v), 3),
                    "Fn": round(float(r["Fn"][i]), 5)}
            for k in ("RF", "RAPP", "RW", "RB", "RTR", "RA", "RR", "RT"):
                if k in r:
                    fila[f"{k} [kN]"] = round(float(r[k][i]) / 1000, 4)
            if "1+k1" in r:
                fila["1+k1"] = round(float(r["1+k1"][i]), 5)
            fila["EHP [kW]"] = round(float(r["EHP"][i]) / 1000, 3)
            filas.append(fila)
    return filas


def valor_a(curva: dict, v_kn: float, clave: str = "RT") -> float:
    return float(np.interp(v_kn, curva["V_kn"], curva[clave]))


def estaciones(H, n: int = 11) -> np.ndarray:
    """Abscisas de n secciones equiespaciadas en la flotación (sin los extremos exactos)."""
    xs = H.x_popa_wl + np.linspace(0, 1, n) * H.lwl
    xs[0] += 0.002 * H.lwl
    xs[-1] -= 0.002 * H.lwl
    return xs


def figuras_comparacion(casco, sup_a, H_a, sup_b, H_b, et_a, et_b, carpeta: Path, prefijo: str = ""):
    graficas.sac(H_a, H_b, et_a, et_b, carpeta / f"{prefijo}sac.png")
    xa, xb = estaciones(H_a), estaciones(H_b)
    sa = hidro.secciones(casco, xa, sup_a, roles_excluidos=(ROL_ESPEJO,))
    sb = hidro.secciones(casco, xb, sup_b, roles_excluidos=(ROL_ESPEJO,))
    graficas.caja_cuadernas(sa, sb, xa, xb, H_a.x_am, H_b.x_am, H_a.dwl, H_b.dwl, et_a, et_b,
                            carpeta / f"{prefijo}caja_cuadernas.png")


def guardar_variante(casco, superficies, ruta: Path, con_malla: bool = True) -> None:
    guardar_3dm(casco, superficies, ruta, con_malla=con_malla)
