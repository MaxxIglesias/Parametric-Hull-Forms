"""Datos compartidos por los tests (se calculan una vez por sesión).

El casco se toma de la variable de entorno CASCO_3DM o, si no existe, del casco padre
incluido en el repositorio (data/Trawler_hull_maxsurf_model.3dm).
"""

import os
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from transfparam import hidrostatica as hidro  # noqa: E402
from transfparam.modelo import cargar_casco  # noqa: E402
from transfparam.transformacion import refinar_para_lackenby, transformar  # noqa: E402

RUTA_DEFECTO = RAIZ / "data" / "Trawler_hull_maxsurf_model.3dm"
DWL = 2.36
DISENO = dict(L=20.45, B=6.16, T=2.36, cb=0.545)


@pytest.fixture(scope="session")
def casco():
    ruta = Path(os.environ.get("CASCO_3DM", RUTA_DEFECTO))
    if not ruta.is_file():
        pytest.skip(f"No se encuentra el casco {ruta} (define CASCO_3DM)")
    return cargar_casco(ruta, RAIZ / "piezas_trawler.json", RAIZ / ".cache")


@pytest.fixture(scope="session")
def H0(casco):
    return hidro.calcular(casco, DWL)


@pytest.fixture(scope="session")
def refinadas(casco, H0):
    return refinar_para_lackenby(casco.superficies(), H0, DISENO["L"])


@pytest.fixture(scope="session")
def transformado(casco, H0, refinadas):
    """Diseño con Cb +3 % y LCB 1 % Lwl más a proa que el casco de partida."""
    return transformar(casco, DWL, L=DISENO["L"], B=DISENO["B"], T=DISENO["T"],
                       cb=DISENO["cb"] * 1.03, lcb_pct=H0.lcb_pct + 1.0, H0=H0, sup_refinadas=refinadas)
