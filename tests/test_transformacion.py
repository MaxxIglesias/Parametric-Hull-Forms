"""Verificación de la transformación sobre el casco real (enunciado, punto 5)."""

import numpy as np
import pytest

from conftest import DISENO, DWL
from transfparam import hidrostatica as hidro
from transfparam.modelo import ROL_ESPEJO, malla_estribor, superficie_rhino_valida
from transfparam.transformacion import Lackenby, escalar, transformar

COEFS = ("cb", "cp", "cm", "cwp", "lcb_pct", "lcf_pct")


def test_convergencia(casco):
    """Doblar la resolución cambia las hidrostáticas menos de un 0.2 %."""
    cv = hidro.convergencia(casco, DWL, 0.05, 200, 200)
    for k, d in cv["diferencias_pct"].items():
        assert d < 0.2, f"{k}: {d:.3f} %"


def test_escalado_sin_lackenby(casco, H0):
    """Δ' = Δ·sx·sy·sz y los coeficientes de forma no cambian."""
    sx, sy, sz = 0.85, 0.9, 1.05
    H = hidro.calcular(casco, DWL * sz, escalar(casco.superficies(), sx, sy, sz))
    assert H.desplazamiento == pytest.approx(H0.desplazamiento * sx * sy * sz, rel=1e-6)
    assert H.lwl == pytest.approx(H0.lwl * sx, rel=1e-6)
    assert H.bwl == pytest.approx(H0.bwl * sy, rel=1e-6)
    assert H.calado == pytest.approx(H0.calado * sz, rel=1e-6)
    for k in COEFS:
        assert getattr(H, k) == pytest.approx(getattr(H0, k), abs=1e-6), k


def test_desplazamiento_cero_casco_identico(casco, H0):
    ley = Lackenby(H0.x_popa_wl, H0.x_am, H0.x_proa_wl, 0.0, 0.0)
    sups = casco.superficies()
    nuevas = ley.aplicar(sups)
    for k in sups:
        assert np.array_equal(sups[k].P, nuevas[k].P)
    H = hidro.calcular(casco, DWL, nuevas)
    assert H.volumen == H0.volumen and H.lcb == H0.lcb


def test_cb_y_lcb_alcanzan_objetivo(transformado, H0):
    H, obj = transformado.H_final, transformado.objetivos
    assert abs(H.cb - obj["Cb"]) < 1e-3
    assert abs(H.lcb_pct - obj["lcb_pct"]) / 100 < 1e-3
    # Eslora, manga y calado no cambian con Lackenby
    assert H.lwl == pytest.approx(DISENO["L"], rel=1e-3)
    assert H.bwl == pytest.approx(DISENO["B"], rel=1e-3)
    assert H.calado == pytest.approx(DISENO["T"], rel=1e-3)
    assert H.cm == pytest.approx(transformado.H_escalado.cm, rel=1e-3)


def test_superficies_validas_y_sin_babor(casco, transformado, refinadas):
    sy = transformado.escalas[1]
    for p in casco.piezas:
        s = transformado.superficies[p.id]
        assert superficie_rhino_valida(s), p.id
        X, tri = malla_estribor(p, s)
        assert X[np.unique(tri)][:, 1].min() >= 0.0, p.id
        # Lackenby no toca la y: cada punto de control tiene y' = sy·y, así que la
        # transformación no lleva ningún punto a babor
        assert np.allclose(s.P[..., 1], sy * refinadas[p.id].P[..., 1], atol=1e-12), p.id


def test_juntas_no_se_separan(casco, transformado):
    """Ninguna junta entre piezas se abre más de 5 mm (incluidas casco-espejo)."""
    assert transformado.juntas_mm, "no se han detectado juntas"
    peor = max(transformado.juntas_mm.values())
    assert peor <= 5.0, f"junta abierta {peor:.2f} mm"
    roles = {p.id: p.rol for p in casco.piezas}
    casco_espejo = [k for k in transformado.juntas_mm
                    if (roles[k[0]] == ROL_ESPEJO) != (roles[k[1]] == ROL_ESPEJO)]
    assert casco_espejo, "no se ha detectado la junta casco-espejo"
    assert max(transformado.juntas_mm[k] for k in casco_espejo) <= 5.0


def test_no_cruza_secciones(transformado):
    ley = transformado.ley
    assert ley is not None and ley.es_monotona()
    x = np.linspace(ley.x_popa - 1, ley.x_proa + 1, 20001)
    xn = x + ley.dx(x)
    assert np.all(np.diff(xn) > 0)
    assert np.all(1 + ley.ddx(x) > 0)


def test_objetivo_imposible_se_rechaza(casco, H0, refinadas):
    with pytest.raises((ValueError, RuntimeError)):
        transformar(casco, DWL, L=DISENO["L"], B=DISENO["B"], T=DISENO["T"], cb=0.85,
                    lcb_pct=H0.lcb_pct + 8, H0=H0, sup_refinadas=refinadas)


def test_juntas_en_variante_extrema(casco, H0, refinadas):
    """Variante más deformada de la matriz (Cb +5 %, LCB -2 % Lwl, L/B +3 % a Δ cte;
    c_popa ≈ 0.77): las juntas siguen por debajo de 5 mm."""
    delta = 1.025 * DISENO["cb"] * DISENO["L"] * DISENO["B"] * DISENO["T"]
    cb = DISENO["cb"] * 1.05
    lb = DISENO["L"] / DISENO["B"] * 1.03
    L = np.sqrt(delta / (1.025 * cb * DISENO["T"]) * lb)
    R = transformar(casco, DWL, L=L, B=L / lb, T=DISENO["T"], cb=cb, lcb_pct=H0.lcb_pct - 2.0,
                    H0=H0, sup_refinadas=refinadas)
    assert abs(R.ley.c_popa) > 0.7
    assert max(R.juntas_mm.values()) <= 5.0
    assert abs(R.H_final.cb - cb) < 1e-3
