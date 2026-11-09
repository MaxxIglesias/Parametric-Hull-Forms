"""Resistencia: comprobación contra ejemplos resueltos."""

import numpy as np
import pytest

from transfparam import resistencia as res


def test_van_oortmerssen_ejemplo_molland():
    """Remolcador de 37 m de Molland et al. (2011), ejemplo 13 (§17.2.13):
    L_D = 36 m, B = 13.6 m, T = 6.1 m, Cp = 0.70, Cm = 0.83, LCB = 2 % a popa,
    iE = 18°, 14 kn -> R_R/Δ = 0.02062."""
    fn = 14 * res.NUDO / np.sqrt(9.81 * 36)
    rr = res.rr_sobre_peso_vo(fn, 36, 13.6, 6.1, 0.70, 0.83, -2.0, 18.0)[0]
    assert rr == pytest.approx(0.02062, rel=0.01)


P_HOJA = {"L": 106.01, "Lpp": 106.01, "B": 18.0, "T": 6.55, "TF": 6.55, "vol": 8617.903,
          "S": 2494.418, "Cb": 0.6895106235676872, "Cp": 0.7043009433786387, "Cm": 0.979,
          "Cwp": 0.776, "lcb": 2.629, "iE": 25.21185526813559, "AT": 0.0, "S_skeg": 0.0}
V_HOJA = 15.5 * 0.5144 / res.NUDO   # la hoja usa 1 kn = 0.5144 m/s


def test_holtrop_1982_hoja_examen():
    """Hoja Holtrop2020B.xlsx (Proyectos del buque, FNB), que aplica Holtrop y Mennen
    (1982): buque de 106 m a 15.5 kn con Cstern = 5. Valores de la hoja, celda a celda."""
    r = res.holtrop_mennen(P_HOJA, [V_HOJA], cstern=5, version="1982")
    assert r["CF"][0] == pytest.approx(0.0015974241633518005, rel=1e-6)
    assert r["RF"][0] / 1000 == pytest.approx(129.82211110847107, rel=1e-6)
    assert r["1+k1"][0] == pytest.approx(1.2230868046408814, rel=1e-6)
    assert r["RW"][0] / 1000 == pytest.approx(112.60609696274655, rel=1e-5)
    assert r["CA"][0] == pytest.approx(0.0005081683533092983, rel=1e-6)


def test_holtrop_1984_solo_cambia_forma_y_olas():
    """La versión 1984 comparte fricción y correlación con la de 1982; RW cambia por el
    término m4 (a Fn = 0.247 el de 1984 da unos 118 kN frente a 112.6 kN)."""
    r82 = res.holtrop_mennen(P_HOJA, [V_HOJA], cstern=5, version="1982")
    r84 = res.holtrop_mennen(P_HOJA, [V_HOJA], cstern=5, version="1984")
    assert r84["RF"][0] == r82["RF"][0] and r84["CA"][0] == r82["CA"][0]
    assert r84["RW"][0] / 1000 == pytest.approx(118.43, rel=1e-3)


def test_holtrop_crece_con_velocidad():
    p = {"L": 20.45, "Lpp": 20.45, "B": 6.16, "T": 2.36, "TF": 2.0, "vol": 162.0, "S": 150.0,
         "Cb": 0.545, "Cp": 0.67, "Cm": 0.82, "Cwp": 0.83, "lcb": -6.0, "iE": 25.0, "AT": 1.8,
         "S_skeg": 0.0}
    r = res.holtrop_mennen(p, np.arange(6, 11.01, 0.5))
    assert np.all(np.diff(r["RT"]) > 0)
    v = res.van_oortmerssen(p, np.arange(6, 11.01, 0.5))
    assert np.all(np.diff(v["RT"]) > 0)


def test_avisos_fuera_de_rango():
    p = {"L": 20.45, "B": 6.16, "T": 2.36, "Cp": 0.67, "Cm": 0.82, "lcb": -6.0, "iE": 25.0,
         "vol": 162.0}
    avisos = res.avisos_validez("Holtrop-Mennen", p, [6, 11])
    assert any("L/B" in a for a in avisos)        # L/B = 3.32 < 3.9
    assert any("Fn" in a for a in avisos)         # 11 kn -> Fn = 0.40 > 0.38
