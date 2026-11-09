"""Transformación paramétrica de cascos (Lackenby, 1950) sobre archivos de Rhino.

Módulos:
    nurbs           B-splines con numpy (Cox-de Boor), inversión de puntos, inserción de nudos.
    modelo          Lectura del .3dm, recortes, teselado y escritura del casco transformado.
    hidrostatica    Hidrostáticas por secciones transversales y curva de áreas (SAC).
    transformacion  Escalado afín + desplazamiento de secciones de Lackenby.
    resistencia     Holtrop-Mennen (1982/1984) y van Oortmerssen (1971).
    serie           Matriz de variantes a desplazamiento constante.
    graficas        Figuras (SAC, caja de cuadernas, mapas de resistencia, curvas Rt-V).
"""

RHO_AGUA_MAR = 1.025  # t/m3
