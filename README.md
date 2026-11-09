# Transformación paramétrica de cascos (Lackenby, 1950) en Python

Reproduce la *Parametric Transformation* de Maxsurf directamente sobre el archivo de
Rhino, sin abrir Rhino ni Maxsurf (solo `rhino3dm`, `numpy`, `scipy`, `matplotlib`).

```
python main.py <casco.3dm> --dwl 2.36 --solo-hidrostaticas --convergencia
python main.py <casco.3dm> --dwl 2.36 --L 20.45 --B 6.16 --T 2.36 --cb 0.545
python main.py <casco.3dm> --dwl 2.36 --L 20.45 --B 6.16 --T 2.36 --cb 0.545 --matriz
python -m pytest tests -q
```

`python main.py -h` lista todas las opciones (Δ fijo con `--delta` y `--libre`, flag
`--sin-skeg`, rangos de la matriz, velocidades, versión de Holtrop, resolución...).

## Archivos

| Archivo | Contenido |
|---|---|
| `main.py` | Programa principal (argparse) |
| `piezas_trawler.json` | Roles (SKEG) y nombres de las caras del `.3dm` de partida |
| `transfparam/nurbs.py` | B-splines con numpy: Cox-de Boor, inversión de puntos, inserción de nudos |
| `transfparam/modelo.py` | Lectura del `.3dm`, recortes, teselado, juntas y escritura |
| `transfparam/hidrostatica.py` | Hidrostáticas por secciones, SAC, superficie mojada, convergencia |
| `transfparam/transformacion.py` | Escalado afín + Lackenby (primer orden y ajuste exacto) |
| `transfparam/resistencia.py` | Holtrop-Mennen 1982/1984 y van Oortmerssen 1971, rangos de validez |
| `transfparam/serie.py` | Matriz de variantes a desplazamiento constante |
| `transfparam/graficas.py` | SAC, caja de cuadernas, mapas de RT, curvas RT-V |
| `tests/` | Verificación (pytest) |

## El casco de partida

`Trawler_hull_maxsurf_model.3dm` (exportado de Maxsurf) está en **mm**, tiene las dos
bandas y es un mosaico de 51 breps sin nombre con caras **recortadas**. El programa:

1. pasa a metros, se queda con las caras de estribor y traslada el origen a los ejes
   del enunciado (espejo en x = 0, punto más bajo en z = 0);
2. como `rhino3dm` no da las curvas de recorte 2D, proyecta las aristas 3D de cada cara
   sobre su superficie (inversión de puntos por Newton) y rehace el contorno de recorte
   en (u, v);
3. tesela cada superficie en una rejilla (u, v) de ~5 cm ajustada al contorno: los
   vértices de los triángulos del borde se llevan sobre el recorte, de modo que las
   piezas vecinas encajan sin huecos ni solapes.

El `.3dm` de salida (en metros) contiene, por pieza: la malla recortada (capa `HULL`,
idéntica a la vista del casco), la superficie NURBS transformada sin recortar (subcapa
oculta `NURBS_sin_recortar`) y las aristas transformadas (subcapa oculta
`Bordes_recorte`, para volver a recortar en Rhino con `_Split` si se quiere la brep).

## Explicación del método (para la memoria)

**Hidrostáticas.** El casco se corta con 200 planos transversales x = cte. Como el
modelo tiene superficies solapadas y caras interiores, en cada sección se toma el
contorno exterior (semimanga máxima y_e(z) en cada cota) y su área sumergida
A(x) = 2∫y_e dz. Volumen, LCB, área de flotación y LCF se integran en x por trapecios;
la superficie mojada es el área de los triángulos exteriores bajo la flotación (sin el
espejo). Al doblar teselado, secciones y cotas, ninguna magnitud cambia más de un 0,1 %.

**Transformación.** Se hace en dos pasos sobre los puntos de control de todas las
superficies a la vez, como en Maxsurf:

1. *Escalado afín* x' = s_x x, y' = s_y y, z' = s_z z, que lleva Lwl, Bwl y T a sus
   valores objetivo. Es lineal, luego exacto sobre los puntos de control: el
   desplazamiento queda multiplicado por s_x s_y s_z y los coeficientes de forma no
   cambian.
2. *Desplazamiento de secciones de Lackenby (1950)*: x' = x + δx(x), donde cada sección
   se mueve en x sin cambiar de forma. Con el casco dividido por la sección de área
   máxima y ξ ∈ [0,1] medida desde ella hasta el extremo de la flotación de cada
   cuerpo, δx = c·L_h·ξ(1−ξ), nulo en la maestra y en los extremos: no cambian la
   eslora, la manga, el calado ni el área maestra, y sí el Cp de cada cuerpo y el LCB.
   En la formulación de Lackenby, con la SAC adimensional de cada medio cuerpo
   (coeficiente prismático Cp_h, centro x̄ y radio de giro r), el cambio es
   δCp_h = c·Cp_h(1−2x̄) y δ(Cp_h x̄) = c·Cp_h(2x̄−3r²). Los cambios de volumen y de
   momento que exigen el Cb y el LCB objetivo dan un sistema lineal en (c_popa, c_proa).
   Como la SAC se traslada exactamente, A'(x+δx) = A(x), el volumen y el momento nuevos
   se calculan sin aproximación y el sistema se resuelve por Newton. Después se corrige
   con las hidrostáticas medidas en el casco transformado hasta una tolerancia de 1e-4
   en Cb y en LCB/Lwl. La transformación no cruza secciones si |c| < 1. Antes de
   aplicarla se insertan nudos en las NURBS (sin cambiar la geometría) para que ningún
   tramo abarque más de 10 cm en x, ni más de 2 cm junto a las perpendiculares y la
   maestra, donde la ley de Lackenby cambia de pendiente. Así mover los puntos de
   control equivale a mover la superficie y las juntas entre piezas no se abren (en la
   variante más deformada de la matriz, la junta máxima es de 3,5 mm; límite 5 mm).

**Resistencia.** Holtrop y Mennen (1982) con la reanálisis de Holtrop (1984), y van
Oortmerssen (1971), con los parámetros medidos en cada casco (Lwl, B, T, ∇, S, Cp, Cm,
Cwp, LCB, iE y área sumergida del espejo A_T). Se avisa cuando un parámetro sale del
rango de validez de cada método. Por defecto se usa Holtrop 1984 (factor de forma y
resistencia por olas de la reanálisis); `--holtrop 1982` usa las del artículo original.
Comprobaciones: van Oortmerssen reproduce el ejemplo del remolcador de Molland et al.
(2011, §17.2.13) y Holtrop 1982 reproduce celda a celda la hoja `Holtrop2020B.xlsx`.

**Calado.** T es la distancia de la DWL al punto más bajo del forro (con skeg), como
pide el enunciado; Cb, Cm y B/T usan ese calado. El calado a proa T_F (solo interviene
en los términos de bulbo y correlación de Holtrop) se obtiene prolongando la línea de
quilla hasta la perpendicular de proa.

## Referencias

- Lackenby, H. (1950). On the systematic geometrical variation of ship forms.
  *Transactions of the Institution of Naval Architects*, 92, 289-316.
- Holtrop, J. y Mennen, G.G.J. (1982). An approximate power prediction method.
  *International Shipbuilding Progress*, 29(335), 166-170.
- Holtrop, J. (1984). A statistical re-analysis of resistance and propulsion data.
  *International Shipbuilding Progress*, 31(363), 272-276.
- van Oortmerssen, G. (1971). A power prediction method and its application to small
  ships. *International Shipbuilding Progress*, 18(207), 397-415.
- Molland, A.F., Turnock, S.R. y Hudson, D.A. (2011). *Ship Resistance and Propulsion*.
  Cambridge University Press (ec. 4.23, 10.24-10.34, 10.49 y tabla A3.23).
- Piegl, L. y Tiller, W. (1997). *The NURBS Book*, 2.ª ed. Springer.
