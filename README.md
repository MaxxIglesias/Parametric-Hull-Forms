# Parametric hull transformation (Lackenby) for a 24 m bottom trawler

Coursework for *280812 – Advanced Ship Design* (Master in Naval and Oceanic Engineering, UPC-FNB).

MAXSURF Modeler's Parametric Transformation distorted the parent hull, a render-type model made of
many trimmed and overlapping NURBS patches. This repository reimplements the same method in Python
and applies it directly to the Rhino `.3dm` file, without Rhino or MAXSURF running. It only uses
`rhino3dm`, `numpy`, `scipy` and `matplotlib`.

- **Hydrostatics** from 200 transverse sections (outer contour only), converged to < 0.1 %.
- **Affine scaling** to the design L, B and T, plus a **Lackenby section shift** (Lackenby, 1950)
  to reach the target C_B and LCB within 1e-4.
- **Resistance** with Holtrop-Mennen (1982/1984) and van Oortmerssen (1971), with validity-range warnings.
- **Variant matrix** at constant displacement (C_B, LCB, L/B), with maps and R_T–V curves.
- **pytest** verification: scaling, zero shift, targets, patch joints < 5 mm, no section crossing,
  and resistance checked against worked examples.

> Code comments, command-line options and output labels are in Spanish.

## Quick start

The parent hull is included in `data/` (see [data/README.md](data/README.md) for its source).

```
pip install -r requirements.txt

# Hydrostatics of the parent hull and convergence check
python main.py data/Trawler_hull_maxsurf_model.3dm --dwl 2.36 --solo-hidrostaticas --convergencia

# Transform the parent to the design hull
python main.py data/Trawler_hull_maxsurf_model.3dm --dwl 2.36 --L 20.45 --B 6.16 --T 2.36 --cb 0.545

# 75-variant matrix at constant displacement (C_B ±5 %, LCB ±2 % Lwl, L/B ±3 %)
python main.py data/Trawler_hull_maxsurf_model.3dm --dwl 2.36 --L 20.45 --B 6.16 --T 2.36 --cb 0.545 --matriz

# Tests (about 90 s)
python -m pytest tests -q
```

`python main.py -h` lists every option. These include a fixed displacement with one free
dimension (`--delta`, `--libre`), excluding the skeg (`--sin-skeg`), the matrix ranges, the speed
range, the Holtrop version (`--holtrop 1982|1984`) and the numerical resolution.

## Results

The results shown in the report are already in `resultados/`:

| Folder | Content |
|---|---|
| `resultados/diseno/` | Design hull: transformed `.3dm`, hydrostatics (parent / scaled / final), convergence, resistance curves, SAC, body plan and R_T–V figures |
| `resultados/matriz/` | Variant study: summary table (`matriz_resumen.csv/.xlsx`), resistance curves of all variants, R_T maps vs C_B and LCB, base vs best comparison |
| `resultados/matriz/variantes/` | `.3dm` of the 75 hull variants (`V000`–`V074`) |

Each output `.3dm` is in metres and holds three layers per patch:
- the trimmed mesh, on the `HULL` layer, which looks exactly like the hull;
- the transformed untrimmed NURBS surface, on the hidden `HULL::NURBS_sin_recortar` sublayer;
- the transformed trim edges, on the hidden `HULL::Bordes_recorte` sublayer, to re-trim in Rhino
  with `_Split` if a brep is needed.

## Code structure

| File | Content |
|---|---|
| `main.py` | Command-line program (argparse) |
| `piezas_trawler.json` | Roles (skeg) and names of the patches of the parent `.3dm` |
| `transfparam/nurbs.py` | B-splines with numpy: Cox-de Boor basis, point inversion, knot insertion |
| `transfparam/modelo.py` | Reading the `.3dm`, trim boundaries, tessellation, patch joints, writing |
| `transfparam/hidrostatica.py` | Hydrostatics by sections, SAC, wetted surface, convergence check |
| `transfparam/transformacion.py` | Affine scaling and Lackenby transformation |
| `transfparam/resistencia.py` | Holtrop-Mennen 1982/1984 and van Oortmerssen 1971, validity ranges |
| `transfparam/serie.py` | Variant matrix at constant displacement |
| `transfparam/graficas.py` | SAC, body plan, R_T maps, R_T–V curves |
| `transfparam/salidas.py` | CSV/Excel tables and reports |
| `tests/` | Verification (pytest) |

## Method

**Parent geometry.** The parent `.3dm` is in millimetres, has both sides and is a mosaic of
trimmed patches. The program converts it to metres and keeps the starboard patches. It then moves
the origin to the transom (x = 0), centreline (y = 0) and lowest point of the hull (z = 0).
`rhino3dm` does not expose the 2D trim curves, so the 3D edges of each face are projected onto
its surface (Newton point inversion) to rebuild the trim boundary in (u, v). Each surface is then
tessellated on a ~5 cm (u, v) grid fitted to that boundary, so neighbouring patches meet without
gaps or overlaps.

**Hydrostatics.** The hull is cut by 200 transverse planes. Because of the overlapping and
internal patches, only the outer contour of each section is kept (maximum half-breadth at each
height), giving A(x) = 2∫y dz. Volume, LCB, waterplane area and LCF are integrated along x with
the trapezoidal rule. The wetted surface is the area of the outer triangles below the waterline,
excluding the transom. Doubling the tessellation, sections and waterlines changes no quantity by
more than 0.1 %.

**Transformation.** Two steps, applied to the control points of all patches at once:

1. *Affine scaling* x' = s_x x, y' = s_y y, z' = s_z z, which brings Lwl, Bwl and T to their
   targets. Being linear, it is exact on the control points: the displacement scales by
   s_x s_y s_z and the form coefficients do not change.
2. *Lackenby section shift* (Lackenby, 1950), x' = x + δx(x): each section moves along the ship
   without changing its shape. The hull is split at the maximum section and, with ξ ∈ [0, 1]
   running from it to the end of the waterline, δx = c·L_h·ξ(1−ξ) for each half body. The shift
   is zero at the maximum section and at both ends, so length, beam, draught and midship area are
   kept, while C_P of each body and the LCB change. Lackenby's first-order relations,
   δC_P = c·C_P(1−2x̄) and δ(C_P x̄) = c·C_P(2x̄−3r²), give the starting amplitudes (x̄ and r:
   centroid and radius of gyration of each half SAC). Since the SAC moves exactly,
   A'(x+δx) = A(x), the new volume and moment are known exactly and both amplitudes are solved
   analytically. They are then corrected with the hydrostatics measured on the transformed hull
   until C_B and LCB/Lwl match within 1e-4. Sections never cross as long as |c| < 1.

Before the shift, knots are inserted in the NURBS surfaces without changing the geometry, so that
no span covers more than 10 cm in x (2 cm next to the perpendiculars and the maximum section,
where the Lackenby law changes slope). Moving the control points is then equivalent to moving the
surface, and the joints between patches stay closed: 3.5 mm in the most deformed variant of the
matrix, against a 5 mm limit.

**Resistance.** Holtrop and Mennen (1982) with Holtrop's re-analysis (1984), and van Oortmerssen
(1971), using the parameters measured on each hull (Lwl, B, T, ∇, S, C_P, C_M, C_WP, LCB, i_E and
immersed transom area A_T). A warning is issued whenever a parameter falls outside the validity
range of a method. Holtrop 1984 is the default; `--holtrop 1982` uses the original paper. Checks:
van Oortmerssen reproduces the tug example of Molland et al. (2011, §17.2.13), and Holtrop 1982
reproduces a worked spreadsheet cell by cell.

**Draught.** T is measured from the waterline to the lowest point of the hull (bottom of the
skeg), and C_B, C_M and B/T use this draught. The forward draught T_F, which only enters the
bulb and correlation terms of Holtrop, is obtained by extending the keel line to the forward
perpendicular.

## Licence

The code and the generated results are released under the MIT licence (see [LICENSE](LICENSE)).
The parent hull model in `data/` belongs to its original author (I. Khlutchin, GrabCAD) and is
not covered by this licence; it is included only for academic use.

## References

- Lackenby, H. (1950). On the systematic geometrical variation of ship forms.
  *Transactions of the Institution of Naval Architects*, 92, 289–316.
- Holtrop, J. and Mennen, G.G.J. (1982). An approximate power prediction method.
  *International Shipbuilding Progress*, 29(335), 166–170.
- Holtrop, J. (1984). A statistical re-analysis of resistance and propulsion data.
  *International Shipbuilding Progress*, 31(363), 272–276.
- van Oortmerssen, G. (1971). A power prediction method and its application to small ships.
  *International Shipbuilding Progress*, 18(207), 397–415.
- Molland, A.F., Turnock, S.R. and Hudson, D.A. (2011). *Ship Resistance and Propulsion*.
  Cambridge University Press (Eqs. 4.23, 10.24–10.34, 10.49 and Table A3.23).
- Piegl, L. and Tiller, W. (1997). *The NURBS Book*, 2nd ed. Springer.
