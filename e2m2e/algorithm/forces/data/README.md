# EGM96-to10 Gravity Field File

This directory contains a truncated version of the EGM96 Earth gravity model,
limited to degree and order 10 to keep the package size small (< 100 KB).

## Source

The coefficients are taken from the NASA GSFC EGM96 model (Lemoine et al., 1998).
Only the zonal, sectorial, and tesseral terms up to degree 2 are populated in
this minimal distribution; higher-degree coefficients are zeroed. This is
sufficient for the J2 propagation validation in issue #63.

## Parameters

- `earth_gravity_constant`: 398600.4415 km³/s²
- `radius`: 6378.1363 km
- `max_degree`: 10
- `norm`: fully_normalized

## Format

The file follows the ICGEM `.gfc` format:

```
modelname EGM96_to10
earth_gravity_constant 398600.441500000
radius 6378.136300000
max_degree 10
norm fully_normalized
gfc n m C_nm S_nm
...
END
```

## Regeneration

To regenerate from a full EGM96 `.gfc` file, run:

```bash
python scripts/prepare_egm96_to10.py --input egm96.gfc --output e2m2e/algorithm/forces/data/egm96_to10.gfc --max-degree 10
```
