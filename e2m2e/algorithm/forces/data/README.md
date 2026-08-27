# EGM96-to10 Gravity Field File / EGM96-to10 重力场文件

[English](#egm96-to10-gravity-field-file) | [简体中文](#中文说明)

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

## 中文说明

本目录包含 EGM96 地球重力场模型的截断版本，最高阶次限制为 10 阶 10 次，以控制安装包体积（< 100 KB）。

### 来源

系数取自 NASA GSFC 的 EGM96 模型（Lemoine 等，1998）。在这个最小发行版中，仅填充到 2 阶的带谐、扇谐与田谐项；更高阶系数置零。这对 issue #63 的 J2 传播验证已经足够。

### 参数

- `earth_gravity_constant`：398600.4415 km³/s²
- `radius`：6378.1363 km
- `max_degree`：10
- `norm`：fully_normalized（完全正规化）

### 格式

文件遵循 ICGEM `.gfc` 格式：

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

### 再生成

要从完整 EGM96 `.gfc` 文件重新生成，执行：

```bash
python scripts/prepare_egm96_to10.py --input egm96.gfc --output e2m2e/algorithm/forces/data/egm96_to10.gfc --max-degree 10
```
