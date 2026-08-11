//! build.rs — 解析仓库根 constants.toml，生成 Rust 物理常量模块。
//!
//! 单一来源：仓库根 ``constants.toml``。本脚本在 cargo build 时把 TOML 中的
//! 通用常量、基准集、天体参数转成 ``const XXX: f64 = ...;`` 写入 OUT_DIR，
//! 再由 ``src/constants.rs`` 以 ``include!`` 引入。
//!
//! 不引入 serde/toml 等第三方依赖，仅用一个手写最小 TOML 解析器，覆盖本项目
//! 用到的标量、字符串、表结构。

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

/// 解析后的 TOML 值（本项目只用得到这几种）。
#[derive(Debug, Clone)]
enum TomlValue {
    Float(f64),
    Int(i64),
    Str(String),
    Table(HashMap<String, TomlValue>),
}

/// 极简 TOML 解析：仅支持本项目 constants.toml 用到的结构。
/// - 跳过注释与空行
/// - 支持 ``key = value``（整数、浮点、字符串）
/// - 支持 ``key = { value = ..., source = ... }`` 内联表
/// - 支持 ``[section]`` / ``[section.sub]`` 表头
struct MiniTomlParser<'a> {
    lines: std::str::Lines<'a>,
}

impl<'a> MiniTomlParser<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            lines: text.lines(),
        }
    }

    fn parse(mut self) -> HashMap<String, TomlValue> {
        let mut root: HashMap<String, TomlValue> = HashMap::new();
        let mut current_table_path: Vec<String> = Vec::new();

        while let Some(line) = self.lines.next() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            if let Some(table_path) = trimmed.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
                current_table_path = table_path
                    .split('.')
                    .map(|s| s.trim().to_string())
                    .collect();
                continue;
            }

            let (key, value_str) = match trimmed.split_once('=') {
                Some((k, v)) => (k.trim().to_string(), v.trim().to_string()),
                None => continue,
            };

            let value = self.parse_value(&value_str);
            Self::insert_at_path(&mut root, &current_table_path, key, value);
        }

        root
    }

    fn parse_value(&self, s: &str) -> TomlValue {
        if s.starts_with('{') && s.ends_with('}') {
            return self.parse_inline_table(s);
        }
        if (s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')) {
            let inner = &s[1..s.len() - 1];
            return TomlValue::Str(inner.to_string());
        }
        if let Ok(i) = s.parse::<i64>() {
            return TomlValue::Int(i);
        }
        if let Ok(f) = s.parse::<f64>() {
            return TomlValue::Float(f);
        }
        // 兜底当字符串处理
        TomlValue::Str(s.to_string())
    }

    fn parse_inline_table(&self, s: &str) -> TomlValue {
        let inner = &s[1..s.len() - 1];
        let mut map: HashMap<String, TomlValue> = HashMap::new();
        for part in inner.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            if let Some((k, v)) = part.split_once('=') {
                let k = k.trim().to_string();
                let v = self.parse_value(v.trim());
                map.insert(k, v);
            }
        }
        TomlValue::Table(map)
    }

    fn insert_at_path(
        root: &mut HashMap<String, TomlValue>,
        path: &[String],
        key: String,
        value: TomlValue,
    ) {
        let mut current = root;
        for segment in path {
            if current
                .get(segment)
                .is_some_and(|v| !matches!(v, TomlValue::Table(_)))
            {
                // 路径冲突：用空表覆盖旧值
                current.insert(segment.clone(), TomlValue::Table(HashMap::new()));
            }
            let entry = current
                .entry(segment.clone())
                .or_insert_with(|| TomlValue::Table(HashMap::new()));
            match entry {
                TomlValue::Table(ref mut t) => current = t,
                _ => unreachable!("just ensured entry is a table"),
            }
        }
        current.insert(key, value);
    }
}

impl TomlValue {
    fn as_table(&self) -> Option<&HashMap<String, TomlValue>> {
        match self {
            TomlValue::Table(t) => Some(t),
            _ => None,
        }
    }

    fn as_f64(&self) -> Option<f64> {
        match self {
            TomlValue::Float(f) => Some(*f),
            TomlValue::Int(i) => Some(*i as f64),
            _ => None,
        }
    }

    #[allow(dead_code)]
    fn as_i64(&self) -> Option<i64> {
        match self {
            TomlValue::Int(i) => Some(*i),
            TomlValue::Float(f) => Some(*f as i64),
            _ => None,
        }
    }
}

fn value_or_inline(root: &HashMap<String, TomlValue>, key: &str) -> Option<f64> {
    root.get(key).and_then(|v| match v {
        TomlValue::Table(t) => t.get("value").and_then(|vv| vv.as_f64()),
        _ => v.as_f64(),
    })
}

fn source_or_inline(root: &HashMap<String, TomlValue>, key: &str) -> Option<String> {
    root.get(key).and_then(|v| match v {
        TomlValue::Table(t) => t.get("source").and_then(|vv| match vv {
            TomlValue::Str(s) => Some(s.clone()),
            _ => None,
        }),
        _ => None,
    })
}

fn table_path<'m>(
    root: &'m HashMap<String, TomlValue>,
    path: &[&str],
) -> Option<&'m HashMap<String, TomlValue>> {
    let mut cur = root;
    for &segment in path {
        match cur.get(segment) {
            Some(TomlValue::Table(t)) => cur = t,
            _ => return None,
        }
    }
    Some(cur)
}

fn write_header(out: &mut String) {
    out.push_str(
        "// Auto-generated by build.rs -- do not edit manually.\n\
         // Single source of truth: constants.toml (repo root).\n\n",
    );
}

fn write_universal(
    out: &mut String,
    root: &HashMap<String, TomlValue>,
    lookup: &mut Vec<(String, String)>,
) {
    let uni = match table_path(root, &["universal"]) {
        Some(t) => t,
        None => return,
    };

    out.push_str(
        "// ----------------------------------------------------------------------------\n",
    );
    out.push_str("// Universal physical constants\n");
    out.push_str(
        "// ----------------------------------------------------------------------------\n",
    );

    let mappings: &[(&str, &str)] = &[
        ("SPEED_OF_LIGHT_KMS", "speed_of_light_kms"),
        ("GRAVITATIONAL_CONSTANT", "gravitational_constant"),
        ("AU_KM", "au_km"),
        ("SECONDS_PER_DAY", "seconds_per_day"),
        ("DAYS_PER_JULIAN_YEAR", "days_per_julian_year"),
        ("DAYS_PER_JULIAN_CENTURY", "days_per_julian_century"),
        ("KM_TO_M", "km_to_m"),
        ("SOLAR_FLUX_W_M2", "solar_flux_w_m2"),
        ("SOLAR_FLUX_TSI_W_M2", "solar_flux_tsi_w_m2"),
        ("RAD_PER_DEG", "rad_per_deg"),
    ];

    for (rust_name, toml_key) in mappings {
        let val = value_or_inline(uni, toml_key)
            .unwrap_or_else(|| panic!("missing universal constant {toml_key}"));
        let src = source_or_inline(uni, toml_key).unwrap_or_else(|| "SI".to_string());
        out.push_str(&format!(
            "/// {rust_name} (source: {src})\npub const {rust_name}: f64 = {val:e};\n\n"
        ));
        lookup.push((format!("universal.{toml_key}"), rust_name.to_string()));
    }

    // 派生量：1 AU 光压（N/m²）
    out.push_str(
        "/// 1 AU 太阳辐射压（N/m²），由 SOLAR_FLUX_W_M2 / (SPEED_OF_LIGHT_KMS * KM_TO_M) 派生。\n",
    );
    out.push_str(
        "pub const SOLAR_PRESSURE_1AU: f64 = SOLAR_FLUX_W_M2 / (SPEED_OF_LIGHT_KMS * KM_TO_M);\n\n",
    );
    lookup.push((
        "universal.solar_pressure_1au".to_string(),
        "SOLAR_PRESSURE_1AU".to_string(),
    ));
}

fn write_datum(
    out: &mut String,
    root: &HashMap<String, TomlValue>,
    lookup: &mut Vec<(String, String)>,
) {
    let datums = match table_path(root, &["datum"]) {
        Some(t) => t,
        None => return,
    };

    out.push_str(
        "// ----------------------------------------------------------------------------\n",
    );
    out.push_str("// Datum constants (DE421 / DE440 / WGS84)\n");
    out.push_str(
        "// ----------------------------------------------------------------------------\n",
    );

    for (datum_name, datum_table) in datums {
        let dt = match datum_table.as_table() {
            Some(t) => t,
            None => continue,
        };
        out.push_str(&format!("// Datum: {datum_name}\n"));
        for (key, value) in dt {
            let val = match value {
                TomlValue::Table(t) => t.get("value").and_then(|v| v.as_f64()),
                _ => value.as_f64(),
            };
            let Some(val) = val else { continue };
            let rust_name = format!("DATUM_{}_{}", datum_name.to_uppercase(), key.to_uppercase());
            out.push_str(&format!(
                "/// {rust_name} (datum {datum_name}, key {key})\npub const {rust_name}: f64 = {val:e};\n\n"
            ));
            lookup.push((format!("datum.{datum_name}.{key}"), rust_name));
        }
    }
}

fn write_body(
    out: &mut String,
    root: &HashMap<String, TomlValue>,
    lookup: &mut Vec<(String, String)>,
) {
    let bodies = match table_path(root, &["body"]) {
        Some(t) => t,
        None => return,
    };

    out.push_str(
        "// ----------------------------------------------------------------------------\n",
    );
    out.push_str("// Body constants\n");
    out.push_str(
        "// ----------------------------------------------------------------------------\n",
    );

    for (body_name, body_table) in bodies {
        let bt = match body_table.as_table() {
            Some(t) => t,
            None => continue,
        };
        let prefix = body_name.to_uppercase();

        // 标量字段
        if let Some(v) = value_or_inline(bt, "mean_radius_km") {
            out.push_str(&format!(
                "pub const {prefix}_MEAN_RADIUS_KM: f64 = {v:e};\n\n"
            ));
            lookup.push((
                format!("body.{body_name}.mean_radius_km"),
                format!("{prefix}_MEAN_RADIUS_KM"),
            ));
        }
        if let Some(v) = value_or_inline(bt, "gravity_ref_radius_km") {
            out.push_str(&format!(
                "pub const {prefix}_GRAVITY_REF_RADIUS_KM: f64 = {v:e};\n\n"
            ));
            lookup.push((
                format!("body.{body_name}.gravity_ref_radius_km"),
                format!("{prefix}_GRAVITY_REF_RADIUS_KM"),
            ));
        }
        if let Some(v) = value_or_inline(bt, "flattening") {
            out.push_str(&format!("pub const {prefix}_FLATTENING: f64 = {v:e};\n\n"));
            lookup.push((
                format!("body.{body_name}.flattening"),
                format!("{prefix}_FLATTENING"),
            ));
        }
        if let Some(v) = value_or_inline(bt, "rotation_rate_iers_rad_s") {
            out.push_str(&format!(
                "pub const {prefix}_ROTATION_RATE_IERS_RAD_S: f64 = {v:e};\n\n"
            ));
            lookup.push((
                format!("body.{body_name}.rotation_rate_iers_rad_s"),
                format!("{prefix}_ROTATION_RATE_IERS_RAD_S"),
            ));
        }
        if let Some(v) = value_or_inline(bt, "rotation_rate_gmat_rad_s") {
            out.push_str(&format!(
                "pub const {prefix}_ROTATION_RATE_GMAT_RAD_S: f64 = {v:e};\n\n"
            ));
            lookup.push((
                format!("body.{body_name}.rotation_rate_gmat_rad_s"),
                format!("{prefix}_ROTATION_RATE_GMAT_RAD_S"),
            ));
        }
        if let Some(v) = value_or_inline(bt, "naif_id") {
            // naif_id 通常作为 i32 常量
            let i = v as i32;
            out.push_str(&format!("pub const {prefix}_NAIF_ID: i32 = {i};\n\n"));
            lookup.push((
                format!("body.{body_name}.naif_id"),
                format!("{prefix}_NAIF_ID as f64"),
            ));
        }

        // GM 按 datum
        if let Some(TomlValue::Table(gm_table)) = bt.get("gm") {
            for (datum, gm_value) in gm_table {
                let val = match gm_value {
                    TomlValue::Table(t) => t.get("value").and_then(|v| v.as_f64()),
                    _ => gm_value.as_f64(),
                };
                let Some(val) = val else { continue };
                let rust_name = format!("{prefix}_GM_{}", datum.to_uppercase());
                out.push_str(&format!("pub const {rust_name}: f64 = {val:e};\n\n"));
                lookup.push((format!("body.{body_name}.gm.{datum}"), rust_name));
            }
        }
    }
}

fn write_lookup(out: &mut String, lookup: &[(String, String)]) {
    out.push_str(
        "// ----------------------------------------------------------------------------\n",
    );
    out.push_str("// Constant lookup table (key -> const)\n");
    out.push_str(
        "// ----------------------------------------------------------------------------\n",
    );
    out.push_str("pub const CONSTANT_LOOKUP: &[(&str, f64)] = &[\n");
    for (key, rust_name) in lookup {
        out.push_str(&format!("    (\"{key}\", {rust_name}),\n"));
    }
    out.push_str("];\n\n");
}

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let constants_toml = manifest_dir
        .parent() // crates/
        .and_then(|p| p.parent()) // repo root
        .map(|p| p.join("constants.toml"))
        .expect("cannot resolve repo root from CARGO_MANIFEST_DIR");

    println!("cargo:rerun-if-changed={}", constants_toml.display());

    let text = fs::read_to_string(&constants_toml)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", constants_toml.display()));
    let root = MiniTomlParser::new(&text).parse();

    let mut generated = String::new();
    write_header(&mut generated);
    let mut lookup: Vec<(String, String)> = Vec::new();
    write_universal(&mut generated, &root, &mut lookup);
    write_datum(&mut generated, &root, &mut lookup);
    write_body(&mut generated, &root, &mut lookup);
    write_lookup(&mut generated, &lookup);

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let out_file = out_dir.join("generated_constants.rs");
    fs::write(&out_file, generated)
        .unwrap_or_else(|e| panic!("cannot write {}: {e}", out_file.display()));
}
