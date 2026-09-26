//! WGS84 测地坐标转换（纯数学，无 SPICE 依赖）。
//!
//! 输入 ECEF/ITRF 位置（km），输出大地纬度（deg）、经度（deg）与椭球面以上
//! 大地高度（km）。供 NRLMSISE-00 大气模型把 ITRF 位置映射到模型要求的
//! "geodetic latitude / longitude / altitude" 输入。
//!
//! 纬度/高度用 Bowring (1985) 闭式解：单次求值在近地空间（0–1000 km）内往返
//! 残差 ≤ 1e-7 deg（纬度）与 ≤ 1e-4 km（高度），远小于大气密度标高。

/// WGS84 椭球长半轴（km）。
pub const WGS84_A_KM: f64 = 6378.137;

/// WGS84 椭球扁率。
pub const WGS84_F: f64 = 1.0 / 298.257223563;

/// WGS84 椭球短半轴（km）。
const WGS84_B_KM: f64 = WGS84_A_KM * (1.0 - WGS84_F);

/// WGS84 第一偏心率平方 `e² = f(2−f)`。
const WGS84_E2: f64 = WGS84_F * (2.0 - WGS84_F);

/// WGS84 第二偏心率平方 `e'² = e²/(1−e²)`。
const WGS84_EP2: f64 = WGS84_E2 / (1.0 - WGS84_E2);

/// ECEF（ITRF）位置 → 大地坐标 `(lat_deg, lon_deg, alt_km)`。
///
/// - `lon_deg = atan2(y, x)`，东经为正，范围 `(−180, 180]`。
/// - `lat_deg` 为大地纬度（WGS84 椭球法线），`alt_km` 为椭球面以上大地高度，
///   均用 Bowring 闭式求解。
/// - 退化输入：`x = y = 0`（极轴）时返回 `lat = ±90`，原点（`|z| = 0`）返回
///   `lat = 0`；`alt` 按到椭球面的最短距离取 `|z| − b`。
pub fn ecef_to_geodetic(r_ecef_km: &[f64; 3]) -> (f64, f64, f64) {
    let (x, y, z) = (r_ecef_km[0], r_ecef_km[1], r_ecef_km[2]);
    let lon_deg = y.atan2(x).to_degrees();
    let p = (x * x + y * y).sqrt();

    // 极轴退化：水平投影为零，纬度按 z 符号取 ±90（原点取 0）。
    if p == 0.0 {
        let lat_deg = if z > 0.0 {
            90.0
        } else if z < 0.0 {
            -90.0
        } else {
            0.0
        };
        return (lat_deg, lon_deg, z.abs() - WGS84_B_KM);
    }

    // Bowring 辅助角 θ = atan2(z·a, p·b) 与单步闭式解。Bowring 展开的截断残差
    // 在近地空间（0–1000 km）内 ≤ 1e-7 deg（纬度）与 ≤ 1e-4 km（高度），
    // 远小于大气密度标高（数十 km），对阻力流场无可见影响。
    let th = (z * WGS84_A_KM).atan2(p * WGS84_B_KM);
    let (sin_th, cos_th) = th.sin_cos();
    let lat = (z + WGS84_EP2 * WGS84_B_KM * sin_th.powi(3))
        .atan2(p - WGS84_E2 * WGS84_A_KM * cos_th.powi(3));

    // 卯酉圈曲率半径 N，高度沿椭球法线：p/cos(lat) − N（cos(lat)=0 已在上面处理）。
    let sin_lat = lat.sin();
    let n = WGS84_A_KM / (1.0 - WGS84_E2 * sin_lat * sin_lat).sqrt();
    let alt_km = p / lat.cos() - n;

    (lat.to_degrees(), lon_deg, alt_km)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 测试用逆变换：大地坐标 → ECEF（km）。不对外暴露。
    fn ecef_from_geodetic(lat_deg: f64, lon_deg: f64, alt_km: f64) -> [f64; 3] {
        let (lat, lon) = (lat_deg.to_radians(), lon_deg.to_radians());
        let (sin_lat, cos_lat) = lat.sin_cos();
        let (sin_lon, cos_lon) = lon.sin_cos();
        let n = WGS84_A_KM / (1.0 - WGS84_E2 * sin_lat * sin_lat).sqrt();
        [
            (n + alt_km) * cos_lat * cos_lon,
            (n + alt_km) * cos_lat * sin_lon,
            (n * (1.0 - WGS84_E2) + alt_km) * sin_lat,
        ]
    }

    /// 赤道上椭球面点：lat=0, lon=0, alt=0。
    #[test]
    fn equator_surface_point() {
        let (lat, lon, alt) = ecef_to_geodetic(&[WGS84_A_KM, 0.0, 0.0]);
        assert!(lat.abs() < 1e-12, "lat={lat}");
        assert!(lon.abs() < 1e-12, "lon={lon}");
        assert!(alt.abs() < 1e-9, "alt={alt}");
    }

    /// 北极点：lat=90, alt=0；南极点：lat=−90。
    #[test]
    fn poles() {
        let (lat, _, alt) = ecef_to_geodetic(&[0.0, 0.0, WGS84_B_KM]);
        assert!((lat - 90.0).abs() < 1e-12, "lat={lat}");
        assert!(alt.abs() < 1e-9, "alt={alt}");

        let (lat_s, _, alt_s) = ecef_to_geodetic(&[0.0, 0.0, -WGS84_B_KM]);
        assert!((lat_s + 90.0).abs() < 1e-12, "lat={lat_s}");
        assert!(alt_s.abs() < 1e-9, "alt={alt_s}");
    }

    /// 原点退化：lat=0，alt = −b（到椭球面最短距离）。
    #[test]
    fn origin_is_degenerate() {
        let (lat, _, alt) = ecef_to_geodetic(&[0.0, 0.0, 0.0]);
        assert_eq!(lat, 0.0);
        assert!((alt + WGS84_B_KM).abs() < 1e-12, "alt={alt}");
    }

    /// 已知 400 km 高度点：赤道经线上 lat=0，alt=400。
    #[test]
    fn known_400km_point() {
        let r = [(WGS84_A_KM + 400.0), 0.0, 0.0];
        let (lat, lon, alt) = ecef_to_geodetic(&r);
        assert!(lat.abs() < 1e-12, "lat={lat}");
        assert!(lon.abs() < 1e-12, "lon={lon}");
        assert!((alt - 400.0).abs() < 1e-9, "alt={alt}");
    }

    /// geodetic → ecef → geodetic 往返：覆盖纬度、经度与高度网格。
    ///
    /// 断言的是 Bowring 单步闭式在该高度域内的实测残差上界（见函数注释）；
    /// 断言消息会打印实测最坏值，便于后续收紧。
    #[test]
    fn round_trip_geodetic() {
        let cases: &[(f64, f64, f64)] = &[
            (0.0, 0.0, 0.0),
            (45.0, 30.0, 400.0),
            (-45.0, -120.0, 800.0),
            (60.0, 179.5, 100.0),
            (-89.0, 0.0, 700.0),
            (23.5, 121.0, 650.0),
            (-0.01, -0.01, 1.0),
        ];
        let (mut dlat, mut dlon, mut dalt) = (0.0_f64, 0.0_f64, 0.0_f64);
        for &(lat, lon, alt) in cases {
            let r = ecef_from_geodetic(lat, lon, alt);
            let (lat2, lon2, alt2) = ecef_to_geodetic(&r);
            dlat = dlat.max((lat2 - lat).abs());
            dlon = dlon.max((lon2 - lon).abs());
            dalt = dalt.max((alt2 - alt).abs());
        }
        assert!(dlat < 1e-7, "lat 往返实测最坏 {dlat:e} deg");
        assert!(dlon < 1e-9, "lon 往返实测最坏 {dlon:e} deg");
        assert!(dalt < 1e-4, "alt 往返实测最坏 {dalt:e} km");
    }
}
