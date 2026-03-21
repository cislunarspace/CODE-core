# EarthMoonTransfer

**文件**: `e2m2e/transfer/earth_moon.py`

## 设计原理

地月转移轨道设计模块，计算从地球停泊轨道到月球轨道的转移轨道。

## 转移类型

### 1. 直接转移 (Direct Transfer)
- Hohmann 转移
- 椭圆转移

### 2. 月球フライバイ (Lunar Flyby)
- 利用月球引力辅助

### 3. 转式轨道 (Transfer Orbit)
- 低能转移路径

## 核心方法

| 方法 | 说明 |
|------|------|
| `design_hohmann_transfer(earth_orbit, moon_orbit)` | 设计 Hohmann 转移 |
| `design_low_energy_transfer()` | 设计低能转移 |
| `optimize_departure_window()` | 优化发射窗口 |

## 相关类

- `EarthDeparture`: 地球出发段
- `MoonArrival`: 月球到达段
