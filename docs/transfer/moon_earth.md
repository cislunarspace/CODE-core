# MoonEarthTransfer

**文件**: `e2m2e/transfer/moon_earth.py`

## 设计原理

月球返回地球的转移轨道设计模块。

## 转移类型

### 1. 直接返回 (Direct Return)
- 自由返回轨迹
- 跳跃式返回

### 2. 引力辅助返回 (Gravity-Assisted Return)
- 月球 Flyby 后返回

## 核心方法

| 方法 | 说明 |
|------|------|
| `design_free_return()` | 设计自由返回轨道 |
| `design_skip_entry()` | 设计跳跃式返回 |
| `compute_entry_state()` | 计算进入条件 |

## 相关类

- `MoonDeparture`: 月球出发段
- `EarthEntry`: 地球进入段
