# 传播中心天体的一等配置（月心传播）

e2m2e 不给轨道预报的传输层提供「中心天体 / 传播原点」一等配置：`PropagationRequest` 及其
CLI/MCP/sidecar 请求模型不加 `central_body` 字段，`propagate_orbit` 也不按中心天体自动生成
对称力配置（中心天体球谐 + 其余天体第三体 + 对应间接项）。传输层的传播原点固定地心。

需要月心系状态的调用方走**进程内手工链路**——这是本就受支持的用法：

```python
system = EphemerisSystem(bodies=["MOON", "EARTH", "SUN"], spice=spice, origin="MOON")
system.coordinate_system = CoordinateSystem(
    axes=ICRSAxes(), origin=CelestialBodyOrigin(body="MOON", spice=spice)
)
force_config = {  # 月心球谐 + 地球/太阳第三体（原点由 system.origin 决定）
    "version": 1,
    "forces": [
        {"name": "gravity_moon", "type": "GravityField",
         "params": {"body": "MOON", "degree": 10, "order": 10, "input_frame": "MOON_PA"}},
        {"name": "third_earth", "type": "ThirdBodyGravity", "params": {"body": "EARTH"}},
        {"name": "third_sun", "type": "ThirdBodyGravity", "params": {"body": "SUN"}},
    ],
}
fm = ForceModel.from_config(force_config, system)
```

## 为什么不在范围内

**缺的是便利封装，不是能力。** #666 分诊期间实测（2026-09-26，一次性脚本，未入库）：

- 上面这条手工链路对 2000 km 环月圆轨道外推 1 天共 289 点，`|r|` 全程保持
  1999.72~2001.03 km，无错误返回；
- 同一物理初值换地心等价配置（`PointMassGravity(EARTH)` + `GravityField(MOON, 10×10)` +
  `IndirectTerm(MOON)` + `ThirdBodyGravity(SUN)`）外推后逐点减月地状态，1 天弧上
  `max|Δr| = 5.9e-4 km`、`max|Δv| = 4.0e-7 km/s`。两条路径是同一物理，差异只在积分容差量级
  —— 说明手工链路不只是「跑得通」，而是算得对。

**自动化本身是风险项。** 传输层默认力模型是地心模型（`PointMassGravity(EARTH)` +
`ThirdBodyGravity(MOON/SUN)`）。只加一个 `central_body` 字段而不重排力模型，等于把地心模型
套进月心系静默算错；要安全就得再引入一套按中心天体派生的默认模型与一致性守卫。维护者判定
这份自动化不值得引入：中心天体与力模型配置全部由调用方手工给，传输层保持地心原点不变。

**已知边界**（现象记录，未立项）：`propagate_orbit` 的原点写死地心，也不校验力配置与原点是否
自洽。把按非地心意图写的力配置交给它（例如地心系里的 `ThirdBodyGravity(EARTH)`，此时
`r_i = 0` 退化），会得到地心系下的错误轨迹，且状态仍报 `CONVERGED`。调用方在传输层混用手工
力配置前需自行确认原点一致。

## 复核路径

这是范围决定，不是失效修复：能力（月心原点 + 手工力配置）在算法层齐备，被拒的只是把它提到
传输层一等配置。若将来 CLI/MCP 成为月心任务的主入口，按以下路径复议：

- 重开 #666，或另开 issue 明确「传输层中心天体」的口径：字段命名（`central_body` 还是沿用
  领域词 `origin`）、默认力模型归属、以及原点与力配置不一致时的守卫策略；
- 届时需先决定是否接受第二套默认力模型，否则只能提供字段 + 显式拒绝不匹配配置的组合。

## 过往请求

- [#666](https://github.com/cislunarspace/CODE-core/issues/666)：`PropagationRequest` 增加
  中心天体参数，`propagate_orbit` 与 `force_mapping` 按中心天体生成对称力配置
