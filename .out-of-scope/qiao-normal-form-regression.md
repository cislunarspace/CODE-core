# qiao normal-form 数值对拍

本项目不把 qiao normal-form 流水线的中间结果作为 e2m2e 的回归基准，也不维护相应的 `.mat` / `.npz` 对拍工具链。

## 为什么不在范围内

qiao 的动力学替代、quasi-Floquet 和中心流形流水线是独立研究代码，其结果依赖另一套 MATLAB 运行环境、生成步骤和大体量中间数据。把这些产物纳入 e2m2e 的回归流程，会让验证依赖外部实现及其工作目录，而不能说明 e2m2e 是否按物理定义正确。

e2m2e 的 normal-form 主链服务于 CR3BP 平动点附近的有界 Lissajous 轨迹生成。其正确性由 Hamilton 结构、辛性、同调方程残差、坐标往返和轨迹有界性等定义级性质验证。SPICE 星历仍是项目支持的标准运行能力；本决策只排除以 qiao 中间结果为 oracle 的对拍工作。

开发者可以临时使用本地脚本调查两套实现的系统性差异，但这类诊断不进入 pytest、发布契约或项目 backlog。

## 过往请求

- [#426](https://github.com/cislunarspace/e2m2e/issues/426)：normal_form 与 qiao 参考数据的数值回归
