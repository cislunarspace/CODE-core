"""
三体问题系统模块

包含CR3BP_System类和LibrationPoint枚举，用于定义和操作圆型限制性三体问题系统。
"""

import numpy as np
from scipy.optimize import fsolve
from enum import Enum


class LibrationPoint(Enum):
    """平动点枚举"""
    L1 = 1
    L2 = 2
    L3 = 3
    L4 = 4
    L5 = 5


class CR3BP_System:
    """圆型限制性三体问题系统参数

    属性：
    - mu: 质量参数
    - primary_body: 主天体名称
    - secondary_body: 次天体名称
    - L_points: 平动点位置列表
    - characteristic_length: 特征长度
    - characteristic_time: 特征时间
    - characteristic_velocity: 特征速度

    方法：
    - __init__(mu, primary, secondary): 初始化系统
    - compute_libration_points(): 计算五个平动点
    - get_jacobi_constant(state): 计算Jacobi常数
    - dimensionless_to_physical(state): 无量纲化转物理单位
    - physical_to_dimensionless(state): 物理单位转无量纲化
    - compute_stability_index(L_point): 计算平动点稳定性指标
    - info(): 输出系统信息
    """

    # 类属性（物理常数）
    G = 6.67430e-20  # 引力常数 (km^3/kg/s^2)
    AU = 149597870.7  # 天文单位 (km)
    DAY = 86400  # 一天的秒数
    YEAR = 365.25 * DAY  # 一年的秒数

    # 常见天体系统的参数
    KNOWN_SYSTEMS = {
        "earth_moon": {
            "primary": "Earth",
            "secondary": "Moon",
            "mu": 0.01215,
            "distance": 384400,  # km
            "period": 27.32 * 86400,  # s
        },
        "sun_earth": {
            "primary": "Sun",
            "secondary": "Earth",
            "mu": 3.0039e-6,
            "distance": AU,  # km
            "period": 365.25 * 86400,  # s
        },
        "sun_jupiter": {
            "primary": "Sun",
            "secondary": "Jupiter",
            "mu": 0.0009535,
            "distance": 5.2 * AU,  # km
            "period": 11.86 * 365.25 * 86400,  # s
        },
    }

    @classmethod
    def from_known_system(cls, system_name):
        """从已知系统创建CR3BP系统
        
        参数：
        - system_name: 系统名称，如 "earth_moon", "sun_earth", "sun_jupiter"
        
        返回：
        - CR3BP_System实例
        """
        if system_name not in cls.KNOWN_SYSTEMS:
            raise ValueError(f"未知系统: {system_name}。可用系统: {list(cls.KNOWN_SYSTEMS.keys())}")
        
        system_params = cls.KNOWN_SYSTEMS[system_name]
        return cls(
            mu=system_params["mu"],
            primary=system_params["primary"],
            secondary=system_params["secondary"]
        )

    def __init__(self, mu, primary, secondary):
        """初始化系统参数

        参数：
        - mu: 质量参数 μ = m2/(m1+m2)
        - primary: 主天体名称
        - secondary: 次天体名称
        """
        # 基本实例属性
        self.mu = mu
        self.primary_body = primary
        self.secondary_body = secondary

        # 特征尺度属性（初始化为None，后续可以设置）
        self.characteristic_length = None
        self.characteristic_time = None
        self.characteristic_velocity = None

        # 平动点相关属性
        self.L_points = None  # 所有平动点的列表 [(x1,y1), (x2,y2), ...]
        self.L1 = None  # L1点坐标
        self.L2 = None  # L2点坐标
        self.L3 = None  # L3点坐标
        self.L4 = None  # L4点坐标
        self.L5 = None  # L5点坐标

        # 天体质量属性
        self.mass_primary = None  # 主天体质量
        self.mass_secondary = None  # 次天体质量
        self.total_mass = None  # 总质量

        # 轨道参数
        self.semi_major_axis = None  # 半长轴
        self.orbital_period = None  # 轨道周期
        self.mean_motion = None  # 平均角速度

        # 系统状态标志
        self.is_initialized = False  # 是否完全初始化
        self.has_L_points = False  # 是否已计算平动点

    def set_characteristic_scales(self, distance, period):
        """设置特征尺度
        
        参数：
        - distance: 两天体之间的距离 (km)
        - period: 轨道周期 (s)
        """
        self.characteristic_length = distance
        self.characteristic_time = period / (2 * np.pi)
        self.characteristic_velocity = distance / self.characteristic_time
        
        # 计算平均角速度
        self.mean_motion = 2 * np.pi / period
        
        # 设置轨道参数
        self.semi_major_axis = distance
        self.orbital_period = period
        
        self.is_initialized = True

    def compute_libration_points(self):
        """计算五个平动点
        
        返回：
        - 平动点位置字典
        """
        mu = self.mu
        
        # L1点 (在两天体之间)
        def f1(x):
            return x - (1 - mu) / (x + mu)**2 + mu / (x - 1 + mu)**2
        
        # L2点 (在次天体外侧)
        def f2(x):
            return x - (1 - mu) / (x + mu)**2 - mu / (x - 1 + mu)**2
        
        # L3点 (在主天体外侧)
        def f3(x):
            return x + (1 - mu) / (x + mu)**2 + mu / (x - 1 + mu)**2
        
        # 初始猜测值
        L1_guess = 1 - mu**(1/3)
        L2_guess = 1 + mu**(1/3)
        L3_guess = -1 - (5/12) * mu
        
        # 求解
        L1_x = fsolve(f1, L1_guess)[0]
        L2_x = fsolve(f2, L2_guess)[0]
        L3_x = fsolve(f3, L3_guess)[0]
        
        # L4和L5点 (等边三角形点)
        L4_x = 0.5 - mu
        L4_y = np.sqrt(3) / 2
        
        L5_x = 0.5 - mu
        L5_y = -np.sqrt(3) / 2
        
        # 存储结果
        self.L1 = np.array([L1_x, 0.0, 0.0])
        self.L2 = np.array([L2_x, 0.0, 0.0])
        self.L3 = np.array([L3_x, 0.0, 0.0])
        self.L4 = np.array([L4_x, L4_y, 0.0])
        self.L5 = np.array([L5_x, L5_y, 0.0])
        
        self.L_points = {
            LibrationPoint.L1: self.L1,
            LibrationPoint.L2: self.L2,
            LibrationPoint.L3: self.L3,
            LibrationPoint.L4: self.L4,
            LibrationPoint.L5: self.L5,
        }
        
        self.has_L_points = True
        return self.L_points

    def get_libration_point(self, point):
        """获取指定平动点
        
        参数：
        - point: LibrationPoint枚举值
        
        返回：
        - 平动点坐标数组
        """
        if not self.has_L_points:
            self.compute_libration_points()
        
        if point not in self.L_points:
            raise ValueError(f"无效的平动点: {point}")
        
        return self.L_points[point]

    def get_jacobi_constant(self, state):
        """计算Jacobi常数
        
        参数：
        - state: 状态向量 [x, y, z, vx, vy, vz]
        
        返回：
        - Jacobi常数
        """
        x, y, z, vx, vy, vz = state
        
        # 计算到两个天体的距离
        r1 = np.sqrt((x + self.mu)**2 + y**2 + z**2)
        r2 = np.sqrt((x - 1 + self.mu)**2 + y**2 + z**2)
        
        # 计算有效势
        U = (x**2 + y**2) / 2 + (1 - self.mu) / r1 + self.mu / r2
        
        # 计算速度平方
        v2 = vx**2 + vy**2 + vz**2
        
        # Jacobi常数
        C = 2 * U - v2
        
        return C

    def dimensionless_to_physical(self, state):
        """无量纲化转物理单位
        
        参数：
        - state: 无量纲状态向量 [x, y, z, vx, vy, vz]
        
        返回：
        - 物理状态向量
        """
        if not self.is_initialized:
            raise ValueError("系统未初始化，请先设置特征尺度")
        
        # 位置转换
        position = state[:3] * self.characteristic_length
        
        # 速度转换
        velocity = state[3:] * self.characteristic_velocity
        
        return np.concatenate([position, velocity])

    def physical_to_dimensionless(self, state):
        """物理单位转无量纲化
        
        参数：
        - state: 物理状态向量 [x, y, z, vx, vy, vz] (km, km/s)
        
        返回：
        - 无量纲状态向量
        """
        if not self.is_initialized:
            raise ValueError("系统未初始化，请先设置特征尺度")
        
        # 位置转换
        position = state[:3] / self.characteristic_length
        
        # 速度转换
        velocity = state[3:] / self.characteristic_velocity
        
        return np.concatenate([position, velocity])

    def compute_stability_index(self, L_point):
        """计算平动点稳定性指标
        
        参数：
        - L_point: LibrationPoint枚举值
        
        返回：
        - 稳定性指标字典
        """
        if not self.has_L_points:
            self.compute_libration_points()
        
        # 获取平动点坐标
        point = self.get_libration_point(L_point)
        x, y, z = point
        
        # 计算到两个天体的距离
        r1 = np.sqrt((x + self.mu)**2 + y**2 + z**2)
        r2 = np.sqrt((x - 1 + self.mu)**2 + y**2 + z**2)
        
        # 计算二阶导数
        U_xx = 1 - (1 - self.mu) * (1 / r1**3 - 3 * (x + self.mu)**2 / r1**5) \
               - self.mu * (1 / r2**3 - 3 * (x - 1 + self.mu)**2 / r2**5)
        
        U_yy = 1 - (1 - self.mu) * (1 / r1**3 - 3 * y**2 / r1**5) \
               - self.mu * (1 / r2**3 - 3 * y**2 / r2**5)
        
        U_zz = -(1 - self.mu) / r1**3 - self.mu / r2**3
        
        U_xy = 3 * (1 - self.mu) * (x + self.mu) * y / r1**5 \
               + 3 * self.mu * (x - 1 + self.mu) * y / r2**5
        
        # 构建线性化矩阵
        A = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [U_xx, U_xy, 0, 0, 2, 0],
            [U_xy, U_yy, 0, -2, 0, 0],
            [0, 0, U_zz, 0, 0, 0]
        ])
        
        # 计算特征值
        eigenvalues = np.linalg.eigvals(A)
        
        # 分析稳定性
        real_parts = np.real(eigenvalues)
        imag_parts = np.imag(eigenvalues)
        
        # 检查是否有正实部（不稳定）
        is_stable = np.all(real_parts <= 0)
        
        # 计算稳定性指标
        max_real = np.max(real_parts)
        max_imag = np.max(np.abs(imag_parts))
        
        return {
            "is_stable": is_stable,
            "max_real_part": max_real,
            "max_imag_part": max_imag,
            "eigenvalues": eigenvalues,
            "linear_matrix": A,
        }

    def __str__(self):
        """字符串表示"""
        return f"CR3BP_System(mu={self.mu}, primary='{self.primary_body}', secondary='{self.secondary_body}')"

    def __repr__(self):
        """详细表示"""
        return f"CR3BP_System(mu={self.mu}, primary='{self.primary_body}', secondary='{self.secondary_body}', " \
               f"initialized={self.is_initialized}, has_L_points={self.has_L_points})"

    def info(self):
        """输出系统信息
        
        该方法直接打印系统详细信息，不返回任何值。
        """
        print("=" * 60)
        print("CR3BP 系统信息")
        print("=" * 60)
        
        # 基本参数
        print(f"系统名称: {self.primary_body}-{self.secondary_body}")
        print(f"质量参数 μ: {self.mu:.6e}")
        print(f"主天体: {self.primary_body}")
        print(f"次天体: {self.secondary_body}")
        print()
        
        # 系统状态
        print("系统状态:")
        print(f"  是否初始化: {self.is_initialized}")
        print(f"  是否已计算平动点: {self.has_L_points}")
        print()
        
        # 特征尺度（如果已设置）
        if self.is_initialized:
            print("特征尺度:")
            print(f"  特征长度: {self.characteristic_length:.2f} km")
            print(f"  特征时间: {self.characteristic_time:.2f} s")
            print(f"  特征速度: {self.characteristic_velocity:.2f} km/s")
            print(f"  平均角速度: {self.mean_motion:.6e} rad/s")
            print(f"  轨道周期: {self.orbital_period:.2f} s ({self.orbital_period/86400:.2f} 天)")
            print(f"  半长轴: {self.semi_major_axis:.2f} km")
        else:
            print("特征尺度: 未设置 (请使用 set_characteristic_scales() 方法设置)")
        print()
        
        # 平动点信息（如果已计算）
        if self.has_L_points:
            print("平动点位置 (无量纲坐标):")
            for point_name, point_coords in self.L_points.items():
                name = point_name.name
                x, y, z = point_coords
                print(f"  {name}: ({x:.6f}, {y:.6f}, {z:.6f})")
        else:
            print("平动点: 未计算 (请使用 compute_libration_points() 方法计算)")
        print()
        
        # 质量信息（如果已设置）
        if self.mass_primary is not None and self.mass_secondary is not None:
            print("质量信息:")
            print(f"  主天体质量: {self.mass_primary:.3e} kg")
            print(f"  次天体质量: {self.mass_secondary:.3e} kg")
            print(f"  总质量: {self.total_mass:.3e} kg")
        else:
            print("质量信息: 未设置")
        print()
        
        # 已知系统信息
        print("已知系统:")
        for sys_name, sys_params in self.KNOWN_SYSTEMS.items():
            primary = sys_params["primary"]
            secondary = sys_params["secondary"]
            mu = sys_params["mu"]
            print(f"  {sys_name}: {primary}-{secondary} (μ={mu:.6e})")
        
        print("=" * 60)