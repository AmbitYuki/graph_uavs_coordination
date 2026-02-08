# quadrotor_formation 代码地图（速查 + 调用链）

> 目标：让你“看文件名就知道它做什么”，并能顺着调用链定位到控制/编队/规划的关键实现。

## 1. 仓库整体在做什么

这个仓库主要用 **MuJoCo** 仿真 `assets/skydio_x2/*.xml` 里的多架四旋翼（Skydio X2 模型），在仿真循环中：

1) 从 MuJoCo 的 `qpos/qvel` 读取每架无人机状态
2) 通过控制器计算每架无人机的控制量（总推力 + 机体系力矩）
3) 通过控制分配（control allocation）把控制量映射到四个电机命令
4) 写回 MuJoCo 的 `ctrl` 并 `mj_step`
5) （可选）引入编队误差，使从机跟随领航机形成固定/旋转编队

最常见的入口不是单独的 main，而是 tests/ 里的仿真测试脚本。

## 2. 关键依赖

- `mujoco`：仿真内核与 viewer
- `numpy`：数值计算
- `matplotlib`：测试脚本绘图
- `spatialmath-python`：SO3/SE3、旋转/位姿运算（编队/几何模块用）
- `pytest`：运行 tests

## 3. 运行入口（建议从 tests 读调用链）

- tests/test_formation_four.py：四机（或三角）编队 + 轨迹跟踪示例
- tests/test_formation.py：大量控制/编队/轨迹的综合测试
- tests/test_cooperation.py：更复杂的协同/切换/可视化脚本

它们都会走同一个核心链路：

`MuJoCo(qpos/qvel)` → `src/parameter/parameter.py: Parameter` →
`src/controller/formation_controller/formation_controller.py: FormationController.control()` →
`src/controller/velocity_controller/velocity_controller.py: VelocityController.control()` +
`src/controller/orientation_controller/orientation_controller.py: OrientationController.control2()` →
`src/model/model.py: Model.assign()` →
`src/control_assignment/x_control_assignment.py: XControlAssignment.assign()` → `MuJoCo(ctrl)`

## 4. src/ 目录逐文件说明

### 4.1 顶层包
- src/__init__.py：空文件，让 `src` 成为包。

### 4.2 常量
- src/constanst/__init__.py：导出常量模块。
- src/constanst/math_const.py：数值阈值（EPS/ERROR），用于几何算法收敛/去抖。

### 4.3 控制分配（Control Allocation）
- src/control_assignment/__init__.py：导出控制分配类。
- src/control_assignment/control_assignment.py：抽象/基类，把控制量 `taus=[u1,u2,u3,u4]` 映射到四电机 `omegas`。
- src/control_assignment/x_control_assignment.py：X 型四旋翼分配矩阵（M0/P），计算逆矩阵完成分配。

### 4.4 动力学/机体模型参数
- src/model/__init__.py：导出 Model/Skydio。
- src/model/model.py：模型基类，提供质量 m、惯量 I、重力 g，并封装 `assign()`。
- src/model/skydio.py：Skydio X2 的具体参数（m、I、XControlAssignment 参数）。

### 4.5 控制器
- src/controller/__init__.py：导出 Controller / FormationController 等。
- src/controller/controller.py：单机控制组合器（PositionController + OrientationController）。

- src/controller/pid_controller/__init__.py：导出 PIDController。
- src/controller/pid_controller/pid_controller.py：离散 PID（后向欧拉积分 + 一阶滤波微分）。

- src/controller/position_controller/__init__.py：导出 PositionController。
- src/controller/position_controller/position_controller.py：位置环（x/y/z）→ 计算 `u1` 与期望倾角 `phi/theta`。

- src/controller/velocity_controller/__init__.py：导出 VelocityController。
- src/controller/velocity_controller/velocity_controller.py：速度环（x/y/z）→ 计算 `u1` 并把编队误差 `e` 作为附加项注入。

- src/controller/orientation_controller/__init__.py：导出 OrientationController。
- src/controller/orientation_controller/orientation_controller.py：姿态环（roll/pitch/yaw）→ 计算 `u2,u3,u4`，带角度 wrap 处理。

- src/controller/formation_controller/__init__.py：导出 FormationController。
- src/controller/formation_controller/formation_controller.py：多机控制与“编队开关/渐入”状态机：
  - `control()`：在 blending/active 期间计算编队误差 e 并注入 VelocityController。
  - `update_switch()` / `request_enable_formation()`：控制从独立飞行 → rendezvous → blending → active 的切换过程。
  - 带推力/倾角软限幅、偏航渐入、误差权重渐入等。

### 4.6 编队几何（相对位置）
- src/formation/__init__.py：导出 Formation + 三种编队。
- src/formation/formation.py：编队基类：
  - `cal_deltas(psi)`：给定领航机偏航角，计算每架机相对其它机的期望相对位移矩阵。
  - `cal_delta0(psi)`：由子类实现第 0 架机到其它机的基准相对向量。
  - `set_spacing(t, adjust_time)`：可选的动态间距渐变（用于“靠拢/展开”）。
- src/formation/line_formation.py：直线编队的 `cal_delta0()`。
- src/formation/triangle_formation.py：三角编队的 `cal_delta0()`。
- src/formation/four_formation.py：四机编队（类似三角+后方一点）的 `cal_delta0()`。

### 4.7 状态参数封装
- src/parameter/__init__.py：导出 Parameter。
- src/parameter/parameter.py：把 MuJoCo 的 `qpos/qvel` 映射成控制器用的：位置/速度/姿态/角速度/期望量。

### 4.8 轨迹规划（Trajectory Planning）
- src/motion_planning/__init__.py：导出轨迹规划相关类；并触发 `Strategy.factory_register()` 做策略注册。
- src/motion_planning/motion_parameter.py：占位（当前未实现）。
- src/motion_planning/motion_planner.py：占位（当前未实现）。

- src/motion_planning/trajectory_planning/trajectory_parameter.py：把 PathParameter 与 VelocityParameter 打包。
- src/motion_planning/trajectory_planning/trajectory_planner.py：速度规划得到标量 s，再用路径规划插值得到 q(t)。

- src/motion_planning/trajectory_planning/path_planning/path_planning_mode_enum.py：路径规划模式枚举（JOINT/CARTESIAN）。
- src/motion_planning/trajectory_planning/path_planning/path_parameter.py：路径参数基类（长度等）。
- src/motion_planning/trajectory_planning/path_planning/path_planner_strategy.py：Path 的 Strategy 基类。
- src/motion_planning/trajectory_planning/path_planning/path_planner.py：StrategyWrapper，按 mode 选择具体 planner。
- src/motion_planning/trajectory_planning/path_planning/joint_planning/joint_parameter.py：关节空间起点/终点（q0,q1）。
- src/motion_planning/trajectory_planning/path_planning/joint_planning/joint_planner.py：关节空间线性插值 q(s)。

- src/motion_planning/trajectory_planning/velocity_planning/velocity_planning_mode_enum.py：速度规划模式（CUBIC/QUINTIC/T_CURVE）。
- src/motion_planning/trajectory_planning/velocity_planning/velocity_parameter.py：速度参数基类。
- src/motion_planning/trajectory_planning/velocity_planning/velocity_planner_strategy.py：Velocity 的 Strategy 基类。
- src/motion_planning/trajectory_planning/velocity_planning/velocity_planner.py：StrategyWrapper，按 mode 选择具体 planner。
- src/motion_planning/trajectory_planning/velocity_planning/cubic_velocity_planning/*：三次多项式 s(t)（边界：s(0)=0,s(tf)=1,速度边界为0）。
- src/motion_planning/trajectory_planning/velocity_planning/quintic_velocity_planning/*：五次多项式 s(t)（更平滑的速度/加速度边界）。

### 4.9 Strategy/Factory 框架
- src/interface/mode_enum.py：ModeEnum 基类（枚举基类）。
- src/interface/register.py：Register 抽象接口（要求提供 mode）。
- src/interface/factory.py：Factory（mode → strategy class 映射表）。
- src/interface/parameter.py：策略参数抽象基类（motion planning 用）。
- src/interface/strategy.py：Strategy 抽象基类 + `factory_register()`（自动注册所有叶子子类）。
- src/interface/strategy_wrapper.py：Wrapper：根据 parameter.get_mode() 实例化对应 strategy。
- src/interface/__init__.py：导出接口并在 import 时调用 `Strategy.factory_register()`。

### 4.10 工具
- src/utils/class_utils.py：找某个基类的“叶子子类”（用于自动注册策略）。
- src/utils/math_utils.py：near_zero 等小工具。

### 4.11 几何库（geometry/）
> 这部分是一个相对独立的几何/碰撞检测小库，和编队控制不是强耦合。

- src/geometry/collision/GJK.py：GJK 算法（支持函数 Support），计算凸体距离/是否相交。
- src/geometry/collision/colliison.py：Collision 包装器（调用 GJK.is_intersecting）。
- src/geometry/collision/distance.py：点/线/线段/砖体等距离计算 + GJK 距离接口。
- src/geometry/collision/*2d.py：2D 场景的距离/相交检测。

- src/geometry/shape/*：基础几何体（球/圆/砖/胶囊/椭球/平面/圆柱等），多数实现 Support 接口或辅助距离计算。
- src/geometry/simplex/*：GJK 用的 simplex 表示（点/线段/三角形/四面体）及最近点/重心坐标计算。
- src/geometry/simplex/factory/*：按点数量构造 simplex 的工厂池（给 GJK 用）。
- src/geometry/rotation/SO3Impl.py、SE3Impl.py：对 spatialmath 的 SO3/SE3 做运算符重载（加减乘标量/复合）。

## 5. tests/ 与 try/ 目录

- tests/test_formation_four.py：四机/三角编队 + 7 段轨迹拼接示例（带 viewer）。
- tests/test_formation.py：覆盖编队/控制/轨迹的综合测试集合（包含较长仿真段）。
- tests/test_cooperation.py：协同控制与可视化（会绘图，字体可能有警告）。
- tests/test_process.py：一些流程性/集成测试。

- try/try_switch.py：实验脚本，通常用于手动试切换逻辑。

## 6. 已知小坑（读代码时注意）

- Parameter 的 `psid` 属性在同一个类里定义了两次：后一个覆盖前一个（可能是历史遗留）。
- MotionPlanner / MotionParameter 目前是占位符，不是主流程的一部分。
- MuJoCo viewer 相关测试在无图形环境下可能需要额外配置（如 EGL），否则会报 OpenGL/显示相关错误。
