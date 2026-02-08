
from unittest import TestCase
import time
import os

import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

from src.model import Skydio
from src.controller import FormationController
from src.formation import LineFormation, TrajectoryFollowingFormation
from src.parameter import Parameter
from src.motion_planning import JointParameter, QuinticVelocityParameter, TrajectoryParameter, TrajectoryPlanner

# 全局时间参量：仿真时间达到该值后，开启编队（秒）
T_SWITCH = 15.0

# 可选：编队飞行一段时间后切换领航机（秒）。设为 None 表示不切换。
T_CHANGE_LEADER = None  #10.0
NEW_LEADER_ID = int(os.environ.get('NEW_LEADER_ID', '1'))

# 可选：再飞一段时间后解散编队（秒）。设为 None 表示不解散。
T_DISBAND = None  #30.0

# 通过“输入参数”切换 leader：
# 运行前设置环境变量，例如：
#   LEADER_ID=2 python -m unittest tests.test_switch.TestSwitch.test_switch
LEADER_ID = int(os.environ.get('LEADER_ID', '1'))

# 默认不阻塞单测：保存图像到文件；需要交互显示时设置 SHOW_PLOT=1
SHOW_PLOT = os.environ.get('SHOW_PLOT', '0') == '1'
PLOT_SAVE_PATH = os.environ.get('PLOT_SAVE_PATH', 'switch_trajectories.png')

# --- 新过渡思路：基于 leader 轨迹 + xyz 偏移生成 follower 参考 ---
# FormationController.tracking_mode = 'offset_velocity' 时：
#   v_i_cmd = v_leader_cmd + Kp * ((p_leader + offset_i) - p_i)
# 其中 offset_i 为相对 leader 的偏移。若使用 TrajectoryFollowingFormation，则偏移会随 leader yaw 旋转，
# 并加入 yaw_rate 产生的切向速度项，实现“围绕 leader 转弯”。
FOLLOWER_TRACKING_MODE = 'offset_velocity'
OFFSET_POS_KP = 1.2
OFFSET_MAX_SPEED = 3.0
OFFSETS_IGNORE_YAW = False

# 可选：显式指定每架机相对 leader 的世界系偏移 (shape: (count,3))。
# None 表示从 formation 形状推导（LineFormation/Triangle/...），并根据 OFFSETS_IGNORE_YAW 决定是否忽略 yaw。
OFFSETS_WORLD_OVERRIDE = None


class TestSwitch(TestCase):
	"""基于 test_cooperation 的四机轨迹，加入“到时间 T_SWITCH 开启编队”的过渡测试。"""

	def test_switch(self):
		model = mujoco.MjModel.from_xml_path("assets/skydio_x2/scene4.xml")
		data = mujoco.MjData(model)

		mujoco.mj_setState(model, data, np.append(model.key_qpos, model.key_qvel), mujoco.mjtState.mjSTATE_PHYSICS)
		mujoco.mj_setState(model, data, model.key_ctrl[0, :], mujoco.mjtState.mjSTATE_CTRL)
		mujoco.mj_forward(model, data)

		skydio = Skydio()
		count = 4
		dt = model.opt.timestep
		kps = [1, 0.5, 0.5, 50.0, 50.0, 50.0]
		kis = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
		kds = [0.1, 0.1, 0.0, 10, 10, 10]
		parameters = [Parameter() for _ in range(count)]

		# 编队形状：几何形状 + 轨迹主导包装器（让转弯时形成围绕 leader 的整体运动）
		formation = TrajectoryFollowingFormation(LineFormation(1.0, count), ignore_yaw=OFFSETS_IGNORE_YAW)
		formation_controller = FormationController(
			kps,
			kis,
			kds,
			skydio,
			count,
			ts=dt,
			position_gain=2.0,
			use_formation=False,
			tracking_mode=FOLLOWER_TRACKING_MODE,
			leader_id=LEADER_ID,
			offset_pos_kp=OFFSET_POS_KP,
			offset_max_speed=OFFSET_MAX_SPEED,
			offsets_world=OFFSETS_WORLD_OVERRIDE,
			offsets_ignore_yaw=OFFSETS_IGNORE_YAW,
		)
		# 如果你想在运行时动态改 offsets，可解开并编辑：
		# formation_controller.set_offsets_world(np.array([
		# 	[0.0, 0.0, 0.0],
		# 	[1.0, 0.0, 0.0],
		# 	[-1.0, 0.0, 0.0],
		# 	[0.0, 1.0, 0.0],
		# ], dtype=float))

		print(f'leader_id={LEADER_ID} tracking_mode={FOLLOWER_TRACKING_MODE} kp={OFFSET_POS_KP} vmax={OFFSET_MAX_SPEED}')

		# 为每架无人机创建轨迹规划器（保持 test_cooperation 原逻辑不变）
		trajectory_planners_list = []
		total_times = []
		phase_counts = []
		phase_times = []
		init_rotate_times = []

		# ----------------------------------------------------------------------------------
		# 无人机1: 螺旋上升（radius减小）
		init_rotate_time1 = 5
		q0_1 = np.array([0.0, 0.0, 0.0, 0.0])
		q1_1 = q0_1 + np.array([0.0, 0.0, 0.0, np.pi])
		joint_param_init1 = JointParameter(q0_1, q1_1)
		vel_param_init1 = QuinticVelocityParameter(init_rotate_time1)
		traj_param_init1 = TrajectoryParameter(joint_param_init1, vel_param_init1)
		trajectory_planner_init1 = TrajectoryPlanner(traj_param_init1)

		radius_init1 = 1.0
		z_final1 = 0.6
		total_angle1 = 2 * 3 * np.pi
		phase_count1 = 3000 * 3
		phase_time1 = 0.005
		circle_total_time1 = phase_count1 * phase_time1
		total_time1 = init_rotate_time1 + circle_total_time1

		waypoints1 = []
		for i in range(phase_count1 + 1):
			angle = i * (total_angle1 / phase_count1)
			radius = radius_init1 - (radius_init1 / phase_count1) * i
			x = q0_1[0] + radius * np.cos(angle)
			y = q0_1[1] + radius * np.sin(angle)
			z = q0_1[2] + (z_final1 / phase_count1) * i
			yaw = angle + np.pi
			waypoints1.append(np.array([x, y, z, yaw]))

		trajectory_planners1 = [trajectory_planner_init1]
		for i in range(phase_count1):
			joint_param = JointParameter(waypoints1[i], waypoints1[i + 1])
			vel_param = QuinticVelocityParameter(phase_time1)
			traj_param = TrajectoryParameter(joint_param, vel_param)
			trajectory_planners1.append(TrajectoryPlanner(traj_param))

		trajectory_planners_list.append(trajectory_planners1)
		total_times.append(total_time1)
		phase_counts.append(phase_count1)
		phase_times.append(phase_time1)
		init_rotate_times.append(init_rotate_time1)

		# ----------------------------------------------------------------------------------
		# 无人机2: 圆形飞行（motion2）
		z_init2 = 0.3
		init_rotate_time2 = 5
		q0_2 = np.array([0.0, 0.0, z_init2, 0.0])
		q1_2 = q0_2 + np.array([0.0, 0.0, 0.0, np.pi])
		joint_param_init2 = JointParameter(q0_2, q1_2)
		vel_param_init2 = QuinticVelocityParameter(init_rotate_time2)
		traj_param_init2 = TrajectoryParameter(joint_param_init2, vel_param_init2)
		trajectory_planner_init2 = TrajectoryPlanner(traj_param_init2)

		radius_xy2 = 1.4
		radius_z2 = 0.8
		total_angle2 = 2 * 3 * np.pi
		phase_count2 = 3000 * 3
		phase_time2 = 0.005
		circle_total_time2 = phase_count2 * phase_time2
		total_time2 = init_rotate_time2 + circle_total_time2

		waypoints2 = []
		for i in range(phase_count2 + 1):
			angle = i * (total_angle2 / phase_count2)
			x = q0_2[0] + radius_xy2 * np.cos(angle)
			y = q0_2[1] + radius_xy2 * np.sin(angle)
			z = radius_z2 * np.sin(4 * angle)
			yaw = angle + np.pi
			waypoints2.append(np.array([x, y, z, yaw]))

		trajectory_planners2 = [trajectory_planner_init2]
		for i in range(phase_count2):
			joint_param = JointParameter(waypoints2[i], waypoints2[i + 1])
			vel_param = QuinticVelocityParameter(phase_time2)
			traj_param = TrajectoryParameter(joint_param, vel_param)
			trajectory_planners2.append(TrajectoryPlanner(traj_param))

		trajectory_planners_list.append(trajectory_planners2)
		total_times.append(total_time2)
		phase_counts.append(phase_count2)
		phase_times.append(phase_time2)
		init_rotate_times.append(init_rotate_time2)

		# ----------------------------------------------------------------------------------
		# 无人机3: 八字飞行（motion3）
		z_init3 = 0.1
		init_rotate_time3 = 5
		q0_3 = np.array([0.0, 0.0, z_init3, 0.0])
		q1_3 = q0_3 + np.array([0.0, 0.0, 0.0, np.pi])
		joint_param_init3 = JointParameter(q0_3, q1_3)
		vel_param_init3 = QuinticVelocityParameter(init_rotate_time3)
		traj_param_init3 = TrajectoryParameter(joint_param_init3, vel_param_init3)
		trajectory_planner_init3 = TrajectoryPlanner(traj_param_init3)

		radius_xy3 = 0.6
		radius_z3 = 0.3
		circle_steps_per3 = 4500
		phase_count3 = circle_steps_per3 * 2
		phase_time3 = 0.005
		circle_total_time3 = phase_count3 * phase_time3
		total_time3 = init_rotate_time3 + circle_total_time3

		waypoints3 = []
		total_z_cycle3 = 2 * np.pi
		for i in range(phase_count3 + 1):
			if i <= circle_steps_per3:
				angle = i * (2 * np.pi / circle_steps_per3)
			else:
				angle = 2 * np.pi - (i - circle_steps_per3) * (2 * np.pi / circle_steps_per3)

			x = q0_3[0] + radius_xy3 * np.cos(angle)
			y = q0_3[1] + radius_xy3 * np.sin(angle)
			z_angle = (i / phase_count3) * total_z_cycle3
			z = q0_3[2] + radius_z3 * np.cos(z_angle)
			yaw = angle + np.pi
			waypoints3.append(np.array([x, y, z, yaw]))

		trajectory_planners3 = [trajectory_planner_init3]
		for i in range(phase_count3):
			joint_param = JointParameter(waypoints3[i], waypoints3[i + 1])
			vel_param = QuinticVelocityParameter(phase_time3)
			traj_param = TrajectoryParameter(joint_param, vel_param)
			trajectory_planners3.append(TrajectoryPlanner(traj_param))

		trajectory_planners_list.append(trajectory_planners3)
		total_times.append(total_time3)
		phase_counts.append(phase_count3)
		phase_times.append(phase_time3)
		init_rotate_times.append(init_rotate_time3)

		# ----------------------------------------------------------------------------------
		# 无人机4: 圆形飞行（motion4）
		z_init4 = 0.1
		init_rotate_time4 = 5
		q0_4 = np.array([0.0, -0.1, z_init4, 0.0])
		q1_4 = q0_4 + np.array([0.0, 0.0, 0.0, np.pi])
		joint_param_init4 = JointParameter(q0_4, q1_4)
		vel_param_init4 = QuinticVelocityParameter(init_rotate_time4)
		traj_param_init4 = TrajectoryParameter(joint_param_init4, vel_param_init4)
		trajectory_planner_init4 = TrajectoryPlanner(traj_param_init4)

		radius4 = 1
		y_final4 = 0.6
		total_angle4 = 2 * 5 * np.pi
		phase_count4 = 3000 * 3
		phase_time4 = 0.005
		circle_total_time4 = phase_count4 * phase_time4
		total_time4 = init_rotate_time4 + circle_total_time4

		waypoints4 = []
		for i in range(phase_count4 + 1):
			angle = i * (total_angle4 / phase_count4)
			x = q0_4[0] + radius4 * np.cos(angle)
			y = q0_4[1] + (y_final4 / phase_count4) * i
			z = radius4 * np.sin(angle)
			yaw = angle + np.pi
			waypoints4.append(np.array([x, y, z, yaw]))

		trajectory_planners4 = [trajectory_planner_init4]
		for i in range(phase_count4):
			joint_param = JointParameter(waypoints4[i], waypoints4[i + 1])
			vel_param = QuinticVelocityParameter(phase_time4)
			traj_param = TrajectoryParameter(joint_param, vel_param)
			trajectory_planners4.append(TrajectoryPlanner(traj_param))

		trajectory_planners_list.append(trajectory_planners4)
		total_times.append(total_time4)
		phase_counts.append(phase_count4)
		phase_times.append(phase_time4)
		init_rotate_times.append(init_rotate_time4)

		# ----------------------------------------------------------------------------------
		total_simulation_time = max(total_times)
		time_step_num = round(total_simulation_time / model.opt.timestep)

		qds_list = [np.zeros((time_step_num, 6)) for _ in range(count)]
		times = np.linspace(0, total_simulation_time, time_step_num)

		for drone_id in range(count):
			joint_position = np.zeros(4)
			for i, timei in enumerate(times):
				if timei > total_times[drone_id]:
					joint_position = qds_list[drone_id][i - 1, [0, 1, 2, 5]]
				else:
					if timei < init_rotate_times[drone_id]:
						joint_position = trajectory_planners_list[drone_id][0].interpolate(timei)
					else:
						circle_time = timei - init_rotate_times[drone_id]
						phase_idx = int(circle_time // phase_times[drone_id])
						if phase_idx >= phase_counts[drone_id]:
							phase_idx = phase_counts[drone_id] - 1
						local_time = circle_time % phase_times[drone_id]
						joint_position = trajectory_planners_list[drone_id][phase_idx + 1].interpolate(local_time)
				qds_list[drone_id][i, [0, 1, 2, 5]] = joint_position

		trajectories = [np.zeros((time_step_num, 3)) for _ in range(count)]
		target_trajectories = [np.zeros((time_step_num, 3)) for _ in range(count)]

		switch_requested = False
		leader_changed = False
		disbanded = False
		current_leader_id = int(LEADER_ID)
		time_num = 0
		with mujoco.viewer.launch_passive(model, data) as viewer:
			while viewer.is_running() and data.time <= total_simulation_time:
				step_start = time.time()

				# 更新状态（让 update_switch 使用最新位置/速度）
				for drone_id in range(count):
					parameters[drone_id].q = data.qpos[7 * drone_id: 7 * (drone_id + 1)]
					parameters[drone_id].dq = data.qvel[6 * drone_id: 6 * (drone_id + 1)]
					trajectories[drone_id][time_num] = parameters[drone_id].q[:3]

				if time_num < time_step_num:
					# 切换前：四机各自按原轨迹运动（保持 test_cooperation 的轨迹不变）
					if (not switch_requested) and data.time < T_SWITCH:
						for drone_id in range(count):
							parameters[drone_id].dposd = qds_list[drone_id][time_num, [0, 1, 2]]
							parameters[drone_id].psid = qds_list[drone_id][time_num, 5]
							target_trajectories[drone_id][time_num] = qds_list[drone_id][time_num, [0, 1, 2]]

					# 编队阶段（包含：开启编队、换 leader、解散编队）
					else:
						# 1) 第一次到达 T_SWITCH：开启编队
						if (not switch_requested) and data.time >= T_SWITCH:
							formation_controller.request_enable_formation(leader_id=current_leader_id)
							switch_requested = True

						# 2) 运行中切换 leader（可选）
						if (not leader_changed) and (T_CHANGE_LEADER is not None) and (data.time >= T_CHANGE_LEADER):
							current_leader_id = int(NEW_LEADER_ID) % count
							formation_controller.request_change_leader(current_leader_id)
							leader_changed = True

						# 3) 解散编队（可选）
						if (not disbanded) and (T_DISBAND is not None) and (data.time >= T_DISBAND):
							formation_controller.request_disable_formation()
							disbanded = True

						# 4) 给当前 leader 写单机轨迹命令。
						#    - 编队开启时：仅 leader 由外部驱动，其余无人机由 controller 覆盖 dposd/psid。
						#    - 编队解散后：所有无人机恢复各自轨迹。
						if not disbanded:
							parameters[current_leader_id].dposd = qds_list[current_leader_id][time_num, [0, 1, 2]]
							parameters[current_leader_id].psid = qds_list[current_leader_id][time_num, 5]
						else:
							for drone_id in range(count):
								parameters[drone_id].dposd = qds_list[drone_id][time_num, [0, 1, 2]]
								parameters[drone_id].psid = qds_list[drone_id][time_num, 5]

						# 目标轨迹记录（用于画图）
						target_trajectories[current_leader_id][time_num] = qds_list[current_leader_id][time_num, [0, 1, 2]]
						try:
							if (not disbanded) and hasattr(formation, 'reference') and callable(getattr(formation, 'reference')):
								leader_pos = parameters[current_leader_id].pos
								leader_vel = parameters[current_leader_id].dpos
								leader_yaw = float(parameters[current_leader_id].psi)
								leader_yaw_rate = float(parameters[current_leader_id].omega[2])
								pos_ref_all, _ = formation.reference(
									leader_pos=leader_pos,
									leader_vel=leader_vel,
									leader_yaw=leader_yaw,
									leader_yaw_rate=leader_yaw_rate,
									leader_id=current_leader_id,
								)
								for i in range(count):
									target_trajectories[i][time_num] = pos_ref_all[i]
							elif disbanded:
								for i in range(count):
									target_trajectories[i][time_num] = qds_list[i][time_num, [0, 1, 2]]
						except Exception:
							pass

				# 推进编队切换状态机（简化版：只推进 blending 计时）
				formation_controller.update_switch(dt, parameters, formation)

				# 计算控制并仿真一步
				torques = formation_controller.control(parameters, formation)
				mujoco.mj_setState(model, data, torques, mujoco.mjtState.mjSTATE_CTRL)
				mujoco.mj_step(model, data)

				with viewer.lock():
					viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
				viewer.sync()

				time_until_next_step = model.opt.timestep - (time.time() - step_start)
				if time_until_next_step > 0:
					time.sleep(time_until_next_step)

				time_num += 1
				if time_num >= time_step_num:
					break

		# 绘制轨迹（可视化对比：切换后 followers 是否逐步贴近编队目标）
		self._plot_trajectories(trajectories, target_trajectories)

	def _plot_trajectories(self, actual_trajectories, target_trajectories=None):
		fig = plt.figure(figsize=(12, 10))
		ax = fig.add_subplot(111, projection='3d')

		colors = ['r', 'g', 'b', 'm']
		labels = [f'无人机 {i + 1}' for i in range(len(actual_trajectories))]

		for i, traj in enumerate(actual_trajectories):
			valid_actual = traj[~np.all(traj == 0, axis=1)]
			if len(valid_actual) == 0:
				continue
			ax.plot(valid_actual[:, 0], valid_actual[:, 1], valid_actual[:, 2],
					color=colors[i], label=f'{labels[i]} 实际轨迹')
			ax.scatter(valid_actual[0, 0], valid_actual[0, 1], valid_actual[0, 2],
					   color=colors[i], marker='o', s=60)
			ax.scatter(valid_actual[-1, 0], valid_actual[-1, 1], valid_actual[-1, 2],
					   color=colors[i], marker='x', s=60)

		if target_trajectories is not None:
			for i, traj in enumerate(target_trajectories):
				valid_target = traj[~np.all(traj == 0, axis=1)]
				if len(valid_target) == 0:
					continue
				ax.plot(valid_target[:, 0], valid_target[:, 1], valid_target[:, 2],
						color=colors[i], linestyle='--', alpha=0.45, label=f'{labels[i]} 目标轨迹')

		ax.set_xlabel('X 坐标 (m)')
		ax.set_ylabel('Y 坐标 (m)')
		ax.set_zlabel('Z 坐标 (m)')
		ax.set_title(f'四机：在 t={T_SWITCH}s 开启编队的过渡测试')
		ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
		ax.view_init(elev=30, azim=45)
		plt.tight_layout()
		if SHOW_PLOT:
			plt.show()
		else:
			fig.savefig(PLOT_SAVE_PATH, dpi=150)
			plt.close(fig)

