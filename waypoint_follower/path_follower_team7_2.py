# python3 src/waypoint_follower/waypoint_follower/path_follower.py /custom_odometry_topic

#!/usr/bin/env python3

import math
import os
import logging
import rclpy
from rclpy.node import Node
import time
from datetime import datetime

import matplotlib.pyplot as plt

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist, PoseStamped

import cvxpy as cp
import numpy as np

def euler_from_quaternion(x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians


def f(x, u):
    b = np.zeros(x.shape, dtype=np.float32)
    _, _, theta = x

    v, dtheta = u

    b[0] = v * np.cos(theta)
    b[1] = v * np.sin(theta)
    b[2] = dtheta

    return b

def df(x, u):
    A = np.zeros((x.shape[0], x.shape[0]), dtype=np.float32)
    B = np.zeros((x.shape[0], u.shape[0]), dtype=np.float32)

    _, _, theta = x
    v, dtheta = u

    # first row of A is for equation for x, second row y, third row theta
    A[0, 0] = 0
    A[0, 1] = 0
    A[0, 2] = v * (-np.sin(theta))
    A[1, 0] = 0
    A[1, 1] = 0
    A[1, 2] = v * np.cos(theta)
    A[2, 0] = 0
    A[2, 1] = 0
    A[2, 2] = 0

    B[0, 0] = np.cos(theta)
    B[0, 1] = 0
    B[1, 0] = np.sin(theta)
    B[1, 1] = 0
    B[2, 0] = 0
    B[2, 1] = 1

    return A, B

def linf(x, u, x2, u2):
    b = f(x,u)
    A, B = df(x,u)
    return b + A @ (x2 - x) + B * (u2 - u)

def make_and_solve_problem(Q, R, ref, x0, u0, dt, horizon, max_abs_speed):
    n_u = 2
    # x, y, theta
    n_state = 3

    if Q is None:
        Q = np.eye(n_state,) * 10 * dt
        Q[2,2] = 0

    if R is None:
        R = np.eye(n_u) * 8 * dt

    u = cp.Variable((horizon, n_u), name="u")
    x = cp.Variable((horizon + 1, n_state), name="x")

    constraints = [x0 == x[0]]

    b = f(x0, u0)
    A, B = df(x0, u0)


    cost = 0.0

    # prev_speed = u0
        # vx_base, vtheta_base = u[i]
        # vel_helper[0] = vx_base * cp.cos(x[i][2])
        # vel_helper[1] = vx_base * cp.sin(x[i][2])
        # vel_helper[2] = vtheta_base

    for i in range(horizon):

        next_target = ref[i]

        change_state = b + A @ (x[i] - x0) + B @ (u[i] - u0)

        next_state = x[i] + dt * change_state
        constraints.append(next_state == x[i + 1])

        cost += cp.quad_form(next_state - next_target, Q) + cp.quad_form(u[i], R)
        # prev_speed = cur_speed

    # constraints += [cp.abs(u) <= self.max_abs_vel_change]
    # constraints.append(cp.abs(u) <= max_abs_speed)
    prob = cp.Problem(cp.Minimize(cost), constraints)

    prob.solve(verbose=True)


    if prob.status != "optimal":
        logging.error(f"error in opto, {prob.status}")

        # u_debug = np.zeros((n+1, n_drones, n_u), dtype=np.float32)
        # u_debug[0] = u0
        # for i in range(n):
        #     u_debug[i+1] = u_debug[i-1] + np.array(u.value[num_controls_per_step*i:num_controls_per_step*i + num_controls_per_step]).reshape((n_drones, n_u))


        # x_debug = np.array(x.value).reshape((n+1, n_drones, n_state))
    
    np_u = np.array(u.value)[0]
    # next_control = np_u[[0,2]] # + u0
    return np_u, np.array(u.value), np.array(x.value)


def make_problem(Q, R, ref, x0, u0, dt: float, horizon: int):
    n_u = 2
    # x, y, theta
    n_state = 3

    if Q is None:
        Q = np.eye(n_state,) * 40 * dt
        Q[2,2] = 0

    if R is None:
        R = np.eye(n_u) * 8 * dt

    u = cp.Variable((horizon + 1, n_u), name="u")
    x = cp.Variable((horizon + 1, n_state), name="x")

    helper_A_x0 = cp.Variable(n_state, name="helper_A_x0")
    helper_B_u0 = cp.Variable(n_state, name="helper_B_u0")

    x0_param = cp.Parameter(n_state, name="x0")
    u0_param = cp.Parameter(n_u, name="u0")
    ref_param = cp.Parameter((horizon, n_state), name="ref")
    b_param = cp.Parameter(n_state, name="b")
    A_param = cp.Parameter((n_state, n_state), name="A")
    B_param = cp.Parameter((n_state, n_u), name="B")

    constraints = [x0_param == x[0]]
    constraints += [u0_param == u[0]]

    cost = 0.0

    constraints.append(helper_A_x0 == A_param @ x[0])
    constraints.append(helper_B_u0 == B_param @ u[0])
    for i in range(horizon):

        next_target = ref_param[i]

        change_state = b_param + A_param @ x[i] - helper_A_x0 + B_param @ u[i + 1] - helper_B_u0

        next_state = x[i] + dt * change_state
        constraints.append(next_state == x[i + 1])

        cost += cp.quad_form(next_state - next_target, Q) + cp.quad_form(u[i + 1], R)
        # prev_speed = cur_speed

    constraints.append(cp.abs(u[:, 0]) <= MAX_LINEAR_SPEED)
    constraints.append(cp.abs(u[:, 1]) <= MAX_ANGULAR_SPEED)
    
    prob = cp.Problem(cp.Minimize(cost), constraints)

    return prob, x0_param, u0_param, ref_param, b_param, A_param, B_param

def solve_problem(prob, x0, ref, u0):

    global x0_param, u0_param, ref_param, b_param, A_param, B_param

    b = f(x0, u0)
    A, B = df(x0, u0)

    ref_param.value = ref
    x0_param.value = x0
    u0_param.value = u0

    b_param.value = b
    A_param.value = A
    B_param.value = B

    prob.solve(verbose=True, enforce_dpp=True)

    if prob.status != "optimal":
        logging.error(f"error in opto, {prob.status}")
    
    u_var = prob.var_dict["u"]
    x_var = prob.var_dict["x"]
    np_u = np.array(u_var.value)[1]
    return np_u, np.array(u_var.value), np.array(x_var.value)


def get_arr_from_path(msg: Path, cur_time, target_seconds: float = 40.0):
    num_poses = len(msg.poses)
    # x, y, theta
    n_state = 3
    # plus time
    n_cols = n_state + 1

    traj = np.zeros((num_poses, n_cols), dtype=np.float32)
    for i in range(num_poses):
        pose = msg.poses[i].pose
        time = msg.poses[i].header.stamp
        time_sec = time.sec + time.nanosec / 1e9

        x, y = pose.position.x, pose.position.y

        quat_arr = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]

        rx, ry, rz = euler_from_quaternion(quat_arr[0], quat_arr[1], quat_arr[2], quat_arr[3])

        traj[i, 0:n_cols] = [time_sec, x, y, rz]
    # actually the time we recieve isn't valid, the publishers of the paths ignore it, so we do too here
    # and overwrite it

    print("cur time")
    print(cur_time)

    print("target seconds")
    print(target_seconds)
    
    print("seconds interpolated")
    print(np.linspace(0, target_seconds, num_poses))

    traj[:, 0] = np.linspace(0, target_seconds, num_poses) + cur_time

    print("first and last times")
    print(traj[0, 0])
    print(traj[num_poses-1, 0])

    # additionally theta is often ignored when the trajectory is published, hence it is not kinematically feasible
    # so we overwrite that to point to the next point
    for i in range(num_poses - 1):
        xy = traj[i, [1, 2]]
        xy_next = traj[i + 1, [1, 2]]
        diff = xy_next - xy
        orientation = np.arctan2(diff[1], diff[0])
        traj[i, 3] = orientation
    
    # last orientation to same as prior
    traj[-1, 3] = traj[-2, 3]

    return traj


DT = 0.1
HORIZON = 20
MAX_ABS_SPEED = 0.5
MAX_LINEAR_SPEED = 0.75
MAX_ANGULAR_SPEED = np.pi
PROBLEM = None

x0_param, u0_param, ref_param, b_param, A_param,  B_param = (None, None, None, None, None, None)
TARGET_SECONDS = 30.0

class PathFollower(Node):
    """
    A ROS2 node that uses a PD controller to follow a path (list of waypoints).
    Subscribes to /path for the target path and a specified odometry topic for the robot's current state,
    and publishes velocity commands to /cmd_vel.
    """

    # NOTE: CANNOT CHANGE
    def __init__(self):
        super().__init__('path_follower')

        self.setup_parameters()

        # sum of derivate
        self.sum_derivate = 0
        self.start_time = 0
        self.end_time = 0

        self.time_list = []
        self.error_x_list = []
        self.error_y_list = []
        self.error_theta_list = []
        self.actual_path_x = []
        self.actual_path_y = []

        self.plot_done = False

        # Path and waypoints
        self.path = None
        self.current_waypoint_index = 0

        # Previous errors (for derivative term)
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.prev_error_theta = 0.0

        # Robot current state
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_orientation = 0.0
        self.is_odom_received = False
        self.current_x_vel = 0.0
        self.current_y_vel = 0.0
        self.current_z_angular_vel = 0.0

        # Thresholds
        self.waypoint_threshold = 0.1  # Distance threshold to consider a waypoint reached
        self.orientation_threshold = 0.05  # Orientation threshold for final alignment

        # Publisher to cmd_vel
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.cmd_vel_pub.publish(Twist())
        # Subscriber to path
        if self.real_robot:
            print("Real Robot")
            # Subscriber to odometry (with the specified topic)
            self.odom_sub = self.create_subscription(
                Odometry,
                "/utlidar/robot_odom",
                self.odom_callback,
                10
            )

        else:
            # Subscriber to odometry (with the specified topic)
            self.odom_sub = self.create_subscription(
                Odometry,
                "/Odometry",
                self.odom_callback,
                10
            )


        self.path_sub = self.create_subscription(
            Path,
            '/planner/path',
            self.path_callback,
            10
        )

        # Timer to periodically publish velocity commands
        timer_period = 0.1  # [s] -> 10 Hz
        self.timer = self.create_timer(timer_period, self.control_loop_callback)

        self.prev_u = np.zeros((2,), dtype=np.float32)

    # NOTE: CANNOT CHANGE
    def normalize_angle(self, angle):
            """
            Normalize an angle to the range [-pi, pi].
            """
            while angle > math.pi:
                angle -= 2.0 * math.pi
            while angle < -math.pi:
                angle += 2.0 * math.pi
            return angle     

    # NOTE: CANNOT CHANGE
    def odom_callback(self, msg: Odometry):
        """
        Extracts and stores the robot's current pose from the odometry.
        """
        self.is_odom_received = True
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y


        self.actual_path_x.append(self.current_x)
        self.actual_path_y.append(self.current_y)

        self.current_x_vel = msg.twist.twist.linear.x
        self.current_y_vel = msg.twist.twist.linear.y
        self.current_z_angular_vel = msg.twist.twist.angular.z

        # Convert quaternion to yaw (theta)
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_orientation = math.atan2(siny_cosp, cosy_cosp)

    # NOTE: CANNOT CHANGE
    def check_ending(self):
        if not self.is_odom_received or self.path is None:
            self.get_logger().info("no odom or no path")
            self.cmd_vel_pub.publish(Twist())
            return True

        # Check if all waypoints are reached
        if self.current_waypoint_index >= len(self.path):
            if self.end_time < 1e-3:
                self.end_time = self.get_clock().now()
                self.end_time = self.end_time.nanoseconds / 1e9
            self.get_logger().info(f"----- All waypoints reached. Stopping. ---- ")
            self.get_logger().info(f"Sum of error: {self.sum_derivate:.3f}, Cost time: {self.end_time - self.start_time:.3f}s.")
            self.cmd_vel_pub.publish(Twist())  # Stop the robot


            if not self.plot_done:
                self.plot_results()
                self.plot_done = True

            if not self.plotted:

                self.plotted = True
                self.get_logger().info("saving plots")

                out_dir = os.path.join(os.environ["HOME"], "tmp", "plot")

                for i in range(len(self.plot_vars)):

                    _, next_times, cur_time, traj_ref, state_pred, x0 = self.plot_vars[i]

                    fig = plt.figure(figsize=(12,12))
                    gs = fig.add_gridspec(2,3)
                    ax_x = fig.add_subplot(gs[0, 0])
                    ax_y = fig.add_subplot(gs[0, 1])
                    ax_theta = fig.add_subplot(gs[0, 2])
                    ax_world = fig.add_subplot(gs[1,:])

                    axes_list = [ax_x, ax_y, ax_theta]
                    labels = ["x", "y", "theta"]
                    
                    # plot global ref, plot interpolated ref, # plot predicted state, # plot actual state
                    global_ref_times = self.np_path[:, 0]
                    interp_ref_times = next_times
                    for i in range(3):
                        label = labels[i]
                        axes_list[i].plot(global_ref_times, self.np_path[:, i + 1], label=f"global_ref_{label}")
                        axes_list[i].plot(interp_ref_times, traj_ref[:, i], label=f"interp_ref_{label}")
                        axes_list[i].plot(interp_ref_times, state_pred[1:, i], label=f"predicted_{label}")
                        axes_list[i].plot(cur_time, x0[i], marker="+", label=f"x0_{label}")
                        axes_list[i].set_xlabel("time")
                        axes_list[i].set_ylabel(label)
                        axes_list[i].legend()
                    ax_world.plot(self.np_path[:, 1], self.np_path[:, 2], label="global_traj")
                    ax_world.plot(traj_ref[:, 0], traj_ref[:, 1], label="interp_ref")
                    ax_world.plot(state_pred[:, 0], state_pred[:, 1], label="predicted_state")
                    ax_world.plot(x0[0], x0[1], marker="+", label=f"x0")
                    ax_world.set_xlabel("x")
                    ax_world.set_ylabel("y")
                    ax_world.legend()


                    date_now = datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
                    print(date_now) 

                    fignamepath = os.path.join(out_dir, f"{date_now}-{cur_time}.png")

                    print(f"Saving to {fignamepath}")
                    fig.savefig(fignamepath)
                    plt.close(fig)
                self.get_logger().info("done saving plots")

            return True
        
        return False 

    # NOTE: STUDENTS ARE ALLOWED TO ADD THEIR OWN FUNCTIONS
    def customized_functions(self):
        pass

    # NOTE: CAN CHANGE
    def setup_parameters(self):

        # PD Controller Gains (tune as necessary)
        self.Kp_linear = 1.0
        self.Kd_linear = 0.1
        self.Kp_angular = 1.0
        self.Kd_angular = 0.1

        self.declare_parameter("traj_time_sec", 30.0)
        self.declare_parameter("max_linear_vel", 0.5)
        self.declare_parameter("max_angular_vel", 1.0)
        self.declare_parameter("real_robot", False)
        self.declare_parameter("horizon", 45)
        self.real_robot = bool(self.get_parameter("real_robot").value)

        global MAX_LINEAR_SPEED, MAX_ANGULAR_SPEED, HORIZON

        MAX_LINEAR_SPEED = float(self.get_parameter("max_linear_vel").value)
        MAX_ANGULAR_SPEED = float(self.get_parameter("max_angular_vel").value)

        HORIZON = int(self.get_parameter("horizon").value)

        

        global TARGET_SECONDS
        self.traj_time_sec = float(self.get_parameter("traj_time_sec").value)
        TARGET_SECONDS = self.traj_time_sec

    # NOTE: CAN CHANGE
    def path_callback(self, msg: Path):
        """
        Updates the target path when a new message is received.
        """
        self.path = msg.poses  # List of PoseStamped messages
        self.current_waypoint_index = 0  # Reset to the first waypoint
        self.get_logger().info(f"Received new path with {len(self.path)} waypoints.")
        self.start_time = self.get_clock().now()
        self.start_time = self.start_time.nanoseconds / 1e9
        self.end_time = 0

        cur_time = self.get_clock().now().nanoseconds / 1e9
        np_path = get_arr_from_path(msg, cur_time, target_seconds=TARGET_SECONDS)
        global PROBLEM, x0_param, u0_param, ref_param, b_param, A_param,  B_param

        print("path")

        # PROBLEM, _, _, _, _ = make_problem(Q=None, R=None, ref=None, x0=None, u0=None, dt=DT, horizon=HORIZON, max_abs_speed=MAX_ABS_SPEED)
        PROBLEM, x0_param, u0_param, ref_param, b_param, A_param,  B_param = make_problem(Q=None, R=None, ref=None, x0=None, u0=None, dt=DT, horizon=HORIZON)
        self.np_path = np_path
        print(self.np_path)

        self.plot_vars = []
        self.plotted = False


    # NOTE: CAN CHANGE
    def control_loop_callback(self):
        """
        Periodic control loop callback. Computes PD control and publishes velocity commands.
        """
        if self.check_ending():
            return


        target_pose = self.path[self.current_waypoint_index].pose
        x_target = target_pose.position.x
        y_target = target_pose.position.y


        q = target_pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        orientation_target = math.atan2(siny_cosp, cosy_cosp)
        
        cur_time = self.get_clock().now()
        cur_time = cur_time.nanoseconds / 1e9


        next_times = (np.arange(HORIZON) + 1) * DT + cur_time
        traj_ref_x = np.interp(next_times, self.np_path[:, 0], self.np_path[:, 1])
        traj_ref_y = np.interp(next_times, self.np_path[:, 0], self.np_path[:, 2])
        traj_ref_theta = np.interp(next_times, self.np_path[:, 0], self.np_path[:, 3])
        traj_ref = np.hstack([traj_ref_x.reshape(len(next_times), 1), traj_ref_y.reshape(len(next_times), 1),
                              traj_ref_theta.reshape(len(next_times), 1)])

        self.current_waypoint_index = np.argmin(np.linalg.norm(self.np_path[:, [1, 2, 3]] - traj_ref[0], axis=1))
        
        if cur_time >= self.np_path[-1, 0]:
            self.current_waypoint_index = self.np_path.shape[0]

        x0 = np.array([self.current_x, self.current_y, self.current_orientation])

        x_target = traj_ref[0, 0]
        y_target = traj_ref[0, 1]
        error_x = x_target - self.current_x
        error_y = y_target - self.current_y
        error_theta = self.normalize_angle(math.atan2(error_y, error_x) - self.current_orientation) # degree

        error_orientation = self.normalize_angle(orientation_target - self.current_orientation)

        # 2) Compute derivative of errors
        derivative_x = error_x - self.prev_error_x
        derivative_y = error_y - self.prev_error_y
        derivative_theta = error_theta - self.prev_error_theta
        self.sum_derivate += math.fabs(derivative_x) + math.fabs(derivative_y) + math.fabs(derivative_theta) / 180.0 * math.pi
        
        print("ref traj")
        print(self.np_path)

        print("cur time")
        print(cur_time)

        print("next ref")
        print(traj_ref[0])

        print("xo")
        print(x0)
        
        # ctrl = solve_problem(PROBLEM, x0, traj_ref, None)
# make_and_solve_problem(Q, R, ref, x0, u0, dt, horizon, max_abs_speed)
        # ctrl, full_ctrl, state_pred = make_and_solve_problem(None, None, traj_ref, x0, self.prev_u, dt=DT, horizon=HORIZON, max_abs_speed=MAX_ABS_SPEED)
        global PROBLEM
        u0 = np.array([self.current_x_vel, self.current_z_angular_vel])
        print("u0")
        print(u0)
        ctrl, full_ctrl, state_pred = solve_problem(PROBLEM, x0, traj_ref, self.prev_u)
        print("next ctrl")
        print(ctrl)

        self.plot_vars.append([np.copy(self.np_path), np.copy(next_times), cur_time, np.copy(traj_ref), np.copy(state_pred), np.copy(x0)])


        current_time = time.time() - self.start_time
        self.time_list.append(current_time)
        self.error_x_list.append(error_x)
        self.error_y_list.append(error_y)
        self.error_theta_list.append(error_theta)

        # 6) Publish velocity commands
        twist_msg = Twist()

        twist_msg.linear.x = ctrl[0]
        twist_msg.angular.z = ctrl[1]

        self.cmd_vel_pub.publish(twist_msg)

        self.prev_u = ctrl


    def plot_results(self):
        """
        Plots the recorded errors vs. time and the actual trajectory versus the planned path.
        """
        # Plot errors vs. time
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.time_list, self.error_x_list, label='Error X')
        plt.plot(self.time_list, self.error_y_list, label='Error Y')
        plt.xlabel('Time (s)')
        plt.ylabel('Error')
        plt.title('Errors vs. Time')
        plt.legend()
        plt.grid(True)

        # Plot the trajectory: actual vs. planned (if available)
        plt.subplot(1, 2, 2)
        plt.plot(self.actual_path_x, self.actual_path_y, 'b-', label='Actual Trajectory')
        if self.path is not None and len(self.path) > 0:
            planned_x = [pose.pose.position.x for pose in self.path]
            planned_y = [pose.pose.position.y for pose in self.path]
            plt.plot(planned_x, planned_y, 'ro--', label='Planned Path')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Trajectory Comparison')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

def main():
    import sys
    args = sys.argv 

    rclpy.init(args=args)

    # Specify the odometry topic (default is '/Odometry')
    node = PathFollower()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt detected, shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
