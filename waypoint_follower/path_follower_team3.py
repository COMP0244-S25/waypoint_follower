# python3 src/waypoint_follower/waypoint_follower/path_follower.py /custom_odometry_topic

#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
import time

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist, PoseStamped


class PathFollower(Node):
    """
    A ROS2 node that uses a PD controller to follow a path (list of waypoints).
    Subscribes to /path for the target path and a specified odometry topic for the robot's current state,
    and publishes velocity commands to /cmd_vel.
    """

    # NOTE: CANNOT CHANGE
    def __init__(self, odometry_topic):
        super().__init__('path_follower')

        self.setup_parameters()

        # sum of derivate
        self.sum_derivate = 0
        self.start_time = 0
        self.end_time = 0

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

        # Thresholds
        self.waypoint_threshold = 0.1  # Distance threshold to consider a waypoint reached
        self.orientation_threshold = 0.05  # Orientation threshold for final alignment

        # Publisher to cmd_vel
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.cmd_vel_pub.publish(Twist())

        # Subscriber to odometry (with the specified topic)
        self.odom_sub = self.create_subscription(
            Odometry,
            odometry_topic,
            self.odom_callback,
            10
        )

        # Subscriber to path
        self.path_sub = self.create_subscription(
            Path,
            '/planner/path',
            self.path_callback,
            10
        )

        # Timer to periodically publish velocity commands
        timer_period = 0.1  # [s] -> 10 Hz
        self.timer = self.create_timer(timer_period, self.control_loop_callback)
        self.get_logger().info(f"Path Follower node started. Subscribing to odometry topic: {odometry_topic}")
        self.get_logger().warn('++++++ Team-3 Starting Testing')

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

        # Convert quaternion to yaw (theta)
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_orientation = math.atan2(siny_cosp, cosy_cosp)

    # NOTE: CANNOT CHANGE
    def check_ending(self):
        if not self.is_odom_received or self.path is None:
            self.cmd_vel_pub.publish(Twist())
            return True

        # Check if all waypoints are reached
        if self.current_waypoint_index >= len(self.path):
            if self.end_time < 1e-3:
                self.end_time = time.time()
            self.get_logger().info(f"----- All waypoints reached. Stopping. ---- ")
            self.get_logger().info(f"Sum of error: {self.sum_derivate:.3f}, Cost time: {self.end_time - self.start_time:.3f}s.")
            self.cmd_vel_pub.publish(Twist())  # Stop the robot
            return True
        
        return False 

    # NOTE: CANNOT CHANGE
    def compute_traj_error(self):
        ################################## Error Computation
        # NOTE: path error is defined as real_path_length / perfect_path_length
        ideal_path_length = 0.0
        for i in range(len(self.path) - 1):
            wp1 = self.path[i].pose.position
            wp2 = self.path[i + 1].pose.position
            dx = wp2.x - wp1.x
            dy = wp2.y - wp1.y
            ideal_path_length += math.hypot(dx, dy)

        # Update total path length
        if not hasattr(self, 'previous_x'):
            self.previous_x = self.current_x
            self.previous_y = self.current_y
        else:
            dx = self.current_x - self.previous_x
            dy = self.current_y - self.previous_y
            self.sum_path_derivate += math.hypot(dx, dy)
            self.previous_x = self.current_x
            self.previous_y = self.current_y

        duration = time.time() - self.start_time
        if ideal_path_length > 0:
            self.sum_derivate = self.sum_path_derivate / ideal_path_length
        else:
            self.sum_derivate = 0.0

        self.get_logger().info(f"Sum of error: {self.sum_derivate:.3f}, Cost time: {time.time() - self.start_time:.3f}s.")

    # NOTE: STUDENTS ARE ALLOWED TO ADD THEIR OWN FUNCTIONS
    def customized_functions(self):
        pass

    # NOTE: CAN CHANGE
    def setup_parameters(self):
        # Max speed parameters
        self.max_linear_vel = 1.0  # m/s
        self.max_angular_vel = 1.0 # rad/s

        # PD controller gains
        self.Kp_linear = 2.0
        self.Kd_linear = 0.2
        self.Kp_angular = 3.0
        self.Kd_angular = 0.3

        # Large turn detection and low-pass filter settings
        self.large_turn_threshold = 0.5  # Large turn angle
        self.velocity_smoothing_alpha = 0.5  # Low-pass filter factor (0~1), closer to 1 means smoother
        self.prev_linear_x = 0.0
        self.prev_angular_z = 0.0

    # NOTE: CAN CHANGE
    def compute_quadratic_bezier(self, p0, p1, p2, t):
        """Return a point on a quadratic Bezier curve at parameter t."""
        x = (1 - t)**2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0]
        y = (1 - t)**2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1]
        return x, y

    # NOTE: CAN CHANGE
    def smooth_corners(self, original_poses, segment_size=5):
        """Smooth path using quadratic Bezier interpolation."""
        new_poses = []
        n = len(original_poses)
        if n < 3:
            return original_poses

        i = 0
        while i < n - 2:
            p0 = original_poses[i].pose.position
            p1 = original_poses[i + 1].pose.position
            p2 = original_poses[i + 2].pose.position

            for s in range(segment_size):
                t = s / (segment_size - 1)
                bx, by = self.compute_quadratic_bezier((p0.x, p0.y), (p1.x, p1.y), (p2.x, p2.y), t)
                new_pose = PoseStamped()
                new_pose.pose.position.x = bx
                new_pose.pose.position.y = by
                new_poses.append(new_pose)

            i += 3

        while i < n:
            new_poses.append(original_poses[i])
            i += 1

        return new_poses

    # NOTE: CAN CHANGE
    def path_callback(self, msg: Path):
        """Receives and processes the path from the planner."""
        self.original_path = msg.poses  # Save original unsmoothed path

        # Apply Bezier smoothing
        bezier_smoothed = self.smooth_corners(msg.poses, segment_size=5)
        self.path = bezier_smoothed
        self.current_waypoint_index = 0

        self.get_logger().info(f"Received new path with {len(self.path)} waypoints (after bezier smoothing).")
        self.start_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        self.sum_path_derivate = 0
        self.end_time = 0

    # NOTE: CAN CHANGE
    def control_loop_callback(self):
        """Main control loop to compute and send velocity commands."""
        if self.check_ending():
            return

        # Get current target waypoint
        target_pose = self.path[self.current_waypoint_index].pose
        x_target = target_pose.position.x
        y_target = target_pose.position.y

        # Convert orientation of target to yaw
        q = target_pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        orientation_target = math.atan2(siny_cosp, cosy_cosp)

        # 1) Position and angle errors
        error_x = x_target - self.current_x
        error_y = y_target - self.current_y
        error_theta = self.normalize_angle(math.atan2(error_y, error_x) - self.current_orientation)
        error_orientation = self.normalize_angle(orientation_target - self.current_orientation)

        # 2) Derivative terms
        derivative_x = error_x - self.prev_error_x
        derivative_y = error_y - self.prev_error_y
        derivative_theta = error_theta - self.prev_error_theta

        # 3) PD control for linear velocity
        vx = self.Kp_linear * error_x + self.Kd_linear * derivative_x
        vy = self.Kp_linear * error_y + self.Kd_linear * derivative_y

        # 4) PD control for angular velocity
        vtheta = self.Kp_angular * error_theta + self.Kd_angular * derivative_theta

        # 5) Save previous errors
        self.prev_error_x = error_x
        self.prev_error_y = error_y
        self.prev_error_theta = error_theta

        # 6) Check if waypoint reached or large turn needed
        twist_msg = Twist()
        distance_to_waypoint = math.hypot(error_x, error_y)
        if distance_to_waypoint < self.waypoint_threshold:
            self.current_waypoint_index += 1
            self.get_logger().info(f"Reached waypoint {self.current_waypoint_index - 1}. Moving to the next one.")
        else:
            if abs(error_theta) > self.large_turn_threshold:
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = max(min(vtheta, self.max_angular_vel), -self.max_angular_vel)
                self.get_logger().info("Large turn: rotating in place")
            else:
                desired_linear = math.hypot(vx, vy)
                desired_angular = vtheta

                desired_linear = max(min(desired_linear, self.max_linear_vel), -self.max_linear_vel)
                desired_angular = max(min(desired_angular, self.max_angular_vel), -self.max_angular_vel)

                twist_msg.linear.x = desired_linear
                twist_msg.angular.z = desired_angular
                self.get_logger().info("Moving forward")

        # 7) Apply low-pass filter to smooth velocities
        alpha = self.velocity_smoothing_alpha
        filtered_linear_x = alpha * self.prev_linear_x + (1 - alpha) * twist_msg.linear.x
        filtered_angular_z = alpha * self.prev_angular_z + (1 - alpha) * twist_msg.angular.z

        twist_msg.linear.x = filtered_linear_x
        twist_msg.angular.z = filtered_angular_z

        self.prev_linear_x = filtered_linear_x
        self.prev_angular_z = filtered_angular_z

        # 8) Publish velocity command
        self.cmd_vel_pub.publish(twist_msg)

        # Compute trajectory error using original path (ideal path)
        if hasattr(self, 'original_path') and self.original_path is not None:
            smoothed_path = self.path
            self.path = self.original_path
            self.compute_traj_error()
            self.path = smoothed_path
        else:
            self.compute_traj_error()


def main():
    import sys
    args = sys.argv

    rclpy.init(args=args)

    odometry_topic = '/Odometry'  # Default topic
    node = PathFollower(odometry_topic)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt detected, shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
