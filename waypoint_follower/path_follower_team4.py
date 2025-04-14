#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
import time

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist, PoseStamped
from math import comb

class PathFollower(Node):
    """
    A ROS2 node that uses a PD controller to follow a path (list of waypoints).
    Subscribes to /planner/path for the target path and a specified odometry topic for the robot's current state,
    and publishes velocity commands to /cmd_vel.
    """

    def __init__(self, odometry_topic):
        super().__init__('path_follower')

        self.setup_parameters()

        # Derivative sum for debugging or analysis
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
        self.prev_angular_z = 0.0

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
        self.get_logger().warn('++++++ Team-4 Starting Testing')
    
    def bezier_curve(self, control_points, num_points=100):
        """
        Generate a Bezier curve from the given control points.
        :param control_points: A list of (x, y) tuples.
        :param num_points: Number of points to sample on the Bezier curve.
        :return: A list of (x, y) points along the Bezier curve.
        """
        n = len(control_points) - 1
        bezier_points = []

        for t in [i / (num_points - 1) for i in range(num_points)]:
            x = 0.0
            y = 0.0
            for i, (px, py) in enumerate(control_points):
                coeff = comb(n, i) * ((1 - t) ** (n - i)) * (t ** i)
                x += coeff * px
                y += coeff * py
            bezier_points.append((x, y))

        return bezier_points

    def is_turn_segment(self, p1, p2, p3, angle_threshold=70.0):
        """
        Determine if the segment formed by three consecutive points is a turn,
        by checking the angle between the two vectors. If the angle is greater
        than angle_threshold, it is considered a turn.
        """
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        mag1 = math.hypot(*v1)
        mag2 = math.hypot(*v2)
        if mag1 == 0 or mag2 == 0:
            return False
        cos_theta = max(min(dot / (mag1 * mag2), 1.0), -1.0)
        angle = math.degrees(math.acos(cos_theta))
        return angle > angle_threshold

    def adaptive_path_smoothing(self, poses, segment_size=4, num_points_per_segment=3):
        """
        Perform adaptive path smoothing. For any set of three consecutive points
        that form a turn (based on is_turn_segment), we apply a Bezier curve fit
        to generate intermediate points for a smoother turn.
        Otherwise, the point is directly added to the smoothed path.
        """
        if len(poses) < 3:
            return poses

        smoothed_poses = []

        i = 0
        while i < len(poses) - 2:
            p1 = poses[i].pose.position
            p2 = poses[i + 1].pose.position
            p3 = poses[i + 2].pose.position

            point1 = (p1.x, p1.y)
            point2 = (p2.x, p2.y)
            point3 = (p3.x, p3.y)

            if self.is_turn_segment(point1, point2, point3):
                # This is a turn segment; apply a Bezier fit for these 3 points
                control_points = [point1, point2, point3]
                bezier_xy = self.bezier_curve(control_points, num_points=num_points_per_segment)

                for x, y in bezier_xy[:-1]:
                    pose = PoseStamped()
                    pose.pose.position.x = x
                    pose.pose.position.y = y
                    smoothed_poses.append(pose)
                i += 2
            else:
                # Otherwise, add the point directly
                pose = PoseStamped()
                pose.pose.position.x = point1[0]
                pose.pose.position.y = point1[1]
                smoothed_poses.append(pose)
                i += 1

        # Add the final points
        for j in range(i, len(poses)):
            smoothed_poses.append(poses[j])

        return smoothed_poses


    def setup_parameters(self):
        """
        Set up initial parameters for velocity limits and PD gains.
        Modify these as needed for your robot or environment.
        """
        # Maximum velocities
        self.max_linear_vel = 1.2  # m/s
        self.max_angular_vel = 2.5 # rad/s

        # PD Controller Gains (tune as necessary)
        self.Kp_linear = 1.5
        self.Kd_linear = 0.1
        self.Kp_angular = 3.0
        self.Kd_angular = 0.3

    def path_callback(self, msg: Path):
        """
        Callback for the incoming path. Stores the original path and generates
        a smoothed path if needed. Resets the waypoint index and timing.
        """
        self.original_path = msg.poses
        self.sum_path_derivate = 0.0
        self.path = self.adaptive_path_smoothing(msg.poses, segment_size=4, num_points_per_segment=2)
        self.current_waypoint_index = 0
        self.get_logger().info(f"Received new path with {len(self.path)} waypoints.")
        self.start_time = time.time()
        self.end_time = 0

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
        
    
    def control_loop_callback(self):
        """
        Periodic control loop callback. Computes PD control and publishes velocity commands.
        """
        # Thresholds
        self.waypoint_threshold = 0.1  # Distance threshold to consider a waypoint reached
        self.orientation_threshold = 0.05  # Orientation threshold for final alignment

        if self.check_ending():
            return
        
        if not self.is_odom_received or self.path is None:
            self.cmd_vel_pub.publish(Twist())
            return

        # Get the current target waypoint
        target_pose = self.path[self.current_waypoint_index].pose
        x_target = target_pose.position.x
        y_target = target_pose.position.y

        # Convert quaternion to yaw (theta) for the target orientation
        q = target_pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        orientation_target = math.atan2(siny_cosp, cosy_cosp)

        # 1) Compute errors
        error_x = x_target - self.current_x
        error_y = y_target - self.current_y
        error_theta = self.normalize_angle(math.atan2(error_y, error_x) - self.current_orientation)
        error_orientation = self.normalize_angle(orientation_target - self.current_orientation)

        # 2) Compute derivative of errors
        derivative_x = error_x - self.prev_error_x
        derivative_y = error_y - self.prev_error_y
        derivative_theta = error_theta - self.prev_error_theta
        self.compute_traj_error()

        # 3) PD control for linear velocities (x, y)
        vx = self.Kp_linear * error_x + self.Kd_linear * derivative_x
        vy = self.Kp_linear * error_y + self.Kd_linear * derivative_y

        # 4) PD control for angular velocity
        vtheta = self.Kp_angular * error_theta + self.Kd_angular * derivative_theta

        # 5) Update previous error terms
        self.prev_error_x = error_x
        self.prev_error_y = error_y
        self.prev_error_theta = error_theta
        
        # 6) Publish velocity commands
        twist_msg = Twist()

        # Check if the robot has reached the current waypoint
        distance_to_waypoint = math.hypot(error_x, error_y)
        if distance_to_waypoint < self.waypoint_threshold:
            self.current_waypoint_index += 1
            self.get_logger().info(f"Reached waypoint {self.current_waypoint_index - 1}. Moving to the next one.")
        else:
            # If the angular error is large, consider reducing linear speed so the robot can turn
            if abs(error_theta) > 1.0 and distance_to_waypoint > self.waypoint_threshold:
                angle_error = abs(error_theta)
                # Scale linear speed according to the angular error
                scale = max(0.1, 1.0 - angle_error * 0.8) 
                linear_speed = min(math.hypot(vx, vy), self.max_linear_vel) * scale
                angular_speed = min(vtheta, self.max_angular_vel)

                # Apply low-pass filter to angular velocity
                alpha = 0.3
                filtered_angular_z = alpha * angular_speed + (1 - alpha) * self.prev_angular_z
                self.prev_angular_z = filtered_angular_z

                twist_msg.linear.x = linear_speed
                twist_msg.angular.z = filtered_angular_z

                self.get_logger().info(
                    f"[Large turn] Turning while creeping forward — angle error: {angle_error:.2f}, scale: {scale:.2f}"
                )
            else:
                # Dynamic speed scaling: the larger the angular error, the lower the forward speed
                angle_error = abs(error_theta)
                scale = max(0.4, 1.0 - angle_error)

                linear_speed = min(math.hypot(vx, vy), self.max_linear_vel) * scale
                angular_speed = min(vtheta, self.max_angular_vel)

                # Raw angular velocity
                raw_angular_z = angular_speed

                # Low-pass filter
                alpha = 0.3  # smaller -> smoother, larger -> more responsive
                filtered_angular_z = alpha * raw_angular_z + (1 - alpha) * self.prev_angular_z
                self.prev_angular_z = filtered_angular_z

                twist_msg.linear.x = linear_speed
                twist_msg.angular.z = filtered_angular_z

                self.get_logger().info(f"Turning while walking — Angle error: {angle_error:.2f}, scale: {scale:.2f}")
                self.get_logger().info("Moving forward")

        self.cmd_vel_pub.publish(twist_msg)

    def normalize_angle(self, angle):
        """
        Normalize an angle to the range [-pi, pi].
        """
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


def main():
    import sys
    args = sys.argv 

    rclpy.init(args=args)

    # Specify the odometry topic (default is '/Odometry')
    odometry_topic = '/Odometry'  # Default value
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
