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
        self.waypoint_threshold = 0.1 # Distance threshold to consider a waypoint reached
        self.orientation_threshold = 0.05  # Orientation threshold for final alignment

        # Publisher to cmd_vel
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 50)
        self.cmd_vel_pub.publish(Twist())

        # Subscriber to odometry (with the specified topic)
        self.odom_sub = self.create_subscription(
            Odometry,
            odometry_topic,
            self.odom_callback,
            50
        )

        # Subscriber to path
        self.path_sub = self.create_subscription(
            Path,
            '/planner/path',
            self.path_callback,
            50
        )

        # Timer to periodically publish velocity commands
        timer_period = 0.05  # [s] -> 10 Hz
        self.timer = self.create_timer(timer_period, self.control_loop_callback)
        self.get_logger().info(f"Path Follower node started. Subscribing to odometry topic: {odometry_topic}")
        self.get_logger().warn('++++++ Team-8H Starting Testing')

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
        # Maximum velocities
        self.max_linear_vel = 1.5   # meter
        self.max_angular_vel = 3.5  # rad

        # PD Controller Gains (tune as necessary)
        self.Kp_linear = 1.5
        self.Kd_linear = 0.3
        self.Kp_angular = 1.5
        self.Kd_angular = 0.3

    # NOTE: CAN CHANGE
    def path_callback(self, msg: Path):
        """
        Updates the target path when a new message is received.
        """
        self.path = msg.poses  # List of PoseStamped messages
        self.current_waypoint_index = 0  # Reset to the first waypoint
        self.get_logger().info(f"Received new path with {len(self.path)} waypoints.")
        self.start_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        self.sum_path_derivate = 0
        self.end_time = 0

    # NOTE: CAN CHANGE
    def control_loop_callback(self):
        """
        Periodic control loop callback. Computes PD control and publishes velocity commands,
        and stops once the final waypoint (and orientation) is reached.
        """
        # Check if the robot has reached the current waypoint
        if self.check_ending():
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
        distance_to_waypoint = math.hypot(error_x, error_y)

        # Heading we want: angle from robot to the waypoint
        heading_to_waypoint = math.atan2(error_y, error_x)
        # Heading error: how far we must turn from current heading
        error_theta = self.normalize_angle(heading_to_waypoint - self.current_orientation)
        # Orientation error is how far from the waypoint's final orientation
        error_orientation = self.normalize_angle(orientation_target - self.current_orientation)

        # 2) Compute derivatives
        derivative_x = error_x - self.prev_error_x
        derivative_y = error_y - self.prev_error_y
        derivative_theta = error_theta - self.prev_error_theta

        # Update total path error ratio
        self.compute_traj_error()

        # 3) PD control for linear velocities (x, y)
        raw_vx = self.Kp_linear * error_x + self.Kd_linear * derivative_x
        raw_vy = self.Kp_linear * error_y + self.Kd_linear * derivative_y
        linear_speed = math.hypot(raw_vx, raw_vy)

        # Scale by how aligned the robot is to the waypoint
        alignment_factor = math.cos(error_theta)
        if alignment_factor < 0:
            alignment_factor = 0.0

        if abs(error_theta) > 0.7:  # Reduce speed for larger turns
            linear_speed *= 0.5
        else:
            linear_speed *= alignment_factor

        # 4) PD control for angular velocity
        vtheta = self.Kp_angular * error_theta + self.Kd_angular * derivative_theta

        # 5) Update previous errors
        self.prev_error_x = error_x
        self.prev_error_y = error_y
        self.prev_error_theta = error_theta

        twist_msg = Twist()

        # Check if we are at the final waypoint
        at_final_waypoint = (self.current_waypoint_index == len(self.path) - 1)

        # Check if the robot has reached the current waypoint (position)
        if distance_to_waypoint < self.waypoint_threshold:
            elapsed_time = time.time() - self.start_time  # Calculate elapsed time
            message = (
                f"Reached waypoint {self.current_waypoint_index}. "
                f"Sum of errors: {self.sum_derivate:.3f}, "
                f"Elapsed time: {elapsed_time:.3f} seconds."
            )
            self.get_logger().info(message)

            # Store the message for later printing
            if not hasattr(self, 'waypoint_messages'):
                self.waypoint_messages = []
            self.waypoint_messages.append(message)

            # If we are at the final waypoint, stop further actions
            if at_final_waypoint:
                self.get_logger().info(
                    f"Final waypoint (#{self.current_waypoint_index}) reached. Stopping all actions."
                )
                # Print all stored messages
                self.get_logger().info("Summary of all waypoints:")
                for msg in self.waypoint_messages:
                    self.get_logger().info(msg)

                self.cmd_vel_pub.publish(Twist())  # Stop the robot
                self.current_waypoint_index += 1  # Increment here to reflect the final waypoint
                return  # Exit the control loop callback
            else:
                # Not the final waypoint => proceed to the next
                self.get_logger().info(f"Waypoint #{self.current_waypoint_index} reached, moving to the next one.")
                self.current_waypoint_index += 1

        else:
            # Not within threshold -> move toward the waypoint
            # If heading error is large, rotate in place first
            if abs(error_theta) > 0.7:  # about 40 degrees
                twist_msg.angular.z = max(-self.max_angular_vel, min(vtheta * 2.5, self.max_angular_vel))
                twist_msg.linear.x = 0.0
                self.get_logger().info("Heading error large -> rotating in place.")
            elif abs(error_theta) > 0.3:  
                twist_msg.angular.z = max(-self.max_angular_vel, min(vtheta * 3.5, self.max_angular_vel))
                twist_msg.linear.x = max(-self.max_linear_vel, min(linear_speed * 0.2, self.max_linear_vel))
                self.get_logger().info("Heading error middle.")
            else:
                # Move forward with scaled linear speed
                twist_msg.angular.z = max(-self.max_angular_vel, min(vtheta * 2.5, self.max_angular_vel))
                twist_msg.linear.x = max(-self.max_linear_vel, min(linear_speed, self.max_linear_vel))
                self.get_logger().info("Moving forward (with heading-based scaling).")

        # Publish Twist
        self.cmd_vel_pub.publish(twist_msg)

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
