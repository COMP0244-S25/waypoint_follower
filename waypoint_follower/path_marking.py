#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
import time

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped


class PathMarker(Node):
    """
    A ROS2 node that tracks path following progress and logs performance metrics.
    Subscribes to the same odometry and path topics as the original controller,
    but doesn't publish any commands.
    """

    def __init__(self, odometry_topic):
        super().__init__('path_marker')

        # Tracking parameters
        self.waypoint_threshold = 0.1  # Must match original controller
        self.orientation_threshold = 0.05

        # Tracking variables
        self.sum_derivate = 0
        self.start_time = 0
        self.end_time = 0
        self.path = None
        self.current_waypoint_index = 0
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.prev_error_theta = 0.0

        # Robot state
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_orientation = 0.0
        self.is_odom_received = False

        # Subscribers (same as original)
        self.odom_sub = self.create_subscription(
            Odometry,
            odometry_topic,
            self.odom_callback,
            10
        )

        self.path_sub = self.create_subscription(
            Path,
            '/planner/path',
            self.path_callback,
            10
        )

        # Same timer interval as original (10Hz)
        self.create_timer(0.1, self.control_loop_callback)
        self.get_logger().info(f"Path Marker started. Tracking odometry on: {odometry_topic}")

    def path_callback(self, msg: Path):
        """
        Updates the target path when a new message is received.
        """
        self.path = msg.poses
        self.current_waypoint_index = 0
        self.get_logger().info(f"Tracking new path with {len(self.path)} waypoints")
        self.start_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        self.sum_path_derivate = 0
        self.end_time = 0

    def odom_callback(self, msg: Odometry):
        """
        Extracts and stores the robot's current pose from the odometry.
        """
        self.is_odom_received = True
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_orientation = math.atan2(siny_cosp, cosy_cosp)

    def check_ending(self):
        if not self.is_odom_received or self.path is None:
            return True

        if self.current_waypoint_index >= len(self.path):
            if self.end_time < 1e-3:
                self.end_time = time.time()
            self.get_logger().info(f"----- All waypoints reached. Stopping. ---- ")
            self.get_logger().info(f"Sum of error: {self.sum_derivate:.3f}, Cost time: {self.end_time - self.start_time:.3f}s.")
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
        if self.check_ending():
            return            

        target_pose = self.path[self.current_waypoint_index].pose
        x_target = target_pose.position.x
        y_target = target_pose.position.y

        # Identical error calculations to original
        error_x = x_target - self.current_x
        error_y = y_target - self.current_y
        error_theta = self.normalize_angle(math.atan2(error_y, error_x) - self.current_orientation)

        # Identical derivative calculations to original
        derivative_x = error_x - self.prev_error_x
        derivative_y = error_y - self.prev_error_y
        derivative_theta = error_theta - self.prev_error_theta
        
        # Identical error accumulation to original (including potential conversion artifact)
        self.compute_traj_error()

        # Update previous errors
        self.prev_error_x = error_x
        self.prev_error_y = error_y
        self.prev_error_theta = error_theta

        # Waypoint progression check (same as original)
        distance = math.hypot(error_x, error_y)
        if distance < self.waypoint_threshold:
            self.current_waypoint_index += 1

    def normalize_angle(self, angle):
        """Identical angle normalization to original"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

def main():
    rclpy.init()
    node = PathMarker('/Odometry')  # Use same default topic as original
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
