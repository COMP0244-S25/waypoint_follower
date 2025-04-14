# python3 src/waypoint_follower/waypoint_follower/path_follower.py /custom_odometry_topic

#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
import time

import numpy as np
from scipy.interpolate import splprep, splev

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist, PoseStamped, Quaternion

def yaw_to_quaternion(yaw):
    """Convert a yaw angle (in radians) to a quaternion (roll and pitch assumed to be 0)."""
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q

def spline_to_poses(spline_tck, num_points=500):
    """
    Given a spline (spline_tck) computed from splprep,
    generate a list of PoseStamped messages corresponding to a dense set of points
    along the spline. The orientation is computed using the first derivative.
    """
    # Generate dense parameter values
    u_fine = np.linspace(0, 1, num=num_points)
    # Evaluate the spline at these parameters
    x_fine, y_fine = splev(u_fine, spline_tck)
    dense_points = np.vstack((x_fine, y_fine)).T  # shape: (num_points, 2)

    poses = []
    for i in range(len(dense_points)):
        pose = PoseStamped()
        # Set position
        pose.pose.position.x = dense_points[i, 0]
        pose.pose.position.y = dense_points[i, 1]
        
        # Compute derivative at the same parameter value for orientation
        dx, dy = splev(u_fine[i], spline_tck, der=1)
        yaw = math.atan2(dy, dx)
        pose.pose.orientation = yaw_to_quaternion(yaw)
        
        poses.append(pose)
    return poses

class PathFollower(Node):
    """
    A ROS2 node that uses a Pure Pursuit controller to follow a path.
    Subscribes to /path for the target path and a specified odometry topic for the robot's current state,
    and publishes velocity commands to /cmd_vel.
    """

    def __init__(self, odometry_topic):
        super().__init__('path_follower')

        self.setup_parameters()

        # sum of derivate for evaluation
        self.sum_derivate = 0
        self.start_time = 0
        self.end_time = 0

        # Path and waypoints
        self.path = None
        self.spline_path = None
        self.lookahead_index = 0

        # Previous errors (for derivative term)
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.prev_error_theta = 0.0

        # Robot current state
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_orientation = 0.0
        self.is_odom_received = False

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
        self.get_logger().info(f"Pure Pursuit Path Follower node started. Subscribing to odometry topic: {odometry_topic}")
        self.get_logger().warn('++++++ Team-1 Starting Testing')

    def setup_parameters(self):
        # Maximum velocities
        self.max_linear_vel = 0.5  # meters/sec
        self.max_angular_vel = 1.0  # rad/sec
        
        # Pure Pursuit parameters
        self.lookahead_distance = 0.3  # meters
        self.min_lookahead_distance = 0.2  # meters
        self.max_lookahead_distance = 0.6 # meters
        
        # NEW: More aggressive steering
        self.steering_gain = 0.8  # Increased from 1.2
        
        # NEW: Speed based on distance to lookahead
        self.min_speed_factor = 0.5  # Minimum speed as fraction of max speed
        
        # Final waypoint thresholds
        self.goal_threshold = 0.05  # meters

    def path_callback(self, msg: Path):
        """
        Callback for receiving a new path.
        Computes the spline representation from the discrete waypoints and stores it.
        """
        self.path = msg.poses  # List of PoseStamped messages
        if self.path is None or len(self.path) < 2:
            self.get_logger().warn("Received path has fewer than 2 waypoints. Cannot compute spline.")
            return

        # Compute the spline from the discrete waypoints
        waypoints = np.array([[wp.pose.position.x, wp.pose.position.y] for wp in self.path])
        self.spline_tck, self.spline_u = splprep(waypoints.T, s=0)
        self.get_logger().info(f"Received new path with {len(self.path)} waypoints; spline computed.")
        
        # Convert the spline back into a dense list of PoseStamped messages
        self.lookahead_index = 0
        self.spline_path = spline_to_poses(self.spline_tck, num_points=1000)
        self.get_logger().info(f"Spline converted to {len(self.spline_path)} poses.")
        
        self.start_time = time.time()
        self.end_time = 0.0

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

    def find_lookahead_point(self):
        """
        Find the lookahead point along the path that is the lookahead distance away from the robot.
        Returns the lookahead point, its index in the path, and the minimum distance to the path.
        """
        # Find the closest point on the path first to estimate cross-track error
        min_dist = float('inf')
        closest_idx = self.lookahead_index
        
        for i in range(self.lookahead_index, min(self.lookahead_index + 20, len(self.spline_path))):
            point = self.spline_path[i].pose.position
            dist = math.hypot(point.x - self.current_x, point.y - self.current_y)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Adjust lookahead more aggressively based on cross-track error.
        # Increase the reduction factor from 0.5 to 1.0 to shorten the lookahead when error is high.
        adjusted_lookahead = max(
            self.min_lookahead_distance,
            min(self.lookahead_distance, self.lookahead_distance - min_dist * 1.0)
        )
        
        # Start searching from the closest index for a point at least the adjusted lookahead distance away
        for i in range(closest_idx, len(self.spline_path)):
            point = self.spline_path[i].pose.position
            distance = math.hypot(point.x - self.current_x, point.y - self.current_y)
            if distance >= adjusted_lookahead:
                return self.spline_path[i].pose, i, min_dist
        
        # If we've reached the end of the path, return the last point
        if len(self.spline_path) > 0:
            return self.spline_path[-1].pose, len(self.spline_path) - 1, min_dist
        
        return None, None, float('inf')

    def control_loop_callback(self):
        """
        Periodic control loop callback. Implements the Pure Pursuit algorithm with adaptive lookahead
        and dynamic steering gain adjustments.
        """
        if not self.is_odom_received or self.spline_path is None:
            self.cmd_vel_pub.publish(Twist())
            return
        
        # Check if reached the end of the path
        if self.lookahead_index >= len(self.spline_path) - 1:
            final_point = self.spline_path[-1].pose.position
            distance_to_goal = math.hypot(final_point.x - self.current_x, final_point.y - self.current_y)
            if distance_to_goal < self.goal_threshold:
                self.get_logger().info("----- Goal reached. Stopping. ----")
                self.get_logger().info(f"Sum of error: {self.sum_derivate:.3f}, Cost time: {self.end_time - self.start_time:.3f}s.")
                self.cmd_vel_pub.publish(Twist())  # Stop the robot
                if self.end_time < 1e-3:
                    self.end_time = time.time()
                return
        
        # Find the lookahead point and compute the cross-track error
        lookahead_pose, new_lookahead_index, cross_track_error = self.find_lookahead_point()
        if lookahead_pose is None:
            self.get_logger().warn("No lookahead point found. Stopping.")
            self.cmd_vel_pub.publish(Twist())
            return
        
        self.lookahead_index = new_lookahead_index
        
        # Compute the goal point relative to the robot's current position
        goal_x_robot = lookahead_pose.position.x - self.current_x
        goal_y_robot = lookahead_pose.position.y - self.current_y
        
        # Transform the goal point to the robot's coordinate frame
        theta = self.current_orientation
        goal_x_robot_frame = goal_x_robot * math.cos(theta) + goal_y_robot * math.sin(theta)
        goal_y_robot_frame = -goal_x_robot * math.sin(theta) + goal_y_robot * math.cos(theta)
        
        # Calculate distance to lookahead point in the robot frame
        distance_to_lookahead = math.sqrt(goal_x_robot_frame**2 + goal_y_robot_frame**2)
        
        # Calculate curvature using the updated steering gain dynamically.
        # Increase the effective steering gain if the cross-track error is high.
        dynamic_steering_gain = self.steering_gain * (1 + 0.5 * cross_track_error)
        if abs(goal_x_robot_frame) < 1e-3:
            curvature = 0
        else:
            curvature = dynamic_steering_gain * 2.0 * goal_y_robot_frame / (distance_to_lookahead**2)
        
        # Adjust linear velocity based on distance to lookahead point
        distance_factor = min(1.0, distance_to_lookahead / self.lookahead_distance)
        linear_velocity = self.max_linear_vel * (self.min_speed_factor + (1 - self.min_speed_factor) * distance_factor)
        
        # Reduce speed for sharp turns (i.e., high curvature)
        curvature_factor = 1.0 - min(1.0, abs(curvature) / 1.0)
        linear_velocity = linear_velocity * max(0.3, curvature_factor)
        
        # Improved handling for targets behind the robot
        if goal_x_robot_frame < 0:
            if abs(goal_y_robot_frame) < 0.15:
                linear_velocity = -0.3 * self.max_linear_vel
            else:
                angle_to_target = math.atan2(goal_y_robot_frame, goal_x_robot_frame)
                if abs(angle_to_target) > math.pi/2:
                    linear_velocity = 0.05 * self.max_linear_vel
                    turn_direction = 1.0 if goal_y_robot_frame > 0 else -1.0
                    angular_velocity = turn_direction * self.max_angular_vel
                else:
                    linear_velocity = 0.2 * self.max_linear_vel
        else:
            angular_velocity = curvature * linear_velocity
            # Additional correction if the cross-track error is significant
            if cross_track_error > 0.1:
                extra_angular = min(0.2, cross_track_error * 0.5) * (1.0 if goal_y_robot_frame > 0 else -1.0)
                angular_velocity += extra_angular
        
        # Apply velocity limits
        linear_velocity = max(-self.max_linear_vel, min(linear_velocity, self.max_linear_vel))
        angular_velocity = max(-self.max_angular_vel, min(angular_velocity, self.max_angular_vel))
        
        # For evaluation, track error derivatives
        error_x = lookahead_pose.position.x - self.current_x
        error_y = lookahead_pose.position.y - self.current_y
        derivative_x = error_x - self.prev_error_x
        derivative_y = error_y - self.prev_error_y
        self.sum_derivate += math.fabs(derivative_x) + math.fabs(derivative_y)
        self.prev_error_x = error_x
        self.prev_error_y = error_y
        
        # Publish the velocity command
        twist_msg = Twist()
        twist_msg.linear.x = linear_velocity
        twist_msg.angular.z = angular_velocity
        self.cmd_vel_pub.publish(twist_msg)
        
        # Log debug information
        self.get_logger().info(f"Target: ({lookahead_pose.position.x:.2f}, {lookahead_pose.position.y:.2f}), " +
                                 f"Curvature: {curvature:.3f}, Distance: {distance_to_lookahead:.2f}, " +
                                 f"v: {linear_velocity:.2f}, w: {angular_velocity:.2f}, " +
                                 f"robot_frame: ({goal_x_robot_frame:.2f}, {goal_y_robot_frame:.2f})")


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