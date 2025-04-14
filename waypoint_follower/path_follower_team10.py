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
    A ROS2 node that uses a Pure Pursuit controller to follow a path (list of waypoints).
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
        self.get_logger().warn('++++++ Team-10 Starting Testing')

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
        self.max_linear_vel = 1.25  # meter
        self.max_angular_vel = 0.5 # rad

        # Lookahead distance for Pure Pursuit 
        self.lookahead_distance = 0.2  # meter

    # NOTE: CAN CHANGE
    def path_callback(self, msg: Path):
        """
        Updates the target path when a new message is received.
        """
        self.path = msg.poses  # List of PoseStamped messages
        self.current_waypoint_index = 0  # Reset to the first waypoint
        self.get_logger().info(f"Received new path with {len(self.path)} waypoints. Switching to Pure Pursuit controller.")
        self.start_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        self.sum_path_derivate = 0
        self.end_time = 0

    # NOTE: CAN CHANGE
    def control_loop_callback(self):
        """
        Periodic control loop callback.
        Uses a Pure Pursuit controller to compute and publish velocity commands.
        """
        # Get the Twist command from the Pure Pursuit controller,
        # which internally triggers check_ending if the end of the path is reached.
        twist_msg = self.pure_pursuit_control()
        self.cmd_vel_pub.publish(twist_msg)

    def pure_pursuit_control(self):
        """
        Implements a Pure Pursuit controller while ensuring:
        - Trajectory error metrics are updated using compute_traj_error.
        - The endpoint condition is handled via check_ending.
        """
        # First, check if the end condition is met or if a valid path exists.
        if self.check_ending():
            return Twist()  # Return a stop command if the path is complete or not yet set

        # Now update trajectory error metrics (safe since self.path is guaranteed to be non-None).
        self.compute_traj_error()

        # Find the target waypoint using the lookahead distance.
        target_pose = None
        for i in range(self.current_waypoint_index, len(self.path)):
            wp = self.path[i].pose.position
            distance = math.hypot(wp.x - self.current_x, wp.y - self.current_y)
            if distance >= self.lookahead_distance:
                target_pose = self.path[i].pose
                self.current_waypoint_index = i
                break
        if target_pose is None:
            # If no waypoint is far enough, use the final waypoint.
            target_pose = self.path[-1].pose
            self.current_waypoint_index = len(self.path) - 1

        # Check if the robot is close to the target waypoint.
        L = math.hypot(target_pose.position.x - self.current_x,
                    target_pose.position.y - self.current_y)
        if L < self.waypoint_threshold:
            self.get_logger().info(f"Reached waypoint {self.current_waypoint_index}.")
            self.current_waypoint_index += 1
            # Check again if we have reached the end before recursively calling.
            if self.check_ending():
                return Twist()
            return self.pure_pursuit_control()

        # Compute the heading to the target.
        angle_to_target = math.atan2(target_pose.position.y - self.current_y,
                                    target_pose.position.x - self.current_x)
        # Calculate the heading error (alpha).
        alpha = self.normalize_angle(angle_to_target - self.current_orientation)

        # If the heading error is large, just turn in place:
        large_angle_threshold = 1.0  # rad ~ 57°
        if abs(alpha) > large_angle_threshold:
            # Turn in place
            v = 0.0
            # For a simple approach, turn at max_angular_vel in the correct direction:
            omega = self.max_angular_vel if alpha > 0.0 else -self.max_angular_vel
            self.get_logger().info(
                f"Large heading error ({alpha:.2f} rad). Turning in place: v={v:.2f}, omega={omega:.2f}"
            )
        else:
            # If heading error is small enough, do the normal Pure Pursuit:
            L = math.hypot(
                target_pose.position.x - self.current_x,
                target_pose.position.y - self.current_y
            )
            curvature = 2.0 * math.sin(alpha) / L if L != 0 else 0.0

            # We scale down speed using cos(alpha) and curvature
            v = self.max_linear_vel * max(0.0, math.cos(alpha)) / (1.0 + abs(curvature))
            
            # Angular velocity = curvature * v
            omega = curvature * v
            if abs(omega) > self.max_angular_vel:
                omega = self.max_angular_vel if omega > 0.0 else -self.max_angular_vel

            self.get_logger().info(
                f"Pure Pursuit: v={v:.2f}, omega={omega:.2f}, alpha={alpha:.2f}, curvature={curvature:.2f}"
            )
        # ------------------------------------------------

        twist_msg = Twist()
        twist_msg.linear.x = v
        twist_msg.angular.z = omega
        return twist_msg

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