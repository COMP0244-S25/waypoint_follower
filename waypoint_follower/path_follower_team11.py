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

        # sum of derivative
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
        self.get_logger().warn('++++++ Team-11 Starting Testing')

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

        # self.get_logger().info(f"Sum of error: {self.sum_derivate:.3f}, Cost time: {time.time() - self.start_time:.3f}s.")

    # NOTE: STUDENTS ARE ALLOWED TO ADD THEIR OWN FUNCTIONS
    def customized_functions(self):
        pass

    # NOTE: CAN CHANGE
    def setup_parameters(self):
        # Maximum velocities
        self.max_linear_vel = 1.0   # meter
        self.max_angular_vel = 0.6  # rad

        # PD Controller Gains (tune as necessary)
        self.Kp_linear = 2.0
        self.Kd_linear = 0.2
        self.Kp_angular = 4.0
        self.Kd_angular = 0.4

        # Used for low-pass filtering
        self.prev_linear_x = 0.0
        self.prev_angular_z = 0.0

        self.large_turn_threshold = 0.56

        # Low pass filter factor (0~1)
        self.velocity_filter_factor = 0.50


    # NOTE: CAN CHANGE
    def path_callback(self, msg: Path):
        """
        Updates the target path when a new message is received.
        """
        # 1) Do corner interpolation on the new path
        new_poses = self.smooth_corners(msg.poses)

        # 2) Assign to self.path
        self.path = new_poses
        self.current_waypoint_index = 0
        # self.get_logger().info(f"Received new path with {len(self.path)} waypoints.")
        self.start_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        self.sum_path_derivate = 0
        self.end_time = 0


    def smooth_corners(self, original_poses):
        """
        Optimizing path smoothing: Using angular adaptive linear interpolation
        """
        if len(original_poses) < 3:
            return original_poses
        
        optimized_poses = [original_poses[0]]
        angle_threshold = math.radians(25)  
        
        for i in range(1, len(original_poses)-1):
            prev = original_poses[i-1].pose.position
            curr = original_poses[i].pose.position
            next = original_poses[i+1].pose.position
            
            # Calculated turning Angle
            v1 = (curr.x - prev.x, curr.y - prev.y)
            v2 = (next.x - curr.x, next.y - curr.y)
            angle = self.angle_between_vectors(v1, v2)
            
            optimized_poses.append(original_poses[i])
            
            if angle > angle_threshold:
                # The number of interpolation points is dynamically calculated according to the Angle
                interp_points = int(angle / math.radians(20)) + 1
                max_points = min(interp_points, 5)
                
                for t in [k/(max_points+1) for k in range(1, max_points+1)]:
                    new_pose = PoseStamped()
                    new_pose.pose.position.x = curr.x + t*(next.x - curr.x)
                    new_pose.pose.position.y = curr.y + t*(next.y - curr.y)
                    optimized_poses.append(new_pose)
        
        optimized_poses.append(original_poses[-1])
        return optimized_poses

    def angle_between_vectors(self, v1, v2):
        """
        Calculate the Angle between v1 and v2 (0 ~ π)
        """
        x1, y1 = v1
        x2, y2 = v2
        dot = x1 * x2 + y1 * y2
        mag1 = math.hypot(x1, y1)
        mag2 = math.hypot(x2, y2)
        if mag1 < 1e-9 or mag2 < 1e-9:
            return 0.0
        cos_angle = dot / (mag1 * mag2)
        cos_angle = max(min(cos_angle, 1.0), -1.0)
        return math.acos(cos_angle)

    # NOTE: CAN CHANGE
    def control_loop_callback(self):
        """
        Periodic control loop callback. Computes PD control and publishes velocity commands.
        """
        if self.check_ending():
            return
        
        lookahead_index = min(self.current_waypoint_index + 1, len(self.path) - 1)
        target_pose = self.path[lookahead_index].pose
        # Get the current target waypoint
        # target_pose = self.path[self.current_waypoint_index].pose
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

        # 6) Whether to reach the current waypoint
        twist_msg = Twist()
        distance_to_waypoint = math.hypot(error_x, error_y)

        if distance_to_waypoint < self.waypoint_threshold:
            self.current_waypoint_index += 1
        else:
            if abs(error_theta) > self.large_turn_threshold:
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = max(min(vtheta, self.max_angular_vel), -self.max_angular_vel)
            else:
                desired_linear = math.hypot(vx, vy)
                desired_angular = vtheta

                # limit
                desired_linear = max(min(desired_linear, self.max_linear_vel), -self.max_linear_vel)
                desired_angular = max(min(desired_angular, self.max_angular_vel), -self.max_angular_vel)

                twist_msg.linear.x = desired_linear
                twist_msg.angular.z = desired_angular

        # 7) Low pass filter factor
        alpha = self.velocity_filter_factor
        filtered_linear_x = alpha * self.prev_linear_x + (1 - alpha) * twist_msg.linear.x
        filtered_angular_z = alpha * self.prev_angular_z + (1 - alpha) * twist_msg.angular.z

        twist_msg.linear.x = filtered_linear_x
        twist_msg.angular.z = filtered_angular_z

        self.prev_linear_x = filtered_linear_x
        self.prev_angular_z = filtered_angular_z

        # 8) Publish control instruction
        self.cmd_vel_pub.publish(twist_msg)

def main():
    import sys
    args = sys.argv

    rclpy.init(args=args)

    # Specify the odometry topic (default is '/Odometry')
    odometry_topic = '/Odometry'
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
