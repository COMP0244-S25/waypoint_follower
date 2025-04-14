#!/usr/bin/env python3

import math
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist, PoseStamped
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import time
import os

class PathFollower(Node):
    """
    Path Tracking Controller
    - Maintains a fixed orientation (the one recorded when the path is received)
    - Uses x and y velocities to move along the path
    - Does not change the robot's orientation during path following
    - Combines target-point vector with path-line vector for better tracking
    """
    def __init__(self, odometry_topic):
        super().__init__('PathFollower')

        self.odometry = odometry_topic

        # ================ Control Parameters ================= #
        self.position_tolerance = 0.08    # Position tolerance
        self.default_linear_speed = 0.4   # Default linear speed
        self.max_linear_speed_x = 0.5     # Maximum forward speed
        self.max_linear_speed_y = 0.5     # Maximum lateral speed
        self.max_angular_speed = 1.0      # Maximum angular speed
        self.max_total_speed = 0.5        # Maximum combined speed
        
        self.target_weight = 1.0
        self.line_weight = 0.4

        # Distance-based control PID (vector toward target point)
        self.Kp_distance = 1.2
        self.Ki_distance = 0.0
        self.Kd_distance = 0.5
        
        # Path-line following PID (vector toward path line)
        self.Kp_path_line = 1.5           # Higher gain to prioritize path-line following
        self.Ki_path_line = 0.0
        self.Kd_path_line = 0.6
        
        # Angular control PID (used only to maintain initial orientation)
        self.Kp_angular = 1.0
        self.Ki_angular = 0.0
        self.Kd_angular = 0.3
        # ================================================= #

        # Current pose
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.is_odom_received = False
        
        # Path related
        self.path_points = []
        self.current_waypoint_index = 0
        self.is_path_completed = False
        self.is_new_path = False
        
        # Fixed orientation to maintain (set when path is received)
        self.fixed_orientation = None
        
        # Trajectory recording for plotting
        self.trajectory_data = {
            'timestamp': [],
            'x_desired': [],
            'y_desired': [],
            'x_current': [],
            'y_current': []
        }
        self.start_time = None

        # PID control related
        self.prev_distance_error = 0.0
        self.distance_error_integral = 0.0
        self.prev_path_line_error = 0.0   # For path-line following
        self.path_line_error_integral = 0.0
        self.prev_angular_error = 0.0
        self.angular_error_integral = 0.0

        # Publish cmd_vel
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscribe to odometry
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odometry,
            self.odom_callback,
            10
        )

        # Subscribe to path
        self.path_sub = self.create_subscription(
            Path,
            '/planner/path',
            self.path_callback,
            10
        )

        # Control loop timer: 10Hz
        self.timer = self.create_timer(0.1, self.control_loop_callback)

        self.get_logger().info("Enhanced path tracking controller started, waiting for path data...")
        self.get_logger().warn('++++++ Team-8M Starting Testing')

    def odom_callback(self, msg: Odometry):
        """Update robot's current position and orientation"""
        self.is_odom_received = True

        pose = msg.pose.pose
        self.current_x = pose.position.x
        self.current_y = pose.position.y

        # Quaternion -> yaw angle
        q = pose.orientation
        siny_cosp = 2.0 * (q.w*q.z + q.x*q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # If a new path is received and odometry data is ready, start processing
        if self.is_new_path and len(self.path_points) > 0:
            self.is_new_path = False
            # Record the current orientation as the fixed orientation to maintain
            if self.fixed_orientation is None:
                self.fixed_orientation = self.current_yaw
                self.get_logger().info(f"Fixed orientation set to: {self.fixed_orientation}")
            self.get_logger().info("Odometry data ready, starting path tracking")

    def path_callback(self, msg: Path):
        """Receive new path and save to self.path_points"""
        raw_poses = msg.poses
        if len(raw_poses) < 2:
            self.get_logger().warn("Received path has insufficient points")
            return

        # Extract path point coordinates
        self.path_points = [(p.pose.position.x, p.pose.position.y) for p in raw_poses]
        self.current_waypoint_index = 0
        self.is_path_completed = False
        self.is_new_path = True  # Mark that a new path was received
        
        # Reset PID integral terms
        self.distance_error_integral = 0.0
        self.path_line_error_integral = 0.0
        self.angular_error_integral = 0.0
        
        # Record the current orientation when path is received
        # Will be updated in the next odom_callback if not yet received
        if self.is_odom_received:
            self.fixed_orientation = self.current_yaw
            self.get_logger().info(f"Fixed orientation set to: {self.fixed_orientation}")
        
        # Reset trajectory recording
        self.trajectory_data = {
            'timestamp': [],
            'x_desired': [],
            'y_desired': [],
            'x_current': [],
            'y_current': []
        }
        self.start_time = time.time()
        
        self.get_logger().info(f"Received new path with {len(self.path_points)} points")

    def calculate_target_vector(self):
        """
        Calculate the vector to the target waypoint
        Returns:
        - distance: distance to the target
        - angle: angle to the target in global frame
        - target_x, target_y: coordinates of the target
        """
        if self.current_waypoint_index >= len(self.path_points):
            return 0.0, 0.0, self.current_x, self.current_y
            
        # Get current waypoint
        target_point = self.path_points[self.current_waypoint_index]
        target_x, target_y = target_point
        
        # Calculate vector to target
        dx = target_x - self.current_x
        dy = target_y - self.current_y
        distance = math.hypot(dx, dy)
        
        # Calculate angle to target in global frame
        target_angle = math.atan2(dy, dx)
        
        return distance, target_angle, target_x, target_y

    def calculate_path_line_vector(self):
        """
        Calculate the vector to the closest point on the path line
        (the line between previous and current waypoint)
        Returns:
        - distance_to_line: perpendicular distance to the path line
        - line_angle: angle perpendicular to the path line, pointing toward the line
        - closest_point_x, closest_point_y: coordinates of the closest point on the line
        - path_line_angle: angle of the path line itself
        """
        # If we're at the first waypoint, or no previous point exists
        if self.current_waypoint_index <= 0 or len(self.path_points) <= 1:
            return 0.0, 0.0, self.current_x, self.current_y, 0.0
            
        # Get current and previous waypoints
        prev_idx = max(0, self.current_waypoint_index - 1)
        curr_idx = self.current_waypoint_index
        
        # If we're at the last waypoint, use the current and previous points
        if curr_idx >= len(self.path_points):
            curr_idx = len(self.path_points) - 1
            prev_idx = curr_idx - 1
            
        prev_point = self.path_points[prev_idx]
        curr_point = self.path_points[curr_idx]
        
        # Line segment parameters
        x1, y1 = prev_point
        x2, y2 = curr_point
        
        # Path line angle
        path_line_angle = math.atan2(y2 - y1, x2 - x1)
        
        # Calculate closest point on the line segment
        # First, calculate the projection factor
        dx_line = x2 - x1
        dy_line = y2 - y1
        line_length_squared = dx_line**2 + dy_line**2
        
        # Avoid division by zero
        if line_length_squared < 1e-10:
            # Points are essentially the same, use the current point
            return 0.0, 0.0, x1, y1, 0.0
            
        # Calculate the projection of the robot position onto the line
        dx_robot = self.current_x - x1
        dy_robot = self.current_y - y1
        
        # The projection factor, clamped to [0, 1] for the line segment
        t = max(0, min(1, (dx_robot * dx_line + dy_robot * dy_line) / line_length_squared))
        
        # Coordinates of the closest point on the line segment
        closest_x = x1 + t * dx_line
        closest_y = y1 + t * dy_line
        
        # Vector from robot to closest point
        dx_to_line = closest_x - self.current_x
        dy_to_line = closest_y - self.current_y
        
        # Distance to the line
        distance_to_line = math.hypot(dx_to_line, dy_to_line)
        
        # Angle to the closest point
        line_angle = math.atan2(dy_to_line, dx_to_line)
        
        return distance_to_line, line_angle, closest_x, closest_y, path_line_angle

    def control_loop_callback(self):
        """Main control loop using path tracking strategy"""
        if not self.is_odom_received or len(self.path_points) < 2 or self.is_path_completed:
            if self.is_path_completed:
                self.stop_robot()
            return

        # Check if fixed orientation is set
        if self.fixed_orientation is None:
            self.fixed_orientation = self.current_yaw
            self.get_logger().info(f"Fixed orientation set to: {self.fixed_orientation}")

        # Get current waypoint index
        if self.current_waypoint_index >= len(self.path_points):
            self.is_path_completed = True
            self.stop_robot()
            self.get_logger().info("Path tracking completed!")
            return

        # Calculate distance and angle to target
        distance, target_angle, target_x, target_y = self.calculate_target_vector()
        
        # Calculate distance and angle to path line
        distance_to_line, line_angle, closest_x, closest_y, path_line_angle = self.calculate_path_line_vector()
        
        # Calculate angular error (difference between current and fixed orientation)
        angle_error = self.normalize_angle(self.fixed_orientation - self.current_yaw)
        
        # Record trajectory data for plotting
        if self.start_time is not None:
            current_time = time.time() - self.start_time
            self.trajectory_data['timestamp'].append(current_time)
            self.trajectory_data['x_current'].append(self.current_x)
            self.trajectory_data['y_current'].append(self.current_y)
            
            # Get desired position (current waypoint)
            self.trajectory_data['x_desired'].append(target_x)
            self.trajectory_data['y_desired'].append(target_y)

        # Use distance-based PID to calculate speed magnitude toward target point
        speed_to_target = self.pid_control(self.Kp_distance, self.Ki_distance, self.Kd_distance,
                                         distance, self.prev_distance_error, self.distance_error_integral)
        
        # Use path-line based PID to calculate speed magnitude toward path line
        speed_to_line = self.pid_control(self.Kp_path_line, self.Ki_path_line, self.Kd_path_line,
                                       distance_to_line, self.prev_path_line_error, self.path_line_error_integral)
        
        # Calculate angular velocity to maintain fixed orientation
        omega = self.pid_control(self.Kp_angular, self.Ki_angular, self.Kd_angular,
                                angle_error, self.prev_angular_error, self.angular_error_integral)

        # Apply speed factors based on distance to waypoint
        if distance > self.position_tolerance * 2:
            speed_factor = 1.0  # Full speed when far from target
        else:
            speed_factor = 0.8  # Reduced speed when approaching target
        
        # Ensure minimum speed for reliable movement
        min_speed = 0.2
        
        # Apply speed factor to target-directed speed
        if distance > self.position_tolerance:
            speed_to_target = max(speed_to_target * speed_factor, min_speed)
        else:
            speed_to_target = max(speed_to_target * speed_factor, 0.0)
        
        # Decompose speeds into global x and y components
        # 1. Vector toward target point
        vx_global_target = speed_to_target * math.cos(target_angle)
        vy_global_target = speed_to_target * math.sin(target_angle)
        
        # 2. Vector toward path line (only when not very close to target and not at first point)
        if distance_to_line > 0.02 and self.current_waypoint_index > 0:
            vx_global_line = speed_to_line * math.cos(line_angle)
            vy_global_line = speed_to_line * math.sin(line_angle)
        else:
            vx_global_line = 0.0
            vy_global_line = 0.0
        
        # Combine vectors with weights (adjust these weights as needed)
        target_weight = self.target_weight
        line_weight = self.line_weight
        
        # Weighted sum of the vectors
        vx_global = target_weight * vx_global_target + line_weight * vx_global_line
        vy_global = target_weight * vy_global_target + line_weight * vy_global_line
        
        # Convert global velocities to robot-relative velocities
        vx = vx_global * math.cos(self.current_yaw) + vy_global * math.sin(self.current_yaw)
        vy = -vx_global * math.sin(self.current_yaw) + vy_global * math.cos(self.current_yaw)
        
        # Ensure combined speed does not exceed maximum
        total_speed = math.hypot(vx, vy)
        if total_speed > self.max_total_speed:
            # Scale both components proportionally
            scale_factor = self.max_total_speed / total_speed
            vx *= scale_factor
            vy *= scale_factor
            
        # Limit individual speeds
        vx = self.saturate(vx, self.max_linear_speed_x)
        vy = self.saturate(vy, self.max_linear_speed_y)
        omega = self.saturate(omega, self.max_angular_speed)

        # Publish velocity commands
        twist = Twist()
        twist.linear.x = vx
        twist.linear.y = vy
        twist.angular.z = omega
        self.cmd_vel_pub.publish(twist)

        # Update PID error values
        self.prev_distance_error = distance
        self.distance_error_integral += distance
        self.prev_path_line_error = distance_to_line
        self.path_line_error_integral += distance_to_line
        self.prev_angular_error = angle_error
        self.angular_error_integral += angle_error

        # Log output
        self.get_logger().debug(
            f"Waypoint index: {self.current_waypoint_index}, " 
            f"Distance to target: {distance:.2f}, Distance to line: {distance_to_line:.2f}, "
            f"Target angle: {target_angle:.2f}, Line angle: {line_angle:.2f}, "
            f"Angle error: {angle_error:.2f}, "
            f"Velocities: ({vx:.2f}, {vy:.2f}, {omega:.2f})"
        )

        # When close enough to current waypoint, switch to next one
        if distance < self.position_tolerance:
            curr_point = self.path_points[self.current_waypoint_index]
            x_curr, y_curr = curr_point
            self.get_logger().info(f"Reached waypoint {self.current_waypoint_index}: ({x_curr:.2f}, {y_curr:.2f})")
            self.current_waypoint_index += 1
            
            # Reset PID integral terms
            self.distance_error_integral = 0.0
            self.path_line_error_integral = 0.0
            
            if self.current_waypoint_index >= len(self.path_points):
                self.is_path_completed = True
                self.stop_robot()
                self.get_logger().info("Path tracking completed!")
                # Generate and save trajectory plots
                self.generate_trajectory_plots()

    def pid_control(self, Kp, Ki, Kd, error, prev_error, integral):
        """PID controller with proportional and derivative control to reduce overshooting but maintain fast response"""
        # Calculate derivative term - helps predict error changes and reduce overshooting
        derivative = error - prev_error
        
        # Calculate output, mainly using proportional and derivative control
        output = Kp * error + Kd * derivative
        
        # Only add integral term when enabled
        if Ki != 0.0:
            max_integral = 5.0  # Integral limit
            integral = max(min(integral + error, max_integral), -max_integral)
            output += Ki * integral
            
        return output

    def stop_robot(self):
        """Stop the robot"""
        twist = Twist()
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

    def normalize_angle(self, angle):
        """Restrict angle to [-pi, pi] range"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def saturate(self, value, limit):
        """Limit value to [-limit, limit] range"""
        return max(min(value, limit), -limit)
        
    def generate_trajectory_plots(self):
        """Generate and save plots of desired vs. current positions over time"""
        if not self.trajectory_data['timestamp']:
            self.get_logger().warn("No trajectory data to plot")
            return
            
        try:
            self.get_logger().info("Generating trajectory plots...")
            
            # Create directory for plots if it doesn't exist
            plot_dir = os.path.expanduser("./trajectory_plots")
            os.makedirs(plot_dir, exist_ok=True)
            
            # Generate timestamp for filenames
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            
            # Plot X position over time
            plt.figure(figsize=(10, 6))
            plt.plot(self.trajectory_data['timestamp'], self.trajectory_data['x_desired'], 'b-', label='X Desired')
            plt.plot(self.trajectory_data['timestamp'], self.trajectory_data['x_current'], 'r--', label='X Current')
            plt.xlabel('Time (seconds)')
            plt.ylabel('X Position (m)')
            plt.title('X Position: Desired vs. Current')
            plt.legend()
            plt.grid(True)
            x_plot_path = os.path.join(plot_dir, f"x_trajectory_{timestamp}.png")
            plt.savefig(x_plot_path)
            plt.close()
            
            # Plot Y position over time
            plt.figure(figsize=(10, 6))
            plt.plot(self.trajectory_data['timestamp'], self.trajectory_data['y_desired'], 'b-', label='Y Desired')
            plt.plot(self.trajectory_data['timestamp'], self.trajectory_data['y_current'], 'r--', label='Y Current')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Y Position (m)')
            plt.title('Y Position: Desired vs. Current')
            plt.legend()
            plt.grid(True)
            y_plot_path = os.path.join(plot_dir, f"y_trajectory_{timestamp}.png")
            plt.savefig(y_plot_path)
            plt.close()
            
            # Plot 2D trajectory
            plt.figure(figsize=(10, 10))
            plt.plot(self.trajectory_data['x_desired'], self.trajectory_data['y_desired'], 'b.-', label='Desired Path')
            plt.plot(self.trajectory_data['x_current'], self.trajectory_data['y_current'], 'r.-', label='Actual Path')
            plt.xlabel('X Position (m)')
            plt.ylabel('Y Position (m)')
            plt.title('2D Trajectory: Desired vs. Actual')
            plt.legend()
            plt.grid(True)
            plt.axis('equal')  # Equal scaling for x and y
            path_plot_path = os.path.join(plot_dir, f"2d_trajectory_{timestamp}.png")
            plt.savefig(path_plot_path)
            plt.close()
            
            # Plot tracking error over time
            x_error = [d - c for d, c in zip(self.trajectory_data['x_desired'], self.trajectory_data['x_current'])]
            y_error = [d - c for d, c in zip(self.trajectory_data['y_desired'], self.trajectory_data['y_current'])]
            total_error = [math.hypot(xe, ye) for xe, ye in zip(x_error, y_error)]
            
            plt.figure(figsize=(10, 6))
            plt.plot(self.trajectory_data['timestamp'], x_error, 'r-', label='X Error')
            plt.plot(self.trajectory_data['timestamp'], y_error, 'g-', label='Y Error')
            plt.plot(self.trajectory_data['timestamp'], total_error, 'b-', label='Total Error')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Error (m)')
            plt.title('Tracking Error Over Time')
            plt.legend()
            plt.grid(True)
            error_plot_path = os.path.join(plot_dir, f"tracking_error_{timestamp}.png")
            plt.savefig(error_plot_path)
            plt.close()
            
            self.get_logger().info(f"Trajectory plots saved to {plot_dir}")
            
        except Exception as e:
            self.get_logger().error(f"Error generating trajectory plots: {e}")

# def main(args=None):
#     rclpy.init(args=args)
#     node = PathTrackingController()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         node.get_logger().info("Keyboard interrupt, shutting down node...")
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()


def main():
    import sys
    args = sys.argv 

    rclpy.init(args=args)

    # Specify the odometry topic (default is '/Odometry')
    odometry_topic = '/Odometry'  # Default value
    # odometry_topic = '/utlidar/robot_odom'

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