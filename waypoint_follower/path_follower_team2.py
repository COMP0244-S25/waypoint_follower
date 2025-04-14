# python3 src/waypoint_follower/waypoint_follower/path_follower.py /custom_odometry_topic

#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
import time

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist, PoseStamped
import numpy as np
from scipy.interpolate import splprep, splev

from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA


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
        self.current_orientation = 0.0
        self.smooth = True


        # Previous errors (for derivative term)
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.prev_error_theta = 0.0
        self.prev_error_orientation = 0.0   # ADD 


        # Robot current state
        self.current_x = 0.0
        self.current_y = 0.0
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

        # ADD: Path visualization publishers
        self.marker_pub = self.create_publisher(MarkerArray, '/path_markers', 10)

        # Timer to periodically publish velocity commands
        timer_period = 0.1  # [s] -> 10 Hz
        self.timer = self.create_timer(timer_period, self.control_loop_callback)
        self.get_logger().info(f"Path Follower node started. Subscribing to odometry topic: {odometry_topic}")
        self.get_logger().warn('++++++ Team-2 Starting Testing')

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

    def publish_path_markers(self):
        """
        Publishes markers (arrows and text) for each waypoint to RViz.
        """
        if self.path is None:
            return

        marker_array = MarkerArray()
        for idx, pose_stamped in enumerate(self.path):
            pos = pose_stamped.pose.position

            # Arrow marker
            arrow = Marker()
            arrow.header = pose_stamped.header
            arrow.ns = "waypoint_arrows"
            arrow.id = idx
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.pose = pose_stamped.pose
            arrow.scale.x = 0.2  # shaft length
            arrow.scale.y = 0.01
            arrow.scale.z = 0.01
            arrow.color = ColorRGBA(r=0.0, g=0.5, b=1.0, a=1.0)  # green
            marker_array.markers.append(arrow)

            # Text marker
            text = Marker()
            text.header = pose_stamped.header
            text.ns = "waypoint_labels"
            text.id = 1000 + idx
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = pos.x
            text.pose.position.y = pos.y
            text.pose.position.z = 0.3  # Slightly above ground
            text.scale.z = 0.05  # Font size
            text.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)  # red
            text.text = str(idx)
            marker_array.markers.append(text)

        self.marker_pub.publish(marker_array)
        self.get_logger().info("Published waypoint markers.")

    # NOTE: CAN CHANGE
    def setup_parameters(self):
        # Maximum velocities
        self.max_linear_vel = 1.2  # meter
        self.max_angular_vel = 1.5 # rad
        self.max_urgent_angular_vel = 3.0 # rad


        # PD Controller Gains (tune as necessary)
        self.Kp_linear = 1.0
        self.Kd_linear = 0.1
        self.Kp_angular = 5.0
        self.Kd_angular = 0.5

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
        self.get_logger().info(f"Sum of error: {self.sum_derivate:.3f}, Cost time: {duration:.3f}s.")

    # NOTE: CAN CHANGE
    def path_callback(self, msg: Path):
        """
        Updates the target path when a new message is received, with Bezier/spline interpolation.
        """
        self.path = msg.poses  # Original Path: List of PoseStamped
        self.current_waypoint_index = 0  # Reset to the first waypoint
        self.get_logger().info(f"Received new path with {len(self.path)} waypoints.")
        self.start_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        self.end_time = 0

        self.sum_path_derivate = 0

        # Extract x and y coordinates from the path
        x = [pose.pose.position.x for pose in self.path]
        y = [pose.pose.position.y for pose in self.path]

        # Perform Bezier/spline interpolation
        if len(x) < 3:
            self.get_logger().warn("Not enough points for Bezier/spline interpolation.")
            self.smoothed_path = self.path
            return

        # Use scipy's spline interpolation
        # tck, u = splprep([x, y], s=0)
        # u_fine = np.linspace(0, 1, num=int(len(self.path) ))
        # x_new, y_new = splev(u_fine, tck)

        x_new, y_new = x, y

        # construct smoothed path
        # self.smoothed_path = []
        # for i in range(len(x_new)):
        #     pose = PoseStamped()
        #     pose.header = msg.header
        #     pose.pose.position.x = x_new[i]
        #     pose.pose.position.y = y_new[i]
        #     pose.pose.position.z = 0.0
        #     self.smoothed_path.append(pose)

        self.smoothed_path = []
        for i in range(len(x_new)):
            pose = PoseStamped()
            pose.header.frame_id = 'world'  # specify the frame_id
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = x_new[i]
            pose.pose.position.y = y_new[i]
            pose.pose.position.z = 0.0
            self.smoothed_path.append(pose)

        self.get_logger().info(f"Generated smoothed path with {len(self.smoothed_path)} points.")

        if self.smooth:
            self.path = self.smoothed_path

        for i in range(len(self.path) - 1):
            current = self.path[i].pose.position
            next_ = self.path[i + 1].pose.position

            dx = next_.x - current.x
            dy = next_.y - current.y
            yaw = math.atan2(dy, dx)

            qz = math.sin(yaw / 2.0)
            qw = math.cos(yaw / 2.0)

            self.path[i].pose.orientation.x = 0.0
            self.path[i].pose.orientation.y = 0.0
            self.path[i].pose.orientation.z = qz
            self.path[i].pose.orientation.w = qw

        if len(self.path) >= 2:
            self.path[-1].pose.orientation = self.path[-2].pose.orientation

        # self.visulize_path()
        self.publish_path_markers()



    # NOTE: CAN CHANGE
    def control_loop_callback(self):
        """
        Periodic control loop callback. Computes PD control and publishes velocity commands.
        """
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

        error_x = x_target - self.current_x
        error_y = y_target - self.current_y
        error_theta = self.normalize_angle(math.atan2(error_y, error_x) - self.current_orientation)
        error_orientation = self.normalize_angle(orientation_target - self.current_orientation)

        # # 1) Compute errors
        # error_x = x_target - self.current_x
        # error_y = y_target - self.current_y
        # error_theta = self.normalize_angle(math.atan2(error_y, error_x)- self.current_orientation)
        # try:
        #     next_target = self.path[self.current_waypoint_index + 1]
        #     orientation_target = self.normalize_angle(math.atan2(next_target.pose.position.y - y_target, next_target.pose.position.x - x_target))
        # except:
        #     orientation_target = self.normalize_angle(math.atan2(error_y, error_x))
        # error_orientation = self.normalize_angle(orientation_target - self.current_orientation)
        # # distance_to_waypoint = math.hypot(error_x, error_y)

        # 2) Compute derivative of errors
        derivative_x = error_x - self.prev_error_x
        derivative_y = error_y - self.prev_error_y
        derivative_theta = error_theta - self.prev_error_theta
        # self.sum_derivate += math.fabs(derivative_x) + math.fabs(derivative_y) + math.fabs(derivative_theta) / 180.0 * math.pi
        self.compute_traj_error()

        derivative_orientation = error_orientation - self.prev_error_orientation
        
        # 3) PD control for linear velocities (x, y)
        vx = self.Kp_linear * error_x + self.Kd_linear * derivative_x
        vy = self.Kp_linear * error_y + self.Kd_linear * derivative_y

        # 4) PD control for angular velocity
        vtheta = self.Kp_angular * error_theta + self.Kd_angular * derivative_theta
        vorientation = self.Kp_angular * error_orientation + self.Kd_angular * derivative_orientation

        # 5) Update previous error terms
        self.prev_error_x = error_x
        self.prev_error_y = error_y
        self.prev_error_theta = error_theta
        self.prev_error_orientation = error_orientation

        # 7) Publish velocity commands
        twist_msg = Twist()
        distance_to_target = math.hypot(error_x, error_y)

        self.get_logger().info(
            f"Index {self.current_waypoint_index}, error_orientation = {error_orientation:.3f}, vorientation = {vorientation:.3f}"
        )
        
        # self.cmd_vel_pub.publish(twist_msg)
        if distance_to_target < self.waypoint_threshold:
            twist_msg.linear.x = 0.0  # 停止前进

            if abs(error_orientation) > 0.05:
                twist_msg.linear.x = 0.05  # move forward a little
                twist_msg.angular.z = np.clip(vorientation, -self.max_urgent_angular_vel, self.max_urgent_angular_vel)
                self.get_logger().info(" Rotating to align with final orientation")

            self.current_waypoint_index += 1  # Move to the next waypoint
            self.get_logger().info(f"Reached waypoint {self.current_waypoint_index - 1}. Moving to the next one.")
        else:
        
            # # 2️  urgent turn
            # if abs(error_orientation) >  0.4 and distance_to_target <= self.orientation_threshold:
            #     twist_msg.linear.x = 0.05  # move forward a little
            #     twist_msg.angular.z = np.clip(vorientation, -self.max_angular_vel, self.max_angular_vel)
            #     # self.last_turn_time = self.get_clock().now().nanoseconds
            #     self.get_logger().info("Urgent Turn")
            
            # 1. urgent rotate
            if abs(error_theta) > 0.5:
                twist_msg.linear.x = 0.01  # move forward a little
                # twist_msg.linear.y = 0.0  
                twist_msg.angular.z = np.clip(vtheta, -self.max_angular_vel, self.max_angular_vel)  # fast 旋转
                # self.last_turn_time = self.get_clock().now().nanoseconds
                self.get_logger().info("Urgent Turn")
            
            # 2. 正常行走逻辑 
            else:
                twist_msg.linear.x = min(math.hypot(vx, vy), self.max_linear_vel)  # 保持最大速度
                twist_msg.angular.z = np.clip(vtheta, -self.max_urgent_angular_vel, self.max_urgent_angular_vel)  # 持续调整方向
                self.get_logger().info("Moving Forward")
                
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
