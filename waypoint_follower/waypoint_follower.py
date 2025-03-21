#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose2D


class WaypointFollower(Node):
    """
    A ROS2 node that uses a PD controller to follow a dynamically received
    waypoint (x, y, theta). Subscribes to /waypoint for the target and
    /odom for the robot's current state, and publishes velocity commands
    to /cmd_vel.
    """

    def __init__(self):
        super().__init__('waypoint_follower')

        # Target waypoint (x, y, theta)
        # Initially, set them to 0. They will be updated by /waypoint subscriber.
        self.x_target = None
        self.y_target = None
        self.orientation_target = None

        self.max_velo = 0.5

        # PD Controller Gains (tune as necessary)
        self.Kp_linear = 10.0
        self.Kd_linear = 0.1
        self.Kp_angular = 5.0
        self.Kd_angular = 0.1

        # Previous errors (for derivative term)
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.prev_error_theta = 0.0
        self.prev_error_orientation = 0.0

        # Robot current state
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_orientation = 0.0
        self.is_odom_received = False
        self.is_arrive_waypoint  = True

        # Publisher to cmd_vel
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscriber to odometry
        self.odom_sub = self.create_subscription(
            Odometry,
            '/Odometry',
            self.odom_callback,
            10
        )

        # Subscriber to waypoint
        self.waypoint_sub = self.create_subscription(
            Pose2D,
            '/waypoint',
            self.waypoint_callback,
            10
        )

        # Timer to periodically publish velocity commands
        timer_period = 0.1  # [s] -> 10 Hz
        self.timer = self.create_timer(timer_period, self.control_loop_callback)

        self.get_logger().info("Waypoint Follower node started. Waiting for waypoints...")

    def waypoint_callback(self, msg: Pose2D):
        """
        Updates the target waypoint when a new message is received.
        """
        if self.x_target==None or abs(self.x_target-msg.x) > 0.2 or abs(self.y_target-msg.y) > 0.2 or abs(self.orientation_target-msg.theta) > 0.2:
            self.is_arrive_waypoint = False

        self.x_target = msg.x
        self.y_target = msg.y
        self.orientation_target = msg.theta
        self.get_logger().info(
            f"Received new waypoint: x={self.x_target}, y={self.y_target}, orientation={self.orientation_target}. current state: x={self.current_x}, y={self.current_y}, orientation={self.current_orientation}"
        )

    def odom_callback(self, msg: Odometry):
        """
        Extracts and stores the robot's current pose from the odometry.
        """
        self.is_odom_received = True
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        
        # Convert quaternion to yaw (theta)
        # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_orientation = math.atan2(siny_cosp, cosy_cosp)

        # keep stop when there is no waypoint published
        if self.x_target is None:
            self.x_target = self.current_x
            self.y_target = self.current_y
            self.orientation_target = self.current_orientation

    def control_loop_callback(self):
        """
        Periodic control loop callback. Computes PD control and publishes velocity commands.
        """
        if not self.is_odom_received:
            return
        # 1) Compute errors
        error_x = self.x_target - self.current_x
        error_y = self.y_target - self.current_y
        error_theta = self.normalize_angle(math.atan2(error_y, error_x)- self.current_orientation)
        error_orientation = self.normalize_angle(self.orientation_target - self.current_orientation)

        # 2) Compute derivative of errors
        derivative_x = error_x - self.prev_error_x
        derivative_y = error_y - self.prev_error_y
        derivative_theta = error_theta - self.prev_error_theta
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

        # 6) Publish velocity commands
        twist_msg = Twist()

        # Before arriving to the waypoint, decide whether to rotate in place or move forward: 
        # If the robot is not heading to the waypoint (with a margin 0.1 rad) and
        # the  waypoint is far away (more than 0.1 m), the robot needs to rotate in place
        # else, once the rotation is completed, enter the next phase, moving forward until
        # the distance between the robot and the waypoint is less than 0.1m
        if not self.is_arrive_waypoint:
            if abs(error_theta)>0.1 and math.hypot(error_x, error_y)>0.1:
                twist_msg.angular.z = min(vtheta, self.max_velo)
                self.get_logger().info( f"rotate before moving forward" )
            elif math.hypot(error_x, error_y)>0.1:
                twist_msg.linear.x = min(math.hypot(vx,vy), self.max_velo)
                twist_msg.angular.z = min(vtheta, self.max_velo)
                self.get_logger().info( f"moving forward" )                      
            else:
                self.is_arrive_waypoint = True
                # arrive the waypoint
                pass
        # After arriving to the waypoint, rotate in place to the target orientation: 
        else:
            if abs(error_orientation)>0.05:
                twist_msg.angular.z = min(vorientation, self.max_velo)
                self.get_logger().info( f"rotating to target orientation" )
            else:
                # arrive the target orientation
                pass

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
    node = WaypointFollower()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt detected, shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
