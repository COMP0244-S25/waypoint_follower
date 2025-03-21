#!/usr/bin/env python3

import math
import sys
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header

class CosineCurvePublisher(Node):
    """
    ROS2 node that generates a cosine curve path starting at the robot's current position.
    The amplitude of the cosine curve reduces from 1.0 to 0.0 over the length of the path.
    The path is published as a nav_msgs/Path message on the /planner/path topic.
    """

    def __init__(self, odom_topic):
        super().__init__('cosine_curve_publisher')
        
        self.odom_topic = odom_topic
        self.initial_position = None  # (x, y)
        
        # Subscribe to odometry to get initial position
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            10
        )
        
        # Publisher for the path
        self.path_pub = self.create_publisher(Path, '/planner/path', 10)
        
        self.get_logger().info(f"Node initialized. Waiting for odometry on {self.odom_topic}...")

    def odom_callback(self, msg):
        """Capture the initial position and generate the path once."""
        if self.initial_position is None:
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            self.initial_position = (x, y)
            self.get_logger().info(f"Initial position received: x={x}, y={y}. Generating cosine curve path...")
            self.generate_and_publish_path(msg.header.frame_id, x, y)

    def generate_and_publish_path(self, frame_id, x0, y0):
        """
        Generate a cosine curve path starting at (x0, y0).
        The amplitude of the cosine curve reduces from 1.0 to 0.0 over the length of the path.
        The curve is defined by:
        - Amplitude (A): Reduces linearly from 1.0 to 0.0
        - Frequency (f): Number of oscillations per unit distance
        - Step size (step_x): Distance between consecutive points
        - Total length (L): Total distance of the path
        """
        f = 0.5  # Frequency of the cosine curve (oscillations per unit distance)
        step_x = 0.1  # Step size for x-axis
        L = 3.0  # Total length of the path
        
        # Generate dense points starting at (x0, y0)
        dense_points = []
        x = x0
        while x <= x0 + L:
            # Calculate the amplitude at this point (reduces linearly from 1.0 to 0.0)
            A = 1.0 - ((x - x0) / L)  # Amplitude reduces from 1.0 to 0.0
            y = y0 + A * math.sin(2 * math.pi * f * (x - x0))  # Use sine to start at (x0, y0)
            dense_points.append((x, y))
            x += step_x

        # Downsample points maintaining minimum distance
        sparse_points = []
        last_point = None
        for point in dense_points:
            if last_point is None:
                sparse_points.append(point)
                last_point = point
            else:
                delta_x = point[0] - last_point[0]
                delta_y = point[1] - last_point[1]
                if math.hypot(delta_x, delta_y) >= 0.5:  # Minimum distance between points
                    sparse_points.append(point)
                    last_point = point

        # Create and publish Path message
        path_msg = Path()
        path_msg.header = Header(
            stamp=self.get_clock().now().to_msg(),
            frame_id=frame_id
        )
        
        for x, y in sparse_points:
            pose = PoseStamped()
            pose.header.stamp = path_msg.header.stamp
            pose.header.frame_id = frame_id
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0  # Neutral orientation
            path_msg.poses.append(pose)
        
        self.path_pub.publish(path_msg)
        self.get_logger().info(f"Published cosine curve path starting at ({x0}, {y0}) with {len(sparse_points)} points.")

def main(args=None):
    rclpy.init(args=args)
    
    # Parse command-line argument for odometry topic
    odom_topic = '/Odometry'  # Default topic
    if len(sys.argv) > 1:
        odom_topic = sys.argv[1]
    
    node = CosineCurvePublisher(odom_topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
