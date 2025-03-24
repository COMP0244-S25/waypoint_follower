#!/usr/bin/env python3

import math
import sys
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header

class ComplexPathPublisher(Node):
    """
    Generates a path containing:
    1. Cosine curve with decaying amplitude (3.0m)
    2. Straight line in original direction (1.5m)
    3. 90-degree turn and straight line in perpendicular direction (1.5m)
    """

    def __init__(self, odom_topic):
        super().__init__('complex_path_publisher')
        self.odom_topic = odom_topic
        self.initial_position = None
        
        # Odometry subscription
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            10
        )
        
        # Path publisher
        self.path_pub = self.create_publisher(Path, '/planner/path', 10)
        self.get_logger().info(f"Waiting for initial position from {self.odom_topic}...")

    def odom_callback(self, msg):
        """Capture initial position once and generate path"""
        if self.initial_position is None:
            x0 = msg.pose.pose.position.x
            y0 = msg.pose.pose.position.y
            self.initial_position = (x0, y0)
            self.get_logger().info(f"Initial position: ({x0:.2f}, {y0:.2f})")
            self.generate_and_publish_path(msg.header.frame_id, x0, y0)

    def generate_and_publish_path(self, frame_id, x0, y0):
        """Generate the complete path with three segments"""
        # Configuration parameters
        step_size = 0.1  # Distance between consecutive points
        min_point_dist = 0.5  # Minimum distance between sparse points
        
        # 1. Cosine curve parameters
        cosine_length = 7.0
        frequency = 0.5  # Oscillations per meter
        
        # 2. First straight line parameters
        straight1_length = 1.5  # In original direction
        
        # 3. Second straight line parameters
        straight2_length = 1.5  # In perpendicular direction

        # Generate dense points for all segments
        dense_points = []
        
        # 1. First straight line (original direction)
        straight1_end = x0 + straight1_length
        x_current = x0 + step_size  # Start from initial position
        while x_current <= straight1_end:
            dense_points.append((x_current, y0))
            x_current += step_size

        # 2. Second straight line (perpendicular direction)
        y_start = y0
        y_end = y_start + straight2_length
        y_current = y_start + step_size
        while y_current <= y_end:
            dense_points.append((straight1_end, y_current))
            y_current += step_size

        # 3. Generate cosine curve starting from end of perpendicular line
        cosine_start_x = straight1_end
        cosine_start_y = y_end
        x = cosine_start_x
        while x <= cosine_start_x + cosine_length:
            # Calculate position relative to cosine start
            rel_x = x - cosine_start_x
            amplitude = 1.0 - (rel_x / cosine_length)
            y = cosine_start_y + amplitude * math.sin(2 * math.pi * frequency * rel_x)
            dense_points.append((x, y))
            x += step_size

        # Downsample points
        sparse_points = self.downsample_points(dense_points, min_point_dist)

        # Create Path message
        path_msg = Path()
        path_msg.header = Header(
            stamp=self.get_clock().now().to_msg(),
            frame_id=frame_id
        )

        for x, y in sparse_points:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0  # Neutral orientation
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)
        self.get_logger().info(f"Published complex path with {len(sparse_points)} points")

    def downsample_points(self, dense_points, min_dist):
        """Downsample points maintaining minimum distance"""
        if not dense_points:
            return []

        sparse_points = [dense_points[0]]
        for point in dense_points[1:]:
            last = sparse_points[-1]
            dx = point[0] - last[0]
            dy = point[1] - last[1]
            if math.hypot(dx, dy) >= min_dist:
                sparse_points.append(point)
        return sparse_points

def main(args=None):
    rclpy.init(args=args)
    odom_topic = sys.argv[1] if len(sys.argv) > 1 else '/Odometry'
    node = ComplexPathPublisher(odom_topic)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
