#!/usr/bin/env python3

import math
import sys
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header

class EightShapePathPublisher(Node):
    """
    A ROS2 node that generates an eight-shaped path once the robot's initial position is received via odometry.
    The path is published as a nav_msgs/Path message on the /path topic.
    """

    def __init__(self, odom_topic):
        super().__init__('eight_shape_path_publisher')
        
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
            self.get_logger().info(f"Initial position received: x={x}, y={y}. Generating path...")
            self.generate_and_publish_path(msg.header.frame_id, x, y)

    def generate_and_publish_path(self, frame_id, x0, y0):
        """Generate an eight-shaped path starting at (x0, y0)."""
        a = 0.6  # Scale factor for the eight shape (adjust as needed)
        dt = 0.01  # Step size for parameter t (smaller = smoother)
        min_distance = 0.2  # Minimum distance between consecutive points
        
        # Generate dense points along the eight shape
        dense_points = []
        t = 0.0
        while t < 2 * math.pi:
            # Parametric equations for eight shape (adjusted to start at bottom)
            x = x0 + a * (math.sin(t) - math.sqrt(2)/2)
            y = y0 + a * (math.sin(2 * t) + 1)
            dense_points.append((x, y))
            t += dt
        
        # Downsample to ensure min_distance spacing between points
        sparse_points = []
        last_point = None
        for point in dense_points:
            if last_point is None:
                sparse_points.append(point)
                last_point = point
            else:
                dx = point[0] - last_point[0]
                dy = point[1] - last_point[1]
                distance = math.hypot(dx, dy)
                if distance >= min_distance:
                    sparse_points.append(point)
                    last_point = point
        
        # Add the first point again to close the loop
        if len(sparse_points) > 0:
            sparse_points.append(sparse_points[0])
        
        # Create Path message
        path_msg = Path()
        path_msg.header = Header(
            stamp=self.get_clock().now().to_msg(),
            frame_id=frame_id
        )
        
        for (x, y) in sparse_points:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        
        self.path_pub.publish(path_msg)
        self.get_logger().info(f"Published eight-shape path with {len(sparse_points)} points.")

def main(args=None):
    rclpy.init(args=args)
    
    # Parse command-line argument for odometry topic
    odom_topic = '/Odometry'  # Default topic
    if len(sys.argv) > 1:
        odom_topic = sys.argv[1]
    
    node = EightShapePathPublisher(odom_topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
