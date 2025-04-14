#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header

class OdometryPathPublisher(Node):
    """
    ROS2 node that:
    1. Subscribes to /odom/groundtruth (Odometry)
    2. Republishes received messages to /Odometry
    3. Maintains and publishes a Path message containing all received poses
    """
    
    def __init__(self):
        super().__init__('odometry_path_publisher')
        
        # Subscribe to ground truth odometry
        self.subscription = self.create_subscription(
            Odometry,
            '/odom/ground_truth',
            self.odom_callback,
            10
        )
        
        # Publisher for republishing Odometry
        self.odom_pub = self.create_publisher(Odometry, '/Odometry', 10)
        
        # Publisher for path history
        self.path_pub = self.create_publisher(Path, '/path', 10)
        
        # Initialize path message
        self.path_msg = Path()
        self.path_msg.header = Header()
        self.path_msg.header.frame_id = ''  # Will be set from first message
        
        self.get_logger().info("Node initialized. Waiting for odometry data...")

    def odom_callback(self, msg):
        """Handle incoming odometry messages"""
        # Republish the odometry message
        self.republish_odometry(msg)
        
        # Update path with new pose
        self.update_path(msg)
        
        # Publish updated path
        self.publish_path()

    def republish_odometry(self, msg):
        """Republish received odometry to new topic"""
        new_msg = Odometry()
        new_msg.header = msg.header
        new_msg.child_frame_id = msg.child_frame_id
        new_msg.pose = msg.pose
        new_msg.twist = msg.twist
        self.odom_pub.publish(new_msg)

    def update_path(self, msg):
        """Update path with new pose from odometry message"""
        # Set frame_id from first message
        if not self.path_msg.header.frame_id:
            self.path_msg.header.frame_id = msg.header.frame_id
            self.get_logger().info(f"Initialized path with frame_id: {self.path_msg.header.frame_id}")
        
        # Create PoseStamped for path
        pose_stamped = PoseStamped()
        pose_stamped.header = msg.header
        pose_stamped.pose = msg.pose.pose
        
        # Add to path
        self.path_msg.poses.append(pose_stamped)
        
        # Update path timestamp
        self.path_msg.header.stamp = self.get_clock().now().to_msg()

    def publish_path(self):
        """Publish the current path"""
        self.path_pub.publish(self.path_msg)
        self.get_logger().debug(f"Published path with {len(self.path_msg.poses)} poses", throttle_duration_sec=1)

def main():
    import sys
    args = sys.argv 

    rclpy.init(args=args)

    node = OdometryPathPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
