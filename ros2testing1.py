#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import numpy as np

class WallDistanceNode(Node):
    def __init__(self):
        super().__init__('wall_distance_node')
        self.bridge = CvBridge()

        # Subscribe to depth image
        self.subscription = self.create_subscription(
            Image,
            '/camera/depth/image_rect_raw',
            self.depth_callback,
            10
        )

        # Publisher for errors [left_error, right_error]
        self.error_publisher = self.create_publisher(
            Float32MultiArray,
            '/wall_following/errors',
            10
        )

        # Desired distance from wall (meters)
        self.target_distance = 0.5

    def depth_callback(self, msg):
        try:
            # Convert ROS2 Image to OpenCV
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth_array = np.array(depth_image, dtype=np.float32) / 1000.0  # Convert mm to meters

            height, width = depth_array.shape
            center_y = height // 2  # Middle row

            # Define sample points (left and right)
            left_x = int(width * 0.2)
            right_x = int(width * 0.8)

            left_distance = depth_array[center_y, left_x]
            right_distance = depth_array[center_y, right_x]

            # Calculate errors
            left_error = left_distance - self.target_distance if left_distance > 0 else 0.0
            right_error = right_distance - self.target_distance if right_distance > 0 else 0.0

            # Log info
            self.get_logger().info(
                f"Left: {left_distance:.2f}m (Error: {left_error:.2f}m) | "
                f"Right: {right_distance:.2f}m (Error: {right_error:.2f}m)"
            )

            # Publish errors
            error_msg = Float32MultiArray()
            error_msg.data = [left_error, right_error]
            self.error_publisher.publish(error_msg)

        except Exception as e:
            self.get_logger().error(f"Error processing depth image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = WallDistanceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
