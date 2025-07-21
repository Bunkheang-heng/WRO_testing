import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth stream
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        # Convert depth frame to numpy array
        depth_image = np.asanyarray(depth_frame.get_data())

        # Colorize depth image for visualization
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Get image dimensions
        height, width = depth_image.shape
        center_y = height // 2  # Middle row

        # Define sample points for left and right wall distances
        left_point = (int(width * 0.2), center_y)   # 20% from left
        right_point = (int(width * 0.8), center_y)  # 80% from left

        # Get distances at these points
        left_distance = depth_frame.get_distance(*left_point)
        right_distance = depth_frame.get_distance(*right_point)

        # Print distances
        print(f"Left wall: {left_distance:.2f} m | Right wall: {right_distance:.2f} m")

        # Draw markers on the depth image
        cv2.circle(depth_colormap, left_point, radius=6, color=(0, 255, 0), thickness=-1)  # Green for Left
        cv2.circle(depth_colormap, right_point, radius=6, color=(255, 0, 0), thickness=-1) # Blue for Right

        # Display the depth image with markers
        cv2.imshow('Depth Frame with Left & Right Markers', depth_colormap)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
