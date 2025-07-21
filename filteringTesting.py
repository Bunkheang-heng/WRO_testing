import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth stream
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

# RealSense filters
spatial = rs.spatial_filter()       # Smooth spatial noise
temporal = rs.temporal_filter()     # Reduce temporal flicker
hole_filling = rs.hole_filling_filter()  # Fill small missing holes

# Complementary filter (EMA) parameters
alpha = 0.8  # Smoothing factor (higher = smoother but slower to react)
smoothed_left = 0
smoothed_right = 0

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        # Apply RealSense filters
        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)
        depth_frame = hole_filling.process(depth_frame)

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

        # Reject invalid/outlier readings
        if left_distance <= 0.2 or left_distance > 4.0:
            left_distance = smoothed_left  # Use previous smoothed value
        if right_distance <= 0.2 or right_distance > 4.0:
            right_distance = smoothed_right

        # Apply complementary filter (EMA smoothing)
        smoothed_left = alpha * smoothed_left + (1 - alpha) * left_distance
        smoothed_right = alpha * smoothed_right + (1 - alpha) * right_distance

        # Print smoothed distances
        print(f"Left wall: {smoothed_left:.2f} m | Right wall: {smoothed_right:.2f} m")

        # Draw markers on depth image
        cv2.circle(depth_colormap, left_point, radius=6, color=(0, 255, 0), thickness=-1)  # Green
        cv2.circle(depth_colormap, right_point, radius=6, color=(255, 0, 0), thickness=-1) # Blue

        # Display the filtered depth image
        cv2.imshow('Filtered Depth Frame', depth_colormap)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
