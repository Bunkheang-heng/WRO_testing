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

        # Colorize depth image
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Get image dimensions
        height, width = depth_image.shape

        # Define the pixel to sample (e.g., center)
        sample_point = (width // 2, height // 2)

        # Get distance at the sample point
        distance = depth_frame.get_distance(*sample_point)
        print(f"Distance at {sample_point}: {distance:.2f} meters")

        # Draw a marker (circle) at the sample point
        cv2.circle(depth_colormap, sample_point, radius=5, color=(0, 255, 0), thickness=-1)

        # Display the depth image with marker
        cv2.imshow('Depth Frame with Marker', depth_colormap)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
