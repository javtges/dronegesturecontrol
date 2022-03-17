import pyrealsense2 as rs
import cv2
import numpy as np

pipeline = rs.pipeline()

config = rs.config()
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

colorizer = rs.colorizer()
pipeline.start(config)

profile = pipeline.get_active_profile()
infrared_profile = rs.video_stream_profile(profile.get_stream(rs.stream.infrared, 2))
infrared_intrinsics = infrared_profile.get_intrinsics()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()

print(infrared_profile)
print(infrared_intrinsics)
print(depth_profile)
print(depth_intrinsics)

while True:

    frames = pipeline.wait_for_frames()

    infrared_frame_zero = frames.get_infrared_frame(1)
    infrared_frame_one  = frames.get_infrared_frame(2)
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    infrared_colormap_zero = np.asanyarray(colorizer.colorize(infrared_frame_zero).get_data())
    infrared_colormap_one = np.asanyarray(colorizer.colorize(infrared_frame_one).get_data())
    depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    colormap = np.asanyarray(colorizer.colorize(color_frame).get_data())

    print(infrared_colormap_one.shape)

    images = np.hstack((depth_colormap, colormap))

    cv2.imshow('RealSense', infrared_colormap_zero)

    if cv2.waitKey(25) == ord('q'):
        break






pipeline.stop()
cv2.destroyAllWindows()
