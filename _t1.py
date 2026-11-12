import pyrealsense2 as rs
print("root-file", len(rs.context().query_devices()))
