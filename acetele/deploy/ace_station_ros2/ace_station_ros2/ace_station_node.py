import rclpy
from rclpy.node import Node

from acetele.station.base_station import make_station


def main():
    rclpy.init()
    station_node = make_station()
    print(type(station_node))
    if not isinstance(station_node, Node):
        raise ValueError("Station node must be a rclpy.node.Node instance")
    try:
        rclpy.spin(station_node)
    except KeyboardInterrupt:
        pass
    finally:
        station_node.close()
        station_node.destroy_node()
