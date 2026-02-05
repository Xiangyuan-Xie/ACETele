import rclpy

from acetele.station.base_station import make_station


def main(args=None):
    rclpy.init(args=args)
    station_node = make_station()
    if not isinstance(station_node, rclpy.node.Node):
        raise ValueError("Station node must be a rclpy.node.Node instance")
    try:
        rclpy.spin(station_node)
    except KeyboardInterrupt:
        pass
    finally:
        station_node.close()
        station_node.destroy_node()
