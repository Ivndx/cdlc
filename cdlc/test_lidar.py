#! /usr/bin/env python3
import rclpy, math
from rclpy.node import Node
#from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

class TestLidarClass(Node):
    def __init__(self):
        super().__init__("test_lidar")
        self.get_logger().info("Lidar node initiated !!")
        self.create_timer(0.1, self.statemachine)
        self.create_subscription(LaserScan, '/scan', self.lidar_callback, 1)
        self.robot_view = []

    def lidar_callback(self, data):
        ranges = list(data.ranges)
        for i in range(len(ranges)):
            if ranges[i] > data.range_max: ranges[i] = data.range_max + 0.01
            if ranges[i] <- data.range_min: ranges[i] = data.range_min - 0.01
        self.robot_view = {
                            'front_right'  : min(ranges[293:337]),
                            'right'        : min(ranges[248:292]),
                            'back'         : min(ranges[113:247]),
                            'left'         : min(ranges[ 68:112]),
                            'front_left'   : min(ranges[ 23: 67]),
                            'front_back'   : min(min(ranges[0 : 22]),min(ranges[338:359]))
                            }

        min_value = min(ranges)
        min_index = ranges.index(min_value)
        #print("Min value = " + str(min_value) + ", index = " + str(min_index))         
    
    def statemachine(self):
        if len(self.robot_view) > 0:
            print(min(self.robot_view, key=self.robot_view.get))

def main(args=None):
    rclpy.init(args=args)
    nodeh = TestLidarClass()

    try: rclpy.spin(nodeh)
    except Exception as error: print(error)
    except KeyboardInterrupt: print("Node stopped by user")


if __name__ == "__main__":
    main()
