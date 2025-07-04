#!/usr/bin/env python3

import rclpy, tf_transformations, math
from rclpy.node import Node
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool

class BugZero(Node):
    def __init__(self):
        super().__init__("BugZero_Node")
        self.get_logger().info("Bug Zero: Started!")
        self.create_timer(0.1, self.state_machine)

        self.pub = self.create_publisher(Twist, "cmd_vel", 1)
        self.create_subscription(Pose, 'target', self.target_callback, 1)
        self.create_subscription(Odometry, 'odom', self.odom_callback, 1)
        self.create_subscription(LaserScan, '/scan', self.lidar_callback, 1)

        # NUEVAS BANDERAS
        self.close_enough_pub = self.create_publisher(Bool, "close_enough", 1)
        self.goal_reached_pub = self.create_publisher(Bool, "goal_reached", 1)

        self.current_pose = []
        self.target_pose = []
        self.target_flag = False
        self.first_time_flag = True
        self.msg = Twist()

        self.tolerance = 0.05
        self.k_linear = 0.55
        self.k_angular = 1.00

        self.state = "StopRobot"
        self.robot_view = []

    def lidar_callback(self, data):
        ranges = list(data.ranges)
        for i in range(len(ranges)):
            if ranges[i] > data.range_max: ranges[i] = data.range_max + 0.01
            if ranges[i] < data.range_min: ranges[i] = data.range_min - 0.01
        
        self.robot_view = {
                            'front_right'  : min(ranges[887:1013]),
                            'right'        : min(ranges[742:886]),
                            'back'         : min(ranges[337:741]),
                            'left'         : min(ranges[202:336]),
                            'front_left'   : min(ranges[ 67: 201]),
                            'front'        : min(min(ranges[:66]),min(ranges[1014:]))
                        }

    def stop_robot(self):
        self.move_robot(0.0, 0.0)
        if self.first_time_flag:
            print("Stopping Robot")
        self.first_time_flag = False

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def go_to_goal(self):
        Ex = self.target_pose[0] - self.current_pose[0]
        Ey = self.target_pose[1] - self.current_pose[1]
        distance_to_target = math.hypot(Ex, Ey)
        desired_heading = math.atan2(Ey, Ex)
        heading_error = self.normalize_angle(desired_heading - self.current_pose[2])

        if abs(heading_error) > 0.1:
            w = self.k_angular * heading_error
            w = max(min(w, 0.25), -0.25)
            v = 0.0
            self.move_robot(v, w)
        else:
            v = self.k_linear * distance_to_target
            v = min(v, 0.1)
            w = self.k_angular * heading_error
            w = max(min(w, 0.35), -0.35)
            self.move_robot(v, w)

    def got_New_Target(self):
        return self.target_flag

    def at_Target(self):
        Ex = self.target_pose[0] - self.current_pose[0]
        Ey = self.target_pose[1] - self.current_pose[1]
        distance_to_target = math.hypot(Ex, Ey)

        print('\x1b[2K', end='\r')
        print("Distance to target = " + str(distance_to_target), end='\r')

        # Publicar close_enough
        if distance_to_target < 0.5:
            self.close_enough_pub.publish(Bool(data=True))
        else:
            self.close_enough_pub.publish(Bool(data=False))

        # Publicar goal_reached
        if distance_to_target < self.tolerance:
            self.goal_reached_pub.publish(Bool(data=True))
            print("\nArrived to Target")
            self.first_time_flag = True
            self.target_pose = []
            self.target_flag = False
            return True
        else:
            self.goal_reached_pub.publish(Bool(data=False))
            return False

    def target_callback(self, msg):
        new_target = [msg.x, msg.y, msg.theta]
        if len(self.target_pose) == 0 or self.target_pose != new_target:
            self.target_pose = new_target
            self.target_flag = True
            print("New target obtained!")

    def move_robot(self, v, w):
        self.msg.linear.x = v
        self.msg.angular.z = w
        self.pub.publish(self.msg)

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = [msg.pose.pose.orientation.x,
             msg.pose.pose.orientation.y,
             msg.pose.pose.orientation.z,
             msg.pose.pose.orientation.w]
        _, _, yaw = tf_transformations.euler_from_quaternion(q)
        self.current_pose = [x, y, yaw]

    def isObstacleTooClose(self):
        distances = [self.robot_view.get("front_left"),
                     self.robot_view.get("front"),
                     self.robot_view.get("front_right")]
        if min(distances) < 0.30:
            if self.first_time_flag:
                print("\nObstacle Too Close!")
            self.first_time_flag = False
            return True
        else:
            return False

    def isPathAheadCleared(self):
        distances = [self.robot_view.get("front_left"),
                     self.robot_view.get("front"),
                     self.robot_view.get("front_right")]
        distance_to_obstacle = min(distances)
        print('\x1b[2K', end='\r')
        print("Distance to clear = " + str(distance_to_obstacle), end='\r')
        if distance_to_obstacle >= 0.30:
            if self.first_time_flag:
                print("\nObstacle Cleared!")
            self.first_time_flag = False
            return True
        else:
            return False

    def follow_wall(self, direction):
        if self.first_time_flag:
            print("Following Wall...")
        self.first_time_flag = False

        if direction == 'left':
            theta2, r2 = math.radians(45), self.robot_view.get('front_left')
            theta1, r1 = math.radians(90), self.robot_view.get('left')
            print('Wall is on the left')
        else:
            theta2, r2 = math.radians(-45), self.robot_view.get('front_right')
            theta1, r1 = math.radians(-90), self.robot_view.get('right')
            print('Wall is on the right')

        P1x, P1y = r1 * math.cos(theta1), r1 * math.sin(theta1)
        Ux_tan, Uy_tan = r2 * math.cos(theta2) - P1x, r2 * math.sin(theta2) - P1y

        norm = math.hypot(Ux_tan, Uy_tan)
        Ux_tan_n, Uy_tan_n = Ux_tan / norm, Uy_tan / norm

        dot = P1x * Ux_tan_n + P1y * Uy_tan_n
        Ux_per, Uy_per = P1x - dot * Ux_tan_n, P1y - dot * Uy_tan_n
        norm_per = math.hypot(Ux_per, Uy_per)
        Ux_per_n, Uy_per_n = Ux_per / norm_per, Uy_per / norm_per

        dwall, betha, Kfw = 0.30, 0.65, 0.55
        Ex_per, Ey_per = Ux_per - dwall * Ux_per_n, Uy_per - dwall * Uy_per_n
        angle_per = math.atan2(Ey_per, Ex_per)
        angle_tan = math.atan2(Uy_tan_n, Ux_tan_n)
        fw_angle = betha * angle_tan + (1 - betha) * angle_per
        fw_angle = math.atan2(math.sin(fw_angle), math.cos(fw_angle))

        v = 0.075 if abs(fw_angle) > 0.1 else 0.075
        w = Kfw * fw_angle
        self.move_robot(v, w)

    def state_machine(self):
        if len(self.current_pose) > 0:
            if self.state == "StopRobot":
                self.stop_robot()
            if self.state == "GoToGoal":
                self.go_to_goal()
            if self.state == "FollowWall":
                self.follow_wall('left')

            if self.state == "StopRobot" and self.got_New_Target():
                self.state = "GoToGoal"
            if self.state == "GoToGoal" and self.at_Target():
                self.state = "StopRobot"
            if self.state == "GoToGoal" and self.isObstacleTooClose():
                self.state = "FollowWall"
            if self.state == "FollowWall" and self.isPathAheadCleared():
                self.state = "GoToGoal"


def main(args=None):
    rclpy.init(args=args)
    node = BugZero()
    try:
        rclpy.spin(node)
    except Exception as error:
        print(error)
    except KeyboardInterrupt:
        print("Node stopped by user!")

if __name__ == "__main__":
    main()