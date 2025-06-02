#!/usr/bin/env python3

import rclpy, tf_transformations, math
from rclpy.node import Node
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool

class BugTwo(Node):
    def __init__(self):
        super().__init__("BugTwo_Node")
        self.get_logger().info("Bug Two: Started!")
        self.create_timer(0.05, self.state_machine)
        self.pub = self.create_publisher(Twist, 'cmd_vel', 1)
        self.create_subscription(Pose, 'target', self.target_callback, 1)
        self.create_subscription(Odometry, 'odom', self.odom_callback, 1)
        self.create_subscription(LaserScan, '/scan', self.lidar_callback, 1)

        self.close_enough_pub = self.create_publisher(Bool, "close_enough", 1)
        self.goal_reached_pub = self.create_publisher(Bool, "goal_reached", 1)  # <- NUEVO

        self.current_pose = []
        self.target_pose = []
        self.target_flag = False
        self.first_time_flag = True
        self.msg = Twist()
        self.avoiding_wall = False

        self.tolerance = 0.05
        self.k_linear = 0.35
        self.k_angular = 0.75

        self.state = "StopRobot"
        self.avoid_direction = "right"

        self.robot_view = []

        self.initialX = 0.0
        self.initialY = 0.0
        self.a = 0.0
        self.b = 0.0
        self.c = 0.0

        self.mdist_epsilon = 0.15
        self.safe_distance = 0.25
        self.last_log = ""

    def lidar_callback(self, data):
        ranges = list(data.ranges)
        for i in range(len(ranges)):
            if ranges[i] > data.range_max: ranges[i] = data.range_max + 0.01
            if ranges[i] < data.range_min: ranges[i] = data.range_min - 0.01

        self.robot_view = {
            'front_right': min(ranges[293:337]),
            'right': min(ranges[248:292]),
            'back': min(ranges[113:247]),
            'left': min(ranges[68:112]),
            'front_left': min(ranges[23:67]),
            'front': min(min(ranges[0:22]), min(ranges[338:359]))
        }

    def stop_robot(self):
        self.move_robot(0.0, 0.0)
        if self.first_time_flag and self.last_log != "Stopping Robot":
            print("Stopping Robot")
        self.last_log = "Stopping Robot"
        self.first_time_flag = False
        if self.last_log != "":
            self.last_log = ""

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def go_to_goal(self):
        if len(self.current_pose) < 3 or len(self.target_pose) < 3:
            return

        Ex = self.target_pose[0] - self.current_pose[0]
        Ey = self.target_pose[1] - self.current_pose[1]
        distance_to_target = math.hypot(Ex, Ey)
        desired_heading = math.atan2(Ey, Ex)
        heading_error = self.normalize_angle(desired_heading - self.current_pose[2])

        if abs(heading_error) > 0.1:
            w = self.k_angular * heading_error
            w = max(min(w, 0.75), -0.75)
            v = 0.0
            self.move_robot(v, w)
        else:
            v = self.k_linear * distance_to_target
            v = min(v, 0.2)
            w = self.k_angular * heading_error
            w = max(min(w, 0.3), -0.3)
            self.move_robot(v, w)

    def got_New_Target(self):
        return self.target_flag

    def at_Target(self):
        if len(self.current_pose) < 3 or len(self.target_pose) < 3:
            return False

        Ex = self.target_pose[0] - self.current_pose[0]
        Ey = self.target_pose[1] - self.current_pose[1]
        distance_to_target = math.hypot(Ex, Ey)

        print('\x1b[2K', end='\r')
        print("Distance to target = " + str(distance_to_target), end='\r')

        # Publicar si está cerca (40 cm)
        if distance_to_target < 0.4:
            self.close_enough_pub.publish(Bool(data=True))
        else:
            self.close_enough_pub.publish(Bool(data=False))

        # Publicar goal_reached si llegó al punto
        if distance_to_target < self.tolerance:
            self.goal_reached_pub.publish(Bool(data=True))
            if self.last_log != "Arrived to Target":
                print("\nArrived to Target")
                self.last_log = "Arrived to Target"
            self.first_time_flag = True
            self.target_pose = []
            self.target_flag = False
            if self.last_log != "":
                self.last_log = ""
            return True
        else:
            self.goal_reached_pub.publish(Bool(data=False))
            return False

    def target_callback(self, msg):
        new_target = [msg.x, msg.y, msg.theta]
        if len(self.target_pose) == 0 or self.target_pose != new_target:
            self.target_pose = new_target
            self.target_flag = True
            self.targetX = self.target_pose[0]
            self.targetY = self.target_pose[1]
            self.initialX = self.current_pose[0]
            self.initialY = self.current_pose[1]
            if self.last_log != "New target obtained!":
                print("New target obtained!")
                self.last_log = "New target obtained!"
            if self.last_log != "":
                self.last_log = ""

    def move_robot(self, v, w):
        self.msg.linear.x = v
        self.msg.angular.z = w
        self.pub.publish(self.msg)

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        _, _, yaw = tf_transformations.euler_from_quaternion(q)
        self.current_pose = [x, y, yaw]

    def state_machine(self):
        if len(self.current_pose) > 0 and len(self.robot_view) > 0:
            if self.state == "StopRobot":
                self.stop_robot()
            if self.state == "GoToGoal":
                self.go_to_goal()
            if self.state == "FollowWall":
                self.follow_wall(self.avoid_direction)

            if self.state == "StopRobot" and self.got_New_Target():
                self.state = "GoToGoal"
            if self.state == "GoToGoal" and self.at_Target():
                self.state = "StopRobot"
            if self.state == "GoToGoal" and self.isObstacleTooClose():
                self.state = "FollowWall"
            if self.state == "FollowWall" and self.lineAgainWithProgress():
                self.state = "GoToGoal"

    # Métodos auxiliares (iguales que antes, omitidos aquí por brevedad)

    def follow_wall(self, direction):
        # (misma lógica de tu código original)
        pass

    def isObstacleTooClose(self):
        # (misma lógica de tu código original)
        pass

    def lineAgainWithProgress(self):
        # (misma lógica de tu código original)
        pass

def main(args=None):
    rclpy.init(args=args)
    nodeh = BugTwo()
    try:
        rclpy.spin(nodeh)
    except Exception as error:
        print(error)
    except KeyboardInterrupt:
        print("Node stopped by user!")

if __name__ == "__main__":
    main()
