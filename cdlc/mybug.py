#! /usr/bin/env python3
import rclpy, math, tf_transformations
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from turtlesim.msg import Pose
from sensor_msgs.msg import LaserScan

class BugAlgorithmClass(Node):
    def __init__(self):
        super().__init__("reactive_nav")
        self.get_logger().info("Reactuve navigation node initiated !!")
        self.create_timer(0.1, self.state_machine)
        self.pub = self.create_publisher(Twist, "cmd_vel", 1)
        self.create_subscription(Odometry, "ground_truth", self.odom_callback, 1)
        self.create_subscription(Pose, "target", self.target_callback, 1)
        self.create_subscription(LaserScan, "scan", self.lidar_callback, 1)
        
        self.current_pose = []
        self.target_pose = []
        self.got_new_target = False
        self.msg = Twist()
        self.state = "stop_robot"
        self.first_time_flag = True
        self.tolerance_to_target = 0.05
        self.Kv = 0.5
        self.Kw = 2.0
        self.L = 0.183

        # Límites de velocidad
        self.max_v = 0.1     # [m/s]
        self.max_w = 0.3     # [rad/s]

        self.min_distance = 0.65
        self.clear_distance = 0.7

        self.robot_view = []
        self.ranges = []

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
                            'front'   : min(min(ranges[0 : 22]),min(ranges[338:359]))
                            }

        min_value = min(ranges)
        min_index = ranges.index(min_value)

    def target_callback(self, msg):
        new_target = [msg.x, msg.y, msg.theta]
        if len(self.target_pose) == 0 or self.target_pose != new_target:
            self.target_pose = new_target
            self.got_new_target = True
            print("New target received ")
    
    def moveRobot(self, v, w):
        self. msg.linear.x = v
        self. msg.angular.z = w
        self.pub.publish(self.msg)


    def odom_callback(self, data):
        x = data.pose.pose.position.x
        y = data.pose.pose.position.y
        q = [data.pose.pose.orientation.x, data.pose.pose.orientation.y, 
             data.pose.pose.orientation.z, data.pose.pose.orientation.w]
        _,_,yaw = tf_transformations.euler_from_quaternion(q)
        self.current_pose = [x, y, yaw]
        
    def stop_robot(self):
        self.moveRobot(0.0, 0.0)
        if self.first_time_flag == True:
             print("Stopping robot... ")
        self.first_time_flag = False

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def go_to_goal(self):
        Ex =  self.target_pose[0] - self.current_pose[0]
        Ey =  self.target_pose[1] - self.current_pose[1]
        distance_to_target = math.hypot(Ex, Ey)
        desired_heading = math.atan2(Ey, Ex)
        heading_error = self.normalize_angle(desired_heading - self.current_pose[2])

        if abs(heading_error) > 0.1:
            w = self.Kw * heading_error
            w = max(min(w, 0.4), -0.4)
            v = 0.0
        
        #Envio de las velocidades para el sistema
        v = self.Kv * distance_to_target
        v = min(v , 0.5)
        w = self.Kw * heading_error
        w = max(min(w, 0.7), -0.7)
        self.moveRobot(v,w)
    
    def gotNewTarget(self):
        return self.got_new_target

    def atTarget(self):
        Ex = self.target_pose[0] - self.current_pose[0]
        Ey = self.target_pose[1] - self.current_pose[1]
        distance_to_target = math.hypot(Ex, Ey)
        print('\x1b[2k', end = '\r')
        print("Distance to target = " + str(distance_to_target), end = '\r')
        if distance_to_target < self.tolerance_to_target:
            print("\nia llegue oli oliiii ... ")
            self.first_time_flag = True
            self.target_pose = []
            self.got_new_target = False
            return True
        else:
            return False
    
    def avoid_obstacle(self):
        if self.first_time_flag == True: print("Avoiding obstacle ")
        self.first_time_flag = False
        angles = [90, 45, 0, -45, -90]
        readings = [self.robot_view.get("left"), 
                    self.robot_view.get("front_left"),
                    self.robot_view.get("front"), 
                    self.robot_view.get("front_right"),
                    self.robot_view.get("right")]
        num, den = 0.0, 0.0
        for i in range(len(angles)):
            num += angles[i]*readings[i]
            den += readings[i]
        v = 0.2
        w = 0.3 * math.radians(num/den)
        self.moveRobot(v, w)
    
    def isObstacleTooClose(self):

        distance_to_obstacle = self.robot_view.get("front")

        if distance_to_obstacle < self.min_distance:
            print("\n Hay Obstaculo aaaaaaaaaaah")
            self.first_time_flag = True
            return True
        else:
            return False

    def isObstacleCleared(self):
        distances = [self.robot_view.get("front_left"),
                     self.robot_view.get("front"),
                     self.robot_view.get("front_right")]
        min_distance_to_obstacle = min(distances)
        print('\x1b[2k', end = '\r')
        print("Distance to clear = " + str(min_distance_to_obstacle), end = '\r')
        if min_distance_to_obstacle > self.clear_distance:
            print("\n ia la libraste carnal  ")
            self.first_time_flag = True
            return True
        else:
            return False 


    def state_machine(self):
        if len(self.current_pose) > 0: 
			#States
            if self.state == "stop_robot": self.stop_robot()
            if self.state == "go_to_goal": self.go_to_goal()
            if self.state == "avoid_obstacle": self.avoid_obstacle()

			#Transitions
            if self.state == "stop_robot" and self.gotNewTarget(): self.state = "go_to_goal"
            if self.state == "go_to_goal" and self.atTarget(): self.state = "stop_robot"
            if self.state == "go_to_goal" and self.isObstacleTooClose(): self.state = "avoid_obstacle"
            if self.state == "avoid_obstacle" and self.isObstacleCleared(): self.state = "go_to_goal"

def main(args=None):
    rclpy.init(args=args)
    nodeh = BugAlgorithmClass()

    try: rclpy.spin(nodeh)
    except Exception as error: print(error)
    except KeyboardInterrupt: print("Node stopped by user")


if __name__ == "__main__":
    main()
