#!/usr/bin/env python3

import rclpy, tf_transformations, math
from rclpy.node import Node
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan



class BugTwo(Node):
    def __init__(self):
        super().__init__("BugTwo_Node")
        self.get_logger().info("Bug Two: Started!")
        self.create_timer(0.1, self.state_machine)
        self.pub = self.create_publisher(Twist, 'cmd_vel', 1)
        self.create_subscription(Pose, 'target', self.target_callback,1)
        self.create_subscription(Odometry, 'ground_truth', self.odom_callback,1)
        self.create_subscription(LaserScan, '/scan', self.lidar_callback,1)

        self.current_pose = []
        self.target_pose = []
        self.target_flag = False
        self.first_time_flag = True
        self.msg = Twist()
        self.distance_to_target = None
        self.avoiding_wall = False

        self.tolerance = 0.05
        self.k_linear = 0.45   # Ganancia para velocidad lineal
        self.k_angular = 1.00 # Ganancia a para velocidad angular

        self.state = "StopRobot"

        self.robot_view = []

        self.initialX = 0.0
        self.initialY = 0.0
        self.initialT = 0.0

        self.mdist_epsilon = 0.10
        self.safe_distance = 0.30

        self.Wall_hit = []
        self.Wall_leave = []

        self.initialX = 0.0
        self.initialY = 0.0
        
        self.a = 0.0
        self.b = 0.0
        self.c = 0.0




    def lidar_callback(self, data):
        ranges = list(data.ranges)
        for i in range(len(ranges)):
            if ranges[i] > data.range_max: ranges[i] = data.range_max + 0.01
            if ranges[i] < data.range_min: ranges[i] = data.range_min - 0.01
        
        self.robot_view = {
                            'front_right'  : min(ranges[293:337]),
                            'right'        : min(ranges[248:292]),
                            'back'         : min(ranges[113:247]),
                            'left'         : min(ranges[ 68:112]),
                            'front_left'   : min(ranges[ 23: 67]),
                            'front'        : min(min(ranges[0 :22]),min(ranges[338:359]))
                        }

    def stop_robot(self):
        self.move_robot(0.0, 0.0)
        if self.first_time_flag == True: print("Stopping Robot")
        self.first_time_flag = False

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))


    def go_to_goal(self):
        Ex =  self.target_pose[0] - self.current_pose[0]
        Ey =  self.target_pose[1] - self.current_pose[1]
        distance_to_target = math.hypot(Ex, Ey)
        desired_heading = math.atan2(Ey, Ex)
        heading_error = self.normalize_angle(desired_heading - self.current_pose[2])

        #El movimiento se divide en 2 fases, para minimizar errores. 
        #Fase 1: Girar hacia el objetivo
        if abs(heading_error) > 0.1:
            w = self.k_angular * heading_error
            w = max(min(w, 0.75), -0.75)
            v = 0.0
            self.move_robot(v,w)
        else:
        #Envio de las velocidades para el sistem
            v = self.k_linear * distance_to_target
            v  = min(v , 0.3)
            w = self.k_angular * heading_error
            w = max(min(w, 0.5), -0.5)
            
            self.move_robot(v,w)


    def got_New_Target(self):
        return self.target_flag
    
    def at_Target(self):
        Ex =  self.target_pose[0] - self.current_pose[0]
        Ey =  self.target_pose[1] - self.current_pose[1]
        distance_to_target = math.hypot(Ex, Ey)
        print('\x1b[2K', end = '\r')
        print("Distance to target = " + str(distance_to_target), end = '\r')
        if distance_to_target < self.tolerance:
            print("\nArrived to Target")
            self.first_time_flag = True
            self.target_pose = []
            self.target_flag = False
            return True
        else:
            return False

    

    def target_callback(self,msg):
        new_target = [msg.x, msg.y, msg.theta]
        if len(self.target_pose) == 0 or self.target_pose != new_target:
            self.target_pose = new_target
            self.target_flag = True
            self.targetX = self.target_pose[0]
            self.targetY = self.target_pose[1]
            self.initialX = self.current_pose[0]
            self.initialY = self.current_pose[1]
            print("New target obtained!")

    def move_robot(self,v,w):
        self.msg.linear.x = v
        self.msg.angular.z = w
        self.pub.publish(self.msg)


    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        _,_,yaw =tf_transformations.euler_from_quaternion(q)
        self.current_pose = [x,y,yaw]

    
    def follow_wall(self, direction):
        if self.first_time_flag == True: print("Following Wall...")
        self.first_time_flag = False

        #Define P1 (r1, theta1) and P2 (r2, theta2)
        if direction == 'left':
            theta2, r2 =  math.radians(45), self.robot_view.get('front_left')
            theta1, r1 =  math.radians(90), self.robot_view.get('left')
            print('\nWall is on the left')
        else:
            theta2, r2 =  math.radians(-45), self.robot_view.get('front_right')
            theta1, r1 =  math.radians(-90), self.robot_view.get('right')
            print('\nWall is on the right')

        #Compute utan = P2 - P1
        P1x, P1y = r1*math.cos(theta1), r1*math.sin(theta1)
        Ux_tan, Uy_tan = r2*math.cos(theta2) - P1x, r2*math.sin(theta2) -P1y

        #Unitary vector
        norm = math.hypot(Ux_tan, Uy_tan)
        Ux_tan_n, Uy_tan_n = Ux_tan/norm, Uy_tan/norm

        #uper and unitary vector
        dot = P1x*Ux_tan_n + P1y * Uy_tan_n
        Ux_per, Uy_per = P1x - dot*Ux_tan_n, P1y - dot*Uy_tan_n
        norm_per = math.hypot(Ux_per, Uy_per)
        Ux_per_n, Uy_per_n = Ux_per/norm_per, Uy_per/norm_per

        #Compute Follow Wall angle
        dwall, betha, Kfw = 0.30, 0.55, 1.0
        Ex_per, Ey_per = Ux_per - dwall*Ux_per_n, Uy_per - dwall*Uy_per_n
        angle_per = math.atan2(Ey_per, Ex_per)
        angle_tan = math.atan2(Uy_tan_n, Ux_tan_n)
        fw_angle = betha*angle_tan + (1-betha)*angle_per
        fw_angle = math.atan2(math.sin(fw_angle),math.cos(fw_angle))
        #Move Robot 
        v = 0.1 if abs(fw_angle) > 0.1 else 0.2
        if abs(fw_angle) < 0.05:
            fw_angle = 0.0
        w = Kfw * fw_angle
        self.move_robot(v,w)
    
    def isPathAheadCleared(self):
        distances= [self.robot_view.get("front_left"),
                     self.robot_view.get("front"),
                     self.robot_view.get("front_right")]
        distance_to_obstacle = min(distances)
        print('\x1b[2k', end = '\r')
        print("Distance to clear = " + str(distance_to_obstacle), end = '\r')
        if distance_to_obstacle >= self.safe_distance:
            if self.first_time_flag == True: print("\nObstacle Cleared!")
            self.first_time_flag = False
            return True
        else:
            return False

    def lineAgainWithProgress(self):

        DX, DY = self.targetX - self.initialX, self.targetY - self.initialY
        
        PX, PY = self.current_pose[0] - self.initialX, self.current_pose[1] - self.initialY
        
        d = (abs(self.a*self.current_pose[0]+self.b*self.current_pose[1]+self.c))/math.sqrt(self.a*2+self.b*2)
        line = self.a*self.current_pose[0]+self.b*self.current_pose[1]+self.c

        #avance actual
        dot = DX * PX + DY * PY
        norm_Mline = math.hypot(DX,DY)
        current_progress = dot/norm_Mline

        self.d_actual = d
        self.progress = current_progress

        on_Mline = d < self.mdist_epsilon
        progress_check = self.progress > self.progress_at_hit


        if on_Mline and progress_check:
            self.avoiding_wall = False
            print("\nLine has been found!")
            return True
        else:
            return False
        
    
    def isObstacleTooClose(self):
        if hasattr(self, 'avoiding_wall') and self.avoiding_wall:
            return False
        
        
        distances= [self.robot_view.get("front_left"),
                     self.robot_view.get("front"),
                     self.robot_view.get("front_right")]
        if min(distances) < self.safe_distance:
            print ("\nWall detected! Avoiding Now")
            
            
            DX, DY = self.targetX - self.initialX, self.targetY - self.initialY
            PX, PY = self.current_pose[0] - self.initialX, self.current_pose[1] - self.initialY

            self.a = self.targetY - self.initialY
            self.b = -(self.targetX - self.initialX)
            self.c = self.targetX*self.initialY - self.targetY*self.initialX

            self.d_hit = (abs(self.a*self.current_pose[0]+self.b*self.current_pose[1]+self.c))/math.sqrt(self.a*2+self.b*2)
            dot = DX * PX + DY * PY
            norm_Mline = math.hypot(DX,DY)
            self.progress_at_hit = dot/norm_Mline

            self.first_time_flag = True
            return True
        else:
            return False
        


    def state_machine(self):
        if len(self.current_pose) > 0 and len(self.robot_view) > 0:
            #states
            if self.state == "StopRobot": self.stop_robot()
            if self.state == "GoToGoal": self.go_to_goal()
            if self.state == "FollowWall": self.follow_wall('left')
            #Changes
            if self.state == "StopRobot" and self.got_New_Target(): self.state = "GoToGoal"
            if self.state == "GoToGoal" and self.at_Target(): self.state = "StopRobot"
            if self.state == "GoToGoal" and self.isObstacleTooClose(): self.state = "FollowWall"
            if self.state == "FollowWall" and self.lineAgainWithProgress():
                self.avoiding_wall = False
                self.state = "GoToGoal"
            


def main(args=None):
    rclpy.init(args=args)
    nodeh = BugTwo()
    try: rclpy.spin(nodeh)
    except Exception as error: print(error)
    except KeyboardInterrupt: print("Node stopped by user!")

if __name__ == "__main__":
    main()