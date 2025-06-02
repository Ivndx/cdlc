## Bug2: Last update: 19/05/25 - 17:15 

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
        self.create_subscription(Pose, 'target', self.target_callback,1)
        self.create_subscription(Odometry, 'odom', self.odom_callback,1)
        self.create_subscription(LaserScan, '/scan', self.lidar_callback,1)
        self.close_enough_pub = self.create_publisher(Bool, "close_enough", 1)

        self.current_pose = []
        self.target_pose = []
        self.target_flag = False
        self.first_time_flag = True
        self.msg = Twist()
        self.avoiding_wall = False

        self.tolerance = 0.05
        self.k_linear = 0.35   # Ganancia para velocidad lineal
        self.k_angular = 0.75 # Ganancia a para velocidad angular

        self.state = "StopRobot"
        self.avoid_direction = "right" #o "left"

        self.robot_view = []

        self.initialX = 0.0
        self.initialY = 0.0

        self.mdist_epsilon = 0.15
        self.safe_distance = 0.25

        self.initialX = 0.0
        self.initialY = 0.0
        
        self.a = 0.0
        self.b = 0.0
        self.c = 0.0

        self.last_log = ""

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
        if self.first_time_flag == True and self.last_log != "Stopping Robot": print("Stopping Robot")
        self.last_log = "Stopping Robot"
        self.first_time_flag = False
        if self.last_log != "": self.last_log = ""

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def go_to_goal(self):
        if len(self.current_pose) < 3 or len(self.target_pose) < 3:
            return
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
        #Envio de las velocidades para el sistema
            v = self.k_linear * distance_to_target
            v  = min(v , 0.2)
            w = self.k_angular * heading_error
            w = max(min(w, 0.3), -0.3)
            
            self.move_robot(v,w)

    def got_New_Target(self):
        return self.target_flag
    
    def at_Target(self):
        if len(self.current_pose) < 3 or len(self.target_pose) < 3:
            return False

        Ex =  self.target_pose[0] - self.current_pose[0]
        Ey =  self.target_pose[1] - self.current_pose[1]
        distance_to_target = math.hypot(Ex, Ey)

        print('\x1b[2K', end = '\r')
        print("Distance to target = " + str(distance_to_target), end = '\r')

        # Publicar si está a menos de 40 cm
        if distance_to_target < 0.4:
            self.close_enough_pub.publish(Bool(data=True))
        else:
            self.close_enough_pub.publish(Bool(data=False))

        if distance_to_target < self.tolerance:
            if self.last_log != "Arrived to Target": 
                print("\nArrived to Target")
                self.last_log = "Arrived to Target"
            self.first_time_flag = True
            self.target_pose = []
            self.target_flag = False
            if self.last_log != "": self.last_log = ""

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
            if self.last_log != "New target obtained!":
                print("New target obtained!")
                self.last_log = "New target obtained!"
            
            if self.last_log != "": self.last_log = ""

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
        if self.first_time_flag == True and self.last_log != "Following Wall...":
            print("Following Wall...")
        self.first_time_flag = False
        self.avoiding_wall = True

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
        dwall, betha, Kfw = 0.20, 0.6 , 1.5
        Ex_per, Ey_per = Ux_per - dwall*Ux_per_n, Uy_per - dwall*Uy_per_n
        angle_per = math.atan2(Ey_per, Ex_per)
        angle_tan = math.atan2(Uy_tan_n, Ux_tan_n)
        fw_angle = betha*angle_tan + (1-betha)*angle_per
        fw_angle = math.atan2(math.sin(fw_angle),math.cos(fw_angle))
        #Move Robot 
        v = max(0.15, 0.25 - abs(fw_angle))   # Reduce velocidad al girar más
        w = Kfw * fw_angle
        w = max(min(w, 4.0), -4.0)
        self.move_robot(v,w)
        if self.last_log != "": self.last_log = ""

    def isObstacleTooClose(self):
        # Si ya hemos llegado al destino, ignorar cualquier pared nueva
        if self.state == "StopRobot":
            return False

        # Si ya estamos evitando una pared, no reiniciamos la lógica
        if self.avoiding_wall:
            return False

        front = self.robot_view.get('front')
        if front < self.safe_distance:
            if self.last_log != "\nWall detected! Avoiding Now": print("\nWall detected! Avoiding Now")
            self.last_log = "\nWall detected! Avoiding Now"
            # Punto de contacto
            self.initialX = self.current_pose[0]
            self.initialY = self.current_pose[1]
            # Parámetros de la M-line ax + by + c = 0
            self.a = self.targetY - self.initialY
            self.b = -(self.targetX - self.initialX)
            self.c = self.targetX * self.initialY - self.targetY * self.initialX
            # Norm of the M-line direction
            DX = self.targetX - self.initialX
            DY = self.targetY - self.initialY
            self.norm_Mline = math.hypot(DX, DY)
            # Avance inicial (en el impacto es cero)
            self.progress_at_hit = 0.0

            self.avoiding_wall = True
        
            return True
        if self.last_log != "": self.last_log = ""

        return False


    def lineAgainWithProgress(self):
        # Si todavía hay obstáculo frontal, no salgas
        if self.robot_view.get("front") < self.safe_distance:
            return False

        # Vector desde el punto de contacto
        PX = self.current_pose[0] - self.initialX
        PY = self.current_pose[1] - self.initialY
        # Dirección de la M-line
        DX = self.targetX - self.initialX
        DY = self.targetY - self.initialY

        # Distancia perpendicular a la M-line
        dist_to_mline = abs(self.a*self.current_pose[0] + self.b*self.current_pose[1] + self.c) \
                        / math.sqrt(self.a*2 + self.b*2)
        # Progreso (proyección escalar normalizada)
        current_progress = (DX*PX + DY*PY) / self.norm_Mline

        # Condición de salida
        if dist_to_mline < self.mdist_epsilon and current_progress > self.progress_at_hit + 0.15:
            if self.last_log != "\nLine has been found! Resuming GoToGoal.": print("\nLine has been found! Resuming GoToGoal.")
            self. last_log != "\nLine has been found! Resuming GoToGoal."
            self.avoiding_wall = False
            return True
        if self.last_log != "": self.last_log = ""
        return False

        
    def state_machine(self):
        if len(self.current_pose) > 0 and len(self.robot_view) > 0:
            #states
            if self.state == "StopRobot": self.stop_robot()
            if self.state == "GoToGoal": self.go_to_goal()
            if self.state == "FollowWall": self.follow_wall(self.avoid_direction)
            #Changes
            if self.state == "StopRobot" and self.got_New_Target(): self.state = "GoToGoal"
            if self.state == "GoToGoal" and self.at_Target(): self.state = "StopRobot"
            if self.state == "GoToGoal" and self.isObstacleTooClose(): self.state = "FollowWall"
            if self.state == "FollowWall" and self.lineAgainWithProgress():self.state = "GoToGoal"

            
def main(args=None):
    rclpy.init(args=args)
    nodeh = BugTwo()
    try: rclpy.spin(nodeh)
    except Exception as error: print(error)
    except KeyboardInterrupt: print("Node stopped by user!")

if __name__ == "__main__":
    main()