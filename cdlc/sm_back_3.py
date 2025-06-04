#!/usr/bin/env python3

import rclpy
import math
import numpy as np
from rclpy.node import Node
from turtlesim.msg import Pose
from std_msgs.msg import Bool
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from aruco_opencv_msgs.msg import ArucoDetection
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

class MainCoordinator(Node):
    def __init__(self):
        super().__init__('main_coordinator')
        self.get_logger().info("SM GOOOOOO ")

        # QoS para ArUco
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )

        # Publishers existentes
        self.target_pub = self.create_publisher(Pose, 'target', 10)
        self.servo_pub = self.create_publisher(Float32, 'ServoAngle', 10)
        self.pub = self.create_publisher(Twist, 'cmd_vel', 1)

        # Subscribers existentes
        self.create_subscription(Bool, 'close_enough', self.close_enough_callback, 10)
        self.create_subscription(Bool, 'goal_reached', self.goal_reached_callback, 10)
        
        # Nuevo subscriber para ArUco
        self.aruco_sub = self.create_subscription(
            ArucoDetection, 
            "aruco_detections", 
            self.aruco_callback, 
            qos_profile
        )
        
        # Subscriber para posición actual del robot
        self.create_subscription(Pose, 'PosicionBot', self.robot_pose_callback, 10)

        self.state = "start"
        self.timer = self.create_timer(0.5, self.state_machine)

        # Variables de estado existentes
        self.close_enough = False
        self.goal_reached = False
        self.dropoff_sent = False
        self.returned_home = False
        self.orientation_started = False
        self.routine_index = 0
        self.servo_counter = 0

        # Variables para ArUco 0
        self.aruco_0_detected = False
        self.aruco_0_position = None  # Posición en coordenadas del mundo
        self.aruco_0_distance = None
        self.last_aruco_position = None  # Para mantener última posición conocida
        self.robot_position = Pose()  # Posición actual del robot
        
        # Parámetros de configuración
        self.aruco_detection_threshold = 0.5  # 50cm para detectar ArUco 0
        self.pickup_final_distance = 0.15     # 15cm distancia final para pick-up
        
        # Matriz de transformación cámara-robot (misma que en odometría)
        self.offset_x = 0.075
        self.offset_z = 0.065
        self.T_cam_robot = np.array([
            [0.0, 0.0, 1.0, self.offset_x],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, self.offset_z],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        # Mensaje de Twist para el movimiento hacia atrás
        self.msg = Twist()

    def aruco_callback(self, msg):
        """
        Callback para procesar detecciones de ArUco
        """
        self.aruco_0_detected = False
        
        for aruco in msg.markers:
            if aruco.marker_id == 0:  # Solo nos interesa el ArUco 0
                self.aruco_0_detected = True
                
                # Transformar posición de cámara a robot
                pos_cam = np.array([
                    [aruco.pose.position.x],
                    [aruco.pose.position.y],
                    [aruco.pose.position.z],
                    [1.0]
                ])
                pos_robot_frame = self.T_cam_robot @ pos_cam
                
                # Calcular posición en coordenadas del mundo
                robot_x = self.robot_position.x
                robot_y = self.robot_position.y
                robot_theta = self.robot_position.theta
                
                # Rotar del frame del robot al frame del mundo
                cos_theta = math.cos(robot_theta)
                sin_theta = math.sin(robot_theta)
                
                aruco_world_x = robot_x + pos_robot_frame[0, 0] * cos_theta - pos_robot_frame[1, 0] * sin_theta
                aruco_world_y = robot_y + pos_robot_frame[0, 0] * sin_theta + pos_robot_frame[1, 0] * cos_theta
                
                self.aruco_0_position = (aruco_world_x, aruco_world_y)
                
                # Calcular distancia al ArUco
                dx = aruco_world_x - robot_x
                dy = aruco_world_y - robot_y
                self.aruco_0_distance = math.sqrt(dx*dx + dy*dy)
                
                self.get_logger().info(f"ArUco 0 detectado en mundo: ({aruco_world_x:.3f}, {aruco_world_y:.3f}), "
                                     f"distancia: {self.aruco_0_distance:.3f}m")
                break

    def robot_pose_callback(self, msg):
        """
        Callback para actualizar la posición actual del robot
        """
        self.robot_position = msg

    def close_enough_callback(self, msg):
        self.close_enough = msg.data

    def goal_reached_callback(self, msg):
        self.goal_reached = msg.data

    def publish_target(self, x, y):
        pose = Pose()
        pose.x = x
        pose.y = y
        pose.theta = 0.0
        self.target_pub.publish(pose)
        self.get_logger().info(f"Sent target: ({x:.2f}, {y:.2f})")

    def generate_straight_approach_points(self, aruco_x, aruco_y):
        """
        Genera puntos de aproximación en línea recta hacia el ArUco
        Mantiene Y constante y solo varía X para crear una línea recta
        """
        # Distancias de aproximación desde el ArUco (en X)
        x_offsets = [0.40, 0.25, 0.10]  # 40cm, 25cm, 10cm antes del ArUco
        
        approach_points = []
        
        # Determinar la dirección de aproximación basada en la posición del robot
        robot_x = self.robot_position.x
        
        # Si el robot está a la izquierda del ArUco, aproximarse desde la izquierda
        # Si está a la derecha, aproximarse desde la derecha
        if robot_x < aruco_x:
            # Aproximarse desde la izquierda (robot va hacia la derecha)
            for offset in x_offsets:
                point_x = aruco_x - offset  # Puntos antes del ArUco en X
                point_y = aruco_y  # Y constante
                approach_points.append((point_x, point_y))
        else:
            # Aproximarse desde la derecha (robot va hacia la izquierda)
            for offset in x_offsets:
                point_x = aruco_x + offset  # Puntos después del ArUco en X
                point_y = aruco_y  # Y constante
                approach_points.append((point_x, point_y))
        
        self.get_logger().info(f"Generados {len(approach_points)} puntos de aproximación recta")
        for i, (px, py) in enumerate(approach_points):
            self.get_logger().info(f"Punto {i+1}: ({px:.3f}, {py:.3f})")
        
        return approach_points

    def state_machine(self):
        if self.state == "start":
            self.get_logger().info("State: GO_TO_PICK_UP")
            angle = 260.0
            self.servo_pub.publish(Float32(data=angle))
            self.state = "go_to_pick_up"
            # Inicialmente ir hacia la posición aproximada
            self.publish_target(1.15, 0.0)
            
        elif self.state == "go_to_pick_up":
            # Siempre ir hacia la coordenada fija inicial
            self.publish_target(1.15, 0.0)
            
            # Verificar si detectamos ArUco 0 y estamos suficientemente cerca (50cm)
            if self.aruco_0_detected and self.aruco_0_distance is not None:
                if self.aruco_0_distance <= self.aruco_detection_threshold:
                    self.get_logger().info(f"ArUco 0 detectado a {self.aruco_0_distance:.2f}m. Switching to PICK_UP_ORIENTATION")
                    self.state = "pick_up_orientation"
                    self.routine_index = 0
                    self.goal_reached = False
                    self.orientation_started = False


        elif self.state == "pick_up_orientation":
            if self.aruco_0_detected and self.aruco_0_position:
                aruco_x, aruco_y = self.aruco_0_position
                
                # Generar puntos de aproximación si no existen
                if not hasattr(self, 'approach_points') or not self.approach_points:
                    self.approach_points = self.generate_straight_approach_points(aruco_x, aruco_y)
                    self.current_approach_index = 0
                
                # Verificar distancia actual al ArUco
                robot_x = self.robot_position.x
                robot_y = self.robot_position.y
                dx = aruco_x - robot_x
                dy = aruco_y - robot_y
                distance_to_aruco = math.sqrt(dx*dx + dy*dy)
                
                # Si estamos muy cerca del ArUco, ir directamente a él
                if distance_to_aruco <= 0.5:
                    self.publish_target(aruco_x, aruco_y)
                    self.get_logger().info(f"Final approach to ArUco 0 at ({aruco_x:.3f}, {aruco_y:.3f})")
                    
                    # Marcar que estuvimos cerca del ArUco
                    self.was_close_to_aruco = True
                    self.get_logger().info(f"Very close to ArUco 0 (distance: {distance_to_aruco:.3f}m). Starting servo.")
                    self.previous_state = "pick_up_orientation"
                    self.state = "start_servo"
                    # Limpiar puntos de aproximación para la próxima vez
                    self.approach_points = []
                
                elif distance_to_aruco <= 0.5:
                    # Aproximación final directa
                    self.publish_target(aruco_x, aruco_y)
                    self.get_logger().info(f"Close approach to ArUco 0 at ({aruco_x:.3f}, {aruco_y:.3f})")
                    
                    # Marcar que estuvimos cerca del ArUco
                    if distance_to_aruco <= 0.5:
                        self.was_close_to_aruco = True
                
                else:
                    # Seguir la secuencia de puntos de aproximación
                    if self.current_approach_index < len(self.approach_points):
                        target_x, target_y = self.approach_points[self.current_approach_index]
                        self.publish_target(target_x, target_y)
                        
                        # Verificar si llegamos al punto actual
                        dx_target = target_x - robot_x
                        dy_target = target_y - robot_y
                        distance_to_target = math.sqrt(dx_target*dx_target + dy_target*dy_target)
                        
                        if distance_to_target <= 0.12:  # 8cm de tolerancia
                            self.current_approach_index += 1
                            if self.current_approach_index < len(self.approach_points):
                                self.get_logger().info(f"Reached approach point {self.current_approach_index}. Moving to next point.")
                            else:
                                self.get_logger().info("All approach points completed. Moving to final approach.")
                        
                        self.get_logger().info(f"Going to approach point {self.current_approach_index+1}/{len(self.approach_points)}: ({target_x:.3f}, {target_y:.3f})")
                    else:
                        # Todos los puntos de aproximación completados, ir al ArUco
                        self.publish_target(aruco_x, aruco_y)
                        self.get_logger().info(f"All approach points completed. Going to ArUco 0 at ({aruco_x:.3f}, {aruco_y:.3f})")
                
                # Guardar última posición conocida del ArUco
                self.last_aruco_position = self.aruco_0_position
                
            else:
                # Si perdemos el ArUco pero estuvimos cerca, proceder con el servo
                if hasattr(self, 'was_close_to_aruco') and self.was_close_to_aruco:
                    self.get_logger().info("Lost ArUco 0 but was close - assuming pickup position reached. Starting servo.")
                    self.previous_state = "pick_up_orientation"
                    self.state = "start_servo"
                    # Limpiar puntos de aproximación
                    self.approach_points = []
                else:
                    # Si perdemos el ArUco sin haber estado cerca, mantener la última posición conocida
                    self.get_logger().warn("ArUco 0 not detected, maintaining position...")
                    if hasattr(self, 'last_aruco_position') and self.last_aruco_position:
                        self.publish_target(self.last_aruco_position[0], self.last_aruco_position[1])


        elif self.state == "start_servo":
            if not hasattr(self, 'servo_start_time'):
                # Publicar solo una vez el ángulo correspondiente
                if self.servo_counter == 0:
                    angle = -300.0
                    self.get_logger().info("Publishing servo angle for pick-up (-300.0)")
                else:
                    angle = 260.0
                    self.get_logger().info("Publishing servo angle for drop-off (260.0)")

                self.servo_pub.publish(Float32(data=angle))
                self.servo_counter = 1 - self.servo_counter

                self.servo_start_time = self.get_clock().now()
            else:
                elapsed_time = (self.get_clock().now() - self.servo_start_time).nanoseconds / 1e9
                if elapsed_time > 3.0:
                    del self.servo_start_time
                    if self.previous_state == "pick_up_orientation":
                        self.dropoff_sent = False
                        self.goal_reached = False
                        self.state = "go_to_drop_off"
                    else:
                        self.state = "go_backwards"

        elif self.state == "go_to_drop_off":
            if not self.dropoff_sent:
                self.publish_target(-1.05, 0.0)
                self.dropoff_sent = True
                self.get_logger().info("Sent drop-off target: (-1.15, 0.0)")
            elif self.goal_reached:
                self.get_logger().info("Reached drop-off target. Switching to START_SERVO.")
                self.previous_state = "go_to_drop_off"
                self.goal_reached = False
                self.dropoff_sent = False
                self.state = "start_servo"

        elif self.state == "go_backwards":
            if not hasattr(self, 'backwards_start_time'):
                self.get_logger().info("State: GO_BACKWARDS - Moving backwards for 2 seconds")
                self.backwards_start_time = self.get_clock().now()
                
                self.msg.linear.x = -0.075
                self.msg.angular.z = 0.0
                self.pub.publish(self.msg)
            else:
                self.msg.linear.x = -0.075
                self.msg.angular.z = 0.0
                self.pub.publish(self.msg)
                
                elapsed_time = (self.get_clock().now() - self.backwards_start_time).nanoseconds / 1e9
                if elapsed_time > 5.0:
                    self.msg.linear.x = 0.0
                    self.msg.angular.z = 0.0
                    self.pub.publish(self.msg)
                    
                    del self.backwards_start_time
                    self.get_logger().info("Finished going backwards. Transitioning to DONE state.")
                    self.state = "done"

        elif self.state == "done":
            if not self.returned_home:
                self.get_logger().info("Returning to origin (0.0, 0.0)")
                self.publish_target(0.0, 0.0)
                self.returned_home = True
            elif self.goal_reached:
                self.get_logger().info("Robot arrived at origin. State machine complete.")
                self.state = "idle"

        elif self.state == "idle":
            pass


def main(args=None):
    """
    Función principal del nodo
    """
    rclpy.init(args=None)
    node = MainCoordinator()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("SM Killed NOOOOOOOOOOOOOOOo ")
    except Exception as e:
        print(f"Error en el nodo: {e}")
    finally:
        try:
            node.destroy_node()
        except:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except:
            pass


if __name__ == "__main__":
    main()