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
        self.timer = self.create_timer(0.1, self.state_machine)  # Más rápido para control visual

        # Variables de estado existentes
        self.close_enough = False
        self.goal_reached = False
        self.dropoff_sent = False
        self.returned_home = False
        self.orientation_started = False
        self.routine_index = 0
        self.servo_counter = 0

        # Variables para control visual del ArUco 0
        self.aruco_0_detected = False
        self.aruco_0_center_x = 0.0
        self.aruco_0_center_y = 0.0
        self.aruco_0_size = 0.0  # Área o diagonal del ArUco para detectar proximidad
        self.aruco_0_distance = 0.0  # Distancia Z en metros
        self.robot_position = Pose()
        
        # Parámetros del controlador visual
        self.image_center_x = 320.0  # Centro de imagen (ajustar según tu cámara)
        self.image_center_y = 240.0  # Centro de imagen (ajustar según tu cámara)
        self.kp_angular = 0.003  # Ganancia proporcional para velocidad angular
        self.linear_velocity = 0.08  # Velocidad lineal constante (8cm/s)
        self.min_aruco_size = 5000.0  # Tamaño mínimo del ArUco para considerar que está cerca
        self.max_distance_for_pickup = 0.20  # Distancia máxima para pick-up (20cm)
        
        # Mensaje de Twist para el control visual
        self.visual_cmd = Twist()

    def calculate_aruco_center(self, corners):
        """
        Calcula el centro del ArUco a partir de sus esquinas (corners)
        """
        if len(corners) < 4:
            return 0.0, 0.0, 0.0
        
        # Calcular centro promediando las coordenadas de las esquinas
        center_x = sum(corner.x for corner in corners) / len(corners)
        center_y = sum(corner.y for corner in corners) / len(corners)
        
        # Calcular tamaño (área aproximada)
        x_coords = [corner.x for corner in corners]
        y_coords = [corner.y for corner in corners]
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        area = width * height
        
        return center_x, center_y, area

    def aruco_callback(self, msg):
        """
        Callback para procesar detecciones de ArUco
        """
        self.aruco_0_detected = False
        
        for aruco in msg.markers:
            if aruco.marker_id == 0:  # Solo nos interesa el ArUco 0
                self.aruco_0_detected = True
                
                # Calcular centro del ArUco usando las esquinas
                center_x, center_y, area = self.calculate_aruco_center(aruco.corners)
                self.aruco_0_center_x = center_x
                self.aruco_0_center_y = center_y
                self.aruco_0_size = area
                
                # Obtener distancia Z (profundidad) desde la pose
                self.aruco_0_distance = aruco.pose.position.z
                
                self.get_logger().info(f"ArUco 0: centro=({self.aruco_0_center_x:.1f}, {self.aruco_0_center_y:.1f}), "
                                     f"tamaño={self.aruco_0_size:.0f}, distancia={self.aruco_0_distance:.3f}m")
                break

    def visual_servo_control(self):
        """
        Controlador visual para mantener el ArUco centrado y avanzar hacia él
        Para que el ArUco esté centrado en X (horizontal) con respecto a la cámara
        """
        if not self.aruco_0_detected:
            # Si no ve el ArUco, detenerse
            self.visual_cmd.linear.x = 0.0
            self.visual_cmd.angular.z = 0.0
            self.pub.publish(self.visual_cmd)
            self.get_logger().warn("ArUco 0 not detected. Stopping.")
            return False
        
        # Calcular error de centrado en X (horizontal)
        # El objetivo es que aruco_0_center_x = image_center_x (centrado)
        error_x = self.aruco_0_center_x - self.image_center_x
        
        # Control proporcional para velocidad angular
        # Si el ArUco está a la derecha (error_x > 0), girar a la derecha (angular negativo)
        # Si el ArUco está a la izquierda (error_x < 0), girar a la izquierda (angular positivo)
        angular_velocity = -self.kp_angular * error_x
        
        # Limitar velocidad angular
        max_angular = 0.5  # rad/s
        angular_velocity = max(-max_angular, min(max_angular, angular_velocity))
        
        # Verificar si el ArUco está suficientemente cerca
        is_close_enough = (self.aruco_0_size > self.min_aruco_size or 
                          self.aruco_0_distance < self.max_distance_for_pickup)
        
        # Verificar también si está centrado (tolerancia en píxeles)
        tolerance_pixels = 10.0  # Tolerancia de centrado
        is_centered = abs(error_x) < tolerance_pixels
        
        if is_close_enough and is_centered:
            # Detener el robot - está listo para pick-up
            self.visual_cmd.linear.x = 0.0
            self.visual_cmd.angular.z = 0.0
            self.pub.publish(self.visual_cmd)
            self.get_logger().info(f"ArUco 0 reached and centered! Size: {self.aruco_0_size:.0f}, "
                                 f"Distance: {self.aruco_0_distance:.3f}m, Error_X: {error_x:.1f}px")
            return True  # Listo para pick-up
        else:
            # Avanzar lentamente mientras se centra
            if is_centered:
                # Si está centrado, avanzar más rápido
                self.visual_cmd.linear.x = self.linear_velocity
            else:
                # Si no está centrado, avanzar más lento para dar tiempo al centrado
                self.visual_cmd.linear.x = self.linear_velocity * 0.5
            
            self.visual_cmd.angular.z = angular_velocity
            self.pub.publish(self.visual_cmd)
            
            self.get_logger().info(f"Visual servo: linear={self.visual_cmd.linear.x:.3f}, "
                                 f"angular={angular_velocity:.3f}, error_x={error_x:.1f}px, "
                                 f"centered={is_centered}, close={is_close_enough}")
            return False

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
            
            # Verificar si detectamos ArUco 0 y estamos suficientemente cerca
            if self.aruco_0_detected and self.aruco_0_distance < 0.6:  # 60cm para empezar control visual
                self.get_logger().info(f"ArUco 0 detected at {self.aruco_0_distance:.2f}m. Switching to VISUAL_SERVO")
                self.state = "visual_servo"

        elif self.state == "visual_servo":
            # Control visual directo hacia el ArUco
            pickup_ready = self.visual_servo_control()
            
            if pickup_ready:
                self.get_logger().info("Visual servo complete. Starting servo for pick-up.")
                self.previous_state = "visual_servo"
                self.state = "start_servo"
            elif not self.aruco_0_detected:
                # Si pierde el ArUco por mucho tiempo, volver a navegación
                if not hasattr(self, 'aruco_lost_time'):
                    self.aruco_lost_time = self.get_clock().now()
                else:
                    elapsed = (self.get_clock().now() - self.aruco_lost_time).nanoseconds / 1e9
                    if elapsed > 3.0:  # 3 segundos sin ver ArUco
                        self.get_logger().warn("Lost ArUco for too long. Returning to navigation.")
                        del self.aruco_lost_time
                        self.state = "go_to_pick_up"
            else:
                # Reset del timer si vuelve a ver el ArUco
                if hasattr(self, 'aruco_lost_time'):
                    del self.aruco_lost_time

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
                    if self.previous_state == "visual_servo":
                        self.dropoff_sent = False
                        self.goal_reached = False
                        self.state = "go_to_drop_off"
                    else:
                        self.state = "go_backwards"

        elif self.state == "go_to_drop_off":
            if not self.dropoff_sent:
                self.publish_target(-1.15, 0.0)
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
                
                self.visual_cmd.linear.x = -0.2
                self.visual_cmd.angular.z = 0.0
                self.pub.publish(self.visual_cmd)
            else:
                self.visual_cmd.linear.x = -0.2
                self.visual_cmd.angular.z = 0.0
                self.pub.publish(self.visual_cmd)
                
                elapsed_time = (self.get_clock().now() - self.backwards_start_time).nanoseconds / 1e9
                if elapsed_time > 2.0:
                    self.visual_cmd.linear.x = 0.0
                    self.visual_cmd.angular.z = 0.0
                    self.pub.publish(self.visual_cmd)
                    
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