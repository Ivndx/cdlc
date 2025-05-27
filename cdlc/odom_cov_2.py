#!/usr/bin/env python3

import rclpy, time, math, tf_transformations
import numpy as np
from rclpy.node import Node
from turtlesim.msg import Pose
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, TransformStamped, PoseStamped
from sensor_msgs.msg import JointState
from aruco_opencv_msgs.msg import ArucoDetection
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from tf_transformations import quaternion_from_euler, euler_from_quaternion
from tf2_ros import TransformBroadcaster
from rosgraph_msgs.msg import Clock
from builtin_interfaces.msg import Time

class DeadReckoning(Node):
    def __init__(self):
        super().__init__("Odometry")
        self.get_logger().info("Odom node started...")

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )

        self.sub_wl = self.create_subscription(Float32, "VelocityEncL", self.callback_L, qos_profile)
        self.sub_wr = self.create_subscription(Float32, "VelocityEncR", self.callback_R, qos_profile)
        self.clock_sub = self.create_subscription(Clock, "clock", self.clock_callback, qos_profile)
        
        # Suscripción para detección de ArUco
        self.aruco_sub = self.create_subscription(ArucoDetection, "aruco_detections", self.aruco_callback, qos_profile)
        
        self.timer = self.create_timer(0.05, self.callback_pub)

        self.pub_pose = self.create_publisher(Pose, "PosicionBot", 10)
        self.pub_odom = self.create_publisher(Odometry, "odom", 10)
        self.pub_js = self.create_publisher(JointState, 'estimated/joint_states', 1)

        self.tf_broadcaster = TransformBroadcaster(self)

        self.t0 = 0.0
        self.t_ant = 0.0
        self.clock_flg = 0
        self.wait = 1

        self.vel_r = 0.0
        self.vel_l = 0.0
        
        self.xf = 0.0
        self.yf = 0.0
        self.angf = 0.0
        self.pose = [self.xf, self.yf, self.angf]

        # Parámetros del robot - CORREGIDOS
        self.r = 0.0505      # Radio de ruedas
        self.L = 0.183        # Distancia entre ruedas (corregido desde 0.183)

        # Parámetros de ruido de las ruedas - CORREGIDOS
        self.sig2r = 0.3     # Varianza rueda derecha
        self.sig2l = 0.3     # Varianza rueda izquierda

        # Matriz de covarianza inicial - SIMPLIFICADA
        self.cov = np.array([
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.05]
        ])
        self.cov_prev = self.cov.copy()

        # Matriz L para ruido de las ruedas
        self.L_noise = np.array([
            [self.sig2r, 0.0],
            [0.0, self.sig2l]
        ])

        # Variables para ArUco
        self.aruco_detected = False
        self.aruco_data = None
        
        # Factor de suavizado para correcciones (ajustable)
        self.correction_smoothing = 0.3  # Entre 0.1 (muy suave) y 1.0 (sin suavizar)
        
        # Mapa de ArUcos conocidos (ID -> (x_world, y_world))
        self.aruco_map = {
            0: (-0.9,  0.0, 0.0),        # yaw = 0 rad
            1: ( 0.1,  1.5, 0.0),        # yaw = 0 rad
            2: ( 1.1, -1.5, 0.0),        # yaw = 0 rad
            3: ( 0.1, -2.0, 0.0),        # yaw = 0 rad
            4: ( 1.5,  4.0, -np.pi/2)    # yaw = -90° (según <pose> ... 4.712 rad)
        }

        
        # Matriz de ruido de medición ArUco - AJUSTADA para ser menos agresiva
        self.R_aruco = np.array([
            [0.05, 0.0],     # Varianza en distancia (reducida)
            [0.0, 0.1]       # Varianza en ángulo (reducida significativamente)
        ])
        
    def clock_callback(self, msg):
        seconds = msg.clock.sec
        nanoseconds = msg.clock.nanosec
        self.t0 = seconds + nanoseconds * 1e-9
        self.clock_flg = 1
        if self.wait == 1:
            self.t_ant = self.t0
            self.wait = 0

    def callback_L(self, msg):
        self.vel_l = msg.data

    def callback_R(self, msg):
        self.vel_r = msg.data
        
    def aruco_callback(self, msg):
        """Callback para recibir detecciones de ArUco"""
        if len(msg.markers) > 0:
            self.aruco_detected = True
            self.aruco_data = msg
            self.get_logger().info(f"ArUco detectado: {len(msg.markers)} marcadores")

    def rotation_matrix(self, theta, axis):
        """Matriz de rotación para transformaciones"""
        if axis == 'x':
            return np.array([
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta),  np.cos(theta)]
            ])
        elif axis == 'y':
            return np.array([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ])
        elif axis == 'z':
            return np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta),  np.cos(theta), 0],
                [0, 0, 1]
            ])

    def transform_aruco_to_robot_frame(self, aruco):
        """Transforma la pose del ArUco del frame de cámara al frame del robot"""
        d = 0.03   # distancia desde base hasta cam en x
        f = 0.087  # altura de la camara sobre la base (z)

        # Orientación del ArUco
        qx, qy, qz, qw = (aruco.pose.orientation.x,
                        aruco.pose.orientation.y,
                        aruco.pose.orientation.z,
                        aruco.pose.orientation.w)
        _, pitch, _ = euler_from_quaternion([qx, qy, qz, qw])

        # Transformación base -> cam (z luego x)
        R_rc = self.rotation_matrix(-np.pi/2, 'z') @ self.rotation_matrix(-np.pi/2, 'x')
        t_rc = np.array([[d], [0], [f]])
        T_rc = np.vstack((np.hstack((R_rc, t_rc)), [[0, 0, 0, 1]]))

        # Transformación cam -> aruco
        R_cm = self.rotation_matrix(pitch, 'y')
        t_cm = np.array([[aruco.pose.position.x],
                        [aruco.pose.position.y],
                        [aruco.pose.position.z]])
        T_cm = np.vstack((np.hstack((R_cm, t_cm)), [[0, 0, 0, 1]]))

        T_rm = T_rc @ T_cm
        x_aruco = T_rm[0, 3]
        y_aruco = T_rm[1, 3]
        return x_aruco, y_aruco


    def ekf_correction(self, aruco_data):
        for aruco in aruco_data.markers:
            if aruco.marker_id not in self.aruco_map:
                continue

            x_aruco_robot, y_aruco_robot = self.transform_aruco_to_robot_frame(aruco)
            aruco_world_x, aruco_world_y, aruco_yaw = self.aruco_map[aruco.marker_id]

            dx = aruco_world_x - self.pose[0]
            dy = aruco_world_y - self.pose[1]

            distance_sq = dx*dx + dy*dy
            distance = max(np.sqrt(distance_sq), 1e-6)

            G = np.array([[-dx/distance, -dy/distance, 0],
                        [dy/distance_sq, -dx/distance_sq, -1]])

            S = G @ self.cov @ G.T + self.R_aruco
            K = self.cov @ G.T @ np.linalg.inv(S)

            z_measured = np.array([[np.sqrt(x_aruco_robot**2 + y_aruco_robot**2)],
                                [np.arctan2(y_aruco_robot, x_aruco_robot) - self.pose[2]]])

            z_expected = np.array([[distance],
                                [aruco_yaw - self.pose[2]]])
            z_expected[1, 0] = np.arctan2(np.sin(z_expected[1, 0]), np.cos(z_expected[1, 0]))

            innovation = z_measured - z_expected
            innovation[1, 0] = np.arctan2(np.sin(innovation[1, 0]), np.cos(innovation[1, 0]))

            max_distance_innovation = 1.5
            max_angle_innovation = np.pi

            if abs(innovation[0, 0]) > max_distance_innovation:
                self.get_logger().warn(f"Innovación de distancia muy grande: {innovation[0, 0]:.3f}m, ignorando ArUco {aruco.marker_id}")
                continue

            if abs(innovation[1, 0]) > max_angle_innovation:
                self.get_logger().warn(f"Innovación de ángulo muy grande: {innovation[1, 0]*180/np.pi:.1f}°, ignorando ArUco {aruco.marker_id}")
                continue

            correction = K @ innovation
            smoothing_factor = 0.3

            self.get_logger().info(f"ArUco {aruco.marker_id}: Innovación dist={innovation[0,0]:.3f}m, ang={innovation[1,0]*180/np.pi:.1f}°")
            self.get_logger().info(f"Corrección: dx={correction[0,0]:.3f}, dy={correction[1,0]:.3f}, dtheta={correction[2,0]*180/np.pi:.1f}°")

            self.pose[0] += correction[0, 0] * smoothing_factor
            self.pose[1] += correction[1, 0] * smoothing_factor
            self.pose[2] += correction[2, 0] * smoothing_factor
            self.pose[2] = np.arctan2(np.sin(self.pose[2]), np.cos(self.pose[2]))

            I = np.eye(3)
            self.cov = (I - K @ G) @ self.cov
            self.cov = (self.cov + self.cov.T) / 2
            if np.any(np.linalg.eigvals(self.cov) <= 0):
                self.get_logger().warn("Covarianza no positiva definida, reinicializando")
                self.cov = np.eye(3) * 0.1

            self.get_logger().info(f"Pose corregida: x={self.pose[0]:.2f}, y={self.pose[1]:.2f}, theta={self.pose[2]*180/np.pi:.1f}°")
            break


    def callback_pub(self):
        if self.wait == 0 and self.clock_flg == 1:
            dt = self.t0 - self.t_ant
            
            if dt <= 0:
                return
                
            self.clock_flg = 0
            
            # Calcular velocidades
            r = self.r
            L = self.L
            wr = self.vel_r
            wl = self.vel_l

            # Velocidad lineal y angular
            V = (r/2.0) * (wr + wl)
            W = (r/L) * (wr - wl)

            # PREDICCIÓN EKF
            theta_prev = self.pose[2]

            # Actualizar pose (predicción)
            self.pose[0] += V * math.cos(theta_prev) * dt
            self.pose[1] += V * math.sin(theta_prev) * dt
            self.pose[2] += W * dt
            self.pose[2] = self.pose[2] % (2 * math.pi)

            # Jacobiano de la predicción H
            H = np.array([
                [1.0, 0.0, -dt * V * np.sin(theta_prev)],
                [0.0, 1.0,  dt * V * np.cos(theta_prev)],
                [0.0, 0.0,  1.0]
            ])

            # Jacobiano del ruido F
            F = 0.5 * r * dt * np.array([
                [np.cos(theta_prev), np.cos(theta_prev)],
                [np.sin(theta_prev), np.sin(theta_prev)],
                [2.0/L, -2.0/L]
            ])

            # Ruido del proceso Q
            Q = F @ self.L_noise @ F.T
            
            # Actualizar covarianza (predicción)
            self.cov = H @ self.cov_prev @ H.T + Q

            # CORRECCIÓN EKF con ArUco (si está disponible)
            if self.aruco_detected and self.aruco_data is not None:
                self.ekf_correction(self.aruco_data)
                # Reset flag
                self.aruco_detected = False
                self.aruco_data = None

            # Actualizar variables de clase para compatibilidad
            self.xf = self.pose[0]
            self.yf = self.pose[1] 
            self.angf = self.pose[2]

            # Publicar Pose
            pose_msg = Pose()
            pose_msg.x = self.xf
            pose_msg.y = self.yf
            pose_msg.theta = self.angf
            self.pub_pose.publish(pose_msg)

            # Publicar Odometry
            qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, self.angf)
            odom_msg = Odometry()

            stamp = Time()
            stamp.sec = int(self.t0)
            stamp.nanosec = int((self.t0 - stamp.sec) * 1e9)

            odom_msg.header.stamp = stamp
            odom_msg.header.frame_id = "world"
            odom_msg.child_frame_id = "base_footprint"
            odom_msg.pose.pose.position.x = self.xf
            odom_msg.pose.pose.position.y = self.yf
            odom_msg.pose.pose.position.z = 0.0
            odom_msg.pose.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)

            # Configuración de covarianza
            odom_msg.pose.covariance = [0.0] * 36
            odom_msg.pose.covariance[0]  = self.cov[0, 0]   # xx
            odom_msg.pose.covariance[1]  = self.cov[0, 1]   # xy
            odom_msg.pose.covariance[5]  = self.cov[0, 2]   # x_theta
            odom_msg.pose.covariance[6]  = self.cov[1, 0]   # yx
            odom_msg.pose.covariance[7]  = self.cov[1, 1]   # yy
            odom_msg.pose.covariance[11] = self.cov[1, 2]   # y_theta
            odom_msg.pose.covariance[30] = self.cov[2, 0]   # theta_x
            odom_msg.pose.covariance[31] = self.cov[2, 1]   # theta_y
            odom_msg.pose.covariance[35] = self.cov[2, 2]   # theta_theta

            self.pub_odom.publish(odom_msg)

            # TF
            tf_msg = TransformStamped()
            tf_msg.header.stamp = stamp
            tf_msg.header.frame_id = "world"
            tf_msg.child_frame_id = "base_footprint"
            tf_msg.transform.translation.x = self.xf
            tf_msg.transform.translation.y = self.yf
            tf_msg.transform.translation.z = 0.0
            tf_msg.transform.rotation = Quaternion(x=qx, y=qy, z=qz, w=qw)
            self.tf_broadcaster.sendTransform(tf_msg)

            # JointState
            js_msg = JointState()
            js_msg.header.stamp = stamp
            js_msg.name = ['estimated/wheel_right_joint', 'estimated/wheel_left_joint']
            js_msg.position = [self.vel_r * dt, self.vel_l * dt]
            self.pub_js.publish(js_msg)

            # Actualizar valores anteriores
            self.cov_prev = self.cov.copy()
            self.t_ant = self.t0


def main(args=None):
    rclpy.init(args=args)
    node = DeadReckoning()
    try:
        rclpy.spin(node)
    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        print("Node terminated by user!")


if __name__ == "__main__":
    main()