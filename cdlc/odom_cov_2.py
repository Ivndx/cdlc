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
from tf_transformations import quaternion_from_euler, quaternion_from_matrix, euler_from_quaternion, quaternion_matrix
from tf2_ros import TransformBroadcaster
from rosgraph_msgs.msg import Clock
from builtin_interfaces.msg import Time

class DeadReckoning(Node):
    def __init__(self):
        super().__init__("Odometry")
        self.get_logger().info("Nodo de odometría EKF con ArUcos iniciado...")

        # Configuración de QoS para comunicación ROS2
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )

        # SUSCRIPTORES
        # Velocidades de encoders de ruedas
        self.sub_wl = self.create_subscription(Float32, "VelocityEncL", self.callback_L, qos_profile)
        self.sub_wr = self.create_subscription(Float32, "VelocityEncR", self.callback_R, qos_profile)
        
        # Reloj de simulación
        self.clock_sub = self.create_subscription(Clock, "clock", self.clock_callback, qos_profile)
        
        # Detecciones de ArUco
        self.aruco_sub = self.create_subscription(ArucoDetection, "aruco_detections", self.aruco_callback, qos_profile)
        
        # Timer para publicar odometría (20 Hz)
        self.timer = self.create_timer(0.05, self.callback_pub)

        # PUBLICADORES
        self.pub_pose = self.create_publisher(Pose, "PosicionBot", 10)
        self.pub_odom = self.create_publisher(Odometry, "odom", 10)
        self.pub_js = self.create_publisher(JointState, 'estimated/joint_states', 1)

        # Broadcaster para transformaciones TF
        self.tf_broadcaster = TransformBroadcaster(self)

        # VARIABLES DE TIEMPO
        self.t0 = 0.0           # Tiempo actual
        self.t_ant = 0.0        # Tiempo anterior
        self.clock_flg = 0      # Bandera de reloj actualizado
        self.wait = 1           # Bandera de espera inicial

        # VELOCIDADES DE RUEDAS (rad/s)
        self.vel_r = 0.0        # Velocidad angular rueda derecha
        self.vel_l = 0.0        # Velocidad angular rueda izquierda
        
        # ESTADO DEL ROBOT [x, y, theta]
        self.xf = 0.0           # Posición X en mundo
        self.yf = 0.0           # Posición Y en mundo
        self.angf = 0.0         # Orientación theta
        self.pose = [self.xf, self.yf, self.angf]

        # PARÁMETROS FÍSICOS DEL ROBOT
        self.r = 0.0505         # Radio de las ruedas (m)
        self.L = 0.183          # Distancia entre ruedas (m)

        # PARÁMETROS DE RUIDO PARA EKF
        self.sig2r = 0.0273        # Varianza rueda derecha (reducida)
        self.sig2l = 0.0067       # Varianza rueda izquierda (reducida)

        # MATRIZ DE COVARIANZA DEL ESTADO (3x3)
        # Representa incertidumbre en [x, y, theta]
        self.cov = np.array([
            [0.5, 0.0,  0.0],   # Varianza en X
            [0.0,  0.5, 0.0],   # Varianza en Y
            [0.0,  0.0,  0.43]   # Varianza en theta
        ])
        self.cov_prev = self.cov.copy()

        # MATRIZ DE RUIDO DE LAS RUEDAS (2x2)
        self.L_noise = np.array([
            [self.sig2r, 0.0],
            [0.0, self.sig2l]
        ])

        # VARIABLES PARA DETECCIÓN DE ARUCO
        self.aruco_detected = False
        self.aruco_data = None
        
        #esta bien 
        # MAPA DE ARUCOS CONOCIDOS: ID -> (x_world, y_world, yaw_world)
        self.aruco_map = {
            0: (-3.95, -1.0, 0.0),
            1: (-1.0, 3.45, 3*np.pi/2),
            2: (1.95, -1.45, 2.356),
            3: (0.1, -2.0, np.pi/2),
            4: (2.5, 4.0, 3*np.pi/2),
            5: (-1.0, -3.5, np.pi/2),
            6: (0.5, 2.9, 3*np.pi/2),
            7: (0.95, 0.0, np.pi),
            8: (-2.0, 1.5, 4.496),
            9: (3.45, 2.95, 3*np.pi/2),
        }
                
        # MATRIZ DE RUIDO DE MEDICIÓN ARUCO (2x2)
        # [distancia, ángulo_relativo]
        self.R_aruco = np.array([
            [0.5, 0.0],        # Varianza en distancia (2cm std)
            [0.0, 0.5]         # Varianza en ángulo (0.05 rad² ≈ 13° std)
        ])
        
        # PARÁMETROS DE TRANSFORMACIÓN CÁMARA-ROBOT (CORREGIDOS)
        self.cam_offset_x =  0.075    # Distancia cámara desde centro del robot en X (m)
        self.cam_offset_y = 0.0 #Robot real = 0.0     # Offset lateral de cámara (m)
        self.cam_offset_z = 0.065   # Altura de cámara sobre la base (m)
        
        # FACTORES DE VALIDACIÓN
        self.max_distance_innovation = 0.5      # Máxima innovación de distancia permitida (m)
        self.max_angle_innovation = np.pi     # Máxima innovación de ángulo permitida (60°)
        
    def clock_callback(self, msg):
        """
        Callback para recibir el tiempo de simulación
        """
        seconds = msg.clock.sec
        nanoseconds = msg.clock.nanosec
        self.t0 = seconds + nanoseconds * 1e-9
        self.clock_flg = 1
        
        # Inicialización del tiempo anterior en la primera iteración
        if self.wait == 1:
            self.t_ant = self.t0
            self.wait = 0

    def callback_L(self, msg):
        """
        Callback para velocidad angular de rueda izquierda
        """
        self.vel_l = msg.data

    def callback_R(self, msg):
        """
        Callback para velocidad angular de rueda derecha
        """
        self.vel_r = msg.data
        
    def aruco_callback(self, msg):
        """
        Callback para recibir detecciones de ArUco
        """
        if len(msg.markers) > 0:
            self.aruco_detected = True
            self.aruco_data = msg
            self.get_logger().info(f"ArUco detectado: {len(msg.markers)} marcadores")

    def rotation_matrix(self, theta, axis):
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
        else:
            raise ValueError(f"Eje desconocido: {axis}")


    def transform_aruco_to_robot_frame(self, aruco):
        # Rotaciones requeridas según la imagen (Y+90°, Z+90°)
        R_y = self.rotation_matrix(np.pi/2, 'y')
        R_z = self.rotation_matrix(np.pi/2, 'z')
        R_rc = R_z @ R_y  # Cámara → robot

        t_rc = np.array([[self.cam_offset_x], [0.0], [self.cam_offset_z]])
        T_rc = np.vstack((np.hstack((R_rc, t_rc)), [[0, 0, 0, 1]]))

        # Obtener matriz de rotación desde quaternion (completo)
        q = aruco.pose.orientation
        R_cm = tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]

        t_cm = np.array([
            [aruco.pose.position.x],
            [aruco.pose.position.y],
            [aruco.pose.position.z]
        ])
        T_cm = np.vstack((np.hstack((R_cm, t_cm)), [[0, 0, 0, 1]]))

        # Transformación final: robot ← cámara ← marcador
        T_rm = T_rc @ T_cm

        x_aruco = T_rm[0, 3]
        y_aruco = T_rm[1, 3]

        return x_aruco, y_aruco

    def normalize_angle(self, angle):
        """
        Normaliza un ángulo al rango [-π, π]
        """
        return np.arctan2(np.sin(angle), np.cos(angle))
    
    def ekf_correction(self, aruco_data):
        for aruco in aruco_data.markers:
            if aruco.marker_id not in self.aruco_map:
                self.get_logger().warn(f"ArUco ID {aruco.marker_id} no está en el mapa")
                continue

            # === TRANSFORMACIÓN CORREGIDA CAMARA → ROBOT ===
            # Orden correcto: Ry(pi/2) @ Rz(pi/2)
            R_y = self.rotation_matrix(np.pi/2, 'y')
            R_z = self.rotation_matrix(np.pi/2, 'z')
            R_rc = R_y @ R_z
            t_rc = np.array([[self.cam_offset_x], [0.0], [self.cam_offset_z]])
            T_rc = np.vstack((np.hstack((R_rc, t_rc)), [[0, 0, 0, 1]]))

            # === TRANSFORMACIÓN DEL ARUCO DETECTADO ===
            q = aruco.pose.orientation
            R_cm = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
            t_cm = np.array([[aruco.pose.position.x],
                            [aruco.pose.position.y],
                            [aruco.pose.position.z]])
            T_cm = np.vstack((np.hstack((R_cm, t_cm)), [[0, 0, 0, 1]]))

            T_rm = T_rc @ T_cm
            x_aruco_robot = T_rm[0, 3]
            y_aruco_robot = T_rm[1, 3]

            aruco_world_x, aruco_world_y, aruco_yaw = self.aruco_map[aruco.marker_id]
            dx_world = aruco_world_x - self.pose[0]
            dy_world = aruco_world_y - self.pose[1]

            cos_theta = np.cos(self.pose[2])
            sin_theta = np.sin(self.pose[2])
            x_aruco_pred = cos_theta * dx_world + sin_theta * dy_world
            y_aruco_pred = -sin_theta * dx_world + cos_theta * dy_world

            distance_measured = np.sqrt(x_aruco_robot**2 + y_aruco_robot**2)
            angle_measured = np.arctan2(y_aruco_robot, x_aruco_robot)
            distance_predicted = np.sqrt(x_aruco_pred**2 + y_aruco_pred**2)
            angle_predicted = np.arctan2(y_aruco_pred, x_aruco_pred)

            if distance_measured <= 0 or distance_predicted <= 0:
                self.get_logger().warn(f"Distancia inválida para ArUco {aruco.marker_id}")
                continue

            innovation_distance = distance_measured - distance_predicted
            innovation_angle = self.normalize_angle(angle_measured - angle_predicted)

            if abs(innovation_distance) > self.max_distance_innovation:
                self.get_logger().warn(f"Innovación de distancia muy grande para ArUco {aruco.marker_id}: {innovation_distance:.3f}m")
                continue

            if abs(innovation_angle) > self.max_angle_innovation:
                self.get_logger().warn(f"Innovación de ángulo muy grande para ArUco {aruco.marker_id}: {innovation_angle*180/np.pi:.1f}°")
                continue

            if distance_predicted < 1e-6:
                continue

            dh_x = dx_world
            dh_y = dy_world
            dx_pred_dtheta = -sin_theta * dx_world + cos_theta * dy_world
            dy_pred_dtheta = -cos_theta * dx_world - sin_theta * dy_world

            H = np.array([
                [-cos_theta * dh_x/distance_predicted - sin_theta * dh_y/distance_predicted,
                sin_theta * dh_x/distance_predicted - cos_theta * dh_y/distance_predicted,
                (x_aruco_pred * dx_pred_dtheta + y_aruco_pred * dy_pred_dtheta) / distance_predicted],
                [sin_theta * dh_x/(distance_predicted**2) - cos_theta * dh_y/(distance_predicted**2),
                cos_theta * dh_x/(distance_predicted**2) + sin_theta * dh_y/(distance_predicted**2),
                (x_aruco_pred * dy_pred_dtheta - y_aruco_pred * dx_pred_dtheta) / (distance_predicted**2) - 1.0]
            ])

            S = H @ self.cov @ H.T + self.R_aruco
            try:
                K = self.cov @ H.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                self.get_logger().warn("Error al invertir matriz S, ignorando medición ArUco")
                continue

            innovation = np.array([[innovation_distance], [innovation_angle]])
            correction = K @ innovation

            smoothing = 0.25
            self.pose[0] += correction[0, 0] * smoothing
            self.pose[1] += correction[1, 0] * smoothing
            self.pose[2] += correction[2, 0] * smoothing
            self.pose[2] = self.normalize_angle(self.pose[2])

            I = np.eye(3)
            self.cov = (I - K @ H) @ self.cov
            self.cov = (self.cov + self.cov.T) / 2
            if np.any(np.linalg.eigvals(self.cov) <= 0):
                self.get_logger().warn("Covarianza no positiva definida, reinicializando")
                self.cov = np.eye(3) * 0.01

            self.get_logger().info(f"ArUco {aruco.marker_id}: dist_innov={innovation_distance:.3f}m, "
                                f"ang_innov={innovation_angle*180/np.pi:.1f}°")
            self.get_logger().info(f"Corrección aplicada: dx={correction[0,0]*smoothing:.3f}, "
                                f"dy={correction[1,0]*smoothing:.3f}, dtheta={correction[2,0]*smoothing*180/np.pi:.1f}°")
            break


    def callback_pub(self):
        """
        Callback principal que ejecuta el ciclo del EKF y publica odometría
        Frecuencia: 20 Hz (cada 0.05s)
        """
        # Verificar que tengamos tiempo válido
        if self.wait == 0 and self.clock_flg == 1:
            dt = self.t0 - self.t_ant
            
            # Ignorar si dt es inválido
            if dt <= 0 or dt > 0.1:  # Máximo 100ms entre actualizaciones
                return
                
            self.clock_flg = 0
            
            # === FASE DE PREDICCIÓN DEL EKF ===
            
            # Parámetros del robot
            r = self.r  # Radio de ruedas
            L = self.L  # Distancia entre ruedas
            wr = self.vel_r  # Velocidad angular rueda derecha
            wl = self.vel_l  # Velocidad angular rueda izquierda

            # CINEMÁTICA DIFERENCIAL CORREGIDA
            # Verificar convención de ruedas: rueda derecha positiva hacia adelante
            V = (r/2.0) * (wr + wl)    # Velocidad lineal del robot
            W = (r/L) * (wr - wl)      # Velocidad angular del robot (positiva = giro antihorario)

            # Estado anterior
            theta_prev = self.pose[2]

            # PREDICCIÓN DEL ESTADO (modelo de movimiento)
            # Integración Euler con coordenadas del robot: X=adelante, Y=izquierda
            self.pose[0] += V * math.cos(theta_prev) * dt  # Nueva posición X
            self.pose[1] += V * math.sin(theta_prev) * dt  # Nueva posición Y  
            self.pose[2] += W * dt                         # Nueva orientación
            self.pose[2] = self.normalize_angle(self.pose[2])  # Normalizar ángulo

            # JACOBIANO DEL MODELO DE ESTADO (H_k)
            # Derivadas parciales del nuevo estado con respecto al estado anterior
            H_k = np.array([
                [1.0, 0.0, -dt * V * np.sin(theta_prev)],  # ∂x_new/∂[x,y,θ]
                [0.0, 1.0,  dt * V * np.cos(theta_prev)],  # ∂y_new/∂[x,y,θ]
                [0.0, 0.0,  1.0]                           # ∂θ_new/∂[x,y,θ]
            ])

            # JACOBIANO DEL RUIDO DE CONTROL (F_u)
            # Derivadas parciales del estado con respecto a las velocidades de ruedas
            F_u = 0.5 * r * dt * np.array([
                [np.cos(theta_prev), np.cos(theta_prev)],   # ∂x/∂[wr,wl]
                [np.sin(theta_prev), np.sin(theta_prev)],   # ∂y/∂[wr,wl]
                [2.0/L, -2.0/L]                             # ∂θ/∂[wr,wl]
            ])

            # MATRIZ DE RUIDO DEL PROCESO (Q)
            Q = F_u @ self.L_noise @ F_u.T
            
            # PREDICCIÓN DE COVARIANZA
            self.cov = H_k @ self.cov_prev @ H_k.T + Q

            # === FASE DE CORRECCIÓN DEL EKF (con ArUco si disponible) ===
            if self.aruco_detected and self.aruco_data is not None:
                self.ekf_correction(self.aruco_data)
                # Reset de banderas
                self.aruco_detected = False
                self.aruco_data = None

            # === PUBLICACIÓN DE MENSAJES ===
            
            # Actualizar variables de clase para compatibilidad
            self.xf = self.pose[0]
            self.yf = self.pose[1] 
            self.angf = self.pose[2]

            # 1. PUBLICAR POSE SIMPLE
            pose_msg = Pose()
            pose_msg.x = self.xf
            pose_msg.y = self.yf
            pose_msg.theta = self.angf
            self.pub_pose.publish(pose_msg)

            # 2. PUBLICAR ODOMETRÍA COMPLETA
            # Convertir orientación a quaternion
            qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, self.angf)
            
            # Crear mensaje de odometría
            odom_msg = Odometry()

            # Timestamp
            stamp = Time()
            stamp.sec = int(self.t0)
            stamp.nanosec = int((self.t0 - stamp.sec) * 1e9)

            # Header
            odom_msg.header.stamp = stamp
            odom_msg.header.frame_id = "world"
            odom_msg.child_frame_id = "base_footprint"
            
            # Pose
            odom_msg.pose.pose.position.x = self.xf
            odom_msg.pose.pose.position.y = self.yf
            odom_msg.pose.pose.position.z = 0.0
            odom_msg.pose.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)

            # Covarianza de pose (6x6, pero solo usamos posición 2D + orientación)
            odom_msg.pose.covariance = [0.0] * 36
            odom_msg.pose.covariance[0]  = self.cov[0, 0]   # var(x)
            odom_msg.pose.covariance[1]  = self.cov[0, 1]   # cov(x,y)
            odom_msg.pose.covariance[5]  = self.cov[0, 2]   # cov(x,θ)
            odom_msg.pose.covariance[6]  = self.cov[1, 0]   # cov(y,x)
            odom_msg.pose.covariance[7]  = self.cov[1, 1]   # var(y)
            odom_msg.pose.covariance[11] = self.cov[1, 2]   # cov(y,θ)
            odom_msg.pose.covariance[30] = self.cov[2, 0]   # cov(θ,x)
            odom_msg.pose.covariance[31] = self.cov[2, 1]   # cov(θ,y)
            odom_msg.pose.covariance[35] = self.cov[2, 2]   # var(θ)

            self.pub_odom.publish(odom_msg)

            # 3. PUBLICAR TRANSFORMACIÓN TF
            tf_msg = TransformStamped()
            tf_msg.header.stamp = stamp
            tf_msg.header.frame_id = "world"
            tf_msg.child_frame_id = "base_footprint"
            tf_msg.transform.translation.x = self.xf
            tf_msg.transform.translation.y = self.yf
            tf_msg.transform.translation.z = 0.0
            tf_msg.transform.rotation = Quaternion(x=qx, y=qy, z=qz, w=qw)
            self.tf_broadcaster.sendTransform(tf_msg)

            # 4. PUBLICAR ESTADOS DE JUNTAS (para visualización)
            js_msg = JointState()
            js_msg.header.stamp = stamp
            js_msg.name = ['estimated/wheel_right_joint', 'estimated/wheel_left_joint']
            js_msg.position = [self.vel_r * dt, self.vel_l * dt]  # Rotación acumulada
            self.pub_js.publish(js_msg)

            # === ACTUALIZACIÓN PARA SIGUIENTE ITERACIÓN ===
            self.cov_prev = self.cov.copy()
            self.t_ant = self.t0


def main(args=None):
    """
    Función principal del nodo
    """
    rclpy.init(args=args)
    node = DeadReckoning()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Nodo terminado por el usuario!")
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