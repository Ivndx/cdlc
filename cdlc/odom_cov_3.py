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
from builtin_interfaces.msg import Time

class DeadReckoning(Node):
    def __init__(self):
        super().__init__("Odometry")
        self.get_logger().info("Kalman I choose you!! ....")

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )

        # SUSCRIPTORES
        # Velocidades de encoders de ruedas
        self.sub_wl = self.create_subscription(Float32, "VelocityEncL", self.callback_L, qos_profile)
        self.sub_wr = self.create_subscription(Float32, "VelocityEncR", self.callback_R, qos_profile)
        
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

        # VARIABLES DE TIEMPO - USANDO time.time()
        self.t0 = time.time()       # Tiempo actual
        self.t_ant = time.time()    # Tiempo anterior
        self.first_run = True       # Bandera para primera ejecución

        # VELOCIDADES DE RUEDAS (rad/s)
        self.vel_r = 0.0        # Velocidad angular rueda derecha
        self.vel_l = 0.0        # Velocidad angular rueda izquierda
        
        # ESTADO DEL ROBOT COMO VECTOR NUMPY (CORREGIDO)
        self.mu = np.zeros((3, 1))  # [x, y, theta] como vector columna
        
        # Para compatibilidad con el resto del código
        self.xf = 0.0
        self.yf = 0.0
        self.angf = 0.0

        # PARÁMETROS FÍSICOS DEL ROBOT
        self.r = 0.0505         # Radio de las ruedas (m)
        self.L = 0.183          # Distancia entre ruedas (m)

        # PARÁMETROS DE RUIDO PARA EKF (SIMPLIFICADOS)
        self.kr = 0.73           # Factor de ruido rueda derecha
        self.kl = 0.63           # Factor de ruido rueda izquierda

        self.offset_x = 0.075   # Offset camara en x
        self.offset_z = 0.065   # Offset camara en z

        # MATRIZ DE COVARIANZA DEL ESTADO (3x3) - INICIALIZACIÓN CONSERVADORA
        self.Sigma = np.eye(3) * 0.01

        # VARIABLES PARA DETECCIÓN DE ARUCO
        self.aruco_detected = False
        self.aruco_data = []

        # MAPA DE ARUCOS MUNDO REAL
        self.aruco_map = {
            #0: (1.15, 0.0),
            1: (-1.535, 0.15),
            2: (1.435, -0.15),
            3: (0.0, -1.03),
            4: (0.0, 1.03),
            5: (-1.535, -0.15)
        }
        
        # MATRIZ DE RUIDO DE MEDICIÓN ARUCO (SIMPLIFICADA)
        self.Rk = np.array([
            [1.0, -0.0002376],
            [-0.0002376, 2.35]
        ])
        
        # MATRIZ DE TRANSFORMACIÓN CÁMARA-ROBOT (SIMPLIFICADA)
        self.T_cam_robot = np.array([
            [0.0, 0.0, 1.0, self.offset_x],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, self.offset_z],
            [0.0, 0.0, 0.0, 1.0]
        ])

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
        Callback para recibir detecciones de ArUco - SIMPLIFICADO
        """
        if len(msg.markers) > 0:
            self.aruco_detected = True
            self.aruco_data = msg.markers
            self.get_logger().info(f"ArUco detectado: {len(msg.markers)} marcadores")

    def normalize_angle(self, angle):
        """
        Normaliza un ángulo al rango [-π, π]
        """
        return math.atan2(math.sin(angle), math.cos(angle))
    
    def ekf_correction(self, aruco_data):
        """
        Corrección EKF simplificada usando mediciones de ArUco
        """
        for aruco in aruco_data:
            id = aruco.marker_id
            if id not in self.aruco_map:
                self.get_logger().warn(f"ArUco ID {id} no está en el mapa")
                continue

            # TRANSFORMACIÓN SIMPLIFICADA CÁMARA → ROBOT
            pos = np.array([
                [aruco.pose.position.x],
                [aruco.pose.position.y],
                [aruco.pose.position.z],
                [1.0]
            ])
            pos_robot = self.T_cam_robot @ pos
            
            # Filtrar ArUcos muy lejanos
            if np.linalg.norm(pos_robot[:2]) > 20:
                self.get_logger().warn(f"ArUco {id} descartado por estar muy lejos: {np.linalg.norm(pos_robot[:2]):.2f}m")
                continue

            # MEDICIÓN: [distancia, ángulo_relativo]
            z = np.array([
                [np.linalg.norm(pos_robot[:2])],
                [math.atan2(pos_robot[1, 0], pos_robot[0, 0])]
            ])

            # PREDICCIÓN DE LA MEDICIÓN
            dx = self.aruco_map[id][0] - self.mu[0, 0]
            dy = self.aruco_map[id][1] - self.mu[1, 0]
            dist = math.hypot(dx, dy)
            angle = self.normalize_angle(math.atan2(dy, dx) - self.mu[2, 0])
            h = np.array([[dist], [angle]])

            # JACOBIANO SIMPLIFICADO
            G = np.array([
                [-dx / dist, -dy / dist, 0.0],
                [dy / (dx**2 + dy**2), -dx / (dx**2 + dy**2), -1.0]
            ])

            # MATRIZ DE INNOVACIÓN
            S = G @ self.Sigma @ G.T + self.Rk
            
            try:
                # GANANCIA DE KALMAN
                K = self.Sigma @ G.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                self.get_logger().warn("Error al invertir matriz S, ignorando medición ArUco")
                continue

            # INNOVACIÓN
            innovation = z - h
            innovation[1, 0] = self.normalize_angle(innovation[1, 0])  # Normalizar ángulo

            self.get_logger().info(f"ArUco {id}: dist_innov={innovation[0,0]:.3f}m, "
                                 f"ang_innov={math.degrees(innovation[1,0]):.1f}°")

            # CORRECCIÓN DEL ESTADO
            self.mu += K @ innovation
            self.mu[2, 0] = self.normalize_angle(self.mu[2, 0])

            # ACTUALIZACIÓN DE COVARIANZA
            self.Sigma = self.Sigma - K @ G @ self.Sigma
            
            # Asegurar que la covarianza sea simétrica y positiva definida
            self.Sigma = (self.Sigma + self.Sigma.T) / 2
            if np.any(np.linalg.eigvals(self.Sigma) <= 0):
                self.get_logger().warn("Covarianza no positiva definida, reinicializando")
                self.Sigma = np.eye(3) * 0.01

            # Solo procesar el primer ArUco válido
            break

    def get_ros_time(self, timestamp):
        """
        Convierte un timestamp de time.time() a formato ROS Time
        """
        stamp = Time()
        stamp.sec = int(timestamp)
        stamp.nanosec = int((timestamp - stamp.sec) * 1e9)
        return stamp

    def callback_pub(self):
        """
        Callback principal que ejecuta el ciclo del EKF y publica odometría
        Frecuencia: 20 Hz (cada 0.05s)
        """
        # Actualizar tiempo usando time.time()
        self.t0 = time.time()
        
        # En la primera ejecución, inicializar t_ant
        if self.first_run:
            self.t_ant = self.t0
            self.first_run = False
            return
            
        dt = self.t0 - self.t_ant
        
        # Ignorar si dt es inválido
        if dt <= 0 or dt > 0.1:  # Máximo 100ms entre actualizaciones
            return
            
        # === FASE DE PREDICCIÓN DEL EKF ===
        
        # Parámetros del robot
        r = self.r  # Radio de ruedas
        L = self.L  # Distancia entre ruedas
        wr = self.vel_r  # Velocidad angular rueda derecha
        wl = self.vel_l  # Velocidad angular rueda izquierda

        # CINEMÁTICA DIFERENCIAL
        v = r * (wr + wl) / 2.0    # Velocidad lineal del robot
        w = r * (wr - wl) / L      # Velocidad angular del robot

        # Estado anterior
        theta = self.mu[2, 0]

        # PREDICCIÓN DEL ESTADO (modelo de movimiento)
        theta_new = self.normalize_angle(theta + w * dt)
        x_new = self.mu[0, 0] + v * math.cos(theta) * dt
        y_new = self.mu[1, 0] + v * math.sin(theta) * dt

        self.mu = np.array([[x_new], [y_new], [theta_new]])

        # MATRIZ DE RUIDO DEL PROCESO (SIMPLIFICADA)
        sigma_w = np.diag([self.kr * abs(wr), self.kl * abs(wl)])
        V = (r * dt / 2.0) * np.array([
            [math.cos(theta), math.cos(theta)],
            [math.sin(theta), math.sin(theta)],
            [2.0 / L, -2.0 / L]
        ])
        Qk = V @ sigma_w @ V.T

        # JACOBIANO DEL MODELO DE ESTADO (SIMPLIFICADO)
        H = np.array([
            [1.0, 0.0, -dt * v * math.sin(theta)],
            [0.0, 1.0,  dt * v * math.cos(theta)],
            [0.0, 0.0, 1.0]
        ])
        
        # PREDICCIÓN DE COVARIANZA
        self.Sigma = H @ self.Sigma @ H.T + Qk

        # === FASE DE CORRECCIÓN DEL EKF (con ArUco si disponible) ===
        if self.aruco_detected and len(self.aruco_data) > 0:
            self.ekf_correction(self.aruco_data)
            # Reset de banderas
            self.aruco_detected = False
            self.aruco_data = []

        # === PUBLICACIÓN DE MENSAJES ===
        
        # Actualizar variables de clase para compatibilidad
        self.xf = self.mu[0, 0]
        self.yf = self.mu[1, 0]
        self.angf = self.mu[2, 0]

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

        # Timestamp usando tiempo real
        stamp = self.get_ros_time(self.t0)

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
        odom_msg.pose.covariance[0]  = self.Sigma[0, 0]   # var(x)
        odom_msg.pose.covariance[1]  = self.Sigma[0, 1]   # cov(x,y)
        odom_msg.pose.covariance[5]  = self.Sigma[0, 2]   # cov(x,θ)
        odom_msg.pose.covariance[6]  = self.Sigma[1, 0]   # cov(y,x)
        odom_msg.pose.covariance[7]  = self.Sigma[1, 1]   # var(y)
        odom_msg.pose.covariance[11] = self.Sigma[1, 2]   # cov(y,θ)
        odom_msg.pose.covariance[30] = self.Sigma[2, 0]   # cov(θ,x)
        odom_msg.pose.covariance[31] = self.Sigma[2, 1]   # cov(θ,y)
        odom_msg.pose.covariance[35] = self.Sigma[2, 2]   # var(θ)

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
        print("Mataron a Kalman (que bueno..) ")
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