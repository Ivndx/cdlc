#!/usr/bin/env python3
import rclpy, time, math, csv, os
import numpy as np
import matplotlib.pyplot as plt
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from tf_transformations import euler_from_quaternion
from datetime import datetime

class PerformanceAnalyzer(Node):
    def __init__(self):
        super().__init__("performance_analyzer")
        self.get_logger().info("Performance Analyzer iniciado - Analizando rendimiento del EKF...")
        
        # Configuración QoS
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Suscriptores
        self.odom_sub = self.create_subscription(
            Odometry, 
            '/odom', 
            self.odom_callback, 
            qos_profile
        )
        
        self.ekf_sub = self.create_subscription(
            Odometry,  # O PoseStamped si tu EKF publica en ese formato
            '/robot_pose_ekf',  # Ajusta al nombre de tu tópico EKF
            self.ekf_callback,
            qos_profile
        )
        
        self.gt_sub = self.create_subscription(
            Odometry,  # O el tipo que use ground_truth
            '/ground_truth',
            self.ground_truth_callback,
            qos_profile
        )
        
        # Almacenamiento de datos
        self.odom_data = []
        self.ekf_data = []
        self.gt_data = []
        
        # Control de tiempo
        self.start_time = None
        self.recording = False
        
        # Timer para análisis periódico
        self.analysis_timer = self.create_timer(10.0, self.periodic_analysis)
        
        # Timer para guardar datos cada 30 segundos
        self.save_timer = self.create_timer(30.0, self.save_unified_data)
        
        self.get_logger().info("Suscripciones creadas. Esperando datos...")
        
    def odom_callback(self, msg):
        """Callback para odometría pura"""
        if not self.recording:
            self.start_recording()
            
        timestamp = self.get_timestamp(msg.header.stamp)
        position = self.extract_position(msg.pose.pose)
        
        self.odom_data.append({
            'timestamp': timestamp,
            'x': position['x'],
            'y': position['y'],
            'theta': position['theta']
        })
        
    def ekf_callback(self, msg):
        """Callback para estimación EKF"""
        if not self.recording:
            return
            
        timestamp = self.get_timestamp(msg.header.stamp)
        position = self.extract_position(msg.pose.pose)
        
        self.ekf_data.append({
            'timestamp': timestamp,
            'x': position['x'],
            'y': position['y'],
            'theta': position['theta']
        })
        
    def ground_truth_callback(self, msg):
        """Callback para ground truth"""
        if not self.recording:
            return
            
        timestamp = self.get_timestamp(msg.header.stamp)
        position = self.extract_position(msg.pose.pose)
        
        self.gt_data.append({
            'timestamp': timestamp,
            'x': position['x'],
            'y': position['y'],
            'theta': position['theta']
        })
        
    def extract_position(self, pose):
        """Extrae posición y orientación de un mensaje Pose"""
        # Posición
        x = pose.position.x
        y = pose.position.y
        
        # Orientación (quaternion a euler)
        quaternion = [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w
        ]
        _, _, theta = euler_from_quaternion(quaternion)
        
        return {'x': x, 'y': y, 'theta': theta}
        
    def get_timestamp(self, stamp):
        """Convierte stamp de ROS a timestamp"""
        return stamp.sec + stamp.nanosec * 1e-9
        
    def start_recording(self):
        """Inicia la grabación de datos"""
        if not self.recording:
            self.recording = True
            self.start_time = time.time()
            self.get_logger().info("¡Iniciando grabación de datos para análisis!")
            
    def find_closest_measurement(self, target_time, data_list, tolerance=0.1):
        """Encuentra la medición más cercana en tiempo"""
        if not data_list:
            return None
            
        closest = min(data_list, key=lambda x: abs(x['timestamp'] - target_time))
        
        if abs(closest['timestamp'] - target_time) <= tolerance:
            return closest
        return None
        
    def calculate_error(self, estimated, ground_truth):
        """Calcula el error entre estimación y ground truth"""
        if estimated is None or ground_truth is None:
            return None
            
        error_x = abs(estimated['x'] - ground_truth['x'])
        error_y = abs(estimated['y'] - ground_truth['y'])
        error_pos = math.sqrt(error_x**2 + error_y**2)
        
        # Error angular (normalizado)
        error_theta = abs(estimated['theta'] - ground_truth['theta'])
        error_theta = min(error_theta, 2*math.pi - error_theta)
        
        return {
            'error_x': error_x,
            'error_y': error_y,
            'error_position': error_pos,
            'error_theta': error_theta
        }
        
    def save_unified_data(self):
        """Guarda datos unificados en un solo archivo CSV con diferencias"""
        if not self.recording or not self.gt_data:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Crear directorio si no existe
        os.makedirs("performance_data", exist_ok=True)
        
        # Preparar datos unificados
        unified_data = []
        
        for gt_point in self.gt_data:
            # Buscar mediciones cercanas en tiempo
            odom_match = self.find_closest_measurement(gt_point['timestamp'], self.odom_data)
            ekf_match = self.find_closest_measurement(gt_point['timestamp'], self.ekf_data)
            
            # Crear fila de datos
            row = {
                'timestamp': gt_point['timestamp'],
                # Ground Truth
                'gt_x': gt_point['x'],
                'gt_y': gt_point['y'],
                'gt_theta': gt_point['theta'],
                # Odometría
                'odom_x': odom_match['x'] if odom_match else None,
                'odom_y': odom_match['y'] if odom_match else None,
                'odom_theta': odom_match['theta'] if odom_match else None,
                # EKF
                'ekf_x': ekf_match['x'] if ekf_match else None,
                'ekf_y': ekf_match['y'] if ekf_match else None,
                'ekf_theta': ekf_match['theta'] if ekf_match else None,
            }
            
            # Calcular diferencias con Ground Truth
            if odom_match:
                odom_error = self.calculate_error(odom_match, gt_point)
                row.update({
                    'odom_error_x': odom_error['error_x'] if odom_error else None,
                    'odom_error_y': odom_error['error_y'] if odom_error else None,
                    'odom_error_pos': odom_error['error_position'] if odom_error else None,
                    'odom_error_theta': odom_error['error_theta'] if odom_error else None,
                })
            else:
                row.update({
                    'odom_error_x': None,
                    'odom_error_y': None,
                    'odom_error_pos': None,
                    'odom_error_theta': None,
                })
                
            if ekf_match:
                ekf_error = self.calculate_error(ekf_match, gt_point)
                row.update({
                    'ekf_error_x': ekf_error['error_x'] if ekf_error else None,
                    'ekf_error_y': ekf_error['error_y'] if ekf_error else None,
                    'ekf_error_pos': ekf_error['error_position'] if ekf_error else None,
                    'ekf_error_theta': ekf_error['error_theta'] if ekf_error else None,
                })
            else:
                row.update({
                    'ekf_error_x': None,
                    'ekf_error_y': None,
                    'ekf_error_pos': None,
                    'ekf_error_theta': None,
                })
            
            unified_data.append(row)
        
        # Guardar archivo unificado
        filename = f"performance_data/unified_analysis_{timestamp}.csv"
        
        if unified_data:
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = [
                    'timestamp',
                    'gt_x', 'gt_y', 'gt_theta',
                    'odom_x', 'odom_y', 'odom_theta',
                    'ekf_x', 'ekf_y', 'ekf_theta',
                    'odom_error_x', 'odom_error_y', 'odom_error_pos', 'odom_error_theta',
                    'ekf_error_x', 'ekf_error_y', 'ekf_error_pos', 'ekf_error_theta'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(unified_data)
                
        self.get_logger().info(f"Datos unificados guardados: {filename}")
        self.get_logger().info(f"Total de filas guardadas: {len(unified_data)}")
        
    def calculate_metrics(self, errors):
        """Calcula métricas estadísticas"""
        if not errors:
            return None
            
        pos_errors = [e['error_position'] for e in errors if e is not None]
        theta_errors = [e['error_theta'] for e in errors if e is not None]
        
        if not pos_errors:
            return None
            
        metrics = {
            'position': {
                'mean': np.mean(pos_errors),
                'rmse': np.sqrt(np.mean(np.square(pos_errors))),
                'max': np.max(pos_errors),
                'std': np.std(pos_errors),
                'count': len(pos_errors)
            },
            'orientation': {
                'mean': np.mean(theta_errors) if theta_errors else 0,
                'rmse': np.sqrt(np.mean(np.square(theta_errors))) if theta_errors else 0,
                'max': np.max(theta_errors) if theta_errors else 0,
                'std': np.std(theta_errors) if theta_errors else 0
            }
        }
        
        return metrics
        
    def periodic_analysis(self):
        """Análisis periódico cada 10 segundos"""
        if not self.recording or len(self.gt_data) < 5:
            return
            
        self.get_logger().info(f"Datos recolectados - GT: {len(self.gt_data)}, "
                             f"Odom: {len(self.odom_data)}, EKF: {len(self.ekf_data)}")
        
        # Analizar últimos datos
        recent_gt = self.gt_data[-20:] if len(self.gt_data) > 20 else self.gt_data
        
        odom_errors = []
        ekf_errors = []
        
        for gt_point in recent_gt:
            # Buscar mediciones cercanas en tiempo
            odom_match = self.find_closest_measurement(gt_point['timestamp'], self.odom_data)
            ekf_match = self.find_closest_measurement(gt_point['timestamp'], self.ekf_data)
            
            if odom_match:
                odom_error = self.calculate_error(odom_match, gt_point)
                if odom_error:
                    odom_errors.append(odom_error)
                    
            if ekf_match:
                ekf_error = self.calculate_error(ekf_match, gt_point)
                if ekf_error:
                    ekf_errors.append(ekf_error)
        
        # Mostrar métricas
        odom_metrics = self.calculate_metrics(odom_errors)
        ekf_metrics = self.calculate_metrics(ekf_errors)
        
        if odom_metrics and ekf_metrics:
            self.get_logger().info("=== ANÁLISIS DE RENDIMIENTO ===")
            self.get_logger().info(f"ODOMETRÍA - Error pos: {odom_metrics['position']['rmse']:.4f}m (RMSE)")
            self.get_logger().info(f"EKF       - Error pos: {ekf_metrics['position']['rmse']:.4f}m (RMSE)")
            
            if odom_metrics['position']['rmse'] > 0:
                improvement = (1 - ekf_metrics['position']['rmse']/odom_metrics['position']['rmse']) * 100
                self.get_logger().info(f"Mejora del EKF: {improvement:.1f}%")
                
    def generate_final_report(self):
        """Genera reporte final al cerrar el nodo"""
        if not self.gt_data:
            self.get_logger().warn("No hay datos de ground truth para generar reporte")
            return
            
        self.get_logger().info("Generando reporte final...")
        
        # Sincronizar todos los datos
        all_odom_errors = []
        all_ekf_errors = []
        
        for gt_point in self.gt_data:
            odom_match = self.find_closest_measurement(gt_point['timestamp'], self.odom_data)
            ekf_match = self.find_closest_measurement(gt_point['timestamp'], self.ekf_data)
            
            if odom_match:
                error = self.calculate_error(odom_match, gt_point)
                if error:
                    all_odom_errors.append(error)
                    
            if ekf_match:
                error = self.calculate_error(ekf_match, gt_point)
                if error:
                    all_ekf_errors.append(error)
        
        # Calcular métricas finales
        odom_final = self.calculate_metrics(all_odom_errors)
        ekf_final = self.calculate_metrics(all_ekf_errors)
        
        self.get_logger().info("========== REPORTE FINAL ==========")
        if odom_final:
            self.get_logger().info(f"ODOMETRÍA:")
            self.get_logger().info(f"  - RMSE posición: {odom_final['position']['rmse']:.4f} m")
            self.get_logger().info(f"  - Error promedio: {odom_final['position']['mean']:.4f} m")
            self.get_logger().info(f"  - Error máximo: {odom_final['position']['max']:.4f} m")
            
        if ekf_final:
            self.get_logger().info(f"EKF:")
            self.get_logger().info(f"  - RMSE posición: {ekf_final['position']['rmse']:.4f} m")
            self.get_logger().info(f"  - Error promedio: {ekf_final['position']['mean']:.4f} m")
            self.get_logger().info(f"  - Error máximo: {ekf_final['position']['max']:.4f} m")
            
        if odom_final and ekf_final and odom_final['position']['rmse'] > 0:
            improvement = (1 - ekf_final['position']['rmse']/odom_final['position']['rmse']) * 100
            self.get_logger().info(f"MEJORA TOTAL DEL EKF: {improvement:.1f}%")
            
        self.get_logger().info("===================================")
        
        # Guardar datos finales unificados
        self.save_unified_data()


def main(args=None):
    """
    Función principal del nodo
    """
    rclpy.init(args=args)
    node = PerformanceAnalyzer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Cerrando Performance Analyzer...")
        node.generate_final_report()
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