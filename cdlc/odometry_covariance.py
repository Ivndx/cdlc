#!/usr/bin/env python3

import rclpy, time, math
import numpy as np
from rclpy.node import Node
from turtlesim.msg import Pose
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from tf_transformations import quaternion_from_euler
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




		self.pub_pose = self.create_publisher(Pose, "PosicionBot", 10)
		self.pub_odom = self.create_publisher(Odometry, "odom", 10)


		self.t0 = 0.0
		self.vel_r = 0.0
		self.vel_l = 0.0
		self.t_ant = 0.0
		self.xf = 0.0
		self.yf = 0.0
		self.angf = 0.0

		self.r = 0.0505
		self.L = 0.183
		self.kr = 0.0273
		self.kl = 0.0067

		self.xx = 0.1186455556
		self.xy = -0.08364444444
		self.yy = -0.08364444444
		self.yx = 0.069688888889
		self.thth = 0.05
		self.thx = 0.05
		self.thy = 0.05
		self.xth = 0.05
		self.yth = 0.05

		self.cov = np.array([
			[self.xx,  self.xy,  self.xth],
			[self.yx,  self.yy,  self.yth],
			[self.thx, self.thy, self.thth]
		])

		self.timer = self.create_timer(0.05, self.callback_pub)

	def clock_callback(self, msg):
    		current_time_sec = msg.clock.sec + msg.clock.nanosec * 1e-9
    		self.t0 = current_time_sec

	def callback_L(self, msg):
		self.vel_l = msg.data

	def callback_R(self, msg):
		self.vel_r = msg.data

	def callback_pub(self):
		dt = 0.05
		r = self.r
		L = self.L
		wr = self.vel_r
		wl = self.vel_l
		v = (r / 2.0) * (wr + wl)

		# Guardar orientación anterior antes de actualizar
		theta_prev = self.angf

		# Jacobiano H_k
		Hk = np.array([
			[1, 0, -dt * v * math.sin(theta_prev)],
			[0, 1,  dt * v * math.cos(theta_prev)],
			[0, 0, 1]
		])

		# Sigma_delta
		Sigma_delta = np.array([
			[abs(self.kr) * abs(wr), 0],
			[0, abs(self.kl) * abs(wl)]
		])

		# V_wk con theta_prev
		Vwk = 0.5 * r * dt * np.array([
			[math.cos(theta_prev), math.cos(theta_prev)],
			[math.sin(theta_prev), math.sin(theta_prev)],
			[2 / L, -2 / L]
		])

		Qk = Vwk @ Sigma_delta @ Vwk.T
		self.cov = Hk @ self.cov @ Hk.T + Qk

		# Actualizar valores individuales
		self.xx  = self.cov[0, 0]
		self.xy  = self.cov[0, 1]
		self.xth = self.cov[0, 2]
		self.yx  = self.cov[1, 0]
		self.yy  = self.cov[1, 1]
		self.yth = self.cov[1, 2]
		self.thx = self.cov[2, 0]
		self.thy = self.cov[2, 1]
		self.thth= self.cov[2, 2]

		# Publicar pose para controladores legacy
		pose_msg = Pose()
		pose_msg.x = self.xf
		pose_msg.y = self.yf
		pose_msg.theta = self.angf
		self.pub_pose.publish(pose_msg)

		# Publicar odometría
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

		odom_msg.pose.covariance = [0.0] * 36
		odom_msg.pose.covariance[0]  = self.xx
		odom_msg.pose.covariance[1]  = self.xy
		odom_msg.pose.covariance[5]  = self.xth
		odom_msg.pose.covariance[6]  = self.yx
		odom_msg.pose.covariance[7]  = self.yy
		odom_msg.pose.covariance[11] = self.yth
		odom_msg.pose.covariance[30] = self.thx
		odom_msg.pose.covariance[31] = self.thy
		odom_msg.pose.covariance[35] = self.thth

		self.pub_odom.publish(odom_msg)

	def principal(self):
		while rclpy.ok():
		
			td = self.t0 -self.t_ant
			

			V = (self.r / 2.0) * (self.vel_r + self.vel_l)
			W = (self.r / self.L) * (self.vel_r - self.vel_l)
			self.angf += td * W
			self.angf = math.atan2(math.sin(self.angf), math.cos(self.angf))
			self.xf += td * V * math.cos(self.angf)
			self.yf += td * V * math.sin(self.angf)
			
			self.t_ant = self.t0
			
			rclpy.spin_once(self)
			
def main(args=None):
	rclpy.init(args=args)
	nodeh = DeadReckoning()
	try:
		nodeh.principal()
	except Exception as error:
		print(error)
	except KeyboardInterrupt:
		print("Node terminated by user!")

if __name__ == "__main__":
	main()
