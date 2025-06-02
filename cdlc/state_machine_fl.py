#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from turtlesim.msg import Pose
from std_msgs.msg import Bool
from std_msgs.msg import Float32

class MainCoordinator(Node):
    def __init__(self):
        super().__init__('main_coordinator')
        self.get_logger().info("MainCoordinator node has been started.")

        self.target_pub = self.create_publisher(Pose, 'target', 10)
        self.create_subscription(Bool, 'close_enough', self.close_enough_callback, 10)
        self.create_subscription(Bool, 'goal_reached', self.goal_reached_callback, 10)
        self.servo_pub = self.create_publisher(Float32, 'ServoAngle', 10)

        self.state = "start"
        self.timer = self.create_timer(0.5, self.state_machine)

        self.close_enough = False
        self.goal_reached = False
        self.dropoff_sent = False
        self.routine_index = 0

        self.precision_path = [
            (1.7, 0.6),
            (1.8, 0.4),
            (1.9, 0.2),
            (2.0, 0.0)
        ]
        self.servo_counter = 0

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
            self.state = "go_to_pick_up"
            self.publish_target(2.0, 0.0)


        elif self.state == "go_to_pick_up":
            self.publish_target(2.0, 0.0)
            if self.close_enough:
                self.get_logger().info("Within 40cm of goal. Switching to PICK_UP_ORIENTATION")
                self.state = "pick_up_orientation"
                self.routine_index = 0
                self.goal_reached = False  # reiniciar bandera


        elif self.state == "pick_up_orientation":
            if self.goal_reached:
                self.goal_reached = False
                self.routine_index += 1
                if self.routine_index < len(self.precision_path):
                    x, y = self.precision_path[self.routine_index]
                    self.publish_target(x, y)
                else:
                    self.get_logger().info("Finished pick-up orientation routine.")
                    self.previous_state = "pick_up_orientation"
                    self.state = "start_servo"
            elif self.routine_index == 0:
                # Primer punto aún no enviado
                x, y = self.precision_path[0]
                self.publish_target(x, y)


        elif self.state == "start_servo":
            if self.servo_counter == 0:
                angle = 260.0
                self.get_logger().info("Publishing servo angle for pick-up (260.0)")
            else:
                angle = -300.0
                self.get_logger().info("Publishing servo angle for drop-off (-300.0)")

            self.servo_pub.publish(Float32(data=angle))
            self.servo_counter = 1 - self.servo_counter

            # Transición después de servo
            if self.previous_state == "pick_up_orientation":
                self.dropoff_sent = False
                self.state = "go_to_drop_off"
            else:
                self.state = "done"


        elif self.state == "go_to_drop_off":
            if not self.dropoff_sent:
                self.publish_target(-2.0, 0.0)
                self.dropoff_sent = True
            if self.close_enough:
                self.get_logger().info("Arrived at drop-off point. Switching to START_SERVO.")
                self.previous_state = "go_to_drop_off"
                self.state = "start_servo"


        elif self.state == "done":
            if not self.returned_home:
                self.get_logger().info("Returning to origin (0.0, 0.0)")
                self.publish_target(0.0, 0.0)
                self.returned_home = True
            elif self.close_enough:
                self.get_logger().info("Robot arrived at origin. State machine complete.")
                self.state = "idle"

        elif self.state == "idle":
            pass  # Aquí acaba


def main(args=None):
    """
    Función principal del nodo
    """
    rclpy.init(args=args)
    node = MainCoordinator()
    
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
