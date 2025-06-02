#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from turtlesim.msg import Pose
from std_msgs.msg import Bool
from std_msgs.msg import Float32

class MainCoordinator(Node):
    def __init__(self):
        super().__init__('main_coordinator')
        self.get_logger().info("SM GOOOOOO ")

        self.target_pub = self.create_publisher(Pose, 'target', 10)
        self.create_subscription(Bool, 'close_enough', self.close_enough_callback, 10)
        self.create_subscription(Bool, 'goal_reached', self.goal_reached_callback, 10)
        self.servo_pub = self.create_publisher(Float32, 'ServoAngle', 10)

        self.state = "start"
        self.timer = self.create_timer(0.5, self.state_machine)

        self.close_enough = False
        self.goal_reached = False
        self.dropoff_sent = False
        self.returned_home = False
        self.orientation_started = False

        self.routine_index = 0

        self.precision_path = [
            (1.5, 0.8),
            (1.75, 0.5),
            (1.95, 0.1),
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
                self.goal_reached = False

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
                    self.orientation_started = False
            elif not self.orientation_started:
                x, y = self.precision_path[0]
                self.publish_target(x, y)
                self.orientation_started = True
                self.get_logger().info("Starting orientation path")

        elif self.state == "start_servo":
            if not hasattr(self, 'servo_start_time'):
                # Publicar solo una vez el ángulo correspondiente
                if self.servo_counter == 0:
                    angle = 260.0
                    self.get_logger().info("Publishing servo angle for pick-up (260.0)")
                else:
                    angle = -300.0
                    self.get_logger().info("Publishing servo angle for drop-off (-300.0)")

                self.servo_pub.publish(Float32(data=angle))
                self.servo_counter = 1 - self.servo_counter

                self.servo_start_time = self.get_clock().now()
            else:
                elapsed_time = (self.get_clock().now() - self.servo_start_time).nanoseconds / 1e9
                if elapsed_time > 3.0:
                    del self.servo_start_time
                    if self.previous_state == "pick_up_orientation":
                        self.dropoff_sent = False
                        self.goal_reached = False  # ← importante reset para drop-off
                        self.state = "go_to_drop_off"
                    else:
                        self.state = "done"

        elif self.state == "go_to_drop_off":
            if not self.dropoff_sent:
                self.publish_target(-2.0, 0.0)
                self.dropoff_sent = True
                self.get_logger().info("Sent drop-off target: (-2.0, 0.0)")
            elif self.goal_reached:
                self.get_logger().info("Reached drop-off target. Switching to START_SERVO.")
                self.previous_state = "go_to_drop_off"
                self.goal_reached = False
                self.dropoff_sent = False  # Reset para futuras ejecuciones
                self.state = "start_servo"

        elif self.state == "done":
            if not self.returned_home:
                self.get_logger().info("Returning to origin (0.0, 0.0)")
                self.publish_target(0.0, 0.0)
                self.returned_home = True
            elif self.goal_reached:
                self.get_logger().info("Robot arrived at origin. State machine complete.")
                self.state = "idle"

        elif self.state == "idle":
            pass  # Aqui muere todo


def main(args=None):
    """
    Función principal del nodo
    """
    rclpy.init(args=args)
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