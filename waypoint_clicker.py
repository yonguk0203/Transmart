#!/usr/bin/env python3
import math
from typing import List

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration

from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion
from std_srvs.srv import Empty
from nav2_msgs.action import FollowWaypoints, NavigateToPose

from tf2_ros import Buffer, TransformListener
from rclpy.time import Time


def yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    # z-yaw quaternion
    q.w = math.cos(yaw/2.0)
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw/2.0)
    return q


class WaypointClicker(Node):
    def __init__(self):
        super().__init__('waypoint_clicker')

        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('auto_yaw', True)
        self.declare_parameter('click_topic', '/clicked_point')

        self.map_frame = self.get_parameter('map_frame').get_parameter_value().string_value
        self.robot_frame = self.get_parameter('robot_frame').get_parameter_value().string_value
        self.auto_yaw = self.get_parameter('auto_yaw').get_parameter_value().bool_value
        click_topic = self.get_parameter('click_topic').get_parameter_value().string_value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.points: List[PoseStamped] = []

        self.sub = self.create_subscription(PointStamped, click_topic, self.on_clicked, 10)
        self.srv_start = self.create_service(Empty, 'start_waypoints', self.handle_start)
        self.srv_clear = self.create_service(Empty, 'clear_waypoints', self.handle_clear)

        self.follow_client = ActionClient(self, FollowWaypoints, 'follow_waypoints')
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.get_logger().info("WaypointClicker ready. Click points in RViz. "
                               "Call '/start_waypoints' to start, '/clear_waypoints' to reset.")

    def on_clicked(self, msg: PointStamped):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = msg.header.frame_id  

        pose.pose.position.x = msg.point.x
        pose.pose.position.y = msg.point.y
        pose.pose.position.z = 0.0
        pose.pose.orientation = yaw_to_quat(0.0)  

        self.points.append(pose)
        self.get_logger().info(f"Added waypoint #{len(self.points)} at "
                               f"({msg.point.x:.2f}, {msg.point.y:.2f}) frame={msg.header.frame_id}")

    def handle_clear(self, request, response):
        self.points.clear()
        self.get_logger().info("Waypoints cleared.")
        return response

    def handle_start(self, request, response):
        if not self.points:
            self.get_logger().warn("No waypoints to follow.")
            return response

        waypoints = self._normalize_and_orient(self.points)

        if self.follow_client.wait_for_server(timeout_sec=0.5):
            self.get_logger().info(f"Sending {len(waypoints)} waypoints via FollowWaypoints...")
            goal = FollowWaypoints.Goal()
            goal.poses = waypoints
            send_future = self.follow_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, send_future)
            if not send_future.result():
                self.get_logger().error("Failed to send FollowWaypoints goal.")
                return response
            result_future = send_future.result().get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            self.get_logger().info("FollowWaypoints finished.")
        else:
            self.get_logger().warn("follow_waypoints server not available. Falling back to NavigateToPose sequential mode.")
            if not self.nav_client.wait_for_server(timeout_sec=5.0):
                self.get_logger().error("navigate_to_pose action server not available.")
                return response

            for i, pose in enumerate(waypoints, 1):
                goal = NavigateToPose.Goal()
                goal.pose = pose
                self.get_logger().info(f"[{i}/{len(waypoints)}] Navigating to ({pose.pose.position.x:.2f}, {pose.pose.position.y:.2f})")
                send_future = self.nav_client.send_goal_async(goal)
                rclpy.spin_until_future_complete(self, send_future)
                if not send_future.result():
                    self.get_logger().error("Failed to send NavigateToPose goal.")
                    break
                result_future = send_future.result().get_result_async()
                rclpy.spin_until_future_complete(self, result_future)
                self.get_logger().info(f"Reached waypoint {i}")

        return response

    def _normalize_and_orient(self, raw: List[PoseStamped]) -> List[PoseStamped]:
        out: List[PoseStamped] = []
        for i, p in enumerate(raw):
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = self.map_frame

            if p.header.frame_id != self.map_frame:
                try:
                    tf = self.tf_buffer.lookup_transform(self.map_frame, p.header.frame_id, Time(), timeout=Duration(seconds=0.5))
                    tx = tf.transform.translation.x
                    ty = tf.transform.translation.y
                    yaw_tf = 2.0 * math.atan2(tf.transform.rotation.z, tf.transform.rotation.w)
                    x = tx + math.cos(yaw_tf) * p.pose.position.x - math.sin(yaw_tf) * p.pose.position.y
                    y = ty + math.sin(yaw_tf) * p.pose.position.x + math.cos(yaw_tf) * p.pose.position.y
                except Exception as e:
                    self.get_logger().warn(f"TF transform failed ({p.header.frame_id}-> {self.map_frame}): {e}")
                    x, y = p.pose.position.x, p.pose.position.y
            else:
                x, y = p.pose.position.x, p.pose.position.y

            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0

            if self.auto_yaw and i < len(raw) - 1:
                nx = raw[i+1].pose.position.x
                ny = raw[i+1].pose.position.y
                yaw = math.atan2(ny - y, nx - x)
            else:
                yaw = 0.0
            pose.pose.orientation = yaw_to_quat(yaw)

            out.append(pose)
        return out


def main():
    rclpy.init()
    node = WaypointClicker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
