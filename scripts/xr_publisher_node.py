#!/usr/bin/env python3
"""
XR Publisher Node
发布 XR 设备数据到 ROS2 话题
高频率发布 (250Hz) 以确保实时性
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import PoseStamped
import numpy as np
import sys

# 添加 xrobotoolkit_sdk 路径
sys.path.insert(0, "/home/ming/miniconda3/envs/xrrobotics/lib/python3.10/site-packages/xrobotoolkit_sdk-1.0.2-py3.10-linux-x86_64.egg")
import xrobotoolkit_sdk as xrt


class XRPublisherNode(Node):
    """ROS2 节点：高频发布 XR 设备数据"""
    
    def __init__(self):
        super().__init__('xr_publisher_node')
        
        # 初始化 XR SDK
        self._init_xr_sdk()
        
        # 创建发布器
        self.right_grip_pub = self.create_publisher(
            Float32, 
            'xr/right_grip', 
            10
        )
        
        self.right_trigger_pub = self.create_publisher(
            Float32, 
            'xr/right_trigger', 
            10
        )
        
        self.right_controller_pose_pub = self.create_publisher(
            PoseStamped, 
            'xr/right_controller_pose', 
            10
        )
        
        # 添加按钮发布器
        self.button_a_pub = self.create_publisher(
            Bool,
            'xr/button_a',
            10
        )
        
        self.button_b_pub = self.create_publisher(
            Bool,
            'xr/button_b',
            10
        )
        
        self.button_x_pub = self.create_publisher(
            Bool,
            'xr/button_x',
            10
        )
        
        self.button_y_pub = self.create_publisher(
            Bool,
            'xr/button_y',
            10
        )
        
        # 创建定时器：250Hz (4ms)
        self.publish_rate = 200  # Hz
        timer_period = 1.0 / self.publish_rate
        self.timer = self.create_timer(timer_period, self.publish_callback)
        
        self.get_logger().info(f'XR Publisher Node started at {self.publish_rate}Hz')
        self.get_logger().info('Publishing topics:')
        self.get_logger().info('  - /xr/right_grip (Float32)')
        self.get_logger().info('  - /xr/right_trigger (Float32)')
        self.get_logger().info('  - /xr/right_controller_pose (PoseStamped)')
        self.get_logger().info('  - /xr/button_a (Bool)')
        self.get_logger().info('  - /xr/button_b (Bool)')
        self.get_logger().info('  - /xr/button_x (Bool)')
        self.get_logger().info('  - /xr/button_y (Bool)')
    
    def _init_xr_sdk(self):
        """初始化 XR SDK"""
        try:
            xrt.init()
            self.get_logger().info('XRoboToolkit SDK initialized successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize XR SDK: {str(e)}')
            raise
    
    def publish_callback(self):
        """定时器回调：发布所有 XR 数据"""
        try:
            timestamp = self.get_clock().now().to_msg()
            
            # 1. 发布右手握持值
            grip_msg = Float32()
            grip_msg.data = float(xrt.get_right_grip())
            self.right_grip_pub.publish(grip_msg)
            
            # 2. 发布右手扳机值
            trigger_msg = Float32()
            trigger_msg.data = float(xrt.get_right_trigger())
            self.right_trigger_pub.publish(trigger_msg)
            
            # 3. 发布右手控制器姿态
            pose_array = xrt.get_right_controller_pose()
            pose_msg = PoseStamped()
            pose_msg.header.stamp = timestamp
            pose_msg.header.frame_id = 'vr_origin'
            
            # pose_array: [x, y, z, qx, qy, qz, qw]
            pose_msg.pose.position.x = float(pose_array[0])
            pose_msg.pose.position.y = float(pose_array[1])
            pose_msg.pose.position.z = float(pose_array[2])
            pose_msg.pose.orientation.x = float(pose_array[3])
            pose_msg.pose.orientation.y = float(pose_array[4])
            pose_msg.pose.orientation.z = float(pose_array[5])
            pose_msg.pose.orientation.w = float(pose_array[6])
            
            self.right_controller_pose_pub.publish(pose_msg)
            
            # 4. 发布按钮状态
            button_a_msg = Bool()
            button_a_msg.data = bool(xrt.get_A_button())
            self.button_a_pub.publish(button_a_msg)
            
            button_b_msg = Bool()
            button_b_msg.data = bool(xrt.get_B_button())
            self.button_b_pub.publish(button_b_msg)
            
            button_x_msg = Bool()
            button_x_msg.data = bool(xrt.get_X_button())
            self.button_x_pub.publish(button_x_msg)
            
            button_y_msg = Bool()
            button_y_msg.data = bool(xrt.get_Y_button())
            self.button_y_pub.publish(button_y_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing XR data: {str(e)}")
    
    def destroy_node(self):
        """清理资源"""
        try:
            xrt.close()
            self.get_logger().info('XR SDK closed successfully')
        except Exception as e:
            self.get_logger().error(f'Error closing XR SDK: {str(e)}')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = XRPublisherNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in XR Publisher Node: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
