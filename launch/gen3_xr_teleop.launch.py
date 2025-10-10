#!/usr/bin/env python3
"""Launch file to start XR publisher, Gen3 teleop, and RViz2."""

import os
from pathlib import Path
import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def _resolve_path(path_str: str, base_directory: Path) -> str:
    """Resolve a potentially relative path using the package share directory."""
    if not path_str:
        return ''
    expanded = os.path.expandvars(os.path.expanduser(path_str))
    path = Path(expanded)
    if path.is_absolute():
        return str(path)
    return str((base_directory / path).resolve())


def _parse_bool(value: str, default: bool) -> bool:
    if value is None or value == '':
        return default
    if isinstance(value, bool):
        return value
    return value.lower() in ('true', '1', 'yes', 'on')


def launch_setup(context, *args, **kwargs):
    pkg_share = Path(get_package_share_directory('gen3_xr_teleop'))
    config_path = Path(LaunchConfiguration('config_file').perform(context))

    if not config_path.is_absolute():
        config_path = (pkg_share / config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f'Config file not found: {config_path}')

    with config_path.open('r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}

    teleop_cfg = config.get('teleop', {})
    xr_cfg = config.get('xr_publisher', {})
    rviz_cfg = config.get('rviz', {})

    robot_ip_override = LaunchConfiguration('robot_ip').perform(context)
    urdf_override = LaunchConfiguration('urdf_path').perform(context)
    log_file_override = LaunchConfiguration('log_file').perform(context)

    robot_ip = robot_ip_override or teleop_cfg.get('robot_ip', '192.168.1.10')
    urdf_path = _resolve_path(urdf_override or teleop_cfg.get('urdf_path', ''), pkg_share)
    log_file = _resolve_path(log_file_override or teleop_cfg.get('log_file', 'joint_tracking_data.csv'), pkg_share)

    publish_rate_override = LaunchConfiguration('publish_rate').perform(context)
    frame_id_override = LaunchConfiguration('frame_id').perform(context)
    publish_rate = xr_cfg.get('publish_rate', 200.0)
    if publish_rate_override:
        publish_rate = float(publish_rate_override)
    frame_id = frame_id_override or xr_cfg.get('frame_id', 'vr_origin')

    rviz_enabled_override = LaunchConfiguration('rviz_enabled').perform(context)
    rviz_config_override = LaunchConfiguration('rviz_config').perform(context)
    rviz_enabled = _parse_bool(rviz_enabled_override, rviz_cfg.get('enabled', True))
    rviz_config_path = _resolve_path(rviz_config_override or rviz_cfg.get('config', ''), pkg_share)

    actions = [
        LogInfo(msg=f'Using config file: {config_path}')
    ]

    teleop_args = [robot_ip]
    if urdf_path:
        teleop_args.append(urdf_path)
    if log_file:
        teleop_args.append(log_file)

    actions.append(
        Node(
            package='gen3_xr_teleop',
            executable='gen3_xr_teleop_node',
            name='gen3_xr_teleop_node',
            output='screen',
            arguments=teleop_args,
        )
    )

    xr_parameters = {
        'publish_rate': publish_rate,
        'frame_id': frame_id,
    }

    actions.append(
        Node(
            package='gen3_xr_teleop',
            executable='xr_publisher_node.py',
            name='xr_publisher_node',
            output='screen',
            parameters=[xr_parameters],
        )
    )

    if rviz_enabled:
        rviz_arguments = []
        if rviz_config_path:
            rviz_arguments.extend(['-d', rviz_config_path])
        actions.append(
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                output='screen',
                arguments=rviz_arguments,
            )
        )

    return actions


def generate_launch_description():
    pkg_share = Path(get_package_share_directory('gen3_xr_teleop'))
    default_config = str(pkg_share / 'config' / 'teleop_config.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config,
            description='Path to YAML configuration file for the teleop launch.'
        ),
        DeclareLaunchArgument(
            'robot_ip',
            default_value='',
            description='Override robot IP address defined in config file.'
        ),
        DeclareLaunchArgument(
            'urdf_path',
            default_value='',
            description='Override URDF path defined in config file.'
        ),
        DeclareLaunchArgument(
            'log_file',
            default_value='',
            description='Override log file path defined in config file.'
        ),
        DeclareLaunchArgument(
            'publish_rate',
            default_value='',
            description='Override XR publisher rate (Hz) from config file.'
        ),
        DeclareLaunchArgument(
            'frame_id',
            default_value='',
            description='Override XR pose frame id from config file.'
        ),
        DeclareLaunchArgument(
            'rviz_enabled',
            default_value='',
            description='Override RViz enabled flag from config file.'
        ),
        DeclareLaunchArgument(
            'rviz_config',
            default_value='',
            description='Override RViz config path from config file.'
        ),
        OpaqueFunction(function=launch_setup),
    ])