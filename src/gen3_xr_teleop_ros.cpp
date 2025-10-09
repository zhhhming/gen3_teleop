/**
 * Gen3 XR Teleoperation ROS2 Node with Data Logging
 * 订阅 XR 数据话题，实时控制 Kinova Gen3 机械臂，并记录位置数据
 */

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/bool.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <iostream>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <vector>
#include <deque>
#include <signal.h>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <iomanip>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// TRAC-IK
#include <trac_ik/trac_ik.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>

// Eigen
#include <Eigen/Dense>
#include <Eigen/Geometry>

// Robot controller
#include "Gen3RobotController.h"

// ====== 工具函数（文件作用域）======
namespace {
inline float clampf(float x, float lo, float hi) {
    return std::max(lo, std::min(hi, x));
}
}

// 全局关闭标志
std::atomic<bool> g_shutdown_requested(false);

void signal_handler(int sig) {
    std::cout << "\nShutdown signal received" << std::endl;
    g_shutdown_requested = true;
}

// 数据记录结构
struct JointDataPoint {
    double timestamp;  // 相对于开始时间的秒数
    std::vector<float> current_positions;  // 当前位置（°）
    std::vector<float> target_positions;   // 目标位置（°，已做就近映射）
    std::vector<float> differences;        // 差值（°）
};

/**
 * ROS2 节点：Gen3 XR 遥操作控制器（带数据记录）
 */
class Gen3XRTeleopNode : public rclcpp::Node {
public:
    Gen3XRTeleopNode(const std::string& robot_urdf_path,
                     const std::string& robot_ip = "192.168.1.10",
                     int tcp_port = 10000,
                     int udp_port = 10001,
                     const std::string& log_file = "joint_tracking_data.csv")
        : Node("gen3_xr_teleop_node"),
          robot_urdf_path_(robot_urdf_path),
          robot_ip_(robot_ip),
          tcp_port_(tcp_port),
          udp_port_(udp_port),
          log_file_path_(log_file),
          shutdown_requested_(false),
          num_joints_(7),
          scale_factor_(1.1f),
          ik_rate_hz_(50),
          control_rate_hz_(1000),
          is_active_(false),
          ref_ee_valid_(false),
          ref_controller_valid_(false),
          filter_initialized_(false),
          filter_alpha_(0.005f),
          gripper_control_mode_(0),  // 0: trigger mode, 1: button mode
          gripper_step_value_(0.1f),
          gripper_button_repeat_interval_(0.1),
          data_logging_enabled_(true)  // 默认开启数据记录
    {
        // 初始化状态向量
        target_joints_.resize(num_joints_, 0.0f);
        current_joints_.resize(num_joints_, 0.0f);
        filtered_joint_state_.resize(num_joints_, 0.0f);
        target_gripper_ = 0.0f;

        // 初始化坐标变换
        initializeTransforms();

        // 初始化 XR 数据（默认值）
        xr_right_grip_ = 0.0f;
        xr_right_trigger_ = 0.0f;
        xr_controller_pose_.resize(7, 0.0);

        // 初始化按钮状态
        button_a_state_ = false;
        button_b_state_ = false;
        button_x_state_ = false;
        button_y_state_ = false;
        button_a_prev_ = false;
        button_b_prev_ = false;
        button_x_prev_ = false;
        button_y_prev_ = false;

        // 初始化按钮计时器
        button_x_last_trigger_time_ = std::chrono::steady_clock::now();
        button_y_last_trigger_time_ = std::chrono::steady_clock::now();

        // 初始化数据记录
        start_time_ = std::chrono::steady_clock::now();

        // 创建订阅器
        grip_sub_ = this->create_subscription<std_msgs::msg::Float32>(
            "xr/right_grip", 10,
            std::bind(&Gen3XRTeleopNode::gripCallback, this, std::placeholders::_1));

        trigger_sub_ = this->create_subscription<std_msgs::msg::Float32>(
            "xr/right_trigger", 10,
            std::bind(&Gen3XRTeleopNode::triggerCallback, this, std::placeholders::_1));

        pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "xr/right_controller_pose", 10,
            std::bind(&Gen3XRTeleopNode::poseCallback, this, std::placeholders::_1));

        button_a_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            "xr/button_a", 10,
            std::bind(&Gen3XRTeleopNode::buttonACallback, this, std::placeholders::_1));

        button_b_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            "xr/button_b", 10,
            std::bind(&Gen3XRTeleopNode::buttonBCallback, this, std::placeholders::_1));

        button_x_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            "xr/button_x", 10,
            std::bind(&Gen3XRTeleopNode::buttonXCallback, this, std::placeholders::_1));

        button_y_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            "xr/button_y", 10,
            std::bind(&Gen3XRTeleopNode::buttonYCallback, this, std::placeholders::_1));

        // 创建 TF broadcaster
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        RCLCPP_INFO(this->get_logger(), "Gen3 XR Teleop Node created");
        RCLCPP_INFO(this->get_logger(), "Data logging: %s",
                   data_logging_enabled_ ? "ENABLED" : "DISABLED");
        RCLCPP_INFO(this->get_logger(), "Log file: %s", log_file_path_.c_str());
        RCLCPP_INFO(this->get_logger(), "Gripper control:");
        RCLCPP_INFO(this->get_logger(), "  - Button A: Toggle control mode (Trigger/Button)");
        RCLCPP_INFO(this->get_logger(), "  - Button B: Cycle step value (0.1/0.01/0.001/0.0001)");
        RCLCPP_INFO(this->get_logger(), "  - Button X: Increase gripper (in button mode)");
        RCLCPP_INFO(this->get_logger(), "  - Button Y: Decrease gripper (in button mode)");
        RCLCPP_INFO(this->get_logger(), "  - Hold button: Repeat every 0.2s");
    }

    ~Gen3XRTeleopNode() {
        shutdown();
    }

    bool initialize() {
        RCLCPP_INFO(this->get_logger(), "Initializing Gen3 XR Teleoperation Controller...");

        // 1. 初始化机械臂控制器
        if (!initializeRobot()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize robot controller");
            return false;
        }

        // 2. 初始化 TRAC-IK
        if (!initializeTracIK()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize TRAC-IK");
            return false;
        }

        RCLCPP_INFO(this->get_logger(), "Controller initialized successfully!");
        return true;
    }

    void run() {
        RCLCPP_INFO(this->get_logger(), "Starting teleoperation threads...");

        // 启动 IK 线程
        std::thread ik_thread(&Gen3XRTeleopNode::ikThread, this);

        // 启动控制线程（高优先级）
        std::thread control_thread(&Gen3XRTeleopNode::controlThread, this);

        // 设置控制线程优先级（Linux）
#ifdef __linux__
        sched_param sch_params;
        sch_params.sched_priority = sched_get_priority_max(SCHED_FIFO) - 1;
        if (pthread_setschedparam(control_thread.native_handle(), SCHED_FIFO, &sch_params)) {
            RCLCPP_WARN(this->get_logger(), "Failed to set control thread priority");
        }
#endif

        // 主线程处理 ROS2 回调
        while (!shutdown_requested_ && !g_shutdown_requested && rclcpp::ok()) {
            rclcpp::spin_some(this->shared_from_this());
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        // 停止线程
        shutdown_requested_ = true;

        if (ik_thread.joinable()) {
            ik_thread.join();
        }
        if (control_thread.joinable()) {
            control_thread.join();
        }

        // 保存数据
        if (data_logging_enabled_) {
            saveLoggedData();
        }

        RCLCPP_INFO(this->get_logger(), "Teleoperation stopped");
    }

    void shutdown() {
        shutdown_requested_ = true;

        // 清理机械臂
        if (robot_controller_) {
            robot_controller_->exitLowLevelMode();
            robot_controller_->stopRobot();
            robot_controller_->shutdown();
            robot_controller_.reset();
        }
    }

private:
    // ========== ROS2 回调函数 ==========

    void gripCallback(const std_msgs::msg::Float32::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(xr_data_mutex_);
        xr_right_grip_ = msg->data;
    }

    void triggerCallback(const std_msgs::msg::Float32::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(xr_data_mutex_);
        xr_right_trigger_ = msg->data;
    }

    void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(xr_data_mutex_);
        xr_controller_pose_[0] = msg->pose.position.x;
        xr_controller_pose_[1] = msg->pose.position.y;
        xr_controller_pose_[2] = msg->pose.position.z;
        xr_controller_pose_[3] = msg->pose.orientation.x;
        xr_controller_pose_[4] = msg->pose.orientation.y;
        xr_controller_pose_[5] = msg->pose.orientation.z;
        xr_controller_pose_[6] = msg->pose.orientation.w;
    }

    void buttonACallback(const std_msgs::msg::Bool::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(xr_data_mutex_);
        button_a_state_ = msg->data;
    }

    void buttonBCallback(const std_msgs::msg::Bool::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(xr_data_mutex_);
        button_b_state_ = msg->data;
    }

    void buttonXCallback(const std_msgs::msg::Bool::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(xr_data_mutex_);
        button_x_state_ = msg->data;
    }

    void buttonYCallback(const std_msgs::msg::Bool::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(xr_data_mutex_);
        button_y_state_ = msg->data;
    }

    // ========== 初始化函数 ==========

    bool initializeRobot() {
        try {
            robot_controller_ = std::make_unique<Gen3RobotController>(
                robot_ip_, tcp_port_, udp_port_, "admin", "admin"
            );

            if (!robot_controller_->initialize()) {
                return false;
            }

            robot_controller_->clearFaults();

            if (!robot_controller_->enterLowLevelMode()) {
                return false;
            }

            auto positions = normalizeAngles(robot_controller_->getJointPositions());
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                current_joints_ = positions;
                target_joints_ = positions;
            }

            initializeFilterState(positions);

            RCLCPP_INFO(this->get_logger(), "Robot controller initialized");
            return true;

        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Robot initialization error: %s", e.what());
            return false;
        }
    }

    bool initializeTracIK() {
        try {
            std::ifstream urdf_file(robot_urdf_path_);
            if (!urdf_file.is_open()) {
                RCLCPP_ERROR(this->get_logger(), "Cannot open URDF: %s", robot_urdf_path_.c_str());
                return false;
            }

            std::string urdf_string((std::istreambuf_iterator<char>(urdf_file)),
                                    std::istreambuf_iterator<char>());

            tracik_solver_ = std::make_unique<TRAC_IK::TRAC_IK>(
                "base_link", "bracelet_link", urdf_string,
                0.005, 0.001, TRAC_IK::Distance
            );

            KDL::Tree kdl_tree;
            if (!kdl_parser::treeFromString(urdf_string, kdl_tree)) {
                RCLCPP_ERROR(this->get_logger(), "Failed to parse URDF to KDL tree");
                return false;
            }

            if (!kdl_tree.getChain("base_link", "bracelet_link", kdl_chain_)) {
                RCLCPP_ERROR(this->get_logger(), "Failed to extract KDL chain");
                return false;
            }

            fk_solver_ = std::make_unique<KDL::ChainFkSolverPos_recursive>(kdl_chain_);

            RCLCPP_INFO(this->get_logger(), "TRAC-IK initialized with %d joints",
                       kdl_chain_.getNrOfJoints());
            return true;

        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "TRAC-IK initialization error: %s", e.what());
            return false;
        }
    }

    void initializeTransforms() {
        R_headset_world_ << 0, 0, -1,
                           -1, 0, 0,
                            0, 1, 0;

        R_z_90_cw_ << 0, 1, 0,
                     -1, 0, 0,
                      0, 0, 1;
    }

    void initializeFilterState(const std::vector<float>& initial_positions) {
        filtered_joint_state_ = initial_positions;
        filter_initialized_ = true;
    }

    // ========== 工具函数 ==========

    float normalizeAngle(float angle) const {
        float normalized = std::fmod(angle + 180.0f, 360.0f);
        if (normalized < 0.0f) {
            normalized += 360.0f;
        }
        return normalized - 180.0f;
    }

    std::vector<float> normalizeAngles(const std::vector<float>& angles) const {
        std::vector<float> normalized;
        normalized.reserve(angles.size());
        for (float angle : angles) {
            normalized.push_back(normalizeAngle(angle));
        }
        return normalized;
    }

    // 将 target 展开到 reference 附近（°）
    float unwrapAngle(float target, float reference) const {
        double unwrapped = static_cast<double>(reference) +
                           std::remainder(static_cast<double>(target) -
                                          static_cast<double>(reference), 360.0);
        return static_cast<float>(unwrapped);
    }

    // 将 target 移到距离 current 最近的等效位置（考虑 360 度环绕，°）
    float moveToNearestEquivalent(float target, float current) const {
        // remainder(x, 360) ∈ (-180, 180]
        const float delta = std::remainder(target - current, 360.0f);
        return current + delta;
    }

    std::vector<float> filterJointPositions(const std::vector<float>& target_positions) {
        if (!filter_initialized_) {
            initializeFilterState(target_positions);
            return target_positions;
        }

        std::vector<float> filtered(num_joints_, 0.0f);
        for (int i = 0; i < num_joints_; ++i) {
            float unwrapped_target = unwrapAngle(target_positions[i], filtered_joint_state_[i]);
            float filtered_angle = filter_alpha_ * unwrapped_target +
                                  (1.0f - filter_alpha_) * filtered_joint_state_[i];
            filtered[i] = filtered_angle;
            filtered_joint_state_[i] = filtered_angle;
        }

        return filtered;
    }

    void processControllerPose(const std::vector<double>& xr_pose,
                              Eigen::Vector3d& delta_pos,
                              Eigen::Vector3d& delta_rot) {
        // 提取位置和四元数
        Eigen::Vector3d controller_pos(xr_pose[0], xr_pose[1], xr_pose[2]);
        Eigen::Quaterniond controller_quat(xr_pose[6], xr_pose[3], xr_pose[4], xr_pose[5]);

        // 转换到世界坐标系
        controller_pos = R_headset_world_ * controller_pos;
        Eigen::Quaterniond R_quat(R_headset_world_);
        controller_quat = R_quat * controller_quat * R_quat.conjugate();

        // 计算增量
        if (!ref_controller_valid_) {
            ref_controller_pos_ = controller_pos;
            ref_controller_quat_ = controller_quat;
            ref_controller_valid_ = true;
            delta_pos.setZero();
            delta_rot.setZero();
        } else {
            delta_pos = (controller_pos - ref_controller_pos_) * scale_factor_;

            Eigen::Quaterniond quat_diff = controller_quat * ref_controller_quat_.conjugate();
            Eigen::AngleAxisd angle_axis(quat_diff);
            delta_rot = angle_axis.angle() * angle_axis.axis();
        }

        // 应用 90 度旋转
        delta_pos = R_z_90_cw_ * delta_pos;
        delta_rot = R_z_90_cw_ * delta_rot;
    }

    KDL::Frame eigenToKDL(const Eigen::Vector3d& pos, const Eigen::Quaterniond& quat) {
        KDL::Frame frame;
        frame.p = KDL::Vector(pos.x(), pos.y(), pos.z());
        frame.M = KDL::Rotation::Quaternion(quat.x(), quat.y(), quat.z(), quat.w());
        return frame;
    }

    void kdlToEigen(const KDL::Frame& frame, Eigen::Vector3d& pos, Eigen::Quaterniond& quat) {
        pos = Eigen::Vector3d(frame.p.x(), frame.p.y(), frame.p.z());
        double x, y, z, w;
        frame.M.GetQuaternion(x, y, z, w);
        quat = Eigen::Quaterniond(w, x, y, z);
    }

    void publishTF(const KDL::Frame& current_frame, const KDL::Frame& target_frame) {
        auto now = this->get_clock()->now();

        // 发布 current frame
        geometry_msgs::msg::TransformStamped current_tf;
        current_tf.header.stamp = now;
        current_tf.header.frame_id = "world";
        current_tf.child_frame_id = "ee_current";
        current_tf.transform.translation.x = current_frame.p.x();
        current_tf.transform.translation.y = current_frame.p.y();
        current_tf.transform.translation.z = current_frame.p.z();

        double x, y, z, w;
        current_frame.M.GetQuaternion(x, y, z, w);
        current_tf.transform.rotation.x = x;
        current_tf.transform.rotation.y = y;
        current_tf.transform.rotation.z = z;
        current_tf.transform.rotation.w = w;

        tf_broadcaster_->sendTransform(current_tf);

        // 发布 target frame
        geometry_msgs::msg::TransformStamped target_tf;
        target_tf.header.stamp = now;
        target_tf.header.frame_id = "world";
        target_tf.child_frame_id = "ee_target";
        target_tf.transform.translation.x = target_frame.p.x();
        target_tf.transform.translation.y = target_frame.p.y();
        target_tf.transform.translation.z = target_frame.p.z();

        target_frame.M.GetQuaternion(x, y, z, w);
        target_tf.transform.rotation.x = x;
        target_tf.transform.rotation.y = y;
        target_tf.transform.rotation.z = z;
        target_tf.transform.rotation.w = w;

        tf_broadcaster_->sendTransform(target_tf);
    }

    // ========== 数据记录函数 ==========

    void logJointData(const std::vector<float>& current, const std::vector<float>& target) {
        if (!data_logging_enabled_) return;

        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time_).count();

        JointDataPoint data_point;
        data_point.timestamp = elapsed;
        data_point.current_positions = current;

        // 处理target positions：移到距离current最近的等效位置
        data_point.target_positions.resize(num_joints_);
        data_point.differences.resize(num_joints_);

        for (int i = 0; i < num_joints_; ++i) {
            float adjusted_target = moveToNearestEquivalent(target[i], current[i]);
            data_point.target_positions[i] = adjusted_target;
            data_point.differences[i] = adjusted_target - current[i];
        }

        std::lock_guard<std::mutex> lock(log_data_mutex_);
        logged_data_.push_back(data_point);
    }

    void saveLoggedData() {
        std::lock_guard<std::mutex> lock(log_data_mutex_);

        if (logged_data_.empty()) {
            RCLCPP_WARN(this->get_logger(), "No data to save");
            return;
        }

        std::ofstream file(log_file_path_);
        if (!file.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open log file: %s",
                        log_file_path_.c_str());
            return;
        }

        // 写入表头
        file << "timestamp";
        for (int i = 0; i < num_joints_; ++i) {
            file << ",current_j" << i
                 << ",target_j" << i
                 << ",diff_j" << i;
        }
        file << "\n";

        // 写入数据
        file << std::fixed << std::setprecision(6);
        for (const auto& data : logged_data_) {
            file << data.timestamp;
            for (int i = 0; i < num_joints_; ++i) {
                file << "," << data.current_positions[i]
                     << "," << data.target_positions[i]
                     << "," << data.differences[i];
            }
            file << "\n";
        }

        file.close();

        RCLCPP_INFO(this->get_logger(), "Saved %zu data points to %s",
                   logged_data_.size(), log_file_path_.c_str());
    }

    // ========== 线程函数 ==========

    void ikThread() {
        RCLCPP_INFO(this->get_logger(), "IK thread started");
        auto dt = std::chrono::duration<double>(1.0 / ik_rate_hz_);

        while (!shutdown_requested_ && !g_shutdown_requested) {
            auto loop_start = std::chrono::steady_clock::now();

            try {
                // 获取 XR 输入（带锁）
                float grip_value, trigger_value;
                bool button_a, button_b, button_x, button_y;
                std::vector<double> controller_pose;
                {
                    std::lock_guard<std::mutex> lock(xr_data_mutex_);
                    grip_value = xr_right_grip_;
                    trigger_value = xr_right_trigger_;
                    controller_pose = xr_controller_pose_;
                    button_a = button_a_state_;
                    button_b = button_b_state_;
                    button_x = button_x_state_;
                    button_y = button_y_state_;
                }

                // 处理按钮 A：切换控制模式
                if (button_a && !button_a_prev_) {
                    gripper_control_mode_ = 1 - gripper_control_mode_;
                    RCLCPP_INFO(this->get_logger(), "Gripper control mode: %s",
                               gripper_control_mode_ == 0 ? "TRIGGER" : "BUTTON");
                }
                button_a_prev_ = button_a;

                // 处理按钮 B：切换步进值
                if (button_b && !button_b_prev_) {
                    if (gripper_step_value_ == 0.1f) {
                        gripper_step_value_ = 0.01f;
                    } else if (gripper_step_value_ == 0.01f) {
                        gripper_step_value_ = 0.001f;
                    } else if (gripper_step_value_ == 0.001f) {
                        gripper_step_value_ = 0.0001f;
                    } else {
                        gripper_step_value_ = 0.1f;
                    }
                    RCLCPP_INFO(this->get_logger(), "Gripper step value: %.4f", gripper_step_value_);
                }
                button_b_prev_ = button_b;

                // 更新夹爪目标
                float new_gripper_target;
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    new_gripper_target = target_gripper_;
                }

                if (gripper_control_mode_ == 0) {
                    // Trigger 模式：直接使用 trigger 值
                    new_gripper_target = std::max(0.0f, std::min(1.0f, trigger_value));
                } else {
                    // Button 模式：使用 X/Y 按钮增减（带重复间隔）
                    auto current_time = std::chrono::steady_clock::now();

                    // 处理按钮 X：增加
                    if (button_x) {
                        bool should_trigger = false;
                        if (!button_x_prev_) {
                            // 首次按下，立即触发
                            should_trigger = true;
                            button_x_last_trigger_time_ = current_time;
                        } else {
                            // 持续按住，检查是否超过重复间隔
                            auto elapsed = std::chrono::duration<double>(
                                current_time - button_x_last_trigger_time_).count();
                            if (elapsed >= gripper_button_repeat_interval_) {
                                should_trigger = true;
                                button_x_last_trigger_time_ = current_time;
                            }
                        }

                        if (should_trigger) {
                            new_gripper_target += gripper_step_value_;
                            new_gripper_target = std::max(0.0f, std::min(1.0f, new_gripper_target));
                            RCLCPP_INFO(this->get_logger(), "Gripper target: %.4f (increased)",
                                       new_gripper_target);
                        }
                    }
                    button_x_prev_ = button_x;

                    // 处理按钮 Y：减少
                    if (button_y) {
                        bool should_trigger = false;
                        if (!button_y_prev_) {
                            // 首次按下，立即触发
                            should_trigger = true;
                            button_y_last_trigger_time_ = current_time;
                        } else {
                            // 持续按住，检查是否超过重复间隔
                            auto elapsed = std::chrono::duration<double>(
                                current_time - button_y_last_trigger_time_).count();
                            if (elapsed >= gripper_button_repeat_interval_) {
                                should_trigger = true;
                                button_y_last_trigger_time_ = current_time;
                            }
                        }

                        if (should_trigger) {
                            new_gripper_target -= gripper_step_value_;
                            new_gripper_target = std::max(0.0f, std::min(1.0f, new_gripper_target));
                            RCLCPP_INFO(this->get_logger(), "Gripper target: %.4f (decreased)",
                                       new_gripper_target);
                        }
                    }
                    button_y_prev_ = button_y;
                }

                // 更新夹爪目标
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    target_gripper_ = new_gripper_target;
                }

                // 获取当前关节位置用于 FK（弧度）
                KDL::JntArray current_joints_kdl(num_joints_);
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    for (int i = 0; i < num_joints_; ++i) {
                        current_joints_kdl(i) = current_joints_[i] * M_PI / 180.0;
                    }
                }

                // 计算当前 end effector frame（每次循环都更新）
                KDL::Frame current_ee_frame;
                fk_solver_->JntToCart(current_joints_kdl, current_ee_frame);

                // 存储 current frame
                {
                    std::lock_guard<std::mutex> lock(frame_mutex_);
                    current_ee_frame_ = current_ee_frame;
                }

                // 检查激活状态
                bool new_active = (grip_value > 0.9f);

                if (new_active != is_active_) {
                    if (new_active) {
                        RCLCPP_INFO(this->get_logger(), "Control activated");
                        ref_ee_valid_ = false;
                        ref_controller_valid_ = false;
                    } else {
                        RCLCPP_INFO(this->get_logger(), "Control deactivated");
                    }
                    is_active_ = new_active;
                }

                if (is_active_) {
                    // 初始化参考坐标系
                    if (!ref_ee_valid_) {
                        ref_ee_frame_ = current_ee_frame;
                        ref_ee_valid_ = true;
                    }

                    // 计算控制器姿态增量
                    Eigen::Vector3d delta_pos, delta_rot;
                    processControllerPose(controller_pose, delta_pos, delta_rot);

                    // 应用增量到参考坐标系
                    Eigen::Vector3d ref_pos;
                    Eigen::Quaterniond ref_quat;
                    kdlToEigen(ref_ee_frame_, ref_pos, ref_quat);

                    Eigen::Vector3d target_pos = ref_pos + delta_pos;

                    double angle = delta_rot.norm();
                    Eigen::Quaterniond target_quat = ref_quat;
                    if (angle > 1e-6) {
                        Eigen::Vector3d axis = delta_rot / angle;
                        Eigen::AngleAxisd delta_rotation(angle, axis);
                        target_quat = delta_rotation * ref_quat;
                    }

                    // 转换为 KDL frame
                    KDL::Frame target_frame = eigenToKDL(target_pos, target_quat);

                    // 求解 IK
                    KDL::JntArray ik_solution(num_joints_);
                    int ret = tracik_solver_->CartToJnt(current_joints_kdl, target_frame, ik_solution);

                    if (ret >= 0) {
                        // IK 成功，更新 target joints 和 target frame
                        {
                            std::lock_guard<std::mutex> lock(state_mutex_);
                            for (int i = 0; i < num_joints_; ++i) {
                                // 直接以“度”存储（不再做 ±180 归一化），便于连续关节跨圈
                                target_joints_[i] = static_cast<float>(ik_solution(i) * 180.0 / M_PI);
                            }
                        }

                        // 更新 target frame（只在IK成功时更新）
                        {
                            std::lock_guard<std::mutex> lock(frame_mutex_);
                            target_ee_frame_ = target_frame;
                        }
                    } else {
                        static int fail_count = 0;
                        if (++fail_count % 2 == 0) {
                            RCLCPP_WARN(this->get_logger(), "IK solution not found");
                        }
                    }
                }

                // 发布 TF（不管是否 active 都发布）
                {
                    std::lock_guard<std::mutex> lock(frame_mutex_);
                    publishTF(current_ee_frame_, target_ee_frame_);
                }

            } catch (const std::exception& e) {
                RCLCPP_ERROR(this->get_logger(), "IK thread error: %s", e.what());
            }

            // 保持循环频率
            auto loop_end = std::chrono::steady_clock::now();
            auto loop_duration = loop_end - loop_start;
            if (loop_duration < dt) {
                std::this_thread::sleep_for(dt - loop_duration);
            }
        }

        RCLCPP_INFO(this->get_logger(), "IK thread stopped");
    }

    void controlThread() {
        RCLCPP_INFO(this->get_logger(), "Control thread started at %dHz", control_rate_hz_);
        auto dt = std::chrono::duration<double>(1.0 / control_rate_hz_);

        std::deque<double> loop_times;
        const size_t max_samples = 1000;
        auto last_report = std::chrono::steady_clock::now();

        // 期望的“最大角速度” (°/s)。可按实际需求调参或做成每关节数组。
        const float max_step_deg  = 0.8f;

        while (!shutdown_requested_ && !g_shutdown_requested) {
            auto loop_start = std::chrono::steady_clock::now();

            try {
                // 获取目标位置（度）
                std::vector<float> target_joints;
                float target_gripper;
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    target_joints = target_joints_;
                    target_gripper = target_gripper_;
                }

                // 应用低通滤波（保持“展开”到最近参考）
                std::vector<float> filtered_joints = filterJointPositions(target_joints);

                // 读取当前关节（度）作为限幅参考
                std::vector<float> current_joints_copy;
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    current_joints_copy = current_joints_;
                }

                // 限幅：基于最短弧差 + 每周期最大步进
                std::vector<float> clamped_joints = current_joints_copy;
                for (int i = 0; i < num_joints_; ++i) {
                    // 最短路径差值，范围 (-180, 180]，单位：度
                    float delta = std::remainder(filtered_joints[i] - current_joints_copy[i], 360.0f);
                    // 限制单步变化
                    delta = clampf(delta, -max_step_deg, max_step_deg);
                    // 应用限幅
                    clamped_joints[i] = current_joints_copy[i] + delta;
                }

                // 发送限幅后的关节位置
                robot_controller_->setJointPositions(clamped_joints);

                // 发送夹爪命令
                robot_controller_->setGripperPosition(target_gripper, 0.9f);

                // 发送命令并刷新反馈
                if (!robot_controller_->sendCommandAndRefresh()) {
                    RCLCPP_ERROR(this->get_logger(), "Failed to send command");
                }

                // 更新当前关节位置（度）——从驱动读取后做到 [-180,180) 仅用于显示/稳定性
                auto current = normalizeAngles(robot_controller_->getJointPositions());
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    current_joints_ = current;
                }

                // 记录数据（原始 target_joints）
                if (data_logging_enabled_) {
                    logJointData(current, target_joints);
                }

                // 性能统计
                auto loop_end = std::chrono::steady_clock::now();
                auto loop_duration = std::chrono::duration<double>(loop_end - loop_start).count();
                loop_times.push_back(loop_duration * 1000.0);

                if (loop_times.size() > max_samples) {
                    loop_times.pop_front();
                }

                if (loop_end - last_report > std::chrono::seconds(2)) {
                    double avg = 0, maxv = 0;
                    for (double t : loop_times) {
                        avg += t;
                        maxv = std::max(maxv, t);
                    }
                    avg /= loop_times.size();

                    RCLCPP_INFO(this->get_logger(),
                               "Control loop: avg=%.2fms, max=%.2fms, rate=%.1fHz, logged=%zu points",
                               avg, maxv, 1000.0/avg, logged_data_.size());

                    last_report = loop_end;
                }

            } catch (const std::exception& e) {
                RCLCPP_ERROR(this->get_logger(), "Control thread error: %s", e.what());
            }

            // 保持循环频率
            auto loop_end = std::chrono::steady_clock::now();
            auto loop_duration = loop_end - loop_start;
            if (loop_duration < dt) {
                std::this_thread::sleep_for(dt - loop_duration);
            }
        }

        RCLCPP_INFO(this->get_logger(), "Control thread stopped");
    }

    // ========== 成员变量 ==========

    // 配置
    std::string robot_urdf_path_;
    std::string robot_ip_;
    int tcp_port_;
    int udp_port_;

    // 控制参数
    double scale_factor_;
    int ik_rate_hz_;
    int control_rate_hz_;
    int num_joints_;

    // 机械臂控制器
    std::unique_ptr<Gen3RobotController> robot_controller_;

    // TRAC-IK
    std::unique_ptr<TRAC_IK::TRAC_IK> tracik_solver_;
    std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_solver_;
    KDL::Chain kdl_chain_;

    // ROS2 订阅器
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr grip_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr trigger_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr button_a_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr button_b_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr button_x_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr button_y_sub_;

    // TF broadcaster
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    // XR 数据（受 xr_data_mutex_ 保护）
    std::mutex xr_data_mutex_;
    float xr_right_grip_;
    float xr_right_trigger_;
    std::vector<double> xr_controller_pose_;

    // 按钮状态
    bool button_a_state_;
    bool button_b_state_;
    bool button_x_state_;
    bool button_y_state_;
    bool button_a_prev_;
    bool button_b_prev_;
    bool button_x_prev_;
    bool button_y_prev_;

    // 按钮重复触发控制
    std::chrono::steady_clock::time_point button_x_last_trigger_time_;
    std::chrono::steady_clock::time_point button_y_last_trigger_time_;
    double gripper_button_repeat_interval_;  // 重复间隔（秒）

    // 线程控制
    std::atomic<bool> shutdown_requested_;

    // 机械臂状态（受 state_mutex_ 保护）
    std::mutex state_mutex_;
    std::vector<float> target_joints_;   // 目标关节角（°，可超±180 做“展开”）
    std::vector<float> current_joints_;  // 当前关节角（°，normalizeAngles 后）

    float target_gripper_;

    // Frame 状态（受 frame_mutex_ 保护）
    std::mutex frame_mutex_;
    KDL::Frame current_ee_frame_;
    KDL::Frame target_ee_frame_;

    // 控制状态
    std::atomic<bool> is_active_;

    // 夹爪控制
    int gripper_control_mode_;  // 0: trigger, 1: button
    float gripper_step_value_;

    // 参考坐标系
    bool ref_ee_valid_;
    KDL::Frame ref_ee_frame_;
    bool ref_controller_valid_;
    Eigen::Vector3d ref_controller_pos_;
    Eigen::Quaterniond ref_controller_quat_;

    // 坐标变换
    Eigen::Matrix3d R_headset_world_;
    Eigen::Matrix3d R_z_90_cw_;

    // 滤波状态
    std::vector<float> filtered_joint_state_;
    bool filter_initialized_;
    const float filter_alpha_;

    // 数据记录相关
    bool data_logging_enabled_;
    std::string log_file_path_;
    std::chrono::steady_clock::time_point start_time_;
    std::mutex log_data_mutex_;
    std::vector<JointDataPoint> logged_data_;
};

// ========== Main ==========

int main(int argc, char** argv) {
    // 安装信号处理器
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // 初始化 ROS2
    rclcpp::init(argc, argv);

    // 配置
    std::string urdf_path = "/home/ming/xrrobotics_new/XRoboToolkit-Teleop-Sample-Python/assets/arx/Gen/GEN3-7DOF.urdf";
    std::string robot_ip = "192.168.1.10";
    std::string log_file = "joint_tracking_data.csv";

    // 解析命令行参数
    if (argc > 1) {
        robot_ip = argv[1];
    }
    if (argc > 2) {
        urdf_path = argv[2];
    }
    if (argc > 3) {
        log_file = argv[3];
    }

    std::cout << "==================================" << std::endl;
    std::cout << "Gen3 XR Teleoperation ROS2 Node" << std::endl;
    std::cout << "==================================" << std::endl;
    std::cout << "Robot IP: " << robot_ip << std::endl;
    std::cout << "URDF: " << urdf_path << std::endl;
    std::cout << "Log file: " << log_file << std::endl;
    std::cout << std::endl;

    try {
        // 创建节点
        auto node = std::make_shared<Gen3XRTeleopNode>(urdf_path, robot_ip);

        // 初始化
        if (!node->initialize()) {
            RCLCPP_ERROR(rclcpp::get_logger("main"), "Failed to initialize controller");
            return 1;
        }

        // 运行主控制循环
        node->run();

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }

    rclcpp::shutdown();
    std::cout << "Program terminated successfully" << std::endl;
    return 0;
}
