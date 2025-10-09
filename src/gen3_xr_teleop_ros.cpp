/**
 * Gen3 XR Teleoperation ROS2 Node
 * 订阅 XR 数据话题，实时控制 Kinova Gen3 机械臂
 * 
 * 架构：
 * - ROS2 回调：接收 XR 数据并更新共享状态
 * - IK 线程：计算逆运动学，生成目标关节角度
 * - Control 线程：1kHz 控制循环，发送命令到机械臂
 */

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>

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

// 全局关闭标志
std::atomic<bool> g_shutdown_requested(false);

void signal_handler(int sig) {
    std::cout << "\nShutdown signal received" << std::endl;
    g_shutdown_requested = true;
}

/**
 * ROS2 节点：Gen3 XR 遥操作控制器
 */
class Gen3XRTeleopNode : public rclcpp::Node {
public:
    Gen3XRTeleopNode(const std::string& robot_urdf_path,
                     const std::string& robot_ip = "192.168.1.10",
                     int tcp_port = 10000,
                     int udp_port = 10001)
        : Node("gen3_xr_teleop_node"),
          robot_urdf_path_(robot_urdf_path),
          robot_ip_(robot_ip),
          tcp_port_(tcp_port),
          udp_port_(udp_port),
          shutdown_requested_(false),
          num_joints_(7),
          scale_factor_(0.7f),
          ik_rate_hz_(50),
          control_rate_hz_(1000),
          is_active_(false),
          ref_ee_valid_(false),
          ref_controller_valid_(false),
          filter_initialized_(false),
          filter_alpha_(0.01f)
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
        xr_joystick_.resize(2, 0.0);
        
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
        
        joystick_sub_ = this->create_subscription<geometry_msgs::msg::Vector3Stamped>(
            "xr/right_joystick", 10,
            std::bind(&Gen3XRTeleopNode::joystickCallback, this, std::placeholders::_1));
        
        RCLCPP_INFO(this->get_logger(), "Gen3 XR Teleop Node created");
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
    
    void joystickCallback(const geometry_msgs::msg::Vector3Stamped::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(xr_data_mutex_);
        xr_joystick_[0] = msg->vector.x;
        xr_joystick_[1] = msg->vector.y;
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
            
            auto positions = normalizeAngles(robot_controller_->getJointPositions());//要确认拿到的角度是degree的，可以在robot initialize的时候看看输出。
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
    
    float unwrapAngle(float target, float reference) const {
        double unwrapped = static_cast<double>(reference) +
                           std::remainder(static_cast<double>(target) - 
                                        static_cast<double>(reference), 360.0);
        return static_cast<float>(unwrapped);
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
    
    // ========== 线程函数 ==========
    
    void ikThread() {
        RCLCPP_INFO(this->get_logger(), "IK thread started");
        auto dt = std::chrono::duration<double>(1.0 / ik_rate_hz_);
        
        while (!shutdown_requested_ && !g_shutdown_requested) {
            auto loop_start = std::chrono::steady_clock::now();
            
            try {
                // 获取 XR 输入（带锁）
                float grip_value, trigger_value;
                std::vector<double> controller_pose;
                {
                    std::lock_guard<std::mutex> lock(xr_data_mutex_);
                    grip_value = xr_right_grip_;
                    trigger_value = xr_right_trigger_;
                    controller_pose = xr_controller_pose_;
                }
                
                // 更新夹爪目标
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    target_gripper_ = std::max(0.0f, std::min(1.0f, trigger_value));
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
                    // 获取当前关节位置用于 FK
                    KDL::JntArray current_joints_kdl(num_joints_);
                    {
                        std::lock_guard<std::mutex> lock(state_mutex_);
                        for (int i = 0; i < num_joints_; ++i) {
                            current_joints_kdl(i) = current_joints_[i] * M_PI / 180.0;
                        }
                    }
                    
                    // 初始化参考坐标系
                    if (!ref_ee_valid_) {
                        fk_solver_->JntToCart(current_joints_kdl, ref_ee_frame_);
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
                        std::lock_guard<std::mutex> lock(state_mutex_);
                        for (int i = 0; i < num_joints_; ++i) {
                            target_joints_[i] = normalizeAngle(ik_solution(i) * 180.0 / M_PI);
                        }
                    } else {
                        static int fail_count = 0;
                        if (++fail_count % 50 == 0) {
                            RCLCPP_WARN(this->get_logger(), "IK solution not found");
                        }
                    }
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
        
        while (!shutdown_requested_ && !g_shutdown_requested) {
            auto loop_start = std::chrono::steady_clock::now();
            
            try {
                // 获取目标位置
                std::vector<float> target_joints;
                float target_gripper;
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    target_joints = target_joints_;
                    target_gripper = target_gripper_;
                }
                
                // 应用滤波
                std::vector<float> filtered_joints = filterJointPositions(target_joints);
                
                // 发送关节位置
                robot_controller_->setJointPositions(filtered_joints);
                
                // 发送夹爪命令
                robot_controller_->setGripperPosition(target_gripper, 1.0f);
                
                // 发送命令并刷新反馈
                if (!robot_controller_->sendCommandAndRefresh()) {
                    RCLCPP_ERROR(this->get_logger(), "Failed to send command");
                }
                
                // 更新当前关节位置
                auto current = normalizeAngles(robot_controller_->getJointPositions());
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    current_joints_ = current;
                }
                
                // 性能监控
                auto loop_end = std::chrono::steady_clock::now();
                auto loop_duration = std::chrono::duration<double>(loop_end - loop_start).count();
                loop_times.push_back(loop_duration * 1000.0);
                
                if (loop_times.size() > max_samples) {
                    loop_times.pop_front();
                }
                
                // 每 2 秒报告统计信息
                if (loop_end - last_report > std::chrono::seconds(2)) {
                    double avg = 0, max = 0;
                    for (double t : loop_times) {
                        avg += t;
                        max = std::max(max, t);
                    }
                    avg /= loop_times.size();
                    
                    RCLCPP_INFO(this->get_logger(), 
                               "Control loop: avg=%.2fms, max=%.2fms, rate=%.1fHz",
                               avg, max, 1000.0/avg);
                    
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
    rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr joystick_sub_;
    
    // XR 数据（受 xr_data_mutex_ 保护）
    std::mutex xr_data_mutex_;
    float xr_right_grip_;
    float xr_right_trigger_;
    std::vector<double> xr_controller_pose_;
    std::vector<double> xr_joystick_;
    
    // 线程控制
    std::atomic<bool> shutdown_requested_;
    
    // 机械臂状态（受 state_mutex_ 保护）
    std::mutex state_mutex_;
    std::vector<float> target_joints_;
    std::vector<float> current_joints_;
    float target_gripper_;
    
    // 控制状态
    std::atomic<bool> is_active_;
    
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
    
    // 解析命令行参数
    if (argc > 1) {
        robot_ip = argv[1];
    }
    if (argc > 2) {
        urdf_path = argv[2];
    }
    
    std::cout << "==================================" << std::endl;
    std::cout << "Gen3 XR Teleoperation ROS2 Node" << std::endl;
    std::cout << "==================================" << std::endl;
    std::cout << "Robot IP: " << robot_ip << std::endl;
    std::cout << "URDF: " << urdf_path << std::endl;
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