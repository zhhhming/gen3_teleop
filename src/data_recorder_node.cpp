#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float32.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <fstream>
#include <filesystem>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <string>
#include <ctime>
#include <iomanip>
#include <cmath>

// ANSI颜色代码
#define COLOR_RESET   "\033[0m"
#define COLOR_GREEN   "\033[92m"
#define COLOR_BLUE    "\033[31m"
#define COLOR_CYAN    "\033[96m"

// 全局关闭标志
std::atomic<bool> g_shutdown_requested{false};

void signalHandler(int signum) {

    g_shutdown_requested = true;
}

class DataRecorderNode : public rclcpp::Node
{
public:
    DataRecorderNode() : Node("data_recorder_node")
    {
        RCLCPP_INFO(this->get_logger(), "Initializing Data Recorder Node...");
        
        // 初始化参数
        num_joints_ = 7;
        log_enable_ = false;
        trigger_trig_ = false;
        xr_left_trigger_ = 0.0f;
        episode_count_ = 0;
        
        // Reference初始化标志
        reference_initial_enable_ = false;
        reference_initialized_ = false;
        enable_unwrap_ = true;  // 默认启用unwrap功能
        
        // 记录启动时间（用于生成时间戳）
        start_time_ = std::chrono::steady_clock::now();
        
        // 初始化关节和夹爪状态向量
        current_joints_.resize(num_joints_, 0.0f);
        target_joints_.resize(num_joints_, 0.0f);
        reference_joints_.resize(num_joints_, 0.0f);
        current_gripper_ = 0.0f;
        target_gripper_ = 0.0f;
        
        // 初始化数据接收标志
        received_cam1_ = false;
        received_cam2_ = false;
        received_digit_D20583_ = false;
        received_digit_D20584_ = false;
        received_current_joints_ = false;
        received_target_joints_ = false;
        received_left_trigger_ = false;
        
        // ========== 创建订阅器 ==========
        
        // 相机 1：/camera/color/image_raw
        color_sub_cam1_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/color/image_raw", 
            rclcpp::SensorDataQoS(),
            [this](const sensor_msgs::msg::Image::SharedPtr msg) {
                std::lock_guard<std::mutex> lk(image_cam1_mutex_);
                last_image_cam1_ = msg;
                received_cam1_ = true;
            });
        
        // 相机 2：/camera/camera/color/image_raw
        color_sub_cam2_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/camera/color/image_raw", 
            rclcpp::SensorDataQoS(),
            [this](const sensor_msgs::msg::Image::SharedPtr msg) {
                std::lock_guard<std::mutex> lk(image_cam2_mutex_);
                last_image_cam2_ = msg;
                received_cam2_ = true;
            });
        
        // DIGIT 1：D20583
        digit_D20583_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/digit/D20583/image_raw", 
            rclcpp::SensorDataQoS(),
            [this](const sensor_msgs::msg::Image::SharedPtr msg) {
                std::lock_guard<std::mutex> lk(digit_D20583_mutex_);
                last_digit_D20583_image_ = msg;
                received_digit_D20583_ = true;
            });
        
        // DIGIT 2：D20584
        digit_D20584_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/digit/D20584/image_raw", 
            rclcpp::SensorDataQoS(),
            [this](const sensor_msgs::msg::Image::SharedPtr msg) {
                std::lock_guard<std::mutex> lk(digit_D20584_mutex_);
                last_digit_D20584_image_ = msg;
                received_digit_D20584_ = true;
            });
        
        // Current Joint States 订阅
        current_joint_states_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/gen3/current_joint_states",
            10,
            std::bind(&DataRecorderNode::currentJointStatesCallback, this, std::placeholders::_1));
        
        // Target Joint States 订阅
        target_joint_states_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/gen3/target_joint_states",
            10,
            std::bind(&DataRecorderNode::targetJointStatesCallback, this, std::placeholders::_1));
        
        // Left Trigger 订阅
        left_trigger_sub_ = this->create_subscription<std_msgs::msg::Float32>(
            "xr/left_trigger",
            10,
            std::bind(&DataRecorderNode::leftTriggerCallback, this, std::placeholders::_1));
        
        RCLCPP_INFO(this->get_logger(), "All subscriptions created.");
        RCLCPP_INFO(this->get_logger(), "Node construction complete. Call waitForData() to initialize.");
    }
    
    // ========== 等待所有订阅都收到数据的初始化函数 ==========
    void waitForData()
    {
        using namespace std::chrono_literals;
        
        RCLCPP_INFO(this->get_logger(), "Waiting for all subscriptions to receive data...");
        
        auto start_wait = std::chrono::steady_clock::now();
        const auto timeout = std::chrono::seconds(30);  // 30秒超时
        rclcpp::Rate rate(10);  // 10 Hz
        
        while (rclcpp::ok() && !g_shutdown_requested) {
            // 使用 rclcpp::spin_some，传递 node_base_interface
            rclcpp::spin_some(this->get_node_base_interface());
            
            if (received_cam1_ && received_cam2_ && 
                received_digit_D20583_ && received_digit_D20584_ &&
                received_current_joints_ && received_target_joints_ &&
                received_left_trigger_) {
                RCLCPP_INFO(this->get_logger(), "All subscriptions have received data!");
                break;
            }
            
            // 显示等待状态（使用 chrono 类型作为参数）
            if (!received_cam1_) {
                RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Waiting for CAM1...");
            }
            if (!received_cam2_) {
                RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Waiting for CAM2...");
            }
            if (!received_digit_D20583_) {
                RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Waiting for DIGIT D20583...");
            }
            if (!received_digit_D20584_) {
                RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Waiting for DIGIT D20584...");
            }
            if (!received_current_joints_) {
                RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Waiting for current joint states...");
            }
            if (!received_target_joints_) {
                RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Waiting for target joint states...");
            }
            if (!received_left_trigger_) {
                RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Waiting for left trigger...");
            }
            
            // 检查超时
            if (std::chrono::steady_clock::now() - start_wait > timeout) {
                RCLCPP_WARN(this->get_logger(), "Timeout waiting for all subscriptions. Starting anyway...");
                break;
            }
            
            rate.sleep();
        }
        
        // ========== 启动数据记录线程 ==========
        logger_thread_ = std::thread(&DataRecorderNode::dataLoggerThread, this);
        
        RCLCPP_INFO(this->get_logger(), "Data Recorder Node initialized successfully!");
        RCLCPP_INFO(this->get_logger(), "Press Left Trigger FIRST TIME to initialize reference angles.");
        RCLCPP_INFO(this->get_logger(), "Then press Left Trigger to start/stop recording.");
        if (enable_unwrap_) {
            RCLCPP_INFO(this->get_logger(), "Unwrap function is ENABLED.");
        } else {
            RCLCPP_INFO(this->get_logger(), "Unwrap function is DISABLED.");
        }
    }
    
    ~DataRecorderNode()
    {
        RCLCPP_INFO(this->get_logger(), "Shutting down Data Recorder Node...");
        shutdown_requested_ = true;
        
        if (logger_thread_.joinable()) {
            logger_thread_.join();
        }
        
        if (csv_.is_open()) {
            csv_.flush();
            csv_.close();
        }
        
        RCLCPP_INFO(this->get_logger(), "Data Recorder Node shut down complete. Total episodes recorded: %d", episode_count_.load());
    }

private:
    // ========== Unwrap函数 ==========
    float unwrap(float target_angle, float reference_angle)
    {
        float diff = target_angle - reference_angle;
        while (diff > M_PI) diff -= 2.0f * M_PI;
        while (diff < -M_PI) diff += 2.0f * M_PI;
        return reference_angle + diff;
    }
    
    // ========== 回调函数 ==========
    
    void currentJointStatesCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        
        // 提取7个关节位置（rad）
        for (int i = 0; i < num_joints_ && i < static_cast<int>(msg->position.size()); ++i) {
            current_joints_[i] = static_cast<float>(msg->position[i]);
        }
        
        // 提取gripper位置（0-1）
        if (msg->position.size() > num_joints_) {
            current_gripper_ = static_cast<float>(msg->position[num_joints_]);
        }
        
        received_current_joints_ = true;
    }
    
    void targetJointStatesCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        
        // 提取7个关节位置（rad）
        for (int i = 0; i < num_joints_ && i < static_cast<int>(msg->position.size()); ++i) {
            target_joints_[i] = static_cast<float>(msg->position[i]);
        }
        
        // 提取gripper位置（0-1）
        if (msg->position.size() > num_joints_) {
            target_gripper_ = static_cast<float>(msg->position[num_joints_]);
        }
        
        received_target_joints_ = true;
    }
    
    void leftTriggerCallback(const std_msgs::msg::Float32::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(xr_data_mutex_);
        float prev_value = xr_left_trigger_;
        xr_left_trigger_ = msg->data;
        
        received_left_trigger_ = true;
        
        // 处理扳机状态切换
        bool prev_trigger_trig = trigger_trig_;
        if (!trigger_trig_ && xr_left_trigger_ > 0.9f) {
            // 扳机按下
            trigger_trig_ = true;
        } else if (trigger_trig_ && xr_left_trigger_ < 0.5f) {
            // 扳机松开
            trigger_trig_ = false;
        }
        
        // 检测从false变为true（按下瞬间）
        if (!prev_trigger_trig && trigger_trig_) {
            // 如果reference还没有初始化，则初始化reference
            if (!reference_initial_enable_) {
                reference_initial_enable_ = true;
                RCLCPP_INFO(this->get_logger(), "Reference initialization triggered. Will use current joint angles as reference.");
            } else {
                // 已经初始化，可以切换log_enable_
                log_enable_ = !log_enable_;
            }
        }
    }
    
    // ========== 数据记录线程 ==========
    
    void dataLoggerThread()
    {
        logger_running_ = true;
        RCLCPP_INFO(this->get_logger(), "Data logger thread started, waiting for reference initialization...");
        
        // 外层循环：等待log_enable_变为true
        while (!shutdown_requested_ && !g_shutdown_requested && rclcpp::ok()) {
            // 首先检查reference是否已经初始化
            if (!reference_initial_enable_) {
                // reference还未初始化，继续等待
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }
            
            // reference_initial_enable_为true，但reference_initialized_还未真正初始化
            if (!reference_initialized_) {
                std::lock_guard<std::mutex> lock(state_mutex_);
                // 使用当前的current_joints_作为初始reference
                reference_joints_ = current_joints_;
                reference_initialized_ = true;
                
                RCLCPP_INFO(this->get_logger(), "Reference angles initialized:");
                for (int i = 0; i < num_joints_; ++i) {
                    RCLCPP_INFO(this->get_logger(), "  Joint %d: %.4f rad", i, reference_joints_[i]);
                }
                RCLCPP_INFO(this->get_logger(), "Ready to record. Press Left Trigger to start/stop recording.");
            }
            
            // 等待log_enable_变为true
            while (!log_enable_ && !shutdown_requested_ && !g_shutdown_requested && rclcpp::ok()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            
            if (shutdown_requested_ || g_shutdown_requested || !rclcpp::ok()) {
                break;
            }
            
            // 新的episode开始
            episode_count_++;
            
            // 开始新的记录会话
            RCLCPP_INFO(this->get_logger(), "===================================================");
            // 使用彩色输出
            std::cout << "Episode " << COLOR_GREEN << episode_count_.load() << COLOR_RESET 
                      << ": Recording " << COLOR_GREEN << "STARTED" << COLOR_RESET << std::endl;
            RCLCPP_INFO(this->get_logger(), "===================================================");
            
            // 1) 生成运行目录（只带时间戳，不带episode_count）
            const auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            std::tm tm_buf{};
            localtime_r(&t, &tm_buf);
            char stamp[32];
            std::snprintf(stamp, sizeof(stamp), "%04d-%02d-%02d_%02d-%02d-%02d",
                        tm_buf.tm_year + 1900, tm_buf.tm_mon + 1, tm_buf.tm_mday,
                        tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec);

            run_dir_ = "data_logs/episode_" + std::string(stamp);

            // 两路相机
            images_cam1_dir_ = run_dir_ + "/images_cam1";
            images_cam2_dir_ = run_dir_ + "/images_cam2";
            std::filesystem::create_directories(images_cam1_dir_);
            std::filesystem::create_directories(images_cam2_dir_);

            // 两路 DIGIT
            digit_images_D20583_dir_ = run_dir_ + "/digit_images/D20583";
            digit_images_D20584_dir_ = run_dir_ + "/digit_images/D20584";
            std::filesystem::create_directories(run_dir_ + "/digit_images");
            std::filesystem::create_directories(digit_images_D20583_dir_);
            std::filesystem::create_directories(digit_images_D20584_dir_);

            // 2) 打开 CSV（带表头）
            const std::string csv_path = run_dir_ + "/joint_gripper_log.csv";
            csv_.open(csv_path, std::ios::out);
            if (!csv_.is_open()) {
                RCLCPP_ERROR(this->get_logger(), "Failed to open CSV: %s", csv_path.c_str());
                continue;  // 跳过本次记录
            }

            // CSV表头：index, 7个current joints (rad), current gripper, 7个target joints (rad), target gripper, 4个图像路径
            csv_ << "index";
            for (int i = 0; i < num_joints_; ++i) {
                csv_ << ",current_joint" << i << "_rad";
            }
            csv_ << ",current_gripper_0to1";
            for (int i = 0; i < num_joints_; ++i) {
                csv_ << ",target_joint" << i << "_rad";
            }
            csv_ << ",target_gripper_0to1";
            csv_ << ",cam1_image_file,cam2_image_file,digit_D20583_file,digit_D20584_file\n";
            csv_.flush();

            // 初始化索引
            int data_index = 0;
            
            // 重置上一帧文件名
            last_cam1_file_rel_.clear();
            last_cam2_file_rel_.clear();
            last_digit_20583_file_rel_.clear();
            last_digit_20584_file_rel_.clear();

            // 3) 15Hz 精准循环
            const auto period = std::chrono::microseconds(66667);  // 15 Hz
            auto next_tick = std::chrono::steady_clock::now();
            auto last_flush = next_tick;
            
            RCLCPP_INFO(this->get_logger(), "Recording to: %s", run_dir_.c_str());

            // 内层循环：记录数据，直到log_enable_变为false
            while (log_enable_ && !shutdown_requested_ && !g_shutdown_requested && rclcpp::ok()) {
                next_tick += period;

                // 生成统一时间戳（用于文件名）
                const auto now = std::chrono::steady_clock::now();
                const uint64_t timestamp_us =
                    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(now - start_time_).count());

                // 拷贝关节与夹爪状态（current + target）
                std::vector<float> joints_current_copy, joints_target_copy;
                float grip_current_copy = 0.0f, grip_target_copy = 0.0f;
                {
                    std::lock_guard<std::mutex> lk(state_mutex_);
                    joints_current_copy = current_joints_;
                    joints_target_copy  = target_joints_;
                    grip_current_copy   = current_gripper_;
                    grip_target_copy    = target_gripper_;
                    
                    // 如果启用unwrap，对target_joints进行unwrap处理
                    if (enable_unwrap_) {
                        for (int i = 0; i < num_joints_; ++i) {
                            joints_current_copy[i] = unwrap(joints_current_copy[i], reference_joints_[i]);
                            reference_joints_[i] = joints_current_copy[i];
                            joints_target_copy[i] = unwrap(joints_target_copy[i], reference_joints_[i]);
                            // 更新reference为unwrapped后的值
                            
                        }
                    }
                }

                // ========== 统一时间戳的四路图像保存 ==========
                // 规则：若该路本周期没有可用图像，则复用上一帧文件名（保持严格对齐）
                std::string cam1_file_rel = last_cam1_file_rel_;
                std::string cam2_file_rel = last_cam2_file_rel_;
                std::string digit_20583_file_rel = last_digit_20583_file_rel_;
                std::string digit_20584_file_rel = last_digit_20584_file_rel_;

                // CAM1
                sensor_msgs::msg::Image::SharedPtr image_copy_cam1;
                {
                    std::lock_guard<std::mutex> lk(image_cam1_mutex_);
                    if (last_image_cam1_) {
                        image_copy_cam1 = last_image_cam1_;
                    }
                }
                if (image_copy_cam1) {
                    const std::string rel = "images_cam1/color_" + std::to_string(timestamp_us) + ".png";
                    const std::string img_path = run_dir_ + "/" + rel;
                    try {
                        // 转换为 OpenCV 图像
                        auto cv_ptr = cv_bridge::toCvCopy(image_copy_cam1, image_copy_cam1->encoding);
                        // 转换为BGR并调整大小
                        cv::Mat bgr;
                        if (image_copy_cam1->encoding == sensor_msgs::image_encodings::RGB8) {
                            cv::cvtColor(cv_ptr->image, bgr, cv::COLOR_RGB2BGR);
                        } else if (image_copy_cam1->encoding == sensor_msgs::image_encodings::BGR8) {
                            bgr = cv_ptr->image;
                        } else {
                            bgr = cv_ptr->image;  // 默认情况
                        }
                        cv::Mat resized;
                        cv::resize(bgr, resized, cv::Size(224, 224), 0, 0, cv::INTER_AREA);
                        
                        // 保存图像
                        if (cv::imwrite(img_path, resized)) {
                            cam1_file_rel = rel;
                            last_cam1_file_rel_ = rel;
                        } else {
                            RCLCPP_WARN(this->get_logger(), "CAM1 failed to write image to: %s", img_path.c_str());
                        }
                    } catch (const cv_bridge::Exception& e) {
                        RCLCPP_WARN(this->get_logger(), "CAM1 cv_bridge error: %s", e.what());
                    } catch (const std::exception& e) {
                        RCLCPP_WARN(this->get_logger(), "CAM1 write error: %s", e.what());
                    }
                }
                
                // CAM2
                sensor_msgs::msg::Image::SharedPtr image_copy_cam2;
                {
                    std::lock_guard<std::mutex> lk(image_cam2_mutex_);
                    if (last_image_cam2_) {
                        image_copy_cam2 = last_image_cam2_;
                    }
                }
                if (image_copy_cam2) {
                    const std::string rel = "images_cam2/color_" + std::to_string(timestamp_us) + ".png";
                    const std::string img_path = run_dir_ + "/" + rel;
                    try {
                        auto cv_ptr = cv_bridge::toCvCopy(image_copy_cam2, image_copy_cam2->encoding);
                        cv::Mat bgr;
                        if (image_copy_cam2->encoding == sensor_msgs::image_encodings::RGB8) {
                            cv::cvtColor(cv_ptr->image, bgr, cv::COLOR_RGB2BGR);
                        } else if (image_copy_cam2->encoding == sensor_msgs::image_encodings::BGR8) {
                            bgr = cv_ptr->image;
                        } else {
                            bgr = cv_ptr->image;
                        }
                        cv::Mat resized;
                        cv::resize(bgr, resized, cv::Size(224, 224), 0, 0, cv::INTER_AREA);

                        if (cv::imwrite(img_path, resized)) {
                            cam2_file_rel = rel;
                            last_cam2_file_rel_ = rel;
                        }
                    } catch (const std::exception& e) {
                        RCLCPP_WARN(this->get_logger(), "CAM2 write error: %s", e.what());
                    }
                }

                // DIGIT D20583
                sensor_msgs::msg::Image::SharedPtr digit_D20583_copy;
                {
                    std::lock_guard<std::mutex> lk(digit_D20583_mutex_);
                    digit_D20583_copy = last_digit_D20583_image_;
                }
                if (digit_D20583_copy) {
                    const std::string rel = "digit_images/D20583/digit_" + std::to_string(timestamp_us) + ".png";
                    const std::string img_path = run_dir_ + "/" + rel;
                    try {
                        auto cv_ptr = cv_bridge::toCvCopy(digit_D20583_copy, digit_D20583_copy->encoding);
                        
                        // 转换为 BGR（PNG 保存需要 BGR 格式）
                        cv::Mat bgr;
                        if (digit_D20583_copy->encoding == sensor_msgs::image_encodings::RGB8) {
                            cv::cvtColor(cv_ptr->image, bgr, cv::COLOR_RGB2BGR);
                        } else if (digit_D20583_copy->encoding == sensor_msgs::image_encodings::BGR8) {
                            bgr = cv_ptr->image;
                        } else if (digit_D20583_copy->encoding == sensor_msgs::image_encodings::MONO8) {
                            bgr = cv_ptr->image;  // 灰度图直接用
                        } else {
                            bgr = cv_ptr->image;  // 其他格式尝试直接保存
                        }
                        
                        if (cv::imwrite(img_path, bgr)) {
                            digit_20583_file_rel = rel;
                            last_digit_20583_file_rel_ = rel;
                        }
                    } catch (const std::exception& e) {
                        RCLCPP_WARN(this->get_logger(), "DIGIT D20583 write error: %s", e.what());
                    }
                }

                // DIGIT D20584
                sensor_msgs::msg::Image::SharedPtr digit_D20584_copy;
                {
                    std::lock_guard<std::mutex> lk(digit_D20584_mutex_);
                    digit_D20584_copy = last_digit_D20584_image_;
                }
                if (digit_D20584_copy) {
                    const std::string rel = "digit_images/D20584/digit_" + std::to_string(timestamp_us) + ".png";
                    const std::string img_path = run_dir_ + "/" + rel;
                    try {
                        auto cv_ptr = cv_bridge::toCvCopy(digit_D20584_copy, digit_D20584_copy->encoding);
                        
                        // 转换为 BGR（PNG 保存需要 BGR 格式）
                        cv::Mat bgr;
                        if (digit_D20584_copy->encoding == sensor_msgs::image_encodings::RGB8) {
                            cv::cvtColor(cv_ptr->image, bgr, cv::COLOR_RGB2BGR);
                        } else if (digit_D20584_copy->encoding == sensor_msgs::image_encodings::BGR8) {
                            bgr = cv_ptr->image;
                        } else if (digit_D20584_copy->encoding == sensor_msgs::image_encodings::MONO8) {
                            bgr = cv_ptr->image;  // 灰度图直接用
                        } else {
                            bgr = cv_ptr->image;  // 其他格式尝试直接保存
                        }
                        
                        if (cv::imwrite(img_path, bgr)) {
                            digit_20584_file_rel = rel;
                            last_digit_20584_file_rel_ = rel;
                        }
                    } catch (const std::exception& e) {
                        RCLCPP_WARN(this->get_logger(), "DIGIT D20584 write error: %s", e.what());
                    }
                }

                // ========== 写入 CSV ==========
                csv_ << data_index;
                for (float rad : joints_current_copy) csv_ << "," << rad;
                csv_ << "," << grip_current_copy;
                for (float rad : joints_target_copy)  csv_ << "," << rad;
                csv_ << "," << grip_target_copy;
                csv_ << "," << cam1_file_rel
                     << "," << cam2_file_rel
                     << "," << digit_20583_file_rel
                     << "," << digit_20584_file_rel
                     << "\n";

                // 每秒 flush，防止异常掉电丢数据
                if (std::chrono::steady_clock::now() - last_flush > std::chrono::seconds(1)) {
                    csv_.flush();
                    last_flush = std::chrono::steady_clock::now();
                }
                
                // 使用\r实现同行刷新显示当前帧数（不会刷屏），Frame数字用青色
                std::cout << "\rEpisode " << COLOR_GREEN << episode_count_.load() << COLOR_RESET << " | Frame: " 
                          << COLOR_CYAN << data_index << COLOR_RESET << std::flush;
                
                data_index++;

                // 精准定时：若提前则 sleep_until，若滞后则立即进入下一周期
                const auto now2 = std::chrono::steady_clock::now();
                if (now2 < next_tick) {
                    std::this_thread::sleep_until(next_tick);
                }
            }
            
            // 记录结束，关闭CSV
            csv_.flush();
            csv_.close();
            
            // 换行，避免与下一条日志混在一起
            std::cout << std::endl;
            
            RCLCPP_INFO(this->get_logger(), "===================================================");
            // 使用彩色输出 - STOPPED用蓝色
            std::cout << "Episode " << episode_count_.load() << ": Recording " 
                      << COLOR_BLUE << "STOPPED" << COLOR_RESET << std::endl;
            RCLCPP_INFO(this->get_logger(), "Total frames: %d", data_index);
            RCLCPP_INFO(this->get_logger(), "Saved to: %s", run_dir_.c_str());
            RCLCPP_INFO(this->get_logger(), "===================================================");
        }
        if (g_shutdown_requested) {
            RCLCPP_INFO(this->get_logger(), "Interrupt signal received. Shutting down...");
        }

        logger_running_ = false;
        RCLCPP_INFO(this->get_logger(), "Data logger thread stopped");
    }
    
    // ========== 成员变量 ==========
    
    // 参数
    int num_joints_;
    bool enable_unwrap_;  // unwrap功能开关
    
    // 状态变量
    std::atomic<bool> log_enable_;
    std::atomic<bool> trigger_trig_;
    std::atomic<float> xr_left_trigger_;
    std::atomic<int> episode_count_{0};
    
    // Reference初始化相关
    std::atomic<bool> reference_initial_enable_;  // 是否触发了reference初始化
    std::atomic<bool> reference_initialized_;     // reference是否已经真正初始化
    std::vector<float> reference_joints_;         // reference角度
    
    // 关节和夹爪状态
    std::vector<float> current_joints_;
    std::vector<float> target_joints_;
    float current_gripper_;
    float target_gripper_;
    
    // 数据接收标志
    std::atomic<bool> received_cam1_;
    std::atomic<bool> received_cam2_;
    std::atomic<bool> received_digit_D20583_;
    std::atomic<bool> received_digit_D20584_;
    std::atomic<bool> received_current_joints_;
    std::atomic<bool> received_target_joints_;
    std::atomic<bool> received_left_trigger_;
    
    // 图像缓存
    sensor_msgs::msg::Image::SharedPtr last_image_cam1_;
    sensor_msgs::msg::Image::SharedPtr last_image_cam2_;
    sensor_msgs::msg::Image::SharedPtr last_digit_D20583_image_;
    sensor_msgs::msg::Image::SharedPtr last_digit_D20584_image_;
    
    // 上一帧文件名（用于复用）
    std::string last_cam1_file_rel_;
    std::string last_cam2_file_rel_;
    std::string last_digit_20583_file_rel_;
    std::string last_digit_20584_file_rel_;
    
    // 目录路径
    std::string run_dir_;
    std::string images_cam1_dir_;
    std::string images_cam2_dir_;
    std::string digit_images_D20583_dir_;
    std::string digit_images_D20584_dir_;
    
    // CSV文件
    std::ofstream csv_;
    
    // 线程和同步
    std::thread logger_thread_;
    std::atomic<bool> logger_running_{false};
    std::atomic<bool> shutdown_requested_{false};
    
    std::mutex state_mutex_;
    std::mutex image_cam1_mutex_;
    std::mutex image_cam2_mutex_;
    std::mutex digit_D20583_mutex_;
    std::mutex digit_D20584_mutex_;
    std::mutex xr_data_mutex_;
    
    // 时间基准
    std::chrono::steady_clock::time_point start_time_;
    
    // 订阅器
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr color_sub_cam1_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr color_sub_cam2_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr digit_D20583_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr digit_D20584_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr current_joint_states_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr target_joint_states_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr left_trigger_sub_;
};

int main(int argc, char** argv)
{
    // 注册信号处理器
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<DataRecorderNode>();
    
    // 等待所有订阅接收数据并启动记录线程
    node->waitForData();
    
    RCLCPP_INFO(node->get_logger(), "Data Recorder Node is running...");
    RCLCPP_INFO(node->get_logger(), "Press Ctrl+C to exit.");
    
    // 持续运行直到收到关闭信号
    while (rclcpp::ok() && !g_shutdown_requested) {
        rclcpp::spin_some(node);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    RCLCPP_INFO(node->get_logger(), "Shutting down gracefully...");
    
    rclcpp::shutdown();
    
    return 0;
}