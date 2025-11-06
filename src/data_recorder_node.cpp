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

// 全局关闭标志
std::atomic<bool> g_shutdown_requested{false};

void signalHandler(int signum) {
    RCLCPP_INFO(rclcpp::get_logger("signal_handler"), "Interrupt signal (%d) received. Shutting down...", signum);
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
        
        // 记录启动时间（用于生成时间戳）
        start_time_ = std::chrono::steady_clock::now();
        
        // 初始化关节和夹爪状态向量
        current_joints_.resize(num_joints_, 0.0f);
        target_joints_.resize(num_joints_, 0.0f);
        current_gripper_ = 0.0f;
        target_gripper_ = 0.0f;
        
        // ========== 创建订阅器 ==========
        
        // 相机 1：/camera/color/image_raw
        color_sub_cam1_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/color/image_raw", 
            rclcpp::SensorDataQoS(),
            [this](const sensor_msgs::msg::Image::SharedPtr msg) {
                std::lock_guard<std::mutex> lk(image_cam1_mutex_);
                last_image_cam1_ = msg;
            });
        
        // 相机 2：/camera/camera/color/image_raw
        color_sub_cam2_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/camera/color/image_raw", 
            rclcpp::SensorDataQoS(),
            [this](const sensor_msgs::msg::Image::SharedPtr msg) {
                std::lock_guard<std::mutex> lk(image_cam2_mutex_);
                last_image_cam2_ = msg;
            });
        
        // DIGIT 1：D20583
        digit_D20583_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/digit/D20583/image_raw", 
            rclcpp::SensorDataQoS(),
            [this](const sensor_msgs::msg::Image::SharedPtr msg) {
                std::lock_guard<std::mutex> lk(digit_D20583_mutex_);
                last_digit_D20583_image_ = msg;
            });
        
        // DIGIT 2：D20584
        digit_D20584_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/digit/D20584/image_raw", 
            rclcpp::SensorDataQoS(),
            [this](const sensor_msgs::msg::Image::SharedPtr msg) {
                std::lock_guard<std::mutex> lk(digit_D20584_mutex_);
                last_digit_D20584_image_ = msg;
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
        
        // ========== 启动数据记录线程 ==========
        logger_thread_ = std::thread(&DataRecorderNode::dataLoggerThread, this);
        
        RCLCPP_INFO(this->get_logger(), "Data Recorder Node initialized successfully!");
        RCLCPP_INFO(this->get_logger(), "Press Left Trigger to start/stop recording.");
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
        
        RCLCPP_INFO(this->get_logger(), "Data Recorder Node shut down complete. Total episodes recorded: %d", episode_count_);
    }

private:
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
    }
    
    void leftTriggerCallback(const std_msgs::msg::Float32::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(xr_data_mutex_);
        float prev_value = xr_left_trigger_;
        xr_left_trigger_ = msg->data;
        
        // 处理扳机状态切换
        bool prev_trigger_trig = trigger_trig_;
        if (!trigger_trig_ && xr_left_trigger_ > 0.9f) {
            // 扳机按下
            trigger_trig_ = true;
        } else if (trigger_trig_ && xr_left_trigger_ < 0.5f) {
            // 扳机松开
            trigger_trig_ = false;
        }
        
        // 检测从false变为true（按下瞬间），切换log_enable_
        if (!prev_trigger_trig && trigger_trig_) {
            log_enable_ = !log_enable_;
            RCLCPP_INFO(this->get_logger(), "Left trigger pressed: Recording %s", 
                       log_enable_ ? "STARTED" : "STOPPED");
        }
    }
    
    // ========== 数据记录线程 ==========
    
    void dataLoggerThread()
    {
        logger_running_ = true;
        RCLCPP_INFO(this->get_logger(), "Data logger thread started, waiting for trigger...");
        
        // 外层循环：等待log_enable_变为true
        while (!shutdown_requested_ && !g_shutdown_requested && rclcpp::ok()) {
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
            RCLCPP_INFO(this->get_logger(), "Episode %d: Recording STARTED", episode_count_);
            RCLCPP_INFO(this->get_logger(), "===================================================");
            
            // 1) 生成运行目录（带时间戳）
            const auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            std::tm tm_buf{};
            localtime_r(&t, &tm_buf);
            char stamp[32];
            std::snprintf(stamp, sizeof(stamp), "%04d-%02d-%02d_%02d-%02d-%02d",
                        tm_buf.tm_year + 1900, tm_buf.tm_mon + 1, tm_buf.tm_mday,
                        tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec);

            run_dir_ = "data_logs/episode_" + std::to_string(episode_count_) + "_" + std::string(stamp);

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
                
                // 使用\r实现同行刷新显示当前帧数（不会刷屏）
                std::cout << "\rEpisode " << episode_count_ << " | Frame: " << data_index << std::flush;
                
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
            RCLCPP_INFO(this->get_logger(), "Episode %d: Recording STOPPED", episode_count_);
            RCLCPP_INFO(this->get_logger(), "Total frames: %d", data_index);
            RCLCPP_INFO(this->get_logger(), "Saved to: %s", run_dir_.c_str());
            RCLCPP_INFO(this->get_logger(), "===================================================");
        }

        logger_running_ = false;
        RCLCPP_INFO(this->get_logger(), "Data logger thread stopped");
    }
    
    // ========== 成员变量 ==========
    
    // 参数
    int num_joints_;
    
    // 状态变量
    std::atomic<bool> log_enable_;
    std::atomic<bool> trigger_trig_;
    std::atomic<float> xr_left_trigger_;
    std::atomic<int> episode_count_;
    
    // 关节和夹爪状态
    std::vector<float> current_joints_;
    std::vector<float> target_joints_;
    float current_gripper_;
    float target_gripper_;
    
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