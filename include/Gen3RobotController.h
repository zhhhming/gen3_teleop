/**
 * Gen3 Robot Controller
 * High-performance C++ controller for Kinova Gen3 robot arm
 * Designed for 1kHz real-time control loops
 */

#ifndef GEN3_ROBOT_CONTROLLER_H
#define GEN3_ROBOT_CONTROLLER_H

#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <cmath>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>


// Kortex API includes

#include <BaseClientRpc.h>
#include <BaseCyclicClientRpc.h>
#include <DeviceConfigClientRpc.h>
#include <ActuatorConfigClientRpc.h>
#include <RouterClient.h>
#include <TransportClientTcp.h>
#include <TransportClientUdp.h>
#include <SessionManager.h>

// Kortex messages
#include <Session.pb.h>
#include <Base.pb.h>
#include <BaseCyclic.pb.h>
#include <ActuatorConfig.pb.h>
#include <Common.pb.h>

namespace k_api = Kinova::Api;

class Gen3RobotController {
public:
    // Constructor with connection parameters
    Gen3RobotController(
        const std::string& ip_address = "192.168.1.10",
        int tcp_port = 10000,
        int udp_port = 10001,
        const std::string& username = "admin",
        const std::string& password = "admin"
    );

    // Destructor - ensures clean shutdown
    ~Gen3RobotController();

    // Initialize the robot controller
    bool initialize();

    // Clear robot faults - should be called before entering low-level mode
    bool clearFaults();

    // Enter/Exit low-level control mode
    bool enterLowLevelMode();
    bool exitLowLevelMode();

    // Main control functions for 1kHz loop
    // Set actuator positions (joint control)
    void setJointPositions(const std::vector<float>& positions);
    
    // Set gripper position with velocity blending
    void setGripperPosition(float position, float speed_ratio = 1.0f);
    
    // Send command and refresh feedback - CALL ONCE PER CONTROL LOOP
    bool sendCommandAndRefresh();

    // Individual control functions (alternative usage)
    bool sendJointPositionsOnly(const std::vector<float>& positions);
    bool sendGripperPositionOnly(float position, float speed_ratio = 1.0f);
    bool refreshOnly(); // Just get feedback without sending commands

    // Getter functions for robot state
    std::vector<float> getJointPositions() const { return joint_positions_; }
    std::vector<float> getJointVelocities() const { return joint_velocities_; }
    float getGripperPosition() const { return gripper_position_; }
    float getGripperVelocity() const { return gripper_velocity_; }
    
    // Get end-effector pose [x, y, z, theta_x, theta_y, theta_z]
    std::vector<float> getEndEffectorPose() const { return end_effector_pose_; }
    
    // Get actuator count
    int getActuatorCount() const { return actuator_count_; }
    
    // Check if in low-level mode
    bool isInLowLevelMode() const { return in_low_level_mode_; }

    // Stop robot motion
    void stopRobot();

    // Clean shutdown
    void shutdown();

private:
    // Connection parameters
    std::string ip_address_;
    int tcp_port_;
    int udp_port_;
    std::string username_;
    std::string password_;

    // Transport and router objects
    k_api::TransportClientTcp* tcp_transport_;
    k_api::TransportClientUdp* udp_transport_;
    k_api::RouterClient* tcp_router_;
    k_api::RouterClient* udp_router_;
    
    // Session managers
    k_api::SessionManager* tcp_session_manager_;
    k_api::SessionManager* udp_session_manager_;

    // Service clients
    k_api::Base::BaseClient* base_client_;
    k_api::BaseCyclic::BaseCyclicClient* base_cyclic_client_;
    k_api::ActuatorConfig::ActuatorConfigClient* actuator_config_client_;

    // Command and feedback
    k_api::BaseCyclic::Command base_command_;
    k_api::BaseCyclic::Feedback base_feedback_;
    
    // Servoing mode management
    k_api::Base::ServoingModeInformation previous_servoing_mode_;
    std::atomic<bool> in_low_level_mode_;

    // Robot state
    int actuator_count_;
    std::vector<float> joint_positions_;
    std::vector<float> joint_velocities_;
    float gripper_position_;
    float gripper_velocity_;
    std::vector<float> end_effector_pose_;

    // Gripper control parameters
    float gripper_target_position_;
    float gripper_last_velocity_;
    float gripper_velocity_blend_alpha_;
    
    // Frame ID for command synchronization
    uint32_t frame_id_;

    // Internal helper functions
    bool createSessions();
    void closeSessions();
    bool initializeBaseCommand();
    void updateStateFromFeedback();
    float calculateGripperVelocity(float target, float current, float max_speed_ratio);
    
    // Error callback
    static void errorCallback(k_api::KError err);
};

#endif // GEN3_ROBOT_CONTROLLER_H