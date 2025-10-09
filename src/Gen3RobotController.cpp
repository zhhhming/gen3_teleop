/**
 * Gen3 Robot Controller Implementation
 * High-performance C++ controller for Kinova Gen3 robot arm
 */

#include "Gen3RobotController.h"
#include <algorithm>
#include <cstring>

// Constructor
Gen3RobotController::Gen3RobotController(
    const std::string& ip_address,
    int tcp_port,
    int udp_port,
    const std::string& username,
    const std::string& password)
    : ip_address_(ip_address),
      tcp_port_(tcp_port),
      udp_port_(udp_port),
      username_(username),
      password_(password),
      tcp_transport_(nullptr),
      udp_transport_(nullptr),
      tcp_router_(nullptr),
      udp_router_(nullptr),
      tcp_session_manager_(nullptr),
      udp_session_manager_(nullptr),
      base_client_(nullptr),
      base_cyclic_client_(nullptr),
      actuator_config_client_(nullptr),
      in_low_level_mode_(false),
      actuator_count_(0),
      gripper_position_(0.0f),
      gripper_velocity_(0.0f),
      gripper_target_position_(0.0f),
      gripper_last_velocity_(0.0f),
      gripper_velocity_blend_alpha_(0.7f),
      frame_id_(0)
{
    end_effector_pose_.resize(6, 0.0f);
}

// Destructor
Gen3RobotController::~Gen3RobotController() {
    shutdown();
}

// Error callback
void Gen3RobotController::errorCallback(k_api::KError err) {
    std::cerr << "Router error: " << err.toString() << std::endl;
}

// Initialize the robot controller
bool Gen3RobotController::initialize() {
    std::cout << "Initializing Gen3 Robot Controller..." << std::endl;
    std::cout << "Connecting to robot at IP: " << ip_address_ << std::endl;

    try {
        // Initialize TCP transport and router
        tcp_transport_ = new k_api::TransportClientTcp();
        tcp_router_ = new k_api::RouterClient(tcp_transport_, errorCallback);
        
        std::cout << "Establishing TCP connection..." << std::endl;
        tcp_transport_->connect(ip_address_, tcp_port_);
        std::cout << "TCP connection established" << std::endl;

        // Initialize UDP transport and router
        udp_transport_ = new k_api::TransportClientUdp();
        udp_router_ = new k_api::RouterClient(udp_transport_, errorCallback);
        
        std::cout << "Establishing UDP connection..." << std::endl;
        udp_transport_->connect(ip_address_, udp_port_);
        std::cout << "UDP connection established" << std::endl;

        // Create sessions
        if (!createSessions()) {
            std::cerr << "Failed to create sessions" << std::endl;
            return false;
        }

        // Initialize service clients
        base_client_ = new k_api::Base::BaseClient(tcp_router_);
        base_cyclic_client_ = new k_api::BaseCyclic::BaseCyclicClient(udp_router_);
        actuator_config_client_ = new k_api::ActuatorConfig::ActuatorConfigClient(tcp_router_);

        std::cout << "Robot clients initialized successfully" << std::endl;

        // Get actuator count
        auto actuator_count = base_client_->GetActuatorCount();
        actuator_count_ = actuator_count.count();
        std::cout << "Robot has " << actuator_count_ << " actuators" << std::endl;

        // Initialize state vectors
        joint_positions_.resize(actuator_count_, 0.0f);
        joint_velocities_.resize(actuator_count_, 0.0f);

        // Get initial feedback
        base_feedback_ = base_cyclic_client_->RefreshFeedback();
        updateStateFromFeedback();

        std::cout << "Current joint positions: ";
        for (float pos : joint_positions_) {
            std::cout << pos << " ";
        }
        std::cout << std::endl;

        // Save current servoing mode
        previous_servoing_mode_ = base_client_->GetServoingMode();

        std::cout << "Gen3 Robot Controller initialization complete!" << std::endl;
        return true;

    } catch (k_api::KDetailedException& ex) {
        std::cerr << "Kortex exception: " << ex.what() << std::endl;
        return false;
    } catch (std::exception& ex) {
        std::cerr << "Standard exception: " << ex.what() << std::endl;
        return false;
    } catch (...) {
        std::cerr << "Unknown exception during initialization" << std::endl;
        return false;
    }
}

// Create TCP and UDP sessions
bool Gen3RobotController::createSessions() {
    try {
        auto session_info = k_api::Session::CreateSessionInfo();
        session_info.set_username(username_);
        session_info.set_password(password_);
        session_info.set_session_inactivity_timeout(60000);   // milliseconds
        session_info.set_connection_inactivity_timeout(2000);  // milliseconds

        std::cout << "Creating TCP session..." << std::endl;
        tcp_session_manager_ = new k_api::SessionManager(tcp_router_);
        tcp_session_manager_->CreateSession(session_info);
        std::cout << "TCP session created" << std::endl;

        std::cout << "Creating UDP session..." << std::endl;
        udp_session_manager_ = new k_api::SessionManager(udp_router_);
        udp_session_manager_->CreateSession(session_info);
        std::cout << "UDP session created" << std::endl;

        return true;
    } catch (k_api::KDetailedException& ex) {
        std::cerr << "Kortex exception during session creation: " << ex.what() << std::endl;
        return false;
    } catch (std::exception& ex) {
        std::cerr << "Session creation failed: " << ex.what() << std::endl;
        return false;
    } catch (...) {
        std::cerr << "Unknown exception during session creation" << std::endl;
        return false;
    }
}

// Clear robot faults
bool Gen3RobotController::clearFaults() {
    if (!base_client_) {
        std::cerr << "Gen3RobotController: base client not available, cannot clear faults." << std::endl;
        return false;
    }

    try {
        std::cout << "Clearing robot faults..." << std::endl;
        base_client_->ClearFaults();
        std::cout << "Faults cleared successfully" << std::endl;
        return true;
    } catch (const k_api::KDetailedException& ex) {
        std::cerr << "Gen3RobotController: failed to clear faults. " << ex.what() << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Gen3RobotController: failed to clear faults. " << ex.what() << std::endl;
    } catch (...) {
        std::cerr << "Gen3RobotController: failed to clear faults due to an unknown error." << std::endl;
    }
    return false;
}

k_api::Base::ServoingMode currentServoingMode(k_api::Base::BaseClient* base)
{
    auto m = base->GetServoingMode();
    return m.servoing_mode();
}

// Enter low-level control mode
bool Gen3RobotController::enterLowLevelMode() {


    if (!base_client_) {
        std::cerr << "Gen3RobotController: base client not available, cannot enter low level mode." << std::endl;
        return false;
    }
        // 与真实状态同步
    auto cur_mode = currentServoingMode(base_client_);
    if (cur_mode == k_api::Base::LOW_LEVEL_SERVOING) {
        // 已经在低级模式：保证标志同步并返回 true（幂等）
        if (!in_low_level_mode_) {
            std::cout << "Already in low-level mode (syncing internal flag)." << std::endl;
            in_low_level_mode_ = true;
        } else {
            std::cout << "Already in low-level mode." << std::endl;
        }
        return true;
    }
    previous_servoing_mode_ = base_client_->GetServoingMode();

    // 若内部标志与真实状态不一致，纠正一下
    if (in_low_level_mode_) {
        std::cout << "Internal flag says low-level, but controller is not. Fixing flag." << std::endl;
        in_low_level_mode_ = false;
    }

    // Clear any existing faults before entering low-level mode
    if (!clearFaults()) {
        std::cerr << "Gen3RobotController: aborting low level mode switch because clearing faults failed." << std::endl;
        return false;
    }

    std::cout << "Entering low-level servoing mode..." << std::endl;

    try {
        // Stop any ongoing motion
        try {
            base_client_->Stop();
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        } catch (const k_api::KDetailedException& ex) {
            std::cerr << "Warning: Stop before switching to low-level failed: " << ex.what() << std::endl;
        }

        // Set to low-level servoing mode
        k_api::Base::ServoingModeInformation servoing_mode;
        servoing_mode.set_servoing_mode(k_api::Base::LOW_LEVEL_SERVOING);
        base_client_->SetServoingMode(servoing_mode);

        // Wait for mode change confirmation
        bool mode_changed = false;
        for (int i = 0; i < 300; i++) {
            auto current_mode = base_client_->GetServoingMode();
            if (current_mode.servoing_mode() == k_api::Base::LOW_LEVEL_SERVOING) {
                mode_changed = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        if (!mode_changed) {
            std::cerr << "Failed to enter LOW_LEVEL_SERVOING mode (timeout)" << std::endl;
            return false;
        }

        // Initialize base command
        if (!initializeBaseCommand()) {
            std::cerr << "Failed to initialize base command" << std::endl;
            return false;
        }

        in_low_level_mode_ = true;
        std::cout << "Low-level servoing mode activated" << std::endl;
        return true;

    } catch (const k_api::KDetailedException& ex) {
        std::cerr << "Gen3RobotController: failed to enter low level mode. " << ex.what() << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Gen3RobotController: failed to enter low level mode. " << ex.what() << std::endl;
    } catch (...) {
        std::cerr << "Gen3RobotController: failed to enter low level mode due to an unknown error." << std::endl;
    }
    return false;
}

// Initialize base command with current positions
bool Gen3RobotController::initializeBaseCommand() {
    try {
        // Get current feedback
        base_feedback_ = base_cyclic_client_->RefreshFeedback();
        
        // Clear and initialize base command
        base_command_.Clear();
        base_command_.set_frame_id(0);
        
        // Initialize actuator commands to current positions
        for (int i = 0; i < actuator_count_; i++) {
            auto* actuator_cmd = base_command_.add_actuators();
            actuator_cmd->set_flags(1);  // Enable
            actuator_cmd->set_position(base_feedback_.actuators(i).position());
            actuator_cmd->set_velocity(0.0f);
            actuator_cmd->set_torque_joint(0.0f);
            actuator_cmd->set_command_id(0);
        }

        // Initialize gripper command
        auto* interconnect_cmd = base_command_.mutable_interconnect();
        interconnect_cmd->mutable_command_id()->set_identifier(0);
        
        if (base_feedback_.interconnect().gripper_feedback().motor_size() > 0) {
            auto* gripper_cmd = interconnect_cmd->mutable_gripper_command();
            auto* motor_cmd = gripper_cmd->add_motor_cmd();
            motor_cmd->set_position(base_feedback_.interconnect().gripper_feedback().motor(0).position());
            motor_cmd->set_velocity(0.0f);
            motor_cmd->set_force(5.0f);  // Default 5% force
        }

        // Send initial command to establish continuity
        base_feedback_ = base_cyclic_client_->Refresh(base_command_);
        frame_id_ = 0;

        return true;
    } catch (k_api::KDetailedException& ex) {
        std::cerr << "Failed to initialize base command: " << ex.what() << std::endl;
        return false;
    } catch (std::exception& ex) {
        std::cerr << "Failed to initialize base command: " << ex.what() << std::endl;
        return false;
    } catch (...) {
        std::cerr << "Failed to initialize base command due to unknown error" << std::endl;
        return false;
    }
}



// Exit low-level control mode
bool Gen3RobotController::exitLowLevelMode() {
    if (!in_low_level_mode_) {
        return true;
    }

    std::cout << "Exiting low-level servoing mode..." << std::endl;

    try {

        // Restore previous servoing mode
        base_client_->SetServoingMode(previous_servoing_mode_);
        bool mode_changed = false;
        for (int i = 0; i < 300; ++i) {
            if (currentServoingMode(base_client_) == previous_servoing_mode_.servoing_mode()) {
                mode_changed = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (!mode_changed) {
            std::cerr << "Failed to leave LOW_LEVEL_SERVOING (timeout switching mode)" << std::endl;
            return false;
        }
        try {
            base_client_->Stop();
        } catch (const k_api::KDetailedException& ex) {
            // 某些固件不要求此步，失败也不致命，记录一下
            std::cerr << "Warning: Stop after mode switch failed: " << ex.what() << std::endl;
        }
        in_low_level_mode_ = false;
        std::cout << "Low-level servoing mode deactivated" << std::endl;
        return true;

    } catch (k_api::KDetailedException& ex) {
        std::cerr << "Failed to exit low-level mode: " << ex.what() << std::endl;
        return false;
    } catch (std::exception& ex) {
        std::cerr << "Failed to exit low-level mode: " << ex.what() << std::endl;
        return false;
    } catch (...) {
        std::cerr << "Failed to exit low-level mode due to unknown error" << std::endl;
        return false;
    }
}

// Set joint positions (for use in control loop)
void Gen3RobotController::setJointPositions(const std::vector<float>& positions) {
    if (positions.size() != actuator_count_) {
        std::cerr << "Invalid joint positions size. Expected " << actuator_count_ 
                  << ", got " << positions.size() << std::endl;
        return;
    }

    // Update actuator commands
    for (int i = 0; i < actuator_count_; i++) {

        base_command_.mutable_actuators(i)->set_position(positions[i]);
   }
}

// Set gripper position with velocity blending
void Gen3RobotController::setGripperPosition(float position, float speed_ratio) {
    // Clamp position to [0, 1] range (0=open, 1=closed)
    gripper_target_position_ = std::max(0.0f, std::min(1.0f, position));
    
    // Convert to percentage (0-100)
    float target_pct = gripper_target_position_ * 100.0f;
    float current_pct = gripper_position_;
    
    // Calculate position error
    float position_error = target_pct - current_pct;
    float tolerance = 0.1f;  // 1.5% tolerance
    
    // Calculate desired velocity with P control
    float kp = 2.0f;  // Proportional gain
    float max_velocity = 30.0f * speed_ratio;  // Max velocity in %/s
    
    float desired_velocity = 0.0f;
    if (std::abs(position_error) > tolerance) {
        desired_velocity = kp * std::abs(position_error);
        desired_velocity = std::min(desired_velocity, max_velocity);
    }
    
    // Velocity blending to avoid sudden changes
    float current_velocity = std::abs(gripper_velocity_);
    float error_normalized = std::min(std::abs(position_error) / 20.0f, 1.0f);
    float dynamic_alpha = gripper_velocity_blend_alpha_ * (1.0f - error_normalized * 0.3f);
    
    float blended_velocity = (1 - dynamic_alpha) * current_velocity + dynamic_alpha * desired_velocity;
    blended_velocity = std::min(blended_velocity, max_velocity);
    
    // Update gripper command
    if (base_command_.interconnect().gripper_command().motor_cmd_size() > 0) {
        auto* motor_cmd = base_command_.mutable_interconnect()
                                      ->mutable_gripper_command()
                                      ->mutable_motor_cmd(0);
        motor_cmd->set_position(target_pct);
        motor_cmd->set_velocity(blended_velocity);
        motor_cmd->set_force(5.0f);  // 5% force
        
        gripper_last_velocity_ = blended_velocity;
    }
}

// Send command and refresh feedback - Main function for control loop
bool Gen3RobotController::sendCommandAndRefresh() {
    if (!in_low_level_mode_) {
        std::cerr << "Not in low-level mode" << std::endl;
        return false;
    }

    try {
        // Update frame ID
        frame_id_ = (frame_id_ + 1) & 0xFFFF;
        base_command_.set_frame_id(frame_id_);
        
        // Update command IDs for all actuators
        for (int i = 0; i < actuator_count_; i++) {
            base_command_.mutable_actuators(i)->set_command_id(frame_id_);
        }
        
        // Send command and get feedback
        base_feedback_ = base_cyclic_client_->Refresh(base_command_);
        
        // Update internal state from feedback
        updateStateFromFeedback();
        
        return true;
    } catch (k_api::KDetailedException& ex) {
        std::cerr << "Failed to send command: " << ex.what() << std::endl;
        return false;
    } catch (std::exception& ex) {
        std::cerr << "Failed to send command: " << ex.what() << std::endl;
        return false;
    } catch (...) {
        std::cerr << "Failed to send command due to unknown error" << std::endl;
        return false;
    }
}

// Send joint positions only
bool Gen3RobotController::sendJointPositionsOnly(const std::vector<float>& positions) {
    if (!in_low_level_mode_) {
        std::cerr << "Not in low-level mode" << std::endl;
        return false;
    }
    
    setJointPositions(positions);
    return sendCommandAndRefresh();
}

// Send gripper position only
bool Gen3RobotController::sendGripperPositionOnly(float position, float speed_ratio) {
    if (!in_low_level_mode_) {
        std::cerr << "Not in low-level mode" << std::endl;
        return false;
    }
    
    setGripperPosition(position, speed_ratio);
    return sendCommandAndRefresh();
}

// Just refresh feedback without changing commands
bool Gen3RobotController::refreshOnly() {
    if (!in_low_level_mode_) {
        // Can still get feedback in high-level mode
        try {
            base_feedback_ = base_cyclic_client_->RefreshFeedback();
            updateStateFromFeedback();
            return true;
        } catch (k_api::KDetailedException& ex) {
            std::cerr << "Failed to refresh feedback: " << ex.what() << std::endl;
            return false;
        } catch (std::exception& ex) {
            std::cerr << "Failed to refresh feedback: " << ex.what() << std::endl;
            return false;
        } catch (...) {
            std::cerr << "Failed to refresh feedback due to unknown error" << std::endl;
            return false;
        }
    }
    
    return sendCommandAndRefresh();
}

// Update internal state from feedback
void Gen3RobotController::updateStateFromFeedback() {
    // Update joint positions and velocities
    for (int i = 0; i < actuator_count_; i++) {
        joint_positions_[i] = base_feedback_.actuators(i).position();
        joint_velocities_[i] = base_feedback_.actuators(i).velocity();
    }
    
    // Update gripper state
    if (base_feedback_.interconnect().gripper_feedback().motor_size() > 0) {
        gripper_position_ = base_feedback_.interconnect().gripper_feedback().motor(0).position();
        gripper_velocity_ = base_feedback_.interconnect().gripper_feedback().motor(0).velocity();
    }
    
    // Update end-effector pose
    end_effector_pose_[0] = base_feedback_.base().tool_pose_x();       // meters
    end_effector_pose_[1] = base_feedback_.base().tool_pose_y();       // meters
    end_effector_pose_[2] = base_feedback_.base().tool_pose_z();       // meters
    end_effector_pose_[3] = base_feedback_.base().tool_pose_theta_x(); // degrees
    end_effector_pose_[4] = base_feedback_.base().tool_pose_theta_y(); // degrees
    end_effector_pose_[5] = base_feedback_.base().tool_pose_theta_z(); // degrees
}

// Stop robot motion
void Gen3RobotController::stopRobot() {
    try {
        if (!base_client_) return;

        auto mode = currentServoingMode(base_client_);
        if (mode == k_api::Base::LOW_LEVEL_SERVOING) {
            std::cout << "Skip Stop(): robot is in LOW_LEVEL_SERVOING, call exitLowLevelMode() first." << std::endl;
            return;
        }

        base_client_->Stop();
        std::cout << "Robot stopped" << std::endl;

    } catch (const k_api::KDetailedException& ex) {
        std::cerr << "Failed to stop robot: " << ex.what() << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Failed to stop robot: " << ex.what() << std::endl;
    } catch (...) {
        std::cerr << "Failed to stop robot due to unknown error" << std::endl;
    }
}

// Clean shutdown
void Gen3RobotController::shutdown() {
    std::cout << "Shutting down Gen3 Robot Controller..." << std::endl;
    
    // Stop robot motion
    stopRobot();
    
    // Exit low-level mode if active
    if (in_low_level_mode_) {
        exitLowLevelMode();
    }
    
    // Close sessions
    closeSessions();
    
    // Disconnect transports
    if (tcp_transport_) {
        tcp_router_->SetActivationStatus(false);
        tcp_transport_->disconnect();
        std::cout << "TCP transport disconnected" << std::endl;
    }
    
    if (udp_transport_) {
        udp_router_->SetActivationStatus(false);
        udp_transport_->disconnect();
        std::cout << "UDP transport disconnected" << std::endl;
    }
    
    // Clean up objects
    delete base_client_;
    delete base_cyclic_client_;
    delete actuator_config_client_;
    delete tcp_session_manager_;
    delete udp_session_manager_;
    delete tcp_router_;
    delete udp_router_;
    delete tcp_transport_;
    delete udp_transport_;
    
    // Reset pointers
    base_client_ = nullptr;
    base_cyclic_client_ = nullptr;
    actuator_config_client_ = nullptr;
    tcp_session_manager_ = nullptr;
    udp_session_manager_ = nullptr;
    tcp_router_ = nullptr;
    udp_router_ = nullptr;
    tcp_transport_ = nullptr;
    udp_transport_ = nullptr;
    
    std::cout << "Gen3 Robot Controller shutdown complete" << std::endl;
}

// Close sessions
void Gen3RobotController::closeSessions() {
    try {
        if (tcp_session_manager_) {
            tcp_session_manager_->CloseSession();
            std::cout << "TCP session closed" << std::endl;
        }
        
        if (udp_session_manager_) {
            udp_session_manager_->CloseSession();
            std::cout << "UDP session closed" << std::endl;
        }
    } catch (k_api::KDetailedException& ex) {
        std::cerr << "Kortex exception closing sessions: " << ex.what() << std::endl;
    } catch (std::exception& ex) {
        std::cerr << "Error closing sessions: " << ex.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown error closing sessions" << std::endl;
    }
}