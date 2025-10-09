cmake_minimum_required(VERSION 3.14)
project(gen3_xr_teleop)

add_compile_definitions(LINUX __linux__ UNIX)
add_compile_definitions(_OS_UNIX OS_LINUX LINUX __linux__ UNIX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- ROS2 ----
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(kdl_parser REQUIRED)
find_package(orocos_kdl REQUIRED)
# TRAC-IK（系统装了 trac_ik_lib 的话）
find_package(trac_ik_lib QUIET)
find_package(tf2_ros REQUIRED)


# Eigen（有些系统需要）
find_package(Eigen3 QUIET)
if(Eigen3_FOUND)
  include_directories(${EIGEN3_INCLUDE_DIR})
endif()

# ============================================
# 1. Gen3 Robot Controller Library（非 ROS）
# ============================================
add_library(gen3_robot_controller SHARED
  src/Gen3RobotController.cpp
)

target_include_directories(gen3_robot_controller PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# -------- Kortex C++ API --------
set(KORTEX_API_ROOT ${CMAKE_SOURCE_DIR}/kortex_api)
set(KORTEX_INC      ${KORTEX_API_ROOT}/include)
set(KORTEX_LIB_DIR  ${KORTEX_API_ROOT}/lib)
if(EXISTS "${KORTEX_API_ROOT}/lib/release")
  set(KORTEX_LIB_DIR "${KORTEX_API_ROOT}/lib/release")
endif()

target_include_directories(gen3_robot_controller PUBLIC
  ${KORTEX_INC}
  ${KORTEX_INC}/client
  ${KORTEX_INC}/common
  ${KORTEX_INC}/client_stubs
  ${KORTEX_INC}/google
  ${KORTEX_INC}/messages
)

target_compile_definitions(gen3_robot_controller PUBLIC LINUX __linux__ UNIX)

# 选择 Kortex 核心库
if(EXISTS "${KORTEX_LIB_DIR}/libKortexApiCpp.a")
  set(KORTEX_CORE_LIB "${KORTEX_LIB_DIR}/libKortexApiCpp.a")
elseif(EXISTS "${KORTEX_LIB_DIR}/libKortexApiCpp_without_protobuf.a")
  set(KORTEX_CORE_LIB "${KORTEX_LIB_DIR}/libKortexApiCpp_without_protobuf.a")
  find_package(Protobuf REQUIRED)
  target_link_libraries(gen3_robot_controller PRIVATE protobuf::libprotobuf)
elseif(EXISTS "${KORTEX_LIB_DIR}/libKortexApiCpp.so")
  set(KORTEX_CORE_LIB "${KORTEX_LIB_DIR}/libKortexApiCpp.so")
else()
  message(FATAL_ERROR "Kortex library not found in ${KORTEX_LIB_DIR}")
endif()

find_package(Threads REQUIRED)
target_link_libraries(gen3_robot_controller PRIVATE
  ${KORTEX_CORE_LIB}
  Threads::Threads
  dl
  rt
  atomic
)

# 让运行时能找到 .so（可选）
if(UNIX AND NOT APPLE)
  set_target_properties(gen3_robot_controller PROPERTIES
    BUILD_RPATH "${KORTEX_LIB_DIR}"
    INSTALL_RPATH "${KORTEX_LIB_DIR}"
  )
endif()

# ============================================
# 2. Gen3 XR Teleop ROS2 Node（可执行文件）
# ============================================
add_executable(gen3_xr_teleop_node
  src/gen3_xr_teleop_ros.cpp
)

target_include_directories(gen3_xr_teleop_node PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${orocos_kdl_INCLUDE_DIRS}
)

ament_target_dependencies(gen3_xr_teleop_node
  rclcpp
  std_msgs
  geometry_msgs
  kdl_parser
  tf2_ros
)

# 链接 TRAC-IK
if(TARGET trac_ik_lib::trac_ik_lib)
  target_link_libraries(gen3_xr_teleop_node trac_ik_lib::trac_ik_lib)
elseif(DEFINED trac_ik_lib_LIBRARIES AND trac_ik_lib_LIBRARIES)
  target_link_libraries(gen3_xr_teleop_node ${trac_ik_lib_LIBRARIES})
else()
  find_library(TRAC_IK_LIB NAMES trac_ik
    PATHS /opt/ros/humble/lib ${CMAKE_PREFIX_PATH}/lib
          ${CMAKE_PREFIX_PATH}/trac_ik_lib/lib
          ${CMAKE_INSTALL_PREFIX}/lib
  )
  if(NOT TRAC_IK_LIB)
    message(FATAL_ERROR "libtrac_ik.so not found. Set CMAKE_PREFIX_PATH or install trac_ik_lib.")
  endif()
  target_link_libraries(gen3_xr_teleop_node ${TRAC_IK_LIB})
endif()

# 链接 orocos-kdl
if(TARGET orocos-kdl)
  target_link_libraries(gen3_xr_teleop_node orocos-kdl)
elseif(DEFINED orocos_kdl_LIBRARIES AND orocos_kdl_LIBRARIES)
  target_link_libraries(gen3_xr_teleop_node ${orocos_kdl_LIBRARIES})
endif()

# 链接 Gen3 控制器库
target_link_libraries(gen3_xr_teleop_node gen3_robot_controller)

# ============================================
# 3. 安装
# ============================================
install(TARGETS gen3_xr_teleop_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS gen3_robot_controller
  EXPORT gen3_robot_controller
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(DIRECTORY include/
  DESTINATION include
)

# Python 节点（确保有 shebang 且 chmod +x）
install(PROGRAMS
  scripts/xr_publisher_node.py
  DESTINATION lib/${PROJECT_NAME}
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/launch")
  install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
endif()

ament_package()
