#!/usr/bin/env python3

import rospy
import numpy as np
import time
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import LanePose, Twist2DStamped, WheelsCmdStamped, StopLineReading, FSMState
from apriltag_ros.msg import AprilTagDetectionArray
from dt_param import DTParam, ParamType
from pid_controller import PIDController
from std_msgs.msg import Header

class LaneControllerNode(DTROS):
    def __init__(self, node_name):
        super().__init__(node_name=node_name, node_type=NodeType.CONTROL)

        # Dynamic Parameters
        self.params = dict()
        self.params["~v_bar"] = DTParam("~v_bar", ParamType.FLOAT, 0.0, 5.0)
        self.params["~k_d"] = DTParam("~k_d", ParamType.FLOAT, -100.0, 100.0)
        self.params["~k_theta"] = DTParam("~k_theta", ParamType.FLOAT, -100.0, 100.0)
        self.params["~k_Id"] = DTParam("~k_Id", ParamType.FLOAT, -100.0, 100.0)
        self.params["~k_Iphi"] = DTParam("~k_Iphi", ParamType.FLOAT, -100.0, 100.0)
        self.params["~theta_thres_min"] = DTParam("~theta_thres_min", ParamType.FLOAT)
        self.params["~theta_thres_max"] = DTParam("~theta_thres_max", ParamType.FLOAT)

        # Regular ROS params
        self.params["~d_thres"] = rospy.get_param("~d_thres", 0.25)
        self.params["~d_offset"] = rospy.get_param("~d_offset", 0.0)
        self.params["~integral_bounds"] = rospy.get_param("~integral_bounds", {'d': {'bot': -0.3, 'top': 0.3}, 'phi': {'bot': -1.2, 'top': 1.2}})
        self.params["~omega_ff"] = rospy.get_param("~omega_ff", 0.0)
        self.params["~verbose"] = rospy.get_param("~verbose", False)
        self.params["~dt"] = rospy.get_param("~dt", 0.05)

        # PID controller setup
        d_bounds = self.params["~integral_bounds"]["d"]
        phi_bounds = self.params["~integral_bounds"]["phi"]

        self.controller = PIDController(
            kp_d=self.params["~k_d"].value,
            ki_d=self.params["~k_Id"].value,
            kp_phi=self.params["~k_theta"].value,
            ki_phi=self.params["~k_Iphi"].value,
            dt=self.params["~dt"],
            d_limits=(d_bounds["bot"], d_bounds["top"]),
            phi_limits=(phi_bounds["bot"], phi_bounds["top"]),
            output_limits=(-8.0, 8.0)
        )

        # Intersection control
        self.tag_behavior_map = {
            1: "left",
            2: "right",
            3: "straight"
        }
        self.current_tag = None
        self.in_intersection = False
        self.last_stop_time = 0

        # Internal state
        self.pose_msg_dict = {}
        self.last_s = None
        self.at_stop_line = False
        self.at_obstacle_stop_line = False
        self.obstacle_stop_line_distance = 1e6
        self.stop_line_distance = 1e6
        self.wheels_cmd_executed = WheelsCmdStamped()
        self.fsm_state = "LANE_FOLLOWING"
        self.current_pose_source = "lane_filter"

        # Subscribers
        self.sub_lane_reading = rospy.Subscriber("~lane_pose", LanePose, self.cbAllPoses, "lane_filter", queue_size=1)
        self.sub_intersection_navigation_pose = rospy.Subscriber("~intersection_navigation_pose", LanePose, self.cbAllPoses, "intersection_navigation", queue_size=1)
        self.sub_stop_line = rospy.Subscriber("~stop_line_reading", StopLineReading, self.cbStopLineReading, queue_size=1)
        self.sub_obstacle_stop_line = rospy.Subscriber("~obstacle_distance_reading", StopLineReading, self.cbObstacleStopLineReading, queue_size=1)
        self.sub_wheels_cmd_executed = rospy.Subscriber("~wheels_cmd", WheelsCmdStamped, self.cbWheelsCmdExecuted, queue_size=1)
        self.sub_fsm_state = rospy.Subscriber("~fsm_node/mode", FSMState, self.cbMode, queue_size=1)
        self.sub_apriltags = rospy.Subscriber("/apriltag_detector_node/detections", AprilTagDetectionArray, self.cbApriltags)

        # Publisher
        self.pub_car_cmd = rospy.Publisher("~car_cmd", Twist2DStamped, queue_size=1)

        self.log("Initialized!")

    def cbApriltags(self, msg):
        if msg.detections:
            self.current_tag = msg.detections[0].id[0]
        else:
            self.current_tag = None

    def cbStopLineReading(self, msg):
        self.stop_line_distance = np.sqrt(msg.stop_line_point.x**2 + msg.stop_line_point.y**2)
        self.at_stop_line = msg.stop_line_detected

    def cbObstacleStopLineReading(self, msg):
        self.obstacle_stop_line_distance = np.sqrt(msg.stop_line_point.x**2 + msg.stop_line_point.y**2)
        self.obstacle_stop_line_detected = msg.stop_line_detected
        self.at_obstacle_stop_line = msg.at_stop_line

    def cbMode(self, fsm_state_msg):
        self.fsm_state = fsm_state_msg.state
        self.current_pose_source = "intersection_navigation" if self.fsm_state == "INTERSECTION_CONTROL" else "lane_filter"

    def cbAllPoses(self, input_pose_msg, pose_source):
        if pose_source == self.current_pose_source:
            self.pose_msg_dict[pose_source] = input_pose_msg
            self.pose_msg = input_pose_msg
            self.getControlAction(self.pose_msg)

    def cbWheelsCmdExecuted(self, msg_wheels_cmd):
        self.wheels_cmd_executed = msg_wheels_cmd

    def publishCmd(self, car_cmd_msg):
        self.pub_car_cmd.publish(car_cmd_msg)

    def getControlAction(self, pose_msg):
        current_s = rospy.Time.now().to_sec()
        dt = current_s - self.last_s if self.last_s is not None else None

        if self.at_stop_line or self.at_obstacle_stop_line:
            v = 0
            omega = 0
        else:
            d_err = pose_msg.d - self.params["~d_offset"]
            phi_err = pose_msg.phi

            if np.abs(d_err) > self.params["~d_thres"]:
                self.log("d_err too large, thresholding it!", "error")
                d_err = np.sign(d_err) * self.params["~d_thres"]

            if phi_err > self.params["~theta_thres_max"].value or phi_err < self.params["~theta_thres_min"].value:
                self.log("phi_err out of bounds, thresholding!", "error")
                phi_err = np.clip(phi_err, self.params["~theta_thres_min"].value, self.params["~theta_thres_max"].value)

            wheels_cmd_exec = [self.wheels_cmd_executed.vel_left, self.wheels_cmd_executed.vel_right]
            stop_dist = self.obstacle_stop_line_distance if self.obstacle_stop_line_detected else self.stop_line_distance

            v, omega = self.controller.compute_control_action(d_err, phi_err, dt, wheels_cmd_exec, stop_dist)

            if self.fsm_state == "INTERSECTION_CONTROL" and self.current_tag is not None:
                behavior = self.tag_behavior_map.get(self.current_tag)
                if behavior == "left":
                    omega += 4.0
                elif behavior == "right":
                    omega -= 4.0

            omega += self.params["~omega_ff"]

        car_cmd_msg = Twist2DStamped()
        car_cmd_msg.header = pose_msg.header
        car_cmd_msg.v = v
        car_cmd_msg.omega = omega

        self.publishCmd(car_cmd_msg)
        self.last_s = current_s

    def cbParametersChanged(self):
        self.controller.update_parameters(self.params)

if __name__ == "__main__":
    node = LaneControllerNode(node_name="lane_controller_node")
    rospy.spin()
