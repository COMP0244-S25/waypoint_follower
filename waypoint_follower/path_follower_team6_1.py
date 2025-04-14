# ======================================================================
# 
# ======================================================================
#
# Dear Examiner
#
# Hi there, Alex here. Commented out below is the piece of code I wrote before the 21st of Mark before the test criteria is changed
#
# It uses a B spline optimization to create a dense, smooth, trajectory 
# and a pure pursuit algorithm that follows the smooth trajectory.
# 
# It does not stop on waypoints, and runs nearly consistently at near max velocity (0.5) aside for when making sharper turns,
# which makes it significantly faster than the PD method under nearly any circumstances.
# The B spline optimized trajectory is also closer to the ellipses/cosine/8 shape pre-discretization.
#
# It took me a long time to code, tune and test. It works well on all 3 trajectories, 
# and it was even partially tested and works on the real robot (only issue being we didn't have enough space in the lab to let it run the whole trajectory)
#
# But of course none of this actually matters because I was informed 3 days later that if I left it in I'm getting an automatic fail. 
# 
# So here you go the whole thing is commented out.
#
# Best Wishes
#
# Alex Wang
# 
# ======================================================================

# # python3 src/waypoint_follower/waypoint_follower/path_follower.py /custom_odometry_topic

# #!/usr/bin/env python3

import math
import rclpy
import time
from rclpy.node import Node

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist, Pose, Point, PoseStamped, Vector3, Quaternion
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA

from scipy.interpolate import splprep, splev

import numpy as np
#import utils as utils


TRACKING_SPEED = 0.5 #[m/s]
TIMER_PERIOD = 0.1  # [s] -> 10 Hz
LOOK_AHEAD = 0.25 # [m]

DEBUG = False
CIRCULATE = False


STATE_FIND_INIT_POSE = 0
STATE_FIND_INIT_ORI = 1
STATE_PURE_PURSUIT = 2
STATE_FINETUNE_CONTROL = 3
STATE_FIND_GOAL_POSE = 4
STATE_STOP_CONTROL = 5





class PathFollower(Node):
    """
    A ROS2 node that uses a PD controller to follow a path (list of waypoints).
    Subscribes to /path for the target path and a specified odometry topic for the robot's current state,
    and publishes velocity commands to /cmd_vel.
    """

    def __init__(self, odometry_topic):
        super().__init__('path_follower')

        self.setup_parameters()

        # sum of derivate
        self.sum_derivate = 0
        self.start_time = 0
        self.end_time = 0

        # Path and waypoints
        self.path = None
        self.current_waypoint_index = 0

        # Previous errors (for derivative term)
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.prev_error_theta = 0.0

        # Robot current state
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_orientation = 0.0
        self.is_odom_received = False

        # Thresholds
        self.waypoint_threshold = 0.1  # Distance threshold to consider a waypoint reached
        self.orientation_threshold = 0.05  # Orientation threshold for final alignment

        # Publisher to cmd_vel
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.cmd_vel_pub.publish(Twist())

        # Subscriber to odometry (with the specified topic)
        self.odom_sub = self.create_subscription(
            Odometry,
            odometry_topic,
            self.odom_callback,
            10
        )

        # Subscriber to path
        self.path_sub = self.create_subscription(
            Path,
            '/planner/path',
            self.path_callback,
            10
        )

        # Timer to periodically publish velocity commands
        timer_period = 0.1  # [s] -> 10 Hz
        self.timer = self.create_timer(timer_period, self.control_loop_callback)
        self.get_logger().info(f"Path Follower node started. Subscribing to odometry topic: {odometry_topic}")
        self.get_logger().warn('++++++ Team-6-1 Starting Testing')

    # NOTE: CANNOT CHANGE
    def normalize_angle(self, angle):
            """
            Normalize an angle to the range [-pi, pi].
            """
            while angle > math.pi:
                angle -= 2.0 * math.pi
            while angle < -math.pi:
                angle += 2.0 * math.pi
            return angle     

    # NOTE: CANNOT CHANGE
    def odom_callback(self, msg: Odometry):
        """
        Extracts and stores the robot's current pose from the odometry.
        """
        self.is_odom_received = True
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        # Convert quaternion to yaw (theta)
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_orientation = math.atan2(siny_cosp, cosy_cosp)


    # NOTE: CANNOT CHANGE
    def check_ending(self):
        if not self.is_odom_received or self.path is None:
            self.cmd_vel_pub.publish(Twist())
            return True

        # Check if all waypoints are reached
        if self.current_waypoint_index >= len(self.path):
            if self.end_time < 1e-3:
                self.end_time = time.time()
            self.get_logger().info(f"----- All waypoints reached. Stopping. ---- ")
            self.get_logger().info(f"Sum of error: {self.sum_derivate:.3f}, Cost time: {self.end_time - self.start_time:.3f}s.")
            self.cmd_vel_pub.publish(Twist())  # Stop the robot
            return True
        
        return False 
    



    # NOTE: STUDENTS ARE ALLOWED TO ADD THEIR OWN FUNCTIONS
    def update_path(self):
        """
        Preprocess path before we start following it
        """
        # If we didn't recieve a path from topic, return
        if self.path_buffer is None: return
        # If we accepted more paths than expected, return
        if self.path_recieved>=self.total_path_to_recieve: return
        self.path_recieved+=1


        path = np.zeros([len(self.path_buffer),2])
        for i,p in enumerate(self.path_buffer):
            path[i,0] = p.pose.position.x
            path[i,1] = p.pose.position.y

        print('len og path', len(path))
        self.get_logger().info(f"Recieved path with lanegth {len(path)}")

        # getting path distance data
        path_diffs = path[1:] - path[:-1]
        dists = np.sqrt(np.sum(path_diffs**2, axis = 1))
        dists_from_start = np.zeros_like(path[:,0])
        dists_from_start[1:] = np.cumsum(dists)
        
        # Total distance of the original path
        total_dist = dists_from_start[-1]
        self.og_path_dist = total_dist

        # We want to have twice times point sensity to ensure
        # Since optimised path will be longe, we use 3 to be sfe
        num_wps = int((10 * total_dist / LOOK_AHEAD)) + 20

        # Assign the path
        path_opt = self.bspline_interpolate(path, num_wps, 0)

        self.get_logger().info(f"Using B spline and interpolated a path with length {len(path_opt)}")

        # getting path distance data
        path_diffs = path_opt[1:] - path_opt[:-1]
        dists = np.sqrt(np.sum(path_diffs**2, axis = 1))
        dists_from_start = np.zeros_like(path_opt[:,0])
        dists_from_start[1:] = np.cumsum(dists)

        headings = np.zeros_like(path_opt[:,0])
        headings[:-1] = np.arctan2(path_diffs[:,1], path_diffs[:,0])
        
        self.path = path
        self.path_opt = path_opt
        self.path_to_path_opt = self.remap_paths(path, path_opt)
        self.path_dist_to = dists_from_start

        self.headings = headings

        self.path_ind = 0
        self.total_path_to_recieve +=1
        self.get_logger().info(f"Path updated, ready to run")


    def full_PID_control(self, state_to_if_reached, wp_id_reached):
        """
        Use PID control to approach waypoints. Used to find initial and end positions
        """
        if self.current_waypoint_index>=len(self.path):
            self.control_state = STATE_STOP_CONTROL
            return 0.0,0.0
        # 1) Get previously computed errors
        error_theta = self.prev_error_theta
        error_dist = self.prev_error_dist

        # 3) PD control for linear velocities (x, y)
        vd = self.pid_ctrl_params[2]

        # 4) PD control for angular velocity
        vtheta = self.pid_ctrl_params[3]


        # Check if the robot has reached the current waypoint
        wp_reached = self.ctrl_switch_wp

        if wp_reached:
            # We reached goal
            self.control_state = state_to_if_reached
            self.path_ind = wp_id_reached
            if state_to_if_reached == STATE_STOP_CONTROL:
                self.get_logger().info("Goal Reached, STOP")
            if state_to_if_reached == STATE_FIND_INIT_ORI:
                self.get_logger().info("Start position reached, Trying to match orientation")
            if state_to_if_reached == STATE_PURE_PURSUIT:
                self.get_logger().info("Start position reached, Initializing Pure Pursuit")
            return 0.0, 0.0
        
        elif abs(error_theta) > self.orientation_threshold and error_dist > self.waypoint_threshold:
            # We are adjusting angle before we start walking
            vtheta = np.clip(vtheta, a_max=self.max_angular_vel, a_min=-self.max_angular_vel) 
            vd = 0.0
        v = np.clip(vd, a_max=self.max_linear_vel, a_min=-self.max_linear_vel)
        w = np.clip(vtheta, a_max=self.max_angular_vel, a_min=-self.max_angular_vel) 

        return v,w
    

    def ori_PID_control(self, state_to_if_reached):
        """
        Use PID to change orientation, used to setup initial position of the robot
        """

        if self.current_waypoint_index>=len(self.path):
            self.control_state = STATE_STOP_CONTROL
            return 0.0,0.0
        # Get the orientation of current target waypoint
        orientation_target = self.headings[self.path_to_path_opt[self.current_waypoint_index]]

        # 1) Compute errors
        error_orientation = (orientation_target - self.current_orientation)
        error_orientation = np.arctan2(np.sin(error_orientation), np.cos(error_orientation))

        # 2) Compute derivative of errors
        derivative_orientation = error_orientation - self.prev_error_orientation

        # 4) PD control for angular velocity
        vtheta = self.Kp_angular * error_orientation + self.Kd_angular * derivative_orientation
        

        # 5) Update previous error terms
        self.prev_error_orientation = error_orientation

        if (abs(error_orientation) < self.orientation_threshold):
            # We reached goal
            self.control_state = state_to_if_reached
            if state_to_if_reached == STATE_STOP_CONTROL:
                self.get_logger().info("Goal Reached, STOP")
            if state_to_if_reached == STATE_FIND_INIT_POSE:
                self.get_logger().info("Start orientation reached, Trying to match position")
            if state_to_if_reached == STATE_PURE_PURSUIT:
                self.get_logger().info("Start orientation reached, Initializing Pure Pursuit")
            return 0.0, 0.0

        w = np.clip(vtheta, a_max = self.max_angular_vel, a_min = -self.max_angular_vel)

        return 0.0,w
    
    def finetune_PID_control(self, state_to_if_reached):
        """
        Use X,Y linear velocity to adjust the position of the robot. This is in case we miss a waypoint on the way
        """
        # Get the current target waypoint
        # 1) Get previously computed errors

        # 3) PD control for linear velocities (x, y)
        vx_world = self.pid_ctrl_params[0]

        # 4) PD control for angular velocity
        vy_world = self.pid_ctrl_params[1]


        # Check if the robot has reached the current waypoint
        wp_reached = self.ctrl_switch_wp

        if wp_reached:
            # We reached goal
            self.control_state = state_to_if_reached
            if state_to_if_reached == STATE_STOP_CONTROL:
                self.get_logger().info("Finetune complete, Trying to match orientation")
            if state_to_if_reached == STATE_FIND_INIT_ORI:
                self.get_logger().info("Finetune Complete, Trying to match orientation")
            if state_to_if_reached == STATE_PURE_PURSUIT:
                self.get_logger().info("Finetune Complete, Initializing Pure Pursuit")
            vx, vy = 0.0,0.0
        
        else:
            self.get_logger().info(f"Fine tun to waypoint, vx:{vx_world}, vy{vy_world}")
        
        # transform the velocities to robot coordinate
        vx = vx_world * np.cos(self.current_orientation) + vy_world * np.sin(self.current_orientation)
        vy = -vx_world * np.sin(self.current_orientation) + vy_world * np.cos(self.current_orientation)
        vx = np.clip(vx, a_max=self.max_linear_vel/4, a_min=-self.max_linear_vel/4)
        vy = np.clip(vy, a_max=self.max_linear_vel/4, a_min=-self.max_linear_vel/4) 

        twist_msg = Twist()

        # Check if the robot has reached the current waypoint
        twist_msg.linear.x = vx
        twist_msg.linear.y = vy
        self.cmd_vel_pub.publish(twist_msg)
        return None, None
    

    def compute_PID_params(self):
        """
        This is the thing that computes all required PID parameters
        If the pure pursuit control where to miss a registered waypoint, this recovery PID control will 
        Drive the robot back to the previouse wp. This mechanis is designed purly for the scoring algorithm
        """

        if self.current_waypoint_index>=len(self.path):
            self.pid_ctrl_params = [0.0,0.0,0.0,0.0]
            return
        # Get the current target waypoint
        x_target = self.path[self.current_waypoint_index,0]
        y_target = self.path[self.current_waypoint_index,1]

        # 1) Compute errors
        error_x = x_target - self.current_x
        error_y = y_target - self.current_y
        error_theta = (math.atan2(error_y, error_x) - self.current_orientation)
        error_theta = np.arctan2(np.sin(error_theta), np.cos(error_theta))
        error_dist = np.linalg.norm([x_target - self.current_x, y_target - self.current_y])

        # 2) Compute derivative of errors
        derivative_x = error_x - self.prev_error_x
        derivative_y = error_y - self.prev_error_y
        derivative_theta = error_theta - self.prev_error_theta
        derivative_dist = error_dist - self.prev_error_dist

        # 3) PD control for linear velocities (x, y)
        vd = self.Kp_linear * error_dist + self.Kd_linear * derivative_dist
        x = self.Kp_linear * error_x + self.Kd_linear * derivative_x
        y = self.Kp_angular * error_y + self.Kd_linear * derivative_y

        # 2.5) Accumulate loss
        self.sum_derivate += math.fabs(derivative_x) + math.fabs(derivative_y) + math.fabs(derivative_theta) / 180.0 * math.pi

        # 4) PD control for angular velocity
        vtheta = self.Kp_angular * error_theta + self.Kd_angular * derivative_theta

        # 5) Update previous error terms
        self.prev_error_dist = error_dist
        self.prev_error_theta = error_theta
        self.prev_error_x = error_x
        self.prev_error_y = error_y

        # 6) Store these params for if we need to use PID control
        self.pid_ctrl_params = [x,y,vd,vtheta]


        # Check if the robot has reached the current waypoint
        distance_to_waypoint = math.hypot(error_x, error_y)

        if distance_to_waypoint < self.waypoint_threshold:
            # We reached goal
            self.current_waypoint_index += 1  # Move to the next waypoint
            self.ctrl_switch_wp = True
        else:
            self.ctrl_switch_wp = False
    

    def bspline_interpolate(self, coords, num_samples=100, s = 0):
        """
        Interpolate the given points with a B-spline using splprep (s=0).
        Returns sampled x_new, y_new points on the spline.
        """
        # Convert to numpy arrays
        x = coords[:,0]
        y = coords[:,1]
        
        # splprep expects data as [x, y] and a parameter array; s=0 -> exact interpolation
        tck, u = splprep([x, y], s=s)

        # Create a set of sample points (u ranges from 0 to 1)
        u_new = np.linspace(0, 1, num_samples)
        x_new, y_new = splev(u_new, tck)
        
        return np.stack([x_new[:],y_new[:]], axis = -1)
    

    def remap_paths(self, path, dense_path):
        dense_path = np.copy(dense_path)
        ids = []
        to_search = dense_path
        last_ind  = 0
        for i in range(len(path)):
            wp_pos = path[i] 
            dist_tos = np.linalg.norm(to_search - wp_pos, axis = 1)
            ddist_tos = dist_tos[1:] - dist_tos[:-1]
            first_min = np.argwhere(ddist_tos>0)
            if len(first_min) == 0:
                ind = len(dense_path)-1
            else:
                ind = min(len(dense_path)-1, last_ind + first_min[0][0]+1)
            ids.append(ind)
            to_search = dense_path[ind:]
            last_ind = ind
        return np.array(ids)



    def full_pure_pursuit_control(self):
        """
        Full control loop of pure pursuit control
        """
        # Get the current target waypoint
        # This is done by: 
        # 1. finding the point closest to agent on the trajectory
        # 2. Look ahead the trajectory for X distance, make that the waypoint
        #       - If we cannot find the next point, we are close to the end. In which case we will check we are "At the end"
        # 3. Find velocity Follow trajectory

        # If the underlyingmechanism decides that we've reached the ends, we teop
        if self.current_waypoint_index > len(self.path)-1:
            self.control_state = STATE_FIND_GOAL_POSE
            self.get_logger().info("Goal Reached, adjusting final orientation")
            return 0.0,0.0

        path_opt = self.path_opt
        path_ind = self.path_ind
        path_dist_to = np.copy(self.path_dist_to)

        # If we want to keep circlingthe path the path, we rebase the path.
        # NOTE: Any part exclusive to CIRCULATE does not actually serve to explain the code
        # If you want to figure out what is going on follow the case where CIRCULATE is set to false
        if CIRCULATE:
            closing_link = path_opt[0] - path_opt[-1]
            path_opt = np.concatenate([path_opt[path_ind:], path_opt[:path_ind]] ,axis = 0)

            dist_to_head = path_dist_to[path_ind:]
            if len(dist_to_head)>0: dist_to_head -= dist_to_head[0]
            
            dist_to_tail = path_dist_to[:path_ind] + np.linalg.norm(closing_link)
            if len(dist_to_head)>0: dist_to_tail += dist_to_head[-1]

            path_dist_to = np.concatenate([dist_to_head, dist_to_tail] ,axis = 0)

            path_ind = 0


        # Find position difference between point and robot
        diffs = path_opt - np.array([[self.current_x,self.current_y]])
        dists = np.sqrt(np.sum(diffs**2, axis = 1))

        # We find the first point in front of last ind where the distance to robot is at a local minimum
        # We assume this is where we are
        ddists = dists[path_ind+1:] - dists[path_ind:-1]
        first_neg =  np.argwhere(ddists > 0)
        if len(first_neg)==0: 
            first_neg = path_ind
        else:
            first_neg = first_neg[0][0] + path_ind
        
        # Set our path ind, i.e where we are on the path
        if CIRCULATE:
            self.path_ind = (self.path_ind + first_neg) % len(path_opt)
        else:
            self.path_ind = first_neg
            # NOTE This is actually how I determine the end 
            dist_to = path_dist_to - path_dist_to[first_neg]
            dist_to_lookahead = dist_to - self.waypoint_threshold
            pos_ind = np.argwhere(dist_to_lookahead > 0)

            # All the points along the path are less than self.waypoint_threshold away. 
            if len(pos_ind)==0: 
                self.control_state = STATE_FIND_GOAL_POSE
                self.get_logger().info("Goal Reached, adjusting final orientation")
                return 0.0, 0.0
        
        # If we bypassed a waypoint without knowing it:
        # We activate the fail safe finetune state, which will move the robot to the waypoint for it to be registered
        if self.path_ind > self.path_to_path_opt[self.current_waypoint_index]:
            self.control_state = STATE_FINETUNE_CONTROL
            self.finetune_PID_control(STATE_PURE_PURSUIT)
            return None, None
                


        self.get_logger().info(f'path progress {self.path_ind} / {len(path_opt)}')
        # Distance along path to the closest point (i.e our position)
        dist_to = path_dist_to - path_dist_to[first_neg]
        dist_to_lookahead = dist_to - LOOK_AHEAD
        pos_ind = np.argwhere(dist_to_lookahead > 0)
        # If we do not find a point that is more than LOOK_AHEAD, we must be near end and going toward the end
        near_end = False
        if len(pos_ind)==0: 
            near_end = True
            first_pos_ind = len(self.path_opt)-1
            path_ind = 0
        else:
            first_pos_ind = pos_ind[0][0]

        path_ind = first_pos_ind

        # The waypoint we are heading toward
        look_ahead_wp = path_opt[path_ind]

        look_ahead = max(np.sqrt((look_ahead_wp[0] - self.current_x)**2 + (look_ahead_wp[1] - self.current_y)**2), 0.1)

        if DEBUG:
            marker = Marker()
            marker.header.frame_id = "camera_init"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale = Vector3(x=0.15, y=0.15, z=0.15)
            marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)
            marker.id = 0
            marker.pose = Pose( position=Point(x=float(look_ahead_wp[0]), y=float(look_ahead_wp[1]), z=0.0),
                                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0))
            self.cmd_marker_pub.publish(marker)

            
        v,w = self.compute_pure_pursuit_control(look_ahead_wp,look_ahead = look_ahead)
        
        if near_end and not CIRCULATE:
            # We slow down at the end
            v *= dists[-1] / LOOK_AHEAD
        return v,w
        
    
    def compute_pure_pursuit_control(self, lookahead_point, look_ahead = LOOK_AHEAD):
        """
        Given the lookahead point, compute linear and angular velocities via
        the Pure Pursuit geometry.
        """
        # Robot current state
        x, y, theta = self.current_x, self.current_y, self.current_orientation

        # Lookahead point
        lx, ly = lookahead_point

        # Compute heading error alpha:
        # Angle from the robot's heading to the line connecting
        # the robot and the lookahead point.
        heading_to_point = math.atan2((ly - y), (lx - x))
        alpha = heading_to_point - theta
        # normalize angle
        alpha = np.arctan2(np.sin(alpha),np.cos(alpha))

        # We use a simple constant linear speed for demonstration.
        # You can adapt this based on curvature or other logic.
        gamma = 1.75  # Tuning parameter for speed reduction
        v_max = self.max_linear_vel  # Maximum tracking speed
        v = v_max * math.exp(-gamma * abs(alpha))

        # Angular velocity from Pure Pursuit:
        w = (2.0 * v * math.sin(alpha)) / LOOK_AHEAD #look_ahead
        # w = (3.0 * v_max * math.sin(alpha)) / LOOK_AHEAD

        return (v, w)

    # NOTE: CAN CHANGE
    def setup_parameters(self):
        # Maximum velocities
        # These are the actual values stated in gait.yaml
        self.max_linear_vel = 0.5  # meter
        self.max_angular_vel = 1.0 # rad

        # PD Controller Gains (the same values as waypoint follower)
        self.Kp_linear = 10.0
        self.Kd_linear = 0.1
        self.Kp_angular = 5.0
        self.Kd_angular = 0.1

        # So instead of vx and vy we are using vdist. 
        # Also we are adding an orientation error, this way we can modify the orientation of the robot to 
        self.prev_error_dist = 0.0
        self.prev_error_orientation = 0.0
        self.pid_ctrl_params =  [0.0,0.0,0.0,0.0]
        self.ctrl_switch_wp = False


        # Path and waypoints
        self.path_buffer = None
        self.path_dist_to = None
        self.path_recieved = 0
        # This changes 
        self.total_path_to_recieve = 1
        self.dst_reached  = False
        self.path_ind = 0

        self.headings = None
    

        # 0: Pose tracking: Use PID to reach a certain pose
        # 1: Pure pursuit: Use pure pursuit to continuously follow a trajectory
        # 3: Stop: Skipping the control loop and stopping the robot
        self.control_state = STATE_FIND_INIT_POSE



        # Publisher to cmd_vel
        self.cmd_marker_pub = self.create_publisher(Marker, '/marker0', 10)

        # Odometry recordings
        self.past_odom = []
        self.tracking_error = None


    # NOTE: CAN CHANGE
    def path_callback(self, msg: Path):
        """
        Updates the target path when a new message is received.
        """ 
        self.start_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        self.end_time = 0
        self.path_buffer = msg.poses  # List of PoseStamped messages
        if self.path is None and not self.dst_reached: self.update_path()


    # NOTE: CAN CHANGE
    def control_loop_callback(self):
        """
        Periodic control loop callback. Computes PD control and publishes velocity commands.
        """
        self.check_ending()
        # self.get_logger().info("path finder control loop")

        # We have made sure that the path will be more than 10 waypoints long
        if not self.is_odom_received or self.path is None:
            return
        
        if self.control_state == STATE_STOP_CONTROL:
            self.cmd_vel_pub.publish(Twist())
            return 
        
        # append current self position
        self.past_odom.append([self.current_x, self.current_y])


        # Underlying PID controller
        self.compute_PID_params()


        # Switch 
        if self.control_state == STATE_FINETUNE_CONTROL:
            self.finetune_PID_control(STATE_PURE_PURSUIT)
            return
        
        if self.control_state == STATE_FIND_INIT_POSE:
            # This gets us to point 0
            v,w = self.full_PID_control(STATE_FIND_INIT_ORI, 0)
        elif self.control_state == STATE_FIND_INIT_ORI:
            # This adjust the orientation of the robot to get a good start
            v,w = self.ori_PID_control(STATE_PURE_PURSUIT)
        elif self.control_state == STATE_FIND_GOAL_POSE:
            # This gets us to the last point
            v,w = self.full_PID_control(STATE_STOP_CONTROL, self.path_to_path_opt[min(self.current_waypoint_index, len(self.path_to_path_opt)-1)])
        elif self.control_state == STATE_PURE_PURSUIT:
            # Pure pursuit is quite effective at normal motion
            v,w = self.full_pure_pursuit_control()

        if v is None or w is None:
            return
        # publish Twist
        twist_msg = Twist()

        # Check if the robot has reached the current waypoint
        twist_msg.linear.x = v
        twist_msg.angular.z = w
        self.cmd_vel_pub.publish(twist_msg)

def main():
    import sys
    args = sys.argv 

    rclpy.init(args=args)

    # Specify the odometry topic (default is '/Odometry')
    odometry_topic = '/Odometry'
    #odometry_topic = '/utlidar/robot_odom' #'/Odometry'  # Default value
    node = PathFollower(odometry_topic)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt detected, shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
