#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This agent demonstrates how to structure your code and visualize camera data in 
an OpenCV window and control the robot with keyboard commands with pynput 
https://pypi.org/project/opencv-python/
https://pypi.org/project/pynput/

"""
import numpy as np
import carla
import cv2 as cv
import random
from math import radians
from pynput import keyboard
import apriltag

""" Import the AutonomousAgent from the Leaderboard. """

from leaderboard.autoagents.autonomous_agent import AutonomousAgent

""" Define the entry point so that the Leaderboard can instantiate the agent class. """

def get_entry_point():
    return 'OpenCVagent'

""" Inherit the AutonomousAgent class. """

class OpenCVagent(AutonomousAgent):

    def setup(self, path_to_conf_file):

        """ This method is executed once by the Leaderboard at mission initialization. We should add any attributes to the class using 
        the 'self' Python keyword that contain data or methods we might need throughout the simulation. If you are using machine learning 
        models for processing sensor data or control, you should load the models here. We encourage the use of class attributes in place
        of using global variables which can cause conflicts. """

        """ Set up a keyboard listener from pynput to capture the key commands for controlling the robot using the arrow keys. """

        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        #detector = apriltag.Detector()
        """ Add some attributes to store values for the target linear and angular velocity. """

        self.current_v = 0
        self.current_w = 0

        """ Initialize a counter to keep track of the number of simulation steps. """

        self.frame = 0
        self.detector = apriltag.Detector()
        self.no_tag_detected_count = 0

    def use_fiducials(self):

        """ We want to use the fiducials, so we return True. """
        return True

    def sensors(self):

        """ In the sensors method, we define the desired resolution of our cameras (remember that the maximum resolution available is 2448 x 2048) 
        and also the initial activation state of each camera and light. Here we are activating the front left camera and light. """

        sensors = {
            carla.SensorPosition.Front: {
                'camera_active': False, 'light_intensity': 1.0, 'width': '1280', 'height': '720'
            },
            carla.SensorPosition.FrontLeft: {
                'camera_active': True, 'light_intensity': 1.0, 'width': '1280', 'height': '720'
            },
            carla.SensorPosition.FrontRight: {
                'camera_active': False, 'light_intensity': 1.0, 'width': '1280', 'height': '720'
            },
            carla.SensorPosition.Left: {
                'camera_active': False, 'light_intensity': 0, 'width': '1280', 'height': '720'
            },
            carla.SensorPosition.Right: {
                'camera_active': False, 'light_intensity': 0, 'width': '1280', 'height': '720'
            },
            carla.SensorPosition.BackLeft: {
                'camera_active': False, 'light_intensity': 0, 'width': '1280', 'height': '720'
            },
            carla.SensorPosition.BackRight: {
                'camera_active': False, 'light_intensity': 0, 'width': '1280', 'height': '720'
            },
            carla.SensorPosition.Back: {
                'camera_active': False, 'light_intensity': 0, 'width': '1280', 'height': '720'
            },
        }
        return sensors

    def run_step(self, input_data):

        """ The run_step method executes in every simulation time-step. Your control logic should go here. """

        """ In the first frame of the simulation we want to raise the robot's excavating arms to remove them from the 
        field of view of the cameras. Remember that we are working in radians. """

        if self.frame == 0:
            self.set_front_arm_angle(radians(60))
            self.set_back_arm_angle(radians(60))

        """ Let's retrieve the front left camera data from the input_data dictionary using the correct dictionary key. We want the 
        grayscale monochromatic camera data, so we use the 'Grayscale' key. """
        cam_positions = [
            carla.SensorPosition.Front, 
            carla.SensorPosition.FrontLeft, 
            carla.SensorPosition.FrontRight,
            carla.SensorPosition.Left,
            carla.SensorPosition.Right,
            carla.SensorPosition.BackLeft,
            carla.SensorPosition.BackRight,
            carla.SensorPosition.Back
        ]

        all_sensor_data = {}
        tag_found = False
        for cam_pos in cam_positions:
            sensor_data = input_data['Grayscale'].get(cam_pos, None)
            all_sensor_data[cam_pos] = sensor_data


        """ We need to check that the sensor data is not None before we do anything with it. The date for each camera will be 
        None for every other simulation step, since the cameras operate at 10Hz while the simulator operates at 20Hz. """

        valid_sensor_data = None
        for cam_pos in cam_positions:
            if all_sensor_data.get(cam_pos) is not None:
                valid_sensor_data = all_sensor_data[cam_pos]
                break

        if valid_sensor_data  is not None:

            """ We use OpenCV's imshow method to render the image to screen, the first time this function is called
            it will open a new OpenCV window. Subsequent calls will render to the same window. """
            image = np.array(valid_sensor_data, dtype=np.uint8)
            #image = np.array(image)
            #detector = apriltag.Detector()
            tags = self.detector.detect(image)
            if tags:
                tag_found = True
                self.no_tag_detected_count = 0
                for tag in tags:
                    tag_id = tag.tag_id
                    center = tag.center
                    corners = tag.corners

                    print(f"Detected AprilTag {tag_id} at {center}")
                    for i in range(4):
                        pt1 = tuple(corners[i].astype(int))
                        pt2 = tuple(corners[(i + 1) % 4].astype(int))
                        cv.line(image, pt1, pt2, (0, 255, 0), 2)

                    cv.circle(image, tuple(center.astype(int)), 5, (0, 0, 255), -1)

                    fx, fy = 1000, 1000
                    cx, cy = 640, 360
                    tag_size = 0.2

                    object_points = np.array([
                        [-tag_size / 2, -tag_size / 2, 0],
                        [tag_size / 2, -tag_size / 2, 0],
                        [tag_size / 2, tag_size / 2, 0],
                        [-tag_size / 2, tag_size / 2, 0]
                    ], dtype=np.float32)

                    image_points = np.array(corners, dtype=np.float32)

                    ret, rvec, tvec = cv.solvePnP(object_points, image_points, 
                                                  np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), 
                                                  None)

                    if ret:
                        print(f"AprilTag {tag_id} Pose:")
                        print(f"Translation (x, y, z): {tvec.ravel()}")
                        print(f"Rotation Vector: {rvec.ravel()}")

                        R, _ = cv.Rodrigues(rvec)
                        T = np.hstack((R, tvec))
                        T = np.vstack((T, [0, 0, 0, 1]))

                        print("Transformation Matrix:")
                        print(T)

                        tag_world_position = np.array([2.0, 3.0, 0.0, 1.0])
                        robot_position_world = np.dot(np.linalg.inv(T), tag_world_position)
                        true_loc = self.get_location()
                        true_x, true_y, true_z = true_loc.x, true_loc.y, true_loc.z
                        print(f"Ground Truth Position (GT): ({true_x}, {true_y}, {true_z})")
                        print(f"Robot Global Position: {robot_position_world[:3]}")

                        error = np.linalg.norm(robot_position_world[:3] - np.array([true_x, true_y, true_z]))
                        print(f"Localization Error: {error:.4f} meters")

            
            cv.imshow("AprilTag Detection", image)
            cv.waitKey(1)

            """ If we want to save camera data, we can use OpenCV to create a PNG file for each camera frame. Uncomment 
            the following line and create a directory named "out" in the root folder of the simulator package """

            #cv.imwrite('out/' + str(self.frame) + '.png', self.sensor_data)

            """ Increment the frame counter. """
        if not tag_found:
            self.no_tag_detected_count += 1
            print(f"WARNING: No AprilTags detected for {self.no_tag_detected_count} frames!")

            if self.no_tag_detected_count > 5:
                print("Reorienting robot to search for AprilTags...")
                self.current_v = 0
                self.current_w = 0.5

        else:
            self.current_v = 0.3
            self.current_w = 0
        
        self.frame += 1 

        """ Now we prepare the control instruction to return to the simulator, with our target linear and angular velocity. """

        control = carla.VehicleVelocityControl(self.current_v, self.current_w)
        
        """ If the simulation has been going for more than 5000 frames, let's stop it. """
        if self.frame >= 5000:
            self.mission_complete()

        return control

    def finalize(self):

        """ In the finalize method, we should clear up anything we've previously initialized that might be taking up memory or resources. 
        In this case, we should close the OpenCV window. """

        cv.destroyAllWindows()

        """ We may also want to add any final updates we have from our mapping data before the mission ends. Let's add some random values 
        to the geometric map to demonstrate how to use the geometric map API. The geometric map should also be updated during the mission
        in the run_step() method, in case the mission is terminated unexpectedly. """

        """ Retrieve a reference to the geometric map object. """

        geometric_map = self.get_geometric_map()

        """ Set some random height values and rock flags. """

        for i in range(100):

            x = 10 * random.random() - 5
            y = 10 * random.random() - 5
            geometric_map.set_height(x, y, random.random())

            rock_flag = random.random() > 0.5
            geometric_map.set_rock(x, y, rock_flag)


    def on_press(self, key):

        """ This is the callback executed when a key is pressed. If the key pressed is either the up or down arrow, this method will add
        or subtract target linear velocity. If the key pressed is either the left or right arrow, this method will set a target angular 
        velocity of 0.6 radians per second. """

        if key == keyboard.Key.up:
            self.current_v += 0.1
            self.current_v = np.clip(self.current_v, 0, 0.3)
        if key == keyboard.Key.down:
            self.current_v -= 0.1
            self.current_v = np.clip(self.current_v, -0.3, 0)
        if key == keyboard.Key.left:
            self.current_w = 0.6
        if key == keyboard.Key.right:
            self.current_w = -0.6      

    def on_release(self, key):

        """ This method sets the angular or linear velocity to zero when the arrow key is released. Stopping the robot. """

        if key == keyboard.Key.up:
            self.current_v = 0
        if key == keyboard.Key.down:
            self.current_v = 0
        if key == keyboard.Key.left:
            self.current_w = 0
        if key == keyboard.Key.right:
            self.current_w = 0

        """ Press escape to end the mission. """
        if key == keyboard.Key.esc:
            self.mission_complete()
            cv.destroyAllWindows()