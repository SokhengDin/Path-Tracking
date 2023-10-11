import numpy as np
import matplotlib.pyplot as plt

class DiffSimulation:

    def __init__(self, r_h=0.2, r_w=0.30, w_h=0.025, w_w=0.075, rot_angle=0.0, w_pos=0.15):

        ## robot shape
        self.robot_h = r_h
        self.robot_w = r_w
        ## wheel shaspe
        self.wheel_h = w_h
        self.wheel_w = w_w
        ## wheel position
        self.w_pos = w_pos
        ## Rotation angle
        self.rotate_angle = rot_angle

    def rotation_angle(self, theta):
        matrix_transform = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]]).T
        return matrix_transform


    def robot_shape(self):
        matrix_shape = np.array([
            [-self.robot_w, self.robot_w, self.robot_w, -self.robot_w, -self.robot_w],
            [self.robot_h, self.robot_h, -self.robot_h, -self.robot_h, self.robot_h]
        ])
        return matrix_shape

    def robot_wheel(self):
        matrix_wheel = np.array([
            [-self.wheel_w, self.wheel_w, self.wheel_w, -self.wheel_w, -self.wheel_w],
            [self.wheel_h, self.wheel_h, -self.wheel_h, -self.wheel_h, self.wheel_h]
        ])
        return matrix_wheel

    def generate_each_wheel_and_draw(self, x, y, yaw):
        # Copy Matrix as robot_shape
        pos_wheel = self.robot_wheel()
        pos_wheel1 = pos_wheel.copy()
        pos_wheel2 = pos_wheel.copy()
        pos_wheel3 = pos_wheel.copy()
        pos_wheel4 = pos_wheel.copy()

        # Push each wheel to where it belong
        pos_wheel1[0, :] += self.w_pos
        pos_wheel1[1, :] -= self.w_pos
        pos_wheel2[0, :] += self.w_pos
        pos_wheel2[1, :] += self.w_pos
        pos_wheel3[0, :] -= self.w_pos
        pos_wheel3[1, :] += self.w_pos
        pos_wheel4[0, :] -= self.w_pos
        pos_wheel4[1, :] -= self.w_pos
        # Matrix Transforms each wheel 1, 2, 3, 4
        pos_wheel1 = np.dot(pos_wheel1.T, self.rotation_angle(yaw)).T
        pos_wheel2 = np.dot(pos_wheel2.T, self.rotation_angle(yaw)).T
        pos_wheel3 = np.dot(pos_wheel3.T, self.rotation_angle(yaw)).T
        pos_wheel4 = np.dot(pos_wheel4.T, self.rotation_angle(yaw)).T

        # Matrix Transforms robot shape
        robot_shape = self.robot_shape()
        robot_shaped = np.dot(robot_shape.T, self.rotation_angle(yaw)).T

        plt.plot(robot_shaped[0, :]+x, robot_shaped[1, :]+y, color="black")
        plt.plot(x, y, marker="x", color="red")
        plt.plot(pos_wheel1[0, :]+x, pos_wheel1[1, :]+y, color="black")
        plt.plot(pos_wheel2[0, :]+x, pos_wheel2[1, :]+y, color="black")
        plt.plot(pos_wheel3[0, :]+x, pos_wheel3[1, :]+y, color="black")
        plt.plot(pos_wheel4[0, :]+x, pos_wheel4[1, :]+y, color="black")