import numpy as np
import matplotlib.pyplot as plt

from bezier_path import calc_4points_bezier_path


v_max = -2.0
v_min = 2.0 #m/s
omega_min = -3.14
omega_max = 3.14 # rad/s

start_x = 0.0
start_y = 0.0
start_yaw = 0.0

end_x = 5.0
end_y = 2.0
end_yaw = 1.57

offset = 2.0


sim_time = 200
sampling_time = 0.01


# Arrow function
def plot_arrow(x, y, yaw, length=0.05, width=0.3, fc="b", ec="k"):
    if not isinstance(x, float):
        for (ix, iy, iyaw) in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * np.cos(yaw), length * np.sin(yaw),
            fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)


class DifferentialDrive:


    def forward_kinematic(self, v, omega, yaw):

        vx = v*np.cos(yaw)
        vy = v*np.sin(yaw)
        vyaw = omega

        return vx, vy, vyaw

    def inverse_kinematic(self, vx, vy, vyaw):

        if vx <0 or vy <0:
            v = -np.sqrt(vx**2+vy**2)
        else:
            v = np.sqrt(vx**2+vy**2)

        omega = vyaw

        return v, omega


    def discrete_state(self, x, y, yaw, v, omega, dt):
        dx, dy, dyaw = self.forward_kinematic(v, omega, yaw)
        x_next = x + dx * dt
        y_next = y + dy * dt
        yaw_next = yaw + dyaw * dt

        return x_next, y_next, yaw_next


class PIDController:

    def __init__(self, kp, ki, kd, dt):

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral = 0

    def calculate_pid(self, errors):
        # Calculate proportional error
        proportional = self.kp * errors[-1]
        # Calculate integration error
        integral = self.integral + self.ki * (errors[-1]) * self.dt
        self.integral = integral
        # Calculate derivative error
        derivative = self.kd * (errors[-1] - errors[-2])/self.dt

        output = proportional + integral + derivative

        return output



# Choosing Kp, Ki, Kd for x

kp_x = 15
ki_x = 5
kd_x = 0

# Choosing Kp, Ki, Kd for y

kp_y = 15
ki_y = 10
kd_y = 0

# Choosing Kp, Ki, Kd for yaw

ki_yaw = 10
kp_yaw = 3.5
kd_yaw = 1

# Initialze position
x0 = start_x
y0 = start_y
yaw0 = start_yaw

current_x = x0
current_y = y0
current_yaw = yaw0

diff_drive = DifferentialDrive() # Create Differential drive robot class

## Create PID for each x, y, yaw

pid_controller_x = PIDController(kp_x, ki_x, kd_x, sampling_time)
pid_controller_y = PIDController(kp_y, ki_y, kd_y, sampling_time)
pid_controller_yaw = PIDController(kp_yaw, ki_yaw, kd_yaw, sampling_time)

ref_path = np.array([end_x, end_y, end_yaw], dtype=np.float32)

error_x = [ref_path[0]-current_x]
error_y = [ref_path[1]-current_y]
error_yaw = [ref_path[2]-current_yaw]


# Calculate limit

vx_min, vy_min, vth_min = diff_drive.forward_kinematic(v_min, omega_min, start_yaw)
vx_max, vy_max, vth_max = diff_drive.forward_kinematic(v_max, omega_max, end_yaw)

plt.figure(figsize=(12, 7)) # Intialize for figure

if __name__ == "__main__":
    for t in range(sim_time):

        error_x.append(ref_path[0]-current_x)
        error_y.append(ref_path[1]-current_y)
        error_yaw.append(ref_path[2]-current_yaw)

        ## Apply PID

        output_vx = pid_controller_x.calculate_pid(error_x)
        output_vy = pid_controller_y.calculate_pid(error_y)
        output_omega = pid_controller_yaw.calculate_pid(error_yaw)

        # if output_vx > vx_max:
        #     output_vx = vx_max
        # elif output_vx < vx_min:
        #     output_vx = vx_min

        # if output_vy > vy_max:
        #     output_vy = vy_max
        # elif output_vy < vy_min:
        #     output_vy = vy_min

        # if output_omega > vth_max:
        #     output_omega = vth_max
        # elif output_omega < vth_min:
        #     output_omega = vth_min

        v_pid, omega_pid = diff_drive.inverse_kinematic(output_vx, output_vy, output_omega)


        ## Discretize for getting new states

        x_next, y_next, yaw_next = diff_drive.discrete_state(current_x, current_y, current_yaw, v_pid, omega_pid, sampling_time) # Skip for omega we only look for x, y

        current_x = x_next
        current_y = y_next
        current_yaw = yaw_next

        print(error_x[-1])

        plt.clf()
        plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
        plot_arrow(current_x, current_y, current_yaw)
        plt.plot(5, 5)
        plt.plot(ref_path[0], ref_path[1], marker="x", color="blue", label="Input Trajectory")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.pause(0.0001)
