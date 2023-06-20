import casadi as ca
import numpy as np
import math
import matplotlib.pyplot as plt

from bezier_path import calc_4points_bezier_path


prediction_horizons = 50
step_horizon = 0.1

Tf = 3.0
sim_time = 20
mpciter = 0
t0 = 0


x_min = -10
y_min = -10
theta_min = -1.57

x_max = 10
y_max = 10
theta_max = 1.57


u_min = -10
u_max = 10

def forward_kinematic(u1, u2, theta):

    rot_mat = np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    for_vec = np.array([
        np.cos(theta)*u1,
        np.sin(theta)*u1,
        u2
    ])

    return for_vec

def plot_arrow(x, y, yaw, length=0.05, width=0.3, fc="b", ec="k"):
    if not isinstance(x, float):
        for (ix, iy, iyaw) in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * np.cos(yaw), length * np.sin(yaw),
            fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)


def shift_timestep(step_horizon, t0, x0, x_f, u, f):
    x0 = x0.reshape((-1,1))
    t = t0 + step_horizon
    f_value = f(x0, u[:, 0])
    st = ca.DM.full(x0 + (step_horizon) * f_value)
    u = np.concatenate((u[:, 1:], u[:, -1:]), axis=1)
    x_f = np.concatenate((x_f[:, 1:], x_f[:, -1:]), axis=1)
    return t, st, x_f, u


x = ca.SX.sym("x")
y = ca.SX.sym("y")
theta = ca.SX.sym("theta")
states = ca.vertcat(x, y, theta)
num_states = states.numel()
u1 = ca.SX.sym("u1")
u2 = ca.SX.sym("u2")
controls = ca.vertcat(u1, u2)
num_controls = controls.numel()

rot_mat = ca.vertcat(
    ca.horzcat(ca.cos(theta), ca.sin(theta), 0),
    ca.horzcat(-ca.sin(theta), ca.cos(theta), 0),
    ca.horzcat(0, 0, 1)
)

vx = u1*ca.cos(theta)
vy = u1*ca.sin(theta)
vyaw = u2

rhs = ca.vertcat(vx, vy, vyaw)

f = ca.Function('f', [states, controls], [rhs])

X = ca.SX.sym('X', num_states, prediction_horizons+1)
X_ref = ca.SX.sym('X_ref', num_states, prediction_horizons+1)
U = ca.SX.sym('U', num_controls, prediction_horizons)
U_ref = ca.SX.sym('U_ref', num_controls, prediction_horizons)

cost_fn = 0.0
g = X[:, 0] - X_ref[:, 0]

Q = np.diag([75, 75, 90])
R = np.diag([0.1, 0.01])

for k in range(prediction_horizons):
    st_err = X[:, k] - X_ref[:, k]
    con_err = U[:, k] - U_ref[:, k]
    cost_fn = cost_fn + st_err.T@st_err + con_err.T@R@con_err
    st_next = X[:, k+1]
    st_next_euler = X[:, k] + step_horizon * f(X[:, k], U[:, k])
    g = ca.vertcat(g, st_next-st_next_euler)

cost_fn = cost_fn + (X[:, prediction_horizons] - X_ref[:, prediction_horizons]).T@Q@(X[:, prediction_horizons] - X_ref[:, prediction_horizons])

opt_var = ca.vertcat(
    ca.reshape(X, -1, 1),
    ca.reshape(U, -1, 1)
)

opt_dec = ca.vertcat(
    ca.reshape(X_ref, -1, 1),
    ca.reshape(U_ref, -1, 1)
)

nlp_prob = {
    'f': cost_fn,
    'x': opt_var,
    'p': opt_dec,
    'g': g
}

nlp_opts = {
    'ipopt.max_iter': 5000,
    'ipopt.print_level': 0,
    'ipopt.acceptable_tol': 1e-6,
    'ipopt.acceptable_obj_change_tol': 1e-4,
    'print_time': 0}

lbx = ca.DM.zeros((num_states*(prediction_horizons+1)+num_controls*prediction_horizons, 1))
ubx = ca.DM.zeros((num_states*(prediction_horizons+1)+num_controls*prediction_horizons, 1))

lbx[0: num_states*(prediction_horizons+1): num_states] = x_min
lbx[1: num_states*(prediction_horizons+1): num_states] = y_min
lbx[2: num_states*(prediction_horizons+1): num_states] = theta_min

ubx[0: num_states*(prediction_horizons+1): num_states] = x_max
ubx[1: num_states*(prediction_horizons+1): num_states] = y_max
ubx[2: num_states*(prediction_horizons+1): num_states] = theta_max

lbx[num_states*(prediction_horizons+1): num_states*(prediction_horizons+1)+num_controls*prediction_horizons: num_controls] = u_min
lbx[num_states*(prediction_horizons+1)+1: num_states*(prediction_horizons+1)+num_controls*prediction_horizons: num_controls] = u_min


ubx[num_states*(prediction_horizons+1): num_states*(prediction_horizons+1)+num_controls*prediction_horizons: num_controls] = u_max
ubx[num_states*(prediction_horizons+1)+1: num_states*(prediction_horizons+1)+num_controls*prediction_horizons: num_controls] = u_max



args = {
    'lbg': ca.DM.zeros((num_states*(prediction_horizons+1), 1)),
    'ubg': ca.DM.zeros((num_states*(prediction_horizons+1), 1)),
    'lbx': lbx,
    'ubx': ubx
}

solver = ca.nlpsol('solver', 'ipopt', nlp_prob, nlp_opts)


path, _ = calc_4points_bezier_path(
    0, 0, 0,
    3, 3, 1.57,
    1.0
)

x = path[:, 0]
y = path[:, 1]

yaw = np.append(np.arctan2(np.diff(y), np.diff(x)), 0)

goal_states = np.vstack([x, y, yaw])


current_states = np.array([0, 0, 0], dtype=np.float64)
current_controls = np.array([0, 0], dtype=np.float64)

states = np.tile(current_states.reshape(3, 1), prediction_horizons+1)
controls = np.tile(current_controls.reshape(2, 1), prediction_horizons)

next_trajectories = np.tile(current_states.reshape(3, 1), prediction_horizons+1)
next_controls = np.tile(current_controls.reshape(2, 1), prediction_horizons)


if __name__ == "__main__":

    while (mpciter * step_horizon < sim_time):

        args['p'] = np.concatenate([
            next_trajectories.T.reshape(-1, 1),
            next_controls.T.reshape(-1, 1)
        ])

        args['x0'] = np.concatenate([
            states.T.reshape(-1, 1),
            controls.T.reshape(-1, 1)
        ])

        sol = solver(
                x0=args['x0'],
                p = args['p'],
                lbx=args['lbx'],
                ubx=args['ubx'],
                lbg=args['lbg'],
                ubg=args['ubg'],
            )

        sol_x = ca.reshape(sol['x'][:num_states*(prediction_horizons+1)], 3, prediction_horizons+1)
        sol_u = ca.reshape(sol['x'][num_states*(prediction_horizons+1):], 2, prediction_horizons)

        for j in range(prediction_horizons):
            index = mpciter + j
            if index >= goal_states.shape[1]:
                index = goal_states.shape[1]-1
            next_trajectories[0, 0] = current_states[0]
            next_trajectories[1, 0] = current_states[1]
            next_trajectories[2, 0] = current_states[2]
            next_trajectories[:, j+1] = goal_states[:, index]
            next_controls = np.tile(np.array([1, 2], dtype=np.float64), prediction_horizons)
            # print(goal_states[:, index])

            # print(next_trajectories[0, j+1])

            # print(index)

        current_states = np.array([
            sol_x.full()[0, 0],
            sol_x.full()[1, 0],
            sol_x.full()[2, 0]
        ])

        current_controls = np.array([
            sol_u.full()[0, 0],
            sol_u.full()[1, 0],
        ])

        u1 = sol_u.full()[0, 0]
        u2 = sol_u.full()[1, 0]
        theta = sol_x.full()[2, 0]

        # x_next = current_states + step_horizon * forward_kinematic(u1, u2, theta)
        # current_states = x_next

        states = np.tile(current_states.reshape(3, 1), prediction_horizons+1)
        controls = np.tile(current_controls.reshape(2, 1), prediction_horizons)

        t0, current_states, states, controls = shift_timestep(step_horizon, t0, current_states, sol_x, sol_u, f)

        # print(current_states)
        print(forward_kinematic(u1, u2, theta))
        plt.clf()
        plt.plot(x, y, "b")
        plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
        plot_arrow(current_states[0], current_states[1], current_states[2])
        plt.plot(x, y, marker="x", color="blue", label="Input Trajectory")
        plt.scatter(sol_x.full()[0, :], sol_x.full()[1, :], marker="*", color="red", label="Predicted value")
        plt.plot(current_states[0], current_states[1], marker="*", color="black")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.pause(0.0001)


        mpciter += 1