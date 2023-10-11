import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import time
import math

from bezier_path import calc_4points_bezier_path

r = 0.06
R = 0.22

a1 = np.pi/4
a2 = 3*np.pi/4
a3 = 5*np.pi/4
a4 = 7*np.pi/4

N = 50
N_IND_SEARCH = 10
step_time = 0.1
mpciter = 0
sim_time = 25

x_min = -ca.inf
y_min = -ca.inf
theta_min = -ca.inf

x_max = ca.inf
y_max = ca.inf
theta_max = ca.inf

u_min = -25
u_max =  25

start_x = 2
start_y = 2
start_yaw = 1.57

end_x = 0
end_y = 0
end_yaw = 0


def calc_index_trajectory(state_x, state_y, cx, cy, pind):

    dx = [state_x - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
    dy = [state_y - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]

    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

    mind = min(d)

    ind = d.index(mind) + pind


    # print(ind)

    return ind

def smooth_yaw(yaw):

    for i in range(len(yaw)-1):
        dyaw = yaw[i+1] - yaw[i]

        while dyaw >= np.pi/2.0:
            yaw[i+1] -= np.pi*2.0
            dyaw = yaw[i+1] - yaw[i]

        while dyaw <= -np.pi/2.0:
            yaw[i+1] += np.pi * 2.0
            dyaw = yaw[i+1] - yaw[i]

    return yaw

def calc_ref_trajectory(state, cx, cy, cyaw, dl, pind, v):
    xref = np.zeros((3, N + 1))
    dref = np.zeros((1, N + 1))
    ncourse = len(cx)

    ind, _ = calc_index_trajectory(state, cx, cy, cyaw, pind)

    if pind >= ind:
        ind = pind

    xref[0, 0] = cx[ind]
    xref[1, 0] = cy[ind]
    xref[2, 0] = cyaw[ind]

    travel = 0.0

    for i in range(N + 1):
        travel += abs(v) * 0.1
        dind = int(round(travel / dl))

        if (ind + dind) < ncourse:
            xref[0, i] = cx[ind + dind]
            xref[1, i] = cy[ind + dind]
            xref[2, i] = cyaw[ind + dind]
        else:
            xref[0, i] = cx[ncourse - 1]
            xref[1, i] = cy[ncourse - 1]
            xref[2, i] = cyaw[ncourse - 1]

    return xref, ind

def pi_2_pi(angle):
    while(angle > math.pi):
        angle = angle - 2.0 * math.pi

    while(angle < -math.pi):
        angle = angle + 2.0 * math.pi

    return angle

def calc_speed_profile(cx, cy, cyaw, target_speed):

    speed_profile = [target_speed] * len(cx)
    direction = 1.0  # forward

    # Set stop point
    for i in range(len(cx) - 1):
        dx = cx[i + 1] - cx[i]
        dy = cy[i + 1] - cy[i]

        move_direction = math.atan2(dy, dx)

        if dx != 0.0 and dy != 0.0:
            dangle = abs(pi_2_pi(move_direction - cyaw[i]))
            if dangle >= math.pi / 4.0:
                direction = -1.0
            else:
                direction = 1.0

        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed

    speed_profile[-1] = 0.0

    return speed_profile

def forward_kinematic(u1, u2, u3, u4, theta):

    rot_mat = np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    J_for = (r/2)*np.array([
        [-np.sin(theta+a1), -np.sin(theta+a2), -np.sin(theta+a3), -np.sin(theta+a4)],
        [np.cos(theta+a1), np.cos(theta+a2), np.cos(theta+a3), np.cos(theta+a4)],
        [1/(2*0.22), 1/(2*0.22), 1/(2*0.22), 1/(2*0.22)]
    ], dtype=np.float64)

    for_vec = J_for@np.array([u1, u2, u3, u4])

    return for_vec

def forward_kinematic_tran(u1, u2, u3, u4, theta):

    rot_mat = np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    J_for = (r/2)*np.array([
        [np.sin(theta+a1), -np.sin(theta+a2), np.sin(theta+a3), -np.sin(theta+a4)],
        [np.cos(theta+a1), -np.cos(theta+a2), np.cos(theta+a3), -np.cos(theta+a4)],
        [1/(2*0.22), 1/(2*0.22), 1/(2*0.22), 1/(2*0.22)]
    ], dtype=np.float64)

    for_vec = J_for@np.array([u1, u2, u3, u4])

    return for_vec

def inverse_kinematic(vx, vy, vyaw, theta):

    rot_mat = np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    J_inv = (1/r)*np.array([
        [-np.sin(theta+a1), -np.sin(theta+a2), -np.sin(theta+a3), -np.sin(theta+a4)],
        [np.cos(theta+a1), np.cos(theta+a2), np.cos(theta+a3), np.cos(theta+a4)],
        [0.22, 0.22, 0.22, 0.22]
    ], dtype=np.float64).T

    inv_vec = J_inv@np.array([vx, vy, vyaw])

    return inv_vec

def inverse_kinematic_tran(vx, vy, vyaw, theta):

    rot_mat = np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    J_inv = (1/r)*np.array([
        [np.sin(theta+a1), -np.sin(theta+a2), np.sin(theta+a3), -np.sin(theta+a4)],
        [np.cos(theta+a1), -np.cos(theta+a2), np.cos(theta+a3), -np.cos(theta+a4)],
        [0.22, 0.22, 0.22, 0.22]
    ], dtype=np.float64).T

    inv_vec = J_inv@np.array([vx, vy, vyaw])

    return inv_vec


def plot_arrow(x, y, yaw, length=0.05, width=0.3, fc="b", ec="k"):
    if not isinstance(x, float):
        for (ix, iy, iyaw) in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * np.cos(yaw), length * np.sin(yaw),
            fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)


# States
x = ca.SX.sym('x')
y = ca.SX.sym('y')
theta = ca.SX.sym('theta')
states = ca.vertcat(x, y, theta)
n_states = states.numel()
u1 = ca.SX.sym('u1')
u2 = ca.SX.sym('u2')
u3 = ca.SX.sym('u3')
u4 = ca.SX.sym('u4')
controls = ca.vertcat(u1, u2, u3, u4)
n_controls = controls.numel()

X = ca.SX.sym('X', n_states, N+1)
X_ref = ca.SX.sym('X_ref', n_states, N+1)
U = ca.SX.sym('U', n_controls, N)
U_ref = ca.SX.sym('U_ref', n_controls, N)


Q = np.diag([750, 750, 900])
R = np.diag([0.01, 0.01, 0.01, 0.01])


cost_fn = 0.0
g = X[:, 0] - X_ref[:, 0]


rot_mat = ca.vertcat(
    ca.horzcat(ca.cos(theta), ca.sin(theta), 0),
    ca.horzcat(-ca.sin(theta), ca.cos(theta), 0),
    ca.horzcat(0, 0, 1)
)

J_for = (r/2)*ca.vertcat(
    ca.horzcat(-ca.sin(theta+a1), -ca.sin(theta+a2), -ca.sin(theta+a3), -ca.sin(theta+a4)),
    ca.horzcat(ca.cos(theta+a1), ca.cos(theta+a2), ca.cos(theta+a3), ca.cos(theta+a4)),
    ca.horzcat(1/(2*0.22), 1/(2*0.22), 1/(2*0.22), 1/(2*0.22))
)


rhs = J_for@ca.vertcat(u1, u2, u3, u4)


f = ca.Function('f', [states, controls], [rhs])

for k in range(N):
    st_err = X[:, k] - X_ref[:, k]
    con_err = U[:, k] - U_ref[:, k]
    cost_fn = cost_fn + st_err.T@Q@st_err + con_err.T@R@con_err
    st_next = X[:, k+1]
    st_euler = X[:, k] + step_time * f(X[:, k], U[:, k])
    g = ca.vertcat(g, st_next-st_euler)

cost_fn = cost_fn + (X[:, N]-X_ref[:, N]).T@Q@(X[:, N]-X_ref[:, N])


opt_dec = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
opt_par = ca.vertcat(ca.reshape(X_ref, -1, 1), ca.reshape(U_ref, -1, 1))


nlp_probs = {
    'f': cost_fn,
    'x': opt_dec,
    'p': opt_par,
    'g': g
}


nlp_opts = {
    'ipopt.max_iter': 5000,
    'ipopt.print_level': 0,
    'ipopt.acceptable_tol': 1e-6,
    'ipopt.acceptable_obj_change_tol': 1e-4,
    'print_time': 0}

solver = ca.nlpsol('solver', 'ipopt', nlp_probs, nlp_opts)


lbx = ca.DM.zeros((n_states*(N+1)+n_controls*N, 1))
ubx = ca.DM.zeros((n_states*(N+1)+n_controls*N, 1))


lbx[0: n_states*(N+1): n_states] = x_min
lbx[1: n_states*(N+1): n_states] = y_min
lbx[2: n_states*(N+1): n_states] = theta_min


ubx[0: n_states*(N+1): n_states] = x_max
ubx[1: n_states*(N+1): n_states] = y_max
ubx[2: n_states*(N+1): n_states] = theta_max


lbx[n_states*(N+1) :  n_states*(N+1)+n_controls*N: n_controls] = u_min
lbx[n_states*(N+1)+1: n_states*(N+1)+n_controls*N: n_controls] = u_min
lbx[n_states*(N+1)+2: n_states*(N+1)+n_controls*N: n_controls] = u_min
lbx[n_states*(N+1)+3: n_states*(N+1)+n_controls*N: n_controls] = u_min


ubx[n_states*(N+1) :  n_states*(N+1)+n_controls*N: n_controls] = u_max
ubx[n_states*(N+1)+1: n_states*(N+1)+n_controls*N: n_controls] = u_max
ubx[n_states*(N+1)+2: n_states*(N+1)+n_controls*N: n_controls] = u_max
ubx[n_states*(N+1)+3: n_states*(N+1)+n_controls*N: n_controls] = u_max


args = {
    'lbg': ca.DM.zeros((n_states*(N+1), 1)),
    'ubg': ca.DM.zeros((n_states*(N+1), 1)),
    'lbx': lbx,
    'ubx': ubx
}


path1, _ = calc_4points_bezier_path(
    0, 0, 0,
    2.7, 1.5, 1.57,
    -1.5
)

# goal1 = np.vstack([path1[:, 0], path1[:, 1], np.append(np.arctan2(np.diff(path1[:, 1]), np.diff(path1[:, 0])), 1.57)])
goal1 = np.vstack([path1[:, 0], path1[:, 1], np.linspace(0, 1.57, 50)])


path2, _ = calc_4points_bezier_path(
    2.7, 1.5, 1.57,
    0.0, 0.0, 0.0,
    1.5
)

# goal2 = np.vstack([path2[:, 0], path2[:, 1], np.append(np.arctan2(np.diff(path2[:, 1]), np.diff(path2[:, 0])), 1.57)])
goal2 = np.vstack([path2[:, 0], path2[:, 1], np.tile(1.57, 50)])

# path3, _ = calc_4points_bezier_path(
#     5.5, -3.0, 1.57,
#     3.5, -3.0, 3.14,
#     10.0, 50
# )

# goal3 = np.vstack([path3[:, 0], path3[:, 1], np.tile(3.14, 50)])

# path4, _ = calc_4points_bezier_path(
#     3.5, -3.0, 3.14,
#     4, 0.0, 0.0,
#     20.0, 50
# )

# goal4 = np.vstack([path4[:, 0], path4[:, 1], np.append(np.arctan2(np.diff(path4[:, 1]), np.diff(path4[:, 0])), 0.0)])

goal_states = np.hstack([goal1, goal2])

# ax = goal_states[0, :]
# ay = goal_states[1, :]

current_states = np.array([0.0, 0.0, 0.0], dtype=np.float64)
current_controls = np.array([0, 0, 0, 0], dtype=np.float64)

states = np.tile(current_states.reshape(3, 1), N+1)
controls = np.tile(current_controls.reshape(4, 1), N)

next_trajectories = np.tile(current_states.reshape(3, 1), N+1)
next_controls = np.tile(current_controls.reshape(4, 1), N)

hist_x = [current_states[0]]
hist_y = [current_states[1]]
hist_th = [current_states[2]]


# ax = [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0]
# ay = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

# rx, ry, ryaw, _, _ = calc_spline_course(
#     ax, ay, ds=0.1
# )

rx = goal_states[0, :]
ry = goal_states[1, :]
ryaw = smooth_yaw(goal_states[2, :])

ax = rx
ay = ry

if __name__ == "__main__":

    target_ind = calc_index_trajectory(current_states[0], current_states[1], rx, ry, 1)
    # old_nearest_index_point = 0
    # new_nearest_index_point = 0
    x_next = np.zeros(3)
    prev_dind = 0
    next_dind = 0
    index = 0
    pred_index = 0

    while (mpciter * step_time < sim_time):
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

        sol_x = ca.reshape(sol['x'][:n_states*(N+1)], 3, N+1)
        sol_u = ca.reshape(sol['x'][n_states*(N+1):], 4, N)

        u1 = sol_u.full()[0, 0]
        u2 = sol_u.full()[1, 0]
        u3 = sol_u.full()[2, 0]
        u4 = sol_u.full()[3, 0]

        theta = sol_x.full()[2, 0]

        # ind = search_index_trajectory(current_states[0], current_states[1], rx, ry)

        target_ind = calc_index_trajectory(x_next[0], x_next[1], rx, ry, target_ind)

        travel = 1.0

        prev_dind = next_dind

        for j in range(N):

            # vx, vy, vyaw = forward_kinematic(sol_u.full()[0, N-1], sol_u.full()[1, N-1], sol_u.full()[2, N-1],sol_u.full()[3, N-1],
            #                                  sol_x.full()[2, N-1])

            for_vec = forward_kinematic(u1, u2, u3, u4, 0.0)

            vx = for_vec[0]
            vy = for_vec[1]

            v = np.sqrt(vx**2+vy**2)
            # # print(vx)
            # old_nearest_index_point = target_ind+dind

            next_trajectories[0, 0] = current_states[0]
            next_trajectories[1, 0] = current_states[1]
            next_trajectories[2, 0] = current_states[2]

            travel += abs(v) * 0.1
            dind = int(round(travel / 1.0))

            pred_index = target_ind + index


            # index = mpciter + j
            # if index >= goal_states.shape[1]:
            #     index = goal_states.shape[1]-1

            if (target_ind + index) < len(rx):
                next_trajectories[0, j+1] = rx[pred_index]
                next_trajectories[1, j+1] = ry[pred_index]
                next_trajectories[2, j+1] = ryaw[pred_index]
            else:
                next_trajectories[0, j+1] = rx[len(rx)-1]
                next_trajectories[1, j+1] = ry[len(ry)-1]
                next_trajectories[2, j+1] = ryaw[len(ryaw)-1]

            next_controls = np.tile(np.array([30, 30, 30, 30]).reshape(4, 1), N)

            # print(next_controls)

        next_dind = dind
        if (next_dind - prev_dind) >=2 :
                index += 1

        if pred_index >= goal_states.shape[1]:
            pred_index = goal_states.shape[1]-1

        x_next = current_states + forward_kinematic(u1, u2, u3, u4, theta) * step_time
        # print(dind)
        print(target_ind)

        # print(pred_index)

        # x_next = current_states + ca.DM.full(f(current_states, sol_u[:, 0])*step_time)


        current_states = x_next

        current_controls = np.array([u1, u2, u3, u4])

        states = np.tile(current_states.reshape(3, 1), N+1)
        controls = np.tile(current_controls.reshape(4, 1), N)

        # print(target_ind + dind)
        # print(travel)

        # print(forward_kinematic(u1, u2, u3, u4, 0.0))

        # print(f(current_states, sol_u[:, 0]))

        # print(sol_u[:, 0])

        plt.clf()
        plt.plot(ax, ay, "b")
        plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
        plot_arrow(current_states[0], current_states[1], current_states[2])
        plt.plot(ax, ay, marker="x", color="blue", label="Input Trajectory")
        plt.scatter(sol_x.full()[0, :], sol_x.full()[1, :], marker="*", color="red", label="Predicted value")
        plt.plot(current_states[0], current_states[1], marker="*", color="black")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.pause(0.0001)

        # print(current_states)

        mpciter += 1
        # print(mpciter)