import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import math

from bezier_path import calc_bezier_path

r = 0.05
lx = 0.165
ly = 0.18
N = 50

N_IND_SEARCH = 10

step_time = 0.1
mpciter = 0
sim_time = 30

x_min = -ca.inf
y_min = -ca.inf
theta_min = -ca.inf

x_max = ca.inf
y_max = ca.inf
theta_max = ca.inf

u_min = -50.0
u_max =  50.0

start_x = 2
start_y = 2
start_yaw = 1.57

end_x = 0
end_y = 0
end_yaw = 0
offset = 1.0


def calc_index_trajectory(state_x, state_y, cx, cy, pind):

    dx = [state_x - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
    dy = [state_y - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]

    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]


    mind = min(d)

    ind = d.index(mind) + pind

    return ind

def smooth_yaw(yaw):

    for i in range(len(yaw) - 1):
        dyaw = yaw[i + 1] - yaw[i]

        while dyaw >= math.pi / 2.0:
            yaw[i + 1] -= math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

        while dyaw <= -math.pi / 2.0:
            yaw[i + 1] += math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

    return yaw

def calc_4points_bezier_path(sx, sy, syaw, ex, ey, eyaw, offset, n_points):
    dist = np.hypot(sx - ex, sy - ey) / offset
    control_points = np.array(
        [[sx, sy],
         [sx + dist * np.cos(syaw), sy + dist * np.sin(syaw)],
         [ex - dist * np.cos(eyaw), ey - dist * np.sin(eyaw)],
         [ex, ey]])

    path = calc_bezier_path(control_points, n_points)

    return path, control_points

def forward_kinematic(u1, u2, u3, u4, theta):
    rot_mat = np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    # rot_mat = np.array([
    #     [-np.sin(theta), np.cos(theta), 0],
    #     [np.cos(theta), np.sin(theta), 0],
    #     [0, 0, 1]
    # ])

    J_for = (r/4)*np.array([
        [1, 1, 1, 1],
        [-1, 1, 1, -1],
        [-1/(lx+ly), 1/(lx+ly), -1/(lx+ly), 1/(lx+ly)]
    ])

    for_vec = rot_mat.T@J_for@np.array([u1, u2, u3, u4])

    return for_vec

def inverse_kinematic(vx, vy, vth, theta):
    rot_mat = np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    J_inv = (r/4)*np.array([
        [1, -1,-(lx+ly)],
        [1,  1,  (lx+ly)],
        [1,  1, -(lx+ly)],
        [1, -1, (lx+ly)]
    ])

    inv_vec = (1/r)*J_inv@rot_mat@np.array([vx, vy, vth])

    return inv_vec



def discrete_velocity(hist_x, hist_y, hist_theta, k, dt):

    vx = (hist_x[k]-hist_x[k-1])/dt
    vy = (hist_y[k]-hist_y[k-1])/dt
    vth = (hist_theta[k]-hist_theta[k-1])/dt

    return vx, vy, vth


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
# u1 = ca.SX.sym('u1')
# u2 = ca.SX.sym('u2')
# u3 = ca.SX.sym('u3')
# u4 = ca.SX.sym('u4')
# controls = ca.vertcat(u1, u2, u3, u4)
v1 = ca.SX.sym('v1')
v2 = ca.SX.sym('v2')
v3 = ca.SX.sym('v3')
v4 = ca.SX.sym('v4')
controls = ca.vertcat(v1, v2, v3, v4)
n_controls = controls.numel()

X = ca.SX.sym('X', n_states, N+1)
X_ref = ca.SX.sym('X_ref', n_states, N+1)
U = ca.SX.sym('U', n_controls, N)
U_ref = ca.SX.sym('U_ref', n_controls, N)

# Q = np.diag([1000, 1000, 2000])
# R = np.diag([0.1, 0.1, 0.1, 0.1])

Q = np.diag([500, 500, 500])
R = np.diag([0.01, 0.01, 0.01, 0.01])

cost_fn = 0.0
g = X[:, 0] - X_ref[:, 0]
rot_mat = ca.vertcat(
    ca.horzcat(ca.cos(theta), ca.sin(theta), 0),
    ca.horzcat(-ca.sin(theta), ca.cos(theta), 0),
    ca.horzcat(0, 0, 1)
)

J_for = (r/4)*ca.DM([
    [1, 1, 1, 1],
    [-1, 1, 1, -1],
    [-1/(lx+ly), 1/(lx+ly), -1/(lx+ly), 1/(lx+ly)]
])


RHS = rot_mat.T@J_for@controls

f = ca.Function('f', [states, controls], [RHS])

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

lbx[n_states*(N+1) : n_states*(N+1)+n_controls*N: n_controls] = u_min
lbx[n_states*(N+1)+1: n_states*(N+1)+n_controls*N: n_controls] = u_min
lbx[n_states*(N+1)+2: n_states*(N+1)+n_controls*N: n_controls] = u_min
lbx[n_states*(N+1)+3: n_states*(N+1)+n_controls*N: n_controls] = u_min

ubx[n_states*(N+1) : n_states*(N+1)+n_controls*N: n_controls] = u_max
ubx[n_states*(N+1)+1: n_states*(N+1)+n_controls*N: n_controls] = u_max
ubx[n_states*(N+1)+2: n_states*(N+1)+n_controls*N: n_controls] = u_max
ubx[n_states*(N+1)+3: n_states*(N+1)+n_controls*N: n_controls] = u_max

args = {
    'lbg': ca.DM.zeros((n_states*(N+1), 1)),
    'ubg': ca.DM.zeros((n_states*(N+1), 1)),
    'lbx': lbx,
    'ubx': ubx
}

# path, _ = calc_4points_bezier_path(
#     start_x, start_y, start_yaw,
#     end_x, end_y, end_yaw,
#     offset
# )


path1, _ = calc_4points_bezier_path(
    0, 0, -1.57,
    5, -5.187, 0.0,
    1.2, 50
)

goal1 = np.vstack([path1[:, 0], path1[:, 1], np.append(np.arctan2(np.diff(path1[:, 1]), np.diff(path1[:, 0])), 0.0)])


path2, _ = calc_4points_bezier_path(
    5, -5.187, 0.0,
    5, -3.487, 2.8,
    3.0, 50
)

goal2 = np.vstack([path2[:, 0], path2[:, 1], np.append(np.arctan2(np.diff(path2[:, 1]), np.diff(path2[:, 0])), 2.8)])

path3, _ = calc_4points_bezier_path(
    5, -3.487, 2.8,
    3.8, -1.0, 0.0,
    2.0, 50
)

goal3 = np.vstack([path3[:, 0], path3[:, 1], np.linspace(2.8, 3.0, 50)])

path4, _ = calc_4points_bezier_path(
    4.0, -1.0, 3.14,
    3.5, 1.0, -0.52,
    1.5, 50
)

goal4 = np.vstack([path4[:, 0], path4[:, 1], np.append(np.arctan2(np.diff(path4[:, 1]), np.diff(path4[:, 0])), -0.52)])

goal_states = np.hstack([goal1, goal2, goal3])
# goal_states = np.loadtxt('/Users/sokhengdin/Desktop/Robocon2023/ocp_ws/test/path.csv', delimiter=',')

# np.savetxt("/Users/sokhengdin/Desktop/Robocon2023/ocp_ws/test/path.csv", goal_states, delimiter=',')

# ax = goal_states[0, :]
# ay = goal_states[1, :]

current_states = np.array([0.0, 0.0, -1.57], dtype=np.float64)
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
ryaw = goal_states[2, :]

ryaw = smooth_yaw(ryaw)

ax = rx
ay = ry
plt.figure(figsize=(12, 7))
if __name__ == "__main__":

    target_ind = calc_index_trajectory(current_states[0], current_states[1], rx, ry, 1)
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

        target_ind = calc_index_trajectory(current_states[0], current_states[1], rx, ry, target_ind)

        # target_ind = calc_index_trajectory(x_next[0], x_next[1], rx, ry, target_ind)

        travel = 0.0

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

            if pred_index >= goal_states.shape[1]:
                pred_index = goal_states.shape[1]-1


            # index = mpciter + j
            # if index >= goal_states.shape[1]:
            #     index = goal_states.shape[1]-1

            if (target_ind + index) < len(rx):
                next_trajectories[0, j+1] = rx[pred_index]
                next_trajectories[1, j+1] = ry[pred_index]
                next_trajectories[2, j+1] = ryaw[pred_index]
                # next_trajectories[0, j+1] = rx[target_ind + dind]
                # next_trajectories[1, j+1] = ry[target_ind + dind]
                # next_trajectories[2, j+1] = ryaw[target_ind + dind]
            else:
                next_trajectories[0, j+1] = rx[len(rx)-1]
                next_trajectories[1, j+1] = ry[len(ry)-1]
                next_trajectories[2, j+1] = ryaw[len(ryaw)-1]

            next_controls = np.tile(np.array([33, 33, 33, 33]).reshape(4, 1), N)

            # print(next_controls)

        next_dind = dind
        if (next_dind - prev_dind) >= 2 :
            index += 1

        x_next = current_states + forward_kinematic(u1, u2, u3, u4, theta) * step_time
        # print(next_dind - prev_dind)

        print(pred_index)
        # print(travel)
        # print(vx, vy)

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