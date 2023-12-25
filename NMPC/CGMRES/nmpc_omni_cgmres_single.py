import numpy as np
import matplotlib.pyplot as plt

from differentialSim import DiffSimulation

from bezier_path import calc_4points_bezier_path

# Arrow function
def plot_arrow(x, y, yaw, length=0.05, width=0.3, fc="b", ec="k"):
    if not isinstance(x, float):
        for (ix, iy, iyaw) in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * np.cos(yaw), length * np.sin(yaw),
            fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)

def plot_robot(x, y, yaw):
    # rotation angle
    theta = 45
    # robot shape
    robot_h = 0.5
    robot_w = 0.5
    # wheel shape
    wh_omnih = 0.025
    wh_omniw = 0.1
    # robot center mass
    center_x = 0
    center_y = 0
    # pos params
    pos = 0.5
    rot = lambda theta: np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])
    rot2 = lambda theta: np.array([[np.cos(theta), np.sin(theta)],
                [-np.sin(theta), np.cos(theta)]])
    robot_shape = np.array([
        [-robot_w,robot_w,robot_w,-robot_w,-robot_w],
        [robot_h,robot_h,-robot_h,-robot_h,robot_h]
        ])
    wheel_shape = np.array([
        [-wh_omniw, wh_omniw, wh_omniw, -wh_omniw, -wh_omniw],
        [wh_omnih, wh_omnih, -wh_omnih,-wh_omnih, wh_omnih]
        ])

    pos_wheel1 = wheel_shape.copy()
    pos_wheel2 = wheel_shape.copy()
    pos_wheel3 = wheel_shape.copy()
    pos_wheel4 = wheel_shape.copy()
    pos_wheel1 = np.dot(pos_wheel1.T, rot(-45)).T
    pos_wheel2 = np.dot(pos_wheel2.T, rot(45)).T
    pos_wheel3 = np.dot(pos_wheel3.T, rot(-45)).T
    pos_wheel4 = np.dot(pos_wheel4.T, rot(45)).T
    pos_wheel1[0, :] += pos
    pos_wheel1[1, :] -= pos
    pos_wheel2[0, :] += pos
    pos_wheel2[1, :] += pos
    pos_wheel3[0, :] -= pos
    pos_wheel3[1, :] += pos
    pos_wheel4[0, :] -= pos
    pos_wheel4[1, :] -= pos


    pos_wheel1 = np.dot(pos_wheel1.T, rot2(yaw)).T
    pos_wheel2 = np.dot(pos_wheel2.T, rot2(yaw)).T
    pos_wheel3 = np.dot(pos_wheel3.T, rot2(yaw)).T
    pos_wheel4 = np.dot(pos_wheel4.T, rot2(yaw)).T


    robot_shape = np.dot(robot_shape.T, rot2(yaw)).T

    plt.plot(robot_shape[0, :]+x, robot_shape[1, :]+y, color="blue")
    plt.plot(pos_wheel1[0, :]+x, pos_wheel1[1, :]+y, color="black")
    plt.plot(pos_wheel2[0, :]+x, pos_wheel2[1, :]+y, color="black")
    plt.plot(pos_wheel3[0, :]+x, pos_wheel3[1, :]+y, color="black")
    plt.plot(pos_wheel4[0, :]+x, pos_wheel4[1, :]+y, color="black")

class OmniModel:
    """
    This class models the kinematics of a omni-wheeled robot.
    """
    def __init__(self, x0, y0, yaw0):
        """
        Initializes the robot model with the given position and orientation.
        
        :param x0: Initial x-coordinate of the robot.
        :param y0: Initial y-coordinate of the robot.
        :param yaw0: Initial yaw (rotation about the z-axis) of the robot.
        """
        self.r = 0.05
        self.L = 0.5

        self.x = x0
        self.y = y0
        self.yaw = yaw0

        self.a1 = np.pi/4
        self.a2 = 3*np.pi/4
        self.a3 = 5*np.pi/4
        self.a4 = 7*np.pi/4

    def rotation_matrix(self, angle):
        rot3dz = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ], dtype=np.float32)

        return rot3dz
    
    def forwardJ_matrix(self, yaw):
        J = (self.r/2)*np.array([
            [-np.sin(yaw+self.a1), -np.sin(yaw+self.a2), -np.sin(yaw+self.a3), -np.sin(yaw+self.a4)],
            [np.cos(yaw+self.a1), np.cos(yaw+self.a2), np.cos(yaw+self.a3), np.cos(yaw+self.a4)],
            [1/(2*self.L), 1/(2*self.L), 1/(2*self.L), 1/(2*self.L)]
        ], dtype=np.float32)

        return J

    def forward_kinematic(self, yaw, u1, u2, u3, u4):
        
        J = self.forwardJ_matrix(yaw)
        u = np.array([u1, u2, u3, u4], dtype=np.float32)

        forvel = J@u

        return forvel


    def inverse_kinematic(self):
        pass

    def update_state(self):
        pass


class HamiltonianEquation:
    """
    This class represents the Hamiltonian function and its derivatives for the NMPC optimization.
    """
    def __init__(self, r, L, Q, R, R_dum, u_max, u_min):
        """
        Initializes the Hamiltonian equation parameters.
        
        :param r: Wheel radius.
        :param lx: Distance from the center of the robot to the wheel along the x-axis.
        :param ly: Distance from the center of the robot to the wheel along the y-axis.
        :param Q: Weight matrix for state error in the cost function.
        :param R: Weight matrix for control input in the cost function.
        :param R_dum: Weight matrix for the dummy control input in the cost function.
        :param u_max: Maximum control input constraints.
        :param u_min: Minimum control input constraints.
        """
        self.q1 = Q[0]
        self.q2 = Q[1]
        self.q3 = Q[2]

        self.r1 = R[0]
        self.r2 = R[1]
        self.r3 = R[2]
        self.r4 = R[3]

        self.r1_dum = R_dum[0]
        self.r2_dum = R_dum[1]
        self.r3_dum = R_dum[2]
        self.r4_dum = R_dum[3]

        self.r = r
        self.L = L
        
        self.a1 = np.pi/4
        self.a2 = 3*np.pi/4
        self.a3 = 5*np.pi/4
        self.a4 = 7*np.pi/4

    def dhdx(self, X, U, X_ref, Lambd):
        dhdx1 = self.q1*(X[0]-X_ref[0])
        dhdx2 = self.q2*(X[1]-X_ref[1])
        
        dhdx31 = self.q3*(X[2]-X_ref[2])
        dhdx32 = self.r*0.5*Lambd[0]*(np.cos(X[2]+self.a1)*(-U[0]+U[2])+np.sin(X[2]+self.a1)*(U[1]-U[3]))
        dhdx33 = self.r*0.5*Lambd[1]*(np.sin(X[2]+self.a1)*(-U[0]+U[2])+np.cos(X[2]+self.a1)*(-U[1]+U[3]))
        dhdx3 = dhdx31+dhdx32+dhdx33

        return dhdx1, dhdx2, dhdx3

    def dhdu(self, X, U, U_ref, Lambd, Mu):
        dhdu1 = (0.25*Lambd[2]*self.r)/self.L+(-0.5*Lambd[0]*self.r*np.sin(X[2]+self.a1))+(0.5*Lambd[1]*self.r*np.cos(X[2]+self.a1))+2*Mu[0]*U[0]+self.r1*2*(U[0]-U_ref[0])
        dhdu2 = (0.25*Lambd[2]*self.r)/self.L+(-0.5*Lambd[0]*self.r*np.cos(X[2]+self.a1))-(0.5*Lambd[1]*self.r*np.sin(X[2]+self.a1))+2*Mu[1]*U[1]+self.r2*2*(U[1]-U_ref[1])
        dhdu3 = (0.25*Lambd[2]*self.r)/self.L+(0.5*Lambd[0]*self.r*np.sin(X[2]+self.a1))-(0.5*Lambd[1]*self.r*np.cos(X[2]+self.a1))+2*Mu[2]*U[2]+self.r3*2*(U[2]-U_ref[2])
        dhdu4 = (0.25*Lambd[2]*self.r)/self.L+(0.5*Lambd[0]*self.r*np.cos(X[2]+self.a1))+(0.5*Lambd[1]*self.r*np.sin(X[2]+self.a1))+2*Mu[3]*U[3]+self.r4*2*(U[3]-U_ref[3])

        return dhdu1, dhdu2, dhdu3, dhdu4

    def dudxdum(self, U_dum, Mu):

        dhdud1 = -self.r1_dum+2*Mu[0]*U_dum[0]
        dhdud2 = -self.r2_dum+2*Mu[1]*U_dum[1]
        dhdud3 = -self.r3_dum+2*Mu[2]*U_dum[2]
        dhdud4 = -self.r4_dum+2*Mu[3]*U_dum[3]

        return dhdud1, dhdud2, dhdud3, dhdud4
    
    def dphidx(self, X, X_ref):

        dphidx1 = self.q1*(X[0]-X_ref[0])
        dphidx2 = self.q2*(X[1]-X_ref[1])
        dphidx3 = self.q3*(X[2]-X_ref[2])

        return dphidx1, dphidx2, dphidx3
    

class NMPCCGMRESSolver:
    """
    This class implements the Nonlinear Model Predictive Control (NMPC) with Continuation Generalized Minimal RESidual (CGMRES) solver.
    """
    def __init__(self) -> None:
        """
        Initializes the NMPC solver with predefined parameters and settings.
        """
        ## NMPC Params
        self.x_dim = 3
        self.u_dim = 4
        self.c_dim = 4
        self.pred_horizons = 10

        ## Continuation Params
        self.ht = 1e-5
        self.zeta = 1/self.ht
        self.alpha = 0.5
        self.tf = 1.0

        ## Tuning Matrix
        self.Q = [75, 75, 100]
        self.R = [0.1, 0.1, 0.1, 0.1]

        ## Dummy Tuning Matrix
        self.R_dum = [1, 1, 1, 1]

        ## Constraint
        self.u_min = [-30, -30, -30, -30]
        self.u_max = [ 30,  30,  30,  30]

        ## Initialize Matrix Solution
        self.dU = np.zeros((self.u_dim, self.pred_horizons))

        self.mpciter = 0

        ## Reference point

        path, _ = calc_4points_bezier_path(
        0, 0, 0,
        3, 3, 0.0,
        1.0
        )
        self.goal = np.vstack([path[:, 0], path[:, 1], np.append(np.arctan2(np.diff(path[:, 1]), np.diff(path[:, 0])), 0.0)])

        self.rx = self.goal[0, :]
        self.ry = self.goal[1, :]
        self.ryaw = self.goal[2, :]

        print(self.goal)

        self.X_r = [3, 3, 0.0]
        self.U_r = [0, 0, 0, 0]

        self.robot_model = OmniModel(0.0, 0.0, 0.0)
        self.hjb_equation = HamiltonianEquation(self.robot_model.r, self.robot_model.L, self.Q, self.R, self.R_dum, self.u_max, self.u_min)

    def calc_state(self, X, U, dt):
        """
        Simulates the state trajectory over the prediction horizon using the provided control inputs.
        
        :param X: Current state trajectory (numpy array).
        :param U: Control inputs trajectory (numpy array).
        :param dt: Time step for discretization.
        :return: Simulated state trajectory.
        """
        for i in range(self.pred_horizons):
            x_dot, y_dot, yaw_dot = self.robot_model.forward_kinematic(X[2, i], U[0, i], U[1, i], U[2, i], U[3, i])

            X[0, i+1] = X[0, i] + x_dot * dt
            X[1, i+1] = X[1, i] + y_dot * dt
            X[2, i+1] = X[2, i] + yaw_dot * dt

        return X

    def calc_costate(self, X, U, Lambda, dt):
        """
        Calculates the costate trajectory using the Hamiltonian dynamics.
        
        :param X: State trajectory (numpy array).
        :param U: Control inputs trajectory (numpy array).
        :param Lambda: Costate trajectory (numpy array).
        :param dt: Time step for discretization.
        :return: Calculated costate trajectory.
        """

        for i in range(self.goal.shape[1]):

            x_N, y_N, yaw_N = self.hjb_equation.dphidx(X[:, -1], self.goal[:, i])

        Lambda[0, -1] = x_N
        Lambda[1, -1] = y_N
        Lambda[2, -1] = yaw_N

        for i in reversed(range(1, self.pred_horizons)):

            dhdx1, dhdx2, dhdx3 = self.hjb_equation.dhdx(
                X[:, i], U[:, i], self.goal[:, i*5], Lambda[:, i+1]
            )

            Lambda[0, i] = Lambda[0, i+1] + dhdx1 * dt
            Lambda[1, i] = Lambda[1, i+1] + dhdx2 * dt
            Lambda[2, i] = Lambda[2, i+1] + dhdx3 * dt
        
        return Lambda

    def calc_f(self, X, U, Lambda, Mu, dt):
        """
        Calculates the optimization function F for the CGMRES solver.
        
        :param X: State trajectory (numpy array).
        :param U: Control inputs trajectory (numpy array).
        :param Lambda: Costate trajectory (numpy array).
        :param Mu: Dual variables trajectory (numpy array).
        :param dt: Time step for discretization.
        :return: Optimization function F.
        """
        X_new = self.calc_state(X, U, dt)

        Lambda_new = self.calc_costate(X_new, U, Lambda, dt)

        F = np.zeros((self.u_dim, self.pred_horizons))

        for i in range(self.pred_horizons):
            dhdu1, dhdu2, dhdu3, dhdu4 = self.hjb_equation.dhdu(
                X[:, i], U[:, i], self.U_r, Lambda_new[:, i+1], Mu[:, i]
            )

            F[0, i] = dhdu1
            F[1, i] = dhdu2
            F[2, i] = dhdu3
            F[3, i] = dhdu4

        F = F.T.ravel().reshape(-1 ,1)

        return F


    def solve_nmpc(self, X, U, Lambda, Mu, time):
        """
        Solves the NMPC optimization problem at the current timestep.
        
        :param X: Current state trajectory (numpy array).
        :param U: Current control inputs trajectory (numpy array).
        :param Lambda: Current costate trajectory (numpy array).
        :param Mu: Current dual variables trajectory (numpy array).
        :param time: Current simulation time.
        :return: Updated control inputs trajectory.
        """
        # Time step horizons
        dt = self.tf * (1-np.exp(-self.alpha*time)) / self.pred_horizons
        # F(x,u,t+h)
        F = self.calc_f(X, U, Lambda, Mu, dt)

        # F(x+hdx,u,t+h)
        x_dot, y_dot, yaw_dot = self.robot_model.forward_kinematic(X[2, 0], U[0, 0], U[1, 0], U[2, 0], U[3, 0])
        X[:, 0] = X[:, 0] + np.array([x_dot, y_dot, yaw_dot]) * self.ht
        Fxt = self.calc_f(X, U, Lambda, Mu, dt)

        # F(x+hdx,u+hdx,t+h)
        Fuxt = self.calc_f(X, U + self.dU*self.ht, Lambda, Mu , dt)

        # Define right-left handside equation
        left = (Fuxt - Fxt) / self.ht
        right = -self.zeta * F - (Fxt - F)/self.ht

        # Define iterations
        m = (self.u_dim ) * self.pred_horizons
        # Define r0
        r0 = right - left
        # print(r0)
        
        # GMRES
        Vm = np.zeros((m, m+1))
        Vm[:, 0:1] = r0 / np.linalg.norm(r0)
        # print("Vm", Vm[:, 0])
        Hm = np.zeros((m+1, m))
        # print(Vm[:, 0])


        for i in range(m):
            Fuxt = self.calc_f(
                X, U + Vm[:, i:i+1].reshape(self.u_dim, self.pred_horizons, order='F') * self.ht,
                Lambda, Mu,
                dt
            )

            Av = (Fuxt - Fxt) / self.ht

            for k in range(i+1):
                Hm[k, i] = np.dot(Av.T, Vm[:, k:k+1])

            temp_vec = np.zeros((m, 1))

            for k in range(i+1):
                temp_vec = temp_vec + np.dot(Hm[k, i], Vm[:, k:k+1])

            v_hat = Av - temp_vec

            Hm[i+1, i] = np.linalg.norm(v_hat)

            Vm[:,i+1:i+2] = v_hat/Hm[i+1, i]

        e = np.zeros((m+1, 1))
        e[0, 0] = 1.0
        gm = np.linalg.norm(r0) * e
        # print(gm)

        UTMat, gm = self.ToUTMat(Hm, gm, m)

        min_y = np.zeros((m, 1))

        for i in range(m):
            min_y[i][0] = (gm[i, 0] - np.dot(UTMat[i:i+1 ,:] ,min_y))/UTMat[i, i]

        dU_new = self.dU + np.dot(Vm[:, 0:m], min_y).reshape(self.u_dim, self.pred_horizons, order='F')

        self.dU = dU_new

        U = U + self.dU * self.ht

        return U
    
    def ToUTMat(self, Hm, gm, m):
        """
        Converts the given Hessenberg matrix to upper triangular form.
        
        :param Hm: The Hessenberg matrix (numpy array).
        :param gm: The GMRES right-hand side vector.
        :param m: Dimension of the subspace for GMRES.
        :return: Upper triangular matrix and updated right-hand side vector.
        """
        for i in range(m):
            nu = np.sqrt(Hm[i, i]**2 + Hm[i+1, i]**2)
            if nu != 0:
                c_i = Hm[i, i]/nu
                s_i = Hm[i+1, i]/nu
            else:
                c_i = 1.0
                s_i = 0.0
            Omega = np.eye(m+1)
            Omega[i, i] = c_i
            Omega[i, i+1] = s_i
            Omega[i+1, i] = -s_i
            Omega[i+1, i+1] = c_i

            Hm = np.matmul(Omega, Hm)
            gm = np.matmul(Omega, gm)

        return Hm, gm




if __name__ == "__main__":

    x_dim = 3
    u_dim = 4
    c_dim = 4

    pred_horizons = 10
    dt = 0.05
    
    x0 = 0.0
    y0 = 0.0
    yaw0 = 0.0

    x_data = []
    y_data = []
    yaw_data = []

    u1_data = []
    u2_data = []
    u3_data = []
    u4_data = []

    count_index = 0
    plot_animation = True

    

    

    X = np.zeros((x_dim, pred_horizons+1))
    X[:, 0] = np.array([x0, y0, yaw0])
    U = np.zeros((u_dim, pred_horizons))
    U[:, 0] = np.array([0.01, 0.01, 0.01, 0.01])

    Lambda = np.zeros((x_dim, pred_horizons+1))
    Mu = np.zeros((u_dim, pred_horizons))

    controller = NMPCCGMRESSolver()
    robot = OmniModel(x0, y0, yaw0)

    tsim = 15.0
    time = 0.0

    x_dot_data = []
    y_dot_data = []
    yaw_dot_data = []


    while time <= tsim:
        
        print("===========Input Control============")
        print(U[:, 0])

        u1_data.append(U[0, 0])
        u2_data.append(U[1, 0])
        u3_data.append(U[2, 0])
        u4_data.append(U[3, 0])


        U = controller.solve_nmpc(X, U, Lambda, Mu, time)
        print("==============Position==============")
        print(X[:, 0])

        x_data.append(X[0, 0])
        y_data.append(X[1, 0])
        yaw_data.append(X[2, 0])

        x_dot, y_dot, yaw_dot = robot.forward_kinematic(X[2, 0], U[0, 0], U[1, 0], U[2, 0], U[3, 0])
        x_dot_data.append(x_dot)
        y_dot_data.append(y_dot)
        yaw_dot_data.append(yaw_dot)


        X[0, 0] = X[0, 0] + x_dot * dt
        X[1, 0] = X[1, 0] + y_dot * dt
        X[2, 0] = X[2, 0] + yaw_dot * dt

        X = controller.calc_state(X, U, dt)

        time += dt
        count_index += 1

    if plot_animation:
        for i in range(count_index):
            plt.clf()
            plt.gcf().canvas.mpl_connect('key_release_event',
                        lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(x_data[-1], y_data[-1], marker="x", color="blue", label="Goal Point")
            plt.plot(np.array(x_data), np.array(y_data), color="red", label="Optimal Trajectory")
            # diff_model.generate_each_wheel_and_draw(x_data[i], y_data[i], yaw_data[i])
            plot_arrow(x_data[i], y_data[i], yaw_data[i])
            plot_robot(x_data[i], y_data[i], yaw_data[i])
            plt.axis("equal")
            plt.legend()
            plt.title("Linear velocity :" + str(round(np.sqrt(x_dot_data[i]**2+y_dot_data[i]**2), 2)) + " m/s")
            plt.grid(True)
            plt.pause(0.01)
        fig, axs = plt.subplots(3)

        axs[0].plot(np.arange(count_index), np.array(x_data))
        axs[0].set_xlabel('t [s]')
        axs[0].set_ylabel('x [m]')
        axs[0].set_title("Point x")
        axs[0].grid(True)

        axs[1].plot(np.arange(count_index), np.array(y_data))
        axs[1].set_xlabel('t [s]')
        axs[1].set_ylabel('y [m]')
        axs[1].set_title("Point y")
        axs[1].grid(True)

        axs[2].plot(np.arange(count_index), np.array(yaw_data))
        axs[2].set_xlabel('t [s]')
        axs[2].set_ylabel(r'$\phi$ [rad]')
        axs[2].set_title("Point yaw")
        axs[2].grid(True)

        plt.tight_layout()
        plt.show()

        fig, axs = plt.subplots(4, figsize=(12, 7))

        axs[0].plot(np.arange(count_index), np.array(u1_data))
        axs[0].set_xlabel('t [s]')
        axs[0].set_ylabel('u1 [m/s]')
        axs[0].set_title("u1")
        axs[0].grid(True)

        axs[1].plot(np.arange(count_index), np.array(u2_data))
        axs[1].set_xlabel('t [s]')
        axs[1].set_ylabel('u2 [rad/s]')
        axs[1].set_title("u2")
        axs[1].grid(True)

        axs[2].plot(np.arange(count_index), np.array(u3_data))
        axs[2].set_xlabel('t [s]')
        axs[2].set_ylabel('u3 [rad/s]')
        axs[2].set_title("u3")
        axs[2].grid(True)

        axs[3].plot(np.arange(count_index), np.array(u4_data))
        axs[3].set_xlabel('t [s]')
        axs[3].set_ylabel('u4 [rad/s]')
        axs[3].set_title("u4")
        axs[3].grid(True)

        plt.tight_layout()
        plt.show()
