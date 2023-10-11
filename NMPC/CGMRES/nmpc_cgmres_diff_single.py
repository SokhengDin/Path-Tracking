import numpy as np
import matplotlib.pyplot as plt

from differentialSim import DiffSimulation

np.set_printoptions(precision=5)

class DifferentialModel:

    def __init__(self, x0, y0, yaw0):
        self.x = x0
        self.y = y0
        self.yaw = yaw0

    def forward_kinematic(self, yaw, u1, u2):
        vx = u1 * np.cos(yaw)
        vy = u1 * np.sin(yaw)
        vyaw = u2

        return vx, vy, vyaw

    def update_state(self, yaw, u1, u2, dt):
        x_dot, y_dot, yaw_dot = self.forward_kinematic(yaw, u1, u2)

        self.x = self.x + x_dot * dt
        self.y = self.y + y_dot * dt
        self.yaw = self.yaw + yaw_dot * dt

class HJBEquation:

    def __init__(self, q1, q2, q3, r1, r2, u_max, u_min):

        self.q1 = q1
        self.q2 = q2
        self.q3 = q3

        self.r1 = r1
        self.r2 = r2

        self.u1_min = u_min[0]
        self.u2_min = u_min[0]

        self.u1_max = u_max[0]
        self.u2_max = u_max[1]

    def dhdx(self, x, y, yaw, x_r, y_r, yaw_r, u1, lam1, lam2):

        dhdx1 = (x-x_r)*self.q1
        dhdx2 = (y-y_r)*self.q2
        dhdx3 = (yaw-yaw_r)*self.q3-u1*lam1*np.sin(yaw)+u1*lam2*np.cos(yaw)

        return dhdx1, dhdx2, dhdx3

    def dhdu(self, yaw, u1, u2, u1_r, u2_r, lam1, lam2, lam3, mu1, mu2):

        dhdu1 = self.r1*(u1-u1_r)+lam1*np.cos(yaw)+lam2*np.sin(yaw)+2*mu1*u1*(u1+(self.u1_max+self.u1_min)/2)
        dhdu2 = self.r2*(u2-u2_r)+lam3+2*mu2*(u2+(self.u2_max+self.u2_min)/2)

        return dhdu1, dhdu2

    def dphidx(self, x, y, yaw, x_r, y_r, yaw_r):
        dphidx1 = self.q1*(x-x_r)
        dphidx2 = self.q2*(y-y_r)
        dphidx3 = self.q3*(yaw-yaw_r)

        return dphidx1, dphidx2, dphidx3
    
class NMPCCGMRES:

    def __init__(self):

        ## NMPC params
        self.x_dim = 3
        self.u_dim = 2
        self.c_dim = 2
        self.pred_horizons = 10

        ### Continuations Params
        self.ht = 1e-5
        self.zeta = 1/self.ht
        self.alpha = 0.5
        self.tf = 1.0

        ### Matrix

        self.q1 = 1
        self.q2 = 6
        self.q3 = 1
        self.r1 = 1
        self.r2 = 0.1

        ### Constraint
        self.u_min = [ 0,  0]
        self.u_max = [ 2,  1.57]
        self.u_dummy = [0.01, 0.01]

        ### Initialize matrix solution
        self.dU = np.zeros((self.u_dim, self.pred_horizons))

        ### Reference point
        self.X_r = [3, 2, 0.0]
        self.U_r = [0, 0]

        self.robot_model = DifferentialModel(0.0, 0.0, 0.0)
        self.hjb_equation = HJBEquation(self.q1, self.q2, self.q3, self.r1, self.r2, self.u_max, self.u_min)

    def calc_state(self, X, U, dt):

        for i in range(self.pred_horizons):
            x_dot, y_dot, yaw_dot = self.robot_model.forward_kinematic(X[2, i], U[0, i], U[1, i])
            
            X[0, i+1] = X[0, i] + x_dot * dt
            X[1, i+1] = X[1, i] + y_dot * dt
            X[2, i+1] = X[2, i] + yaw_dot * dt

        return X
    
    def calc_costate(self, X, U, Lambda, dt):

        x_N, y_N, yaw_N = self.hjb_equation.dphidx(X[0, -1], X[1, -1], X[2, -1],
                                                   self.X_r[0], self.X_r[1], self.X_r[2])
        Lambda[0, -1] = x_N
        Lambda[1, -1] = y_N
        Lambda[2, -1] = yaw_N


        for i in reversed(range(1, self.pred_horizons)):

            dhdx1, dhdx2, dhdx3 = self.hjb_equation.dhdx(
                X[0, i], X[1, i], X[2, i],
                self.X_r[0], self.X_r[1], self.X_r[2],
                U[0, i], Lambda[0, i+1], Lambda[1, i+1]
            )

            Lambda[0, i] = Lambda[0, i+1] + dhdx1 * dt
            Lambda[1, i] = Lambda[1, i+1] + dhdx2 * dt
            Lambda[2, i] = Lambda[2, i+1] + dhdx3 * dt
    

        return Lambda
    
    def calc_f(self, X, U, Lambda, Mu, dt):

        X_new = self.calc_state(X, U, dt)

        Lambda_new = self.calc_costate(X_new, U, Lambda, dt)

        F = np.zeros((self.u_dim, self.pred_horizons))

        for i in range(self.pred_horizons):
            dhdu1, dhdu2 = self.hjb_equation.dhdu(X_new[2, i], U[0, i], U[1, i],
                                                  self.U_r[0], self.U_r[1],
                                                  Lambda_new[0, i+1], Lambda_new[1, i+1], Lambda_new[2, i+1],
                                                  Mu[0, i], Mu[1, i])
            F[0, i] = dhdu1
            F[1, i] = dhdu2
            
        F = F.T.ravel().reshape(-1, 1)

        return F
    
    def solve_nmpc(self, X, U, Lambda, Mu, time):
        # Time step horizons
        dt = self.tf * (1-np.exp(-self.alpha*time)) / self.pred_horizons
        # F(x,u,t+h)
        F = self.calc_f(X, U, Lambda, Mu, dt)
        # F(x+hdx,u,t+h)
        x_dot, y_dot, yaw_dot = self.robot_model.forward_kinematic(X[2, 0], U[0, 0], U[1, 0])
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
    u_dim = 2
    c_dim = 2

    pred_horizons = 10
    dt = 0.01
    
    x0 = 0.0
    y0 = 0.0
    yaw0 = 0.0

    x_data = []
    y_data = []
    yaw_data = []

    u1_data = []
    u2_data = []

    count_index = 0
    plot_animation = True

    diff_model = DiffSimulation()


    X = np.zeros((x_dim, pred_horizons+1))
    X[:, 0] = np.array([x0, y0, yaw0])
    U = np.zeros((u_dim, pred_horizons))
    U[:, 0] = np.array([0.01, 0.01])

    Lambda = np.zeros((x_dim, pred_horizons+1))
    Mu = np.zeros((u_dim, pred_horizons))

    robot = DifferentialModel(x0, y0, yaw0)
    controller = NMPCCGMRES()

    tsim = 10.0
    time = 0.0


    while time <= tsim:
        
        print("===========Input Control============")
        print(U[:, 0])

        u1_data.append(U[0, 0])
        u2_data.append(U[1, 0])

        U = controller.solve_nmpc(X, U, Lambda, Mu, time)
        print("==============Position==============")
        print(X[:, 0])

        x_data.append(X[0, 0])
        y_data.append(X[1, 0])
        yaw_data.append(X[2, 0])

        x_dot, y_dot, yaw_dot = robot.forward_kinematic(X[2, 0], U[0, 0], U[1, 0])

        X[0, 0] = X[0, 0] + x_dot * dt
        X[1, 0] = X[1, 0] + y_dot * dt
        X[2, 0] = X[2, 0] + yaw_dot * dt

        X = controller.calc_state(X, U, dt)

        time += dt
        count_index += 1

    if plot_animation:
        plt.figure(figsize=(12, 7))
        for i in range(count_index):
            plt.clf()
            plt.gcf().canvas.mpl_connect('key_release_event',
                        lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(x_data[-1], y_data[-1], marker="x", color="blue", label="Goal Point")
            plt.plot(np.array(x_data), np.array(y_data), color="red", label="Optimal Trajectory")
            diff_model.generate_each_wheel_and_draw(x_data[i], y_data[i], yaw_data[i])
            plt.axis("equal")
            plt.legend()
            plt.title("Linear velocity :" + str(round(u1_data[i], 2)) + " m/s")
            plt.grid(True)
            plt.pause(0.01)

        fig, axes = plt.subplots(2, 3, layout="constrained", figsize=(12, 7))
        axes[0, 0].plot(np.array(x_data), np.array(y_data), color="red")
        axes[0, 0].set_xlabel('x [m]')
        axes[0, 0].set_ylabel('y [m]')
        axes[0, 0].set_title("Optimal Trajectory")
        axes[0, 0].grid(True)
        axes[0, 1].plot(np.arange(count_index), np.array(x_data), color="green")
        axes[0, 1].set_xlabel('t [s]')
        axes[0, 1].set_ylabel('x [m]')
        axes[0, 1].set_title("Point x")
        axes[0, 1].grid(True)
        axes[0, 2].plot(np.arange(count_index), np.array(y_data), color="orange")
        axes[0, 2].set_xlabel('t [s]')
        axes[0, 2].set_ylabel('y [m]')
        axes[0, 2].set_title("Point y")
        axes[0, 2].grid(True)
        axes[1, 0].plot(np.arange(count_index), np.array(yaw_data), color="blue")
        axes[1, 0].set_xlabel('t [s]')
        axes[1, 0].set_ylabel(r'$\phi$ [rad]')
        axes[1, 0].set_title("Point yaw")
        axes[1, 0].grid(True)
        axes[1, 1].plot(np.arange(count_index), np.array(u1_data), color="red")
        axes[1, 1].set_xlabel('t [s]')
        axes[1, 1].set_ylabel('u1 [m/s]')
        axes[1, 1].set_title("u1")
        axes[1, 1].grid(True)
        axes[1, 2].plot(np.arange(count_index), np.array(u2_data), color="red")
        axes[1, 2].set_xlabel('t [s]')
        axes[1, 2].set_ylabel('u2 [rad/s]')
        axes[1, 2].set_title("u2")
        axes[1, 2].grid(True)
        plt.show()


