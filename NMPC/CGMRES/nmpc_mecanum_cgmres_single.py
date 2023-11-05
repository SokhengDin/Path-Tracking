import numpy as np
import matplotlib.pyplot as plt


class MecanumModel:
    """
    This class models the kinematics of a mecanum-wheeled robot.
    """
    def __init__(self, x0, y0, yaw0):
        """
        Initializes the robot model with the given position and orientation.
        
        :param x0: Initial x-coordinate of the robot.
        :param y0: Initial y-coordinate of the robot.
        :param yaw0: Initial yaw (rotation about the z-axis) of the robot.
        """
        self.r = 0.05
        self.lx = 0.5
        self.ly = 0.5

        self.x = x0
        self.y = y0
        self.yaw = yaw0

    def rotation_matrix(self, angle):
        rot3dz = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ], dtype=np.float32)

        return rot3dz
    
    def inverseJ_matrix(self):
        J = np.array([
            [1, 1, 1, 1],
            [-1, 1, 1, -1],
            [-1/(self.lx+self.ly), 1/(self.lx+self.ly), -1/(self.lx+self.ly), 1/(self.lx+self.ly)]
        ], dtype=np.float32)

        return J

    def forward_kinematic(self, yaw, u1, u2, u3, u4):
        
        rot3dz = self.rotation_matrix(yaw)
        J = self.inverseJ_matrix()
        u = np.array([u1, u2, u3, u4], dtype=np.float32)

        forvel = rot3dz@J@u

        return forvel


    def inverse_kinematic(self):
        pass

    def update_state(self):
        pass


class HamiltonianEquation:
    """
    This class represents the Hamiltonian function and its derivatives for the NMPC optimization.
    """
    def __init__(self, r, lx, ly, Q, R, R_dum, u_max, u_min):
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
        self.lx = lx
        self.ly = ly


    def dhdx(self, X, U, X_ref, Lambd):
        dhdx1 = self.q1*(X[0]-X_ref[0])
        dhdx2 = self.q2*(X[1]-X_ref[1])
        dhdx31 = self.q3*(X[2]-X_ref[2])
        dhdx32 = Lambd[0]*self.r*((U[0]+U[3])*(-np.sin(X[2])+np.cos(X[2]))/4+(U[1]+U[2])*(-np.sin(X[2])-np.cos(X[2]))/4)
        dhdx33 = Lambd[1]*self.r*((U[0]+U[3])*(np.sin(X[2])+np.cos(X[2]))/4 +(U[1]+U[2])*(-np.sin(X[2])+np.cos(X[2]))/4)
        dhdx3 = dhdx31+dhdx32+dhdx33

        return dhdx1, dhdx2, dhdx3

    def dhdu(self, X, U, U_ref, Lambd, Mu):
        dhdu11 = Lambd[0]*self.r*(np.sin(X[2])+np.cos(X[2]))/4+Lambd[1]*self.r*(np.sin(X[2])-np.cos(X[2]))/4
        dhdu12 = -(self.r*Lambd[2])/(4*(self.lx+self.ly))+2*Mu[0]*U[0]+self.r1*(2*U[0]-2*U_ref[0])
        dhdu1 = dhdu11+dhdu12
        dhdu21 = Lambd[0]*self.r*(-np.sin(X[2])+np.cos(X[2]))/4+Lambd[1]*self.r*(np.sin(X[2])+np.cos(X[2]))/4
        dhdu22 = (self.r*Lambd[2])/(4*(self.lx+self.ly))+2*Mu[1]*U[1]+self.r2*(2*U[1]-2*U_ref[1])
        dhdu2 = dhdu21+dhdu22
        dhdu31 = Lambd[0]*self.r*(-np.sin(X[2])+np.cos(X[2]))/4+Lambd[1]*self.r*(np.sin(X[2])+np.cos(X[2]))/4
        dhdu32 = -(self.r*Lambd[2])/(4*(self.lx+self.ly))+2*Mu[2]*U[2]+self.r3*(2*U[2]-2*U_ref[2])
        dhdu3 = dhdu31+dhdu32
        dhdu41 = Lambd[0]*self.r*(np.sin(X[2])+np.cos(X[2]))/4+Lambd[1]*self.r*(np.sin(X[2])-np.cos(X[2]))/4
        dhdu42 = +(self.r*Lambd[2])/(4*(self.lx+self.ly))+2*Mu[3]*U[3]+self.r4*(2*U[3]-2*U_ref[3])
        dhdu4 = dhdu41+dhdu42

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
        self.Q = [1, 1, 1]
        self.R = [1, 1, 1, 1]

        ## Dummy Tuning Matrix
        self.R_dum = [1, 1, 1, 1]

        ## Constraint
        self.u_min = [-30, -30, -30, -30]
        self.u_max = [ 30,  30,  30,  30]

        ## Initialize Matrix Solution
        self.dU = np.zeros((self.u_dim, self.pred_horizons))

        ## Reference point
        self.X_r = [3.0, 2.0, 0.0]
        self.U_r = [0.0, 0.0, 0.0, 0.0]

        self.robot_model = MecanumModel(0.0, 0.0, 0.0)
        self.hjb_equation = HamiltonianEquation(self.robot_model.r, self.robot_model.lx, self.robot_model.ly, self.Q, self.R, self.R_dum, self.u_max, self.u_min)

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
        x_N, y_N, yaw_N = self.hjb_equation.dphidx(X[:, -1], self.X_r)

        Lambda[0, -1] = x_N
        Lambda[1, -1] = y_N
        Lambda[2, -1] = yaw_N

        for i in reversed(range(1, self.pred_horizons)):

            dhdx1, dhdx2, dhdx3 = self.hjb_equation.dhdx(
                X[:, i], U[:, i], self.X_r, Lambda[:, i+1]
            )

            Lambda[0, i] = Lambda[0, i+1] + dhdx1 * dt
            Lambda[1, i] = Lambda[1, i+1] + dhdx2 * dt
            Lambda[2, i] = Lambda[2, i+1] + dhdx3 * dt
        
        return Lambda

    def calc_f(self, X, U, Lamda, Mu, dt):
        """
        Calculates the optimization function F for the CGMRES solver.
        
        :param X: State trajectory (numpy array).
        :param U: Control inputs trajectory (numpy array).
        :param Lambda: Costate trajectory (numpy array).
        :param Mu: Dual variables trajectory (numpy array).
        :param dt: Time step for discretization.
        :return: Optimization function F.
        """
        X = self.calc_state(X, U, dt)

        Lambda = self.calc_costate(X, U, Lamda, dt)

        F = np.zeros((self.u_dim, self.pred_horizons))

        for i in range(self.pred_horizons):
            dhdu1, dhdu2, dhdu3, dhdu4 = self.hjb_equation.dhdu(
                X[:, i], U[:, i], self.U_r, Lambda[:, i], Mu[:, i]
            )

            F[0, i] = dhdu1
            F[1, i] = dhdu2
            F[2, i] = dhdu3
            F[3, i] = dhdu4

        F = F.T.ravel().reshape(-1 ,1)

        return F


    def solve_nmpc(self, X, U, Lamda, Mu, time):
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
        # F(x,u, t+h)
        F = self.calc_f(X, U, Lambda, Mu, dt)
        #F(x+hdx,u,t+h)
        x_dot, y_dot, yaw_dot = self.robot_model.forward_kinematic(X[2, 0], U[0, 0], U[1, 0], U[2, 0], U[3, 0])
        X[:, 0] = X[:, 0] + np.array([x_dot, y_dot, yaw_dot]) * dt
        Fxt = self.calc_f(X, U, Lambda, Mu, dt)
        # F(x+hdx,u+hdu,t+h)
        Fuxt = self.calc_f(X, U + self.dU*self.ht, Lambda, Mu, dt)

        # Define right-left handside equation
        left = (Fuxt - Fxt) / self.ht
        right = -self.zeta * F - (Fxt - F)/self.ht

        # Define iterations
        m = (self.u_dim)*self.pred_horizons
        # Define r0
        r0 = right - right

        # GMRES with givens rotation
        Vm = np.zeros((m, m+1))
        Vm[:, 0:1] = r0 / np.linalg.norm(r0)
        Hm = np.zeros((m+1, m))

        for i in range(m):
            Fuxt = self.calc_f(
                X, U + Vm[:, i:i+1].reshape(self.u_dim, self.pred_horizons, order='F')*self.ht,
                Lambda, Mu, dt
            )

            Av = (Fuxt - Fxt)/self.ht

            for k in range(i+1):
                Hm[k, i] = np.dot(Av.T, Vm[:, k:k+1])

            temp_vec = np.zeros((m, 1))

            for k in range(i+1):
                temp_vec = temp_vec + np.dot(Hm[k, i], Vm[:, k:k+1])
            
            v_hat = Av - temp_vec

            Hm[i+1, i] = np.linalg.norm(v_hat)

            Vm[:,i+1:i+2] = v_hat/Hm[i+1, i]
        
        e = np.zeros((m+1, 1))
        e[0, 0] = 1
        gm = np.linalg.norm(r0)*e

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
    dt = 0.01
    
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
    robot = MecanumModel(x0, y0, yaw0)

    tsim = 10.0
    time = 0.0


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

        X[0, 0] = X[0, 0] + x_dot * dt
        X[1, 0] = X[1, 0] + y_dot * dt
        X[2, 0] = X[2, 0] + yaw_dot * dt

        X = controller.calc_state(X, U, dt)

        time += dt
        count_index += 1
