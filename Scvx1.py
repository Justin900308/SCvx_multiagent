from scipy import signal
from numpy import linalg as LA
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.interpolate import InterpolatedUnivariateSpline

# Input

P_des = np.array(([[12, 12],[15, 15]]))
P_ini = np.array(([[0, 0],[0, 0]]))

Ts = 0.1  # Sampling time
T_end = 12  # End time
T = int(T_end / Ts + 1)  # Total time steps

# Specify the obstacles
Num_obs = 2
obs_center = np.zeros((T, 2 * Num_obs))
for i in range(T):  # obs centers
    obs_center[i, :] = np.array([3 - 0.0 * i, 2 - 0.0 * i, 7 + 0.0 * i, 10 + 0.0 * i])

# Num_obs = 1
# obs_center = np.zeros((T, 2 * Num_obs))
# for i in range(T):  # obs centers
#     obs_center[i, :] = np.array([5 - 0.0 * i, 5 - 0.0 * i])

R = np.array(([3, 2]))  # obs r
Num_agent = 2  # number of agent


def trajectory_gen(P_des, P_ini, obs_center, R, T, Num_agent, Num_obs):
    # Define system matrices
    A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
    B = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
    C = np.eye(4)
    D = np.zeros((4, 2))

    # Continuous-time system
    sys = signal.StateSpace(A, B, C, D)

    # Discretize the system
    sysd = sys.to_discrete(Ts)
    Ad = sysd.A
    Bd = sysd.B

    m=2
    n=4
    # Initialize state and input arrays
    count = 0
    X = np.zeros((n * Num_agent, T+1))  # State trajectory
    X[:, 0] = np.array([P_ini[0, 0], P_ini[0, 1], 0, 0,
                        P_ini[1, 0], P_ini[1, 1], 0, 0])  # Initial state

    u = -np.zeros((m * Num_agent, T))  # Initialize u as an empty 2x0 array to store control inputs

    # Compute the initial trajectory
    # for i in range(T - 1):
    #     X[:, i + 1] = np.zeros((Num_agent * 4))

    # Plot the trajectory
    plt.plot(X[0, :], X[1, :], '.')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Trajectory Plot')
    plt.grid(True)
    # plt.show()

    #######################################################################

    alpha = 1
    r_default = 0.1

    lambda_param = 10000

    # Time length N and trajectory state X
    N = X.shape[1]

    # Convert u to a numpy array for proper matrix operations
    # u = np.array(u)
    # u = np.reshape(u, (2, int(T_end / Ts + 1)))

    # Plot the obstacle (circle)
    theta = np.linspace(0, 2 * np.pi, 201)

    for i in range(int(T)):
        for j in range(Num_agent):
            plt.plot(X[n * j:j + 1, i], X[n * j + 1:j + 2, i], 'r.', markersize=2)
            for k in range(Num_obs):
                x_theta = R[k] * np.cos(theta)
                y_theta = R[k] * np.sin(theta)
                plt.plot(obs_center[i, 2 * k:2 * k + 1] + x_theta, obs_center[i, 2 * k + 1:2 * k + 2] + y_theta)

    # Start the iterative optimization process
    linear_cost = np.zeros((201, 1))
    for interation in range(30):

        # Define variables for optimization
        w1 = cp.Variable((m , N - 1))
        v1 = cp.Variable((n , N - 1))
        d1 = cp.Variable((n , N))
        s1 = cp.Variable(((N) * Num_obs, 1))

        w2 = cp.Variable((m * 1, N - 1))
        v2 = cp.Variable((n * 1, N - 1))
        d2 = cp.Variable((n * 1, N))
        s2 = cp.Variable(((N) * Num_obs, 1))
        # Define the cost function

        Linear_cost = (1 * cp.norm(((u[0:2,:] + w1)), 2) + 1 * lambda_param * cp.sum(
            cp.sum(cp.abs(v1))) + 1 * lambda_param * cp.sum(cp.pos(s1)) +
                       1 * cp.norm(((u[2:4,:] + w2)), 2) + 1 * lambda_param * cp.sum(
            cp.sum(cp.abs(v2))) + 1 * lambda_param * cp.sum(cp.pos(s2)))


        # Linear_cost = (cp.sum(cp.abs(u[0:2,:] + w1)) + 1 * lambda_param * cp.sum(
        #     cp.sum(cp.abs(v1))) + 1 * lambda_param * cp.sum(cp.pos(s1)) +
        #                cp.sum(cp.abs(u[2:4,:] + w2)) + 1 * lambda_param * cp.sum(
        #     cp.sum(cp.abs(v2))) + 1 * lambda_param * cp.sum(cp.pos(s2)))
        #


        constraints = [d1[:, 0] == np.zeros(n*1)]
        #constraints = [d2[:, 0] == np.zeros(n * 1)]
        constraints.append(d2[:, 0] == np.zeros(n * 1))
        E = np.eye(n)

        for i in range(N - 1):
            j = 0
            constraints.append(
                X[n * j:n * j + n, i + 1] + d1[:, i + 1] == (
                        Ad @ X[n * j:n * j + n, i] + Ad @ d1[:, i]) + (
                        Bd @ u[m * j:m * j + m, i] + Bd @ w1[:, i]) + E @ v1[:, i])
            j = 1
            constraints.append(
                X[n * j:n * j + n, i + 1] + d2[:, i + 1] == (
                        Ad @ X[n * j:n * j + n, i] + Ad @ d2[:, i]) + (
                        Bd @ u[m * j:m * j + m, i] + Bd @ w2[:, i]) + E @ v2[:, i])
            # constraints.append(cp.abs(w[0, i]) <= r_default)
            j = 0
            constraints.append(w1[:, i] <= r_default)
            constraints.append(-r_default <= w1[0, i])
            constraints.append(w1[:, i] <= r_default)
            constraints.append(-r_default <= w1[1, i])
            j = 1
            constraints.append(w2[:, i] <= r_default)
            constraints.append(-r_default <= w2[0, i])
            constraints.append(w2[:, i] <= r_default)
            constraints.append(-r_default <= w2[1, i])
            # Obstacle avoidance constraint
            for k in range(Num_obs):
                J = 0
                # constraints.append(
                #     2 * R[k] - cp.norm(X[n * J:n*J + 2, i] - obs_center[i, 2 * k:+2 * k + 2], 2) - (
                #             X[n * J:n*J + 2, i] - obs_center[i, 2 * k:+2 * k + 2]).T @ (
                #             X[n * J:n*J + 2, i] + d[n * J:n*J + 2, i] - obs_center[i, 2 * k:+2 * k + 2]) /
                #     cp.norm(X[n * J:n*J + 2, i] - obs_center[i, 2 * k:+2 * k + 2], 2) <= 0)
                constraints.append(
                    2 * R[k] - cp.norm(X[0:2, i] - obs_center[i, 2 * k:+2 * k + 2], 2) -
                    (X[0:2, i] - obs_center[i, 2 * k:+2 * k + 2]).T @
                    (X[0:2, i] + d1[0:2, i] - obs_center[i, 2 * k:+2 * k + 2]) /
                    cp.norm(X[0:2, i] - obs_center[i, 2 * k:+2 * k + 2], 2) <= s1[(N) * k + i, 0]
                )
                constraints.append(
                    2 * R[k] - cp.norm(X[4:6, i] - obs_center[i, 2 * k:+2 * k + 2], 2) -
                    (X[4:6, i] - obs_center[i, 2 * k:+2 * k + 2]).T @
                    (X[4:6, i] + d2[0:2, i] - obs_center[i, 2 * k:+2 * k + 2]) /
                    cp.norm(X[4:6, i] - obs_center[i, 2 * k:+2 * k + 2], 2) <= s2[(N) * k + i, 0]

                )
                # constraints.append(
                #     2 * R[k] - cp.norm(X[n * J:n*J + 2, i] - obs_center[i, 2 * k:+2 * k + 2], 2) - (
                #             X[n * J:n*J + 2, i] - obs_center[i, 2 * k:+2 * k + 2]).T @ (
                #             X[n * J:n*J + 2, i] + d[n * J:n*J + 2, i] - obs_center[i, 2 * k:+2 * k + 2]) /
                #     cp.norm(X[n * J:n*J + 2, i] - obs_center[i, 2 * k:+2 * k + 2], 2) <= s[(N)*k+i,J]*0)

                constraints.append(s1[(N) * k + i, 0] >= 0)
                constraints.append(s2[(N) * k + i, 0] >= 0)

        # Terminal condition
        constraints.append(X[0:4, N - 1] + d1[:, N - 1] == np.array([P_des[0, 0], P_des[0, 1], 0, 0]))
        constraints.append(X[4:8, N - 1] + d2[:, N - 1] == np.array([P_des[1, 0], P_des[1, 1], 0, 0]))
        # Define the problem
        problem = cp.Problem(cp.Minimize(Linear_cost), constraints)
        if interation >=5:
            kkkk=5
        # Solve the optimization problem
        problem.solve(solver=cp.CLARABEL)
        #problem.solve(solver=cp.SCS)
        # Update the variables after solving
        w1_val=w1.value
        w2_val = w2.value
        w_val=np.vstack((w1_val,w2_val))

        v1_val=v1.value
        v2_val = v2.value
        v_val=np.vstack((v1_val,v2_val))

        d1_val=d1.value
        d2_val = d2.value
        d_val=np.vstack((d1_val,d2_val))

        s1_val=s1.value
        s2_val = s2.value
        s_val=np.vstack((s1_val,s2_val))

        # U_val = U.value
        j = 1

        linear_cost[interation, 0] = (1 * LA.norm(((u + w_val) * Ts), ord=1) +
                                      1 * lambda_param * np.sum(np.sum(np.abs(v_val))) +
                                      1 * lambda_param * np.sum(s_val))

        rho0 = 0
        rho1 = 0.25
        rho2 = 0.7
        if interation >= 1:
            delta_L = (linear_cost[interation, 0] - linear_cost[interation - 1, 0]) / linear_cost[interation, 0]
        else:
            delta_L = 1
        #print(np.abs(delta_L))
        if np.abs(delta_L) <= rho0:
            r_default = np.max((r_default, 1.5))
            X = X + d_val
            u = u + w_val
        elif np.abs(delta_L) <= rho1:
            r_default = r_default
            X = X + d_val
            u = u + w_val
        elif np.abs(delta_L) <= rho2:
            r_default = r_default / 3.2
            X = X + d_val
            u = u + w_val
        else:
            X = X + d_val
            u = u + w_val
            r_default = 1.5
        #r_default = 3.5
        # Update the trajectory

        # Plot the updated trajectory
        for i in range(Num_agent):
            if i == 0:
                plt.plot(X[0:1, :], X[1:2, :], 'g.', markersize=2)
            elif i==1:
                plt.plot(X[4:5, :], X[5:6, :], 'b.', markersize=2)
            plt.xlim((-10, 20))
            plt.ylim((-10, 20))
            plt.pause(0.001)
        # plt.clf()
        ss = np.zeros(((N-1)*Num_obs,Num_agent))

        ss_max = np.array([0])
        for i in range(T):
            for j in range(Num_agent):
                for k in range(Num_obs):
                    ss[(N-1)*k+i,J] = LA.norm(X[n * J:n*J + 2, i] - obs_center[i, 2 * k:+2 * k + 2], 2) - R[k]
        print(np.min(ss))
        if (np.min(ss) >= 0) and (interation > 10):

            break
        # if  (interation > 10):
        #
        #     break
        # print(np.min(ss) )
        print('Iteration:  ', interation + 1)

    # Final trajectory plot
    plt.clf()
    return (X, u)

X, u = trajectory_gen(P_des, P_ini, obs_center, R, T, Num_agent, Num_obs)
######################### Main
# X, u = trajectory_gen(P_des, P_ini, obs_center, R, T, Num_agent, Num_obs)
#
# T_series = np.zeros((1, T + 1))
# for i in range(T + 1):
#     T_series[0, i] = Ts * i
#
# Discre_x = X[0:1, :]
# Discre_y = X[1:2, :]
#
# spline_x = InterpolatedUnivariateSpline(T_series, X[0:1, :], k=2)
# spline_y = InterpolatedUnivariateSpline(T_series, X[1:2, :], k=2)
# spline_u = InterpolatedUnivariateSpline(T_series, X[2:3, :], k=2)
# spline_v = InterpolatedUnivariateSpline(T_series, X[3:4, :], k=2)
#
# t_continuous = np.linspace(T_series[0], T_series[-1], 1000)
#
# X_continuous = spline_x(t_continuous)
# Y_continuous = spline_y(t_continuous)
# u_continuous = spline_u(t_continuous)
# v_continuous = spline_v(t_continuous)
#
# # Create a 2x2 grid of subplots
# fig, axs = plt.subplots(2, 2, figsize=(10, 8))
#
# # Plot each subplot
# axs[0, 0].plot(t_continuous[0, :], X_continuous[0, :], 'r')
# axs[0, 0].set_title('X Position data')
#
# axs[0, 1].plot(t_continuous[0, :], Y_continuous[0, :], 'r')
# axs[0, 1].set_title('Y Position data')
#
# axs[1, 0].plot(t_continuous[0, :], u_continuous[0, :], 'r')
# axs[1, 0].set_title('u data')
#
# axs[1, 1].plot(t_continuous[0, :], v_continuous[0, :], 'r')
# axs[1, 1].set_title('v data')
#
# # Add some space between plots and display
# plt.tight_layout()
# plt.show()

for i in range(int(T)):
    theta = np.linspace(0, 2 * np.pi, 201)
    for j in range(Num_agent):
        if j==0:
            plt.plot(X[0:1, :i], X[1:2, :i], 'r.', markersize=2)
        elif j==1:
            plt.plot(X[4:5, :i], X[5:6, :i], 'g.', markersize=2)


        for k in range(Num_obs):
            x_theta = R[k] * np.cos(theta)
            y_theta = R[k] * np.sin(theta)
            plt.plot(obs_center[i, 2 * k:2 * k + 1] + x_theta, obs_center[i, 2 * k + 1:2 * k + 2] + y_theta)
            plt.xlim((-10, 20))
            plt.ylim((-10, 20))
    plt.pause(0.001)
    plt.clf()

# for i in range(int(T)):
#     plt.plot(X[0, :i], X[1, :i], '.')
#     theta = np.linspace(0, 2 * np.pi, 201)
#     x_theta = R * np.cos(theta)
#     y_theta = R * np.sin(theta)
#     plt.plot(obs_center[i, 0] + x_theta, obs_center[i, 1] + y_theta)
#     plt.xlim((P_ini[0, 0] - 0, P_des[0, 0] + 5))
#     plt.ylim((P_ini[0, 1] - 0, P_des[0, 1] + 5))
#     plt.plot(P_des[0, 0], P_des[0, 1], 'r.', markersize=10)
#     plt.pause(0.001)
#     plt.clf()
