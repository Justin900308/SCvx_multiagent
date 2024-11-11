from scipy import signal
from numpy import linalg as LA
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.interpolate import InterpolatedUnivariateSpline

# Input

P_des = np.array(([[15, 15], [0, 0], [0, 15], [15, 0]]))
P_ini = np.array(([[0, 0], [15, 15], [15, 0], [0, 15]]))

Ts = 0.1  # Sampling time
T_end = 12  # End time
T = int(T_end / Ts + 1)  # Total time steps

# Specify the obstacles
Num_obs = 2
obs_center = np.zeros((T, 2 * Num_obs))
for i in range(T):  # obs centers
    obs_center[i, :] = np.array([4 - 0.0 * i, 4 + 0.0 * i, 8 + 0.0 * i, 8 + 0.00 * i])

# Num_obs = 1
# obs_center = np.zeros((T, 2 * Num_obs))
# for i in range(T):  # obs centers
#     obs_center[i, :] = np.array([5 - 0.0 * i, 5 - 0.0 * i])

R = np.array(([2, 2]))  # obs r
Num_agent = 4  # number of agent


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

    m = 2
    n = 4
    # Initialize state and input arrays
    count = 0
    X = np.zeros((n * Num_agent, T + 1))  # State trajectory
    X[:, 0] = np.array([P_ini[0, 0], P_ini[0, 1], 0, 0,
                        P_ini[1, 0], P_ini[1, 1], 0, 0,
                        P_ini[2, 0], P_ini[2, 1], 0, 0,
                        P_ini[3, 0], P_ini[3, 1], 0, 0])  # Initial state

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
        d = cp.Variable((n * Num_agent, N))
        v = cp.Variable((n * Num_agent, N - 1))
        w1 = cp.Variable((m, N - 1))
        w2 = cp.Variable((m, N - 1))
        w3 = cp.Variable((m, N - 1))
        w4 = cp.Variable((m, N - 1))
        s1 = cp.Variable(((N) * (Num_obs + Num_agent - 1), 1))
        s2 = cp.Variable(((N) * (Num_obs + Num_agent - 1), 1))
        s3 = cp.Variable(((N) * (Num_obs + Num_agent - 1), 1))
        s4 = cp.Variable(((N) * (Num_obs + Num_agent - 1), 1))
        # Define the cost function

        # Linear_cost = (1 * cp.norm(((u[0:2,:] + w1)), 2) + 1 * lambda_param * cp.sum(
        #     cp.sum(cp.abs(v1))) + 1 * lambda_param * cp.sum(cp.pos(s1)) +
        #                1 * cp.norm(((u[2:4,:] + w2)), 2) + 1 * lambda_param * cp.sum(
        #     cp.sum(cp.abs(v2))) + 1 * lambda_param * cp.sum(cp.pos(s2)))
        Linear_cost = (1 * cp.norm(((u[0:2, :] + w1)), 2) + 1 * lambda_param * cp.sum(
            cp.sum(cp.abs(v))) + 1 * lambda_param * cp.sum(cp.pos(s1)) +
                       1 * cp.norm(((u[2:4, :] + w2)), 2) + 1 * lambda_param * cp.sum(cp.pos(s2)) +
                       1 * cp.norm(((u[4:6, :] + w3)), 2) + 1 * lambda_param * cp.sum(cp.pos(s3)) +
                       1 * cp.norm(((u[6:8, :] + w4)), 2) + 1 * lambda_param * cp.sum(cp.pos(s4)))

        # Linear_cost = (cp.sum(cp.abs(u[0:2,:] + w1)) + 1 * lambda_param * cp.sum(
        #     cp.sum(cp.abs(v1))) + 1 * lambda_param * cp.sum(cp.pos(s1)) +
        #                cp.sum(cp.abs(u[2:4,:] + w2)) + 1 * lambda_param * cp.sum(
        #     cp.sum(cp.abs(v2))) + 1 * lambda_param * cp.sum(cp.pos(s2)))
        #

        constraints = [d[0:4, 0] == np.zeros(n * 1)]
        constraints.append(d[4:8, 0] == np.zeros(n * 1))
        constraints.append(d[8:12, 0] == np.zeros(n * 1))
        constraints.append(d[12:16, 0] == np.zeros(n * 1))
        E = np.eye(n)
        for ii in range(N * (Num_obs + Num_obs - 1)):
            constraints.append(s1[ii, 0] >= 0)
            constraints.append(s2[ii, 0] >= 0)
            constraints.append(s3[ii, 0] >= 0)
            constraints.append(s4[ii, 0] >= 0)
        for i in range(N - 1):
            j = 0
            constraints.append(
                X[n * j:n * j + n, i + 1] + d[n * j:n * j + n, i + 1] == (
                        Ad @ X[n * j:n * j + n, i] + Ad @ d[n * j:n * j + n, i]) + (
                        Bd @ u[m * j:m * j + m, i] + Bd @ w1[:, i]) + E @ v[n * j:n * j + n, i])
            j = 1
            constraints.append(
                X[n * j:n * j + n, i + 1] + d[n * j:n * j + n, i + 1] == (
                        Ad @ X[n * j:n * j + n, i] + Ad @ d[n * j:n * j + n, i]) + (
                        Bd @ u[m * j:m * j + m, i] + Bd @ w2[:, i]) + E @ v[n * j:n * j + n, i])
            j = 2
            constraints.append(
                X[n * j:n * j + n, i + 1] + d[n * j:n * j + n, i + 1] == (
                        Ad @ X[n * j:n * j + n, i] + Ad @ d[n * j:n * j + n, i]) + (
                        Bd @ u[m * j:m * j + m, i] + Bd @ w3[:, i]) + E @ v[n * j:n * j + n, i])
            j = 3
            constraints.append(
                X[n * j:n * j + n, i + 1] + d[n * j:n * j + n, i + 1] == (
                        Ad @ X[n * j:n * j + n, i] + Ad @ d[n * j:n * j + n, i]) + (
                        Bd @ u[m * j:m * j + m, i] + Bd @ w4[:, i]) + E @ v[n * j:n * j + n, i])
            # constraints.append(cp.abs(w[0, i]) <= r_default)
            j = 0
            constraints.append(w1[:, i] <= r_default)
            constraints.append(-r_default <= w1[0, i])
            constraints.append(-r_default <= w1[1, i])
            j = 1
            constraints.append(w2[:, i] <= r_default)
            constraints.append(-r_default <= w2[0, i])
            constraints.append(-r_default <= w2[1, i])
            j = 2
            constraints.append(w3[:, i] <= r_default)
            constraints.append(-r_default <= w3[0, i])
            constraints.append(-r_default <= w3[1, i])
            j = 3
            constraints.append(w4[:, i] <= r_default)
            constraints.append(-r_default <= w4[0, i])
            constraints.append(-r_default <= w4[1, i])

            # Obstacle avoidance constraint
            Current_obs_list = np.array([obs_center[i, 0:2],
                                         obs_center[i, 2:4],
                                         X[0:2, i],
                                         X[4:6, i],
                                         X[8:10, i],
                                         X[12:14, i]])
            R_list = np.array([R[0]+1, R[1]+1, 2, 2, 2, 2])

            count_agen0 = 0
            count_agen1 = 0
            count_agen2 = 0
            count_agen3 = 0
            for k in range(Num_obs + Num_agent - 1):
                if k != Num_obs + 0: # agent 0 obs constraints
                    constraints.append(
                        2 * R_list[k] - cp.norm(X[0:2, i] - Current_obs_list[k, :], 2) -
                        (X[0:2, i] - Current_obs_list[k, :]).T @
                        (X[0:2, i] + d[0:2, i] - Current_obs_list[k, :]) /
                        cp.norm(X[0:2, i] - Current_obs_list[k, :], 2) <= s1[(N) * count_agen0 + i, 0]
                    )
                    count_agen0 = count_agen0 + 1
                if k != Num_obs + 1:# agent 1 obs constraints
                    constraints.append(
                        2 * R_list[k] - cp.norm(X[4:6, i] - Current_obs_list[k, :], 2) -
                        (X[4:6, i] - Current_obs_list[k, :]).T @
                        (X[4:6, i] + d[4:6, i] - Current_obs_list[k, :]) /
                        cp.norm(X[4:6, i] - Current_obs_list[k, :], 2) <= s2[(N) * count_agen1 + i, 0]

                    )
                    count_agen1 = count_agen1 + 1
                if k != Num_obs + 2:# agent 2 obs constraints
                    constraints.append(
                        2 * R_list[k] - cp.norm(X[8:10, i] - Current_obs_list[k, :], 2) -
                        (X[8:10, i] - Current_obs_list[k, :]).T @
                        (X[8:10, i] + d[8:10, i] - Current_obs_list[k, :]) /
                        cp.norm(X[8:10, i] - Current_obs_list[k, :], 2) <= s2[(N) * count_agen1 + i, 0]

                    )
                    count_agen2 = count_agen2 + 1
                if k != Num_obs + 3:# agent 3 obs constraints
                    constraints.append(
                        2 * R_list[k] - cp.norm(X[12:14, i] - Current_obs_list[k, :], 2) -
                        (X[12:14, i] - Current_obs_list[k, :]).T @
                        (X[12:14, i] + d[12:14, i] - Current_obs_list[k, :]) /
                        cp.norm(X[12:14, i] - Current_obs_list[k, :], 2) <= s2[(N) * count_agen1 + i, 0]

                    )
                    count_agen3 = count_agen3 + 1

        # Terminal condition
        constraints.append(X[0:4, N - 1] + d[0:4, N - 1] == np.array([P_des[0, 0], P_des[0, 1], 0, 0]))
        constraints.append(X[4:8, N - 1] + d[4:8, N - 1] == np.array([P_des[1, 0], P_des[1, 1], 0, 0]))
        constraints.append(X[8:12, N - 1] + d[8:12, N - 1] == np.array([P_des[2, 0], P_des[2, 1], 0, 0]))
        constraints.append(X[12:16, N - 1] + d[12:16, N - 1] == np.array([P_des[3, 0], P_des[3, 1], 0, 0]))
        # Define the problem
        problem = cp.Problem(cp.Minimize(Linear_cost), constraints)
        if interation >= 5:
            kkkk = 5
        # Solve the optimization problem
        problem.solve(solver=cp.CLARABEL)
        # problem.solve(solver=cp.SCS)
        # Update the variables after solving
        w1_val = w1.value
        w2_val = w2.value
        w3_val = w3.value
        w4_val = w4.value
        w_val = np.vstack((w1_val, w2_val,w3_val,w4_val))

        # v1_val=v1.value
        # v2_val = v2.value
        # v_val=np.vstack((v1_val,v2_val))
        v_val = v.value
        # d1_val=d1.value
        # d2_val = d2.value
        # d_val=np.vstack((d1_val,d2_val))
        d_val = d.value

        s1_val = s1.value
        s2_val = s2.value
        s3_val = s3.value
        s4_val = s4.value
        s_val = np.vstack((s1_val, s2_val,s3_val,s4_val))

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
        # print(np.abs(delta_L))
        if np.abs(delta_L) <= rho0:
            r_default = np.max((r_default, 1.3))
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
            r_default = 1.3
        # r_default = 3.5
        print(delta_L)
        # Update the trajectory

        # Plot the updated trajectory
        for i in range(Num_agent):
            if i == 0:
                plt.plot(X[0:1, :], X[1:2, :], 'g.', markersize=2)
            elif i == 1:
                plt.plot(X[4:5, :], X[5:6, :], 'b.', markersize=2)
            elif i == 2:
                plt.plot(X[8:9, :], X[9:10, :], 'b.', markersize=2)
            elif i == 3:
                plt.plot(X[12:13, :], X[13:14, :], 'b.', markersize=2)
            plt.xlim((-10, 20))
            plt.ylim((-10, 20))
            plt.pause(0.001)
        # plt.clf()
        ss = np.zeros((N * (Num_obs + Num_agent - 1), Num_agent))

        Current_obs_list = np.array([obs_center[i, 0:2],
                                     obs_center[i, 2:4],
                                     X[0:2, i],
                                     X[4:6, i]])
        R_list = np.array([2, 2, 1, 1])

        ss_max = np.array([0])
        # for i in range(N):
        #     for j in range(Num_agent):
        #         for k in range(Num_obs+Num_agent-1):
        #             ss[(N-1)*k+i,j] = LA.norm(X[n * j:n*j + 2, i] - obs_center[i, 2 * k:+2 * k + 2], 2) - R[k]
        #print(np.min(s_val))
        # if (np.min(ss) >= 0) and (interation > 15):
        #
        #     break
        if (interation > 15):
            break
        # print(np.min(ss) )
        print('Iteration:  ', interation + 1)

    # Final trajectory plot
    plt.clf()
    return (X, u)


X, u = trajectory_gen(P_des, P_ini, obs_center, R, T, Num_agent, Num_obs)


for i in range(int(T)):
    theta = np.linspace(0, 2 * np.pi, 201)
    for j in range(Num_agent):
        x_theta = 1 * np.cos(theta)
        y_theta = 1 * np.sin(theta)
        if j == 0:
            plt.plot(X[0:1, :i], X[1:2, :i], 'r.', markersize=2)
            plt.plot(X[0:1, i] + x_theta, X[1:2, i] + y_theta, 'r.', markersize=2)

        elif j == 1:
            plt.plot(X[4:5, :i], X[5:6, :i], 'g.', markersize=2)
            plt.plot(X[4:5, i] + x_theta, X[5:6, i] + y_theta, 'g.', markersize=2)
        if j == 2:
            plt.plot(X[8:9, :i], X[9:10, :i], 'r.', markersize=2)
            plt.plot(X[8:9, i] + x_theta, X[9:10, i] + y_theta, 'r.', markersize=2)

        elif j == 3:
            plt.plot(X[12:13, :i], X[13:14, :i], 'g.', markersize=2)
            plt.plot(X[12:13, i] + x_theta, X[13:14, i] + y_theta, 'g.', markersize=2)

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
