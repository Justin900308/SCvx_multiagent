" Multiprocessing only supports single input, single output "

from scipy import signal
from numpy import linalg as LA
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from multiprocessing import Process, Queue
import multiprocessing as mp


def dfdx_discrete(
        x: np.ndarray,
        u: np.ndarray,
        dt: float
) -> np.ndarray:
    A = dfdx(x, u)
    B = dfdu(x, u)
    C = np.eye(4)
    D = np.zeros((4, 2))
    sys = signal.StateSpace(A, B, C, D)
    sysd = sys.to_discrete(dt)
    Ad = sysd.A
    return (Ad)


def dfdu_discrete(
        x: np.ndarray,
        u: np.ndarray,
        dt: float
) -> np.ndarray:
    A = dfdx(x, u)
    B = dfdu(x, u)
    C = np.eye(4)
    D = np.zeros((4, 2))
    sys = signal.StateSpace(A, B, C, D)
    sysd = sys.to_discrete(dt)
    Bd = sysd.B
    return (Bd)


def f(
        x: np.ndarray,
        u: np.ndarray,
        dt: float
) -> np.ndarray:
    Ad = dfdx_discrete(x, u, dt)
    Bd = dfdu_discrete(x, u, dt)

    out = Ad @ x + Bd @ u
    return out


def dfdx(
        x: np.ndarray, u: np.ndarray
) -> np.ndarray:
    A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
    return A


def dfdu(
        x: np.ndarray, u: np.ndarray
) -> np.ndarray:
    B = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
    return B


def s_fun(
        x_t: np.ndarray,
        R: np.ndarray,
        obs_t_k: np.ndarray
) -> np.ndarray:
    S = 2 * R - LA.norm(x_t[0:2] - obs_t_k, 2)
    return S


def dsdx(
        x_t: np.ndarray,
        R: np.ndarray,
        obs_t_k: np.ndarray
) -> np.ndarray:
    S_grad = (x_t[0:2] - obs_t_k).T / cp.norm(x_t[0:2] - obs_t_k, 2)

    return S_grad


def ini_traj(
        x_ini: np.ndarray,
        x_des: np.ndarray,
        T: int
) -> np.ndarray:
    x_traj_ini = np.zeros((n, T))
    u_traj_ini = np.zeros((m, T - 1))
    x_traj_ini_x = np.linspace(x_ini[0, 0], x_des[0, 0], T)
    x_traj_ini_y = np.linspace(x_ini[1, 0], x_des[1, 0], T)
    x_traj_ini[0, :] = x_traj_ini_x
    x_traj_ini[1, :] = x_traj_ini_y
    return x_traj_ini, u_traj_ini


def solve_convex_optimal_control_subproblem(
        X_traj: np.ndarray,
        U_traj: np.ndarray,
        obs_trajs: np.ndarray,
        x_des: np.ndarray,
        num_obs: int,
        R: np.ndarray,
        r: float,  # thrust region
) -> np.ndarray:
    lambda_param = 10000
    T = X_traj.shape[1]
    # Define variables for optimization
    w = cp.Variable((2 * num_agent, T - 1))
    v = cp.Variable((4 * num_agent, T - 1))
    d = cp.Variable((4 * num_agent, T))
    s = cp.Variable((1 * num_obs, T - 1))
    constraints = [d[:, 0] == np.zeros(4)]
    for t in range(T - 1):

        # Define the cost function
        sup_problem_cost = 1 * cp.norm(((U_traj + w)), 1) + 1 * lambda_param * cp.sum(
            cp.sum(cp.abs(v))) + 1 * lambda_param * cp.sum(cp.pos(s))

        # Define constraints
        constraints.append(d[:, 0] == np.zeros(4))

        E = np.eye(4)

        x_t = X_traj[:, t]
        x_tp1 = X_traj[:, t + 1]
        d_t = d[:, t]
        d_tp1 = d[:, t + 1]
        u_t = U_traj[:, t]
        w_t = w[:, t]
        v_t = v[:, t]
        s_prime_t = s[:, t]
        obs_t = obs_trajs[:, :, t]

        Ad = dfdx_discrete(x_t, u_t, Ts)
        Bd = dfdu_discrete(x_t, u_t, Ts)
        constraints.append(
            x_tp1 + d_tp1 == (
                    Ad @ x_t + Ad @ d_t) + (
                    Bd @ u_t + Bd @ w_t) + E @ v_t)
        constraints.append(cp.abs(w_t) <= r)
        for k in range(num_obs):
            S = s_fun(x_t, R[k], obs_t[k])
            S_grad = dsdx(x_t, R[k], obs_t[k])
            constraints.append(
                S - S_grad @ (x_t[0:2] + d_t[0:2] - obs_t[k]) <= s_prime_t[k:k + 1]
            )
            constraints.append(s_prime_t[k:k + 1] >= 0)

    # Terminal condition
    constraints.append(X_traj[:, T - 1] + d[:, T - 1] == np.array([x_des[0, 0], x_des[1, 0], 0, 0]))

    # Define the problem
    problem = cp.Problem(cp.Minimize(sup_problem_cost), constraints)

    # Solve the optimization problem
    problem.solve(solver=cp.CLARABEL)
    cost = problem.value
    w_traj_val = w.value
    d_traj_val = d.value
    sss = 5
    return cost, d_traj_val, w_traj_val


def thrust_region_update(
        cost_list: np.ndarray,
        iter: int,
        r_current: float
) -> np.ndarray:
    rho0 = 0.1
    rho1 = 0.25
    rho2 = 0.7

    if iter >= 1:
        delta_L = (cost_list[0, iter] - cost_list[0, iter - 1]) / cost_list[0, iter]
    else:
        delta_L = 1

    if np.abs(delta_L) <= rho0:
        r_next = np.max((r_current / 2, 0.002))
    elif np.abs(delta_L) <= rho1:
        r_next = r_current / 1.2
    elif np.abs(delta_L) <= rho2:
        r_next = r_current / 2.2
    else:
        r_next = r_current / 1.1

    return r_next


def traj_gen(
        input_list: list,
        q
) -> list:
    ## Full input
    agent_index = input_list[0]
    R = input_list[1]
    x_des = input_list[2]
    obs_trajs = input_list[3]
    X = input_list[4]
    u = input_list[5]
    num_obs = np.size(R) - 1

    ## Prevent self infereencing
    R = np.delete(R, agent_index)
    obs_trajs = np.delete(obs_trajs, agent_index, 0)

    num_iter = 10
    cost_list = np.zeros((1, num_iter))
    r = 0.5
    for iter in range(num_iter):

        plt.plot(X[0, :], X[1, :], '.')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('Trajectory Plot')
        plt.grid(True)
        plt.xlim(-5, 20)
        plt.ylim(-5, 20)
        # plt.show()
        theta = np.linspace(0, 2 * np.pi, 201)
        for t in range(int(T)):
            plt.plot(X[0:1, t], X[1:2, t], 'r.', markersize=2)
            for k in range(num_obs):
                x_theta = R[k] * np.cos(theta)
                y_theta = R[k] * np.sin(theta)
                plt.plot(obs_trajs[k, 0, t] + x_theta, obs_trajs[k, 1, t] + y_theta, 'g')

        [sup_problem_cost, d_traj_val, w_traj_val] = (
            solve_convex_optimal_control_subproblem(X, u, obs_trajs, x_des, num_obs, R, r))

        X = X + d_traj_val
        u = u + w_traj_val
        actual_cost = LA.norm((u + w_traj_val), ord=1)
        cost_list[0, iter] = sup_problem_cost

        r = thrust_region_update(cost_list, iter, r)

        ss = np.zeros((1 * num_obs, T))
        #plt.pause(0.001)
        ss_max = np.array([0])
        for t in range(T):
            for j in range(num_agent):
                for k in range(num_obs):
                    ss[k:k + 1, t + T * j] = LA.norm(X[4 * j:j + 2, t] - obs_trajs[k, 0, t], 2) - R[k]

        if (np.min(ss) > 0) and (iter > 100):
            break
        print('Agent:    ', agent_index ,'  Iteration:  ', iter + 1, '  Subproblem Cost:   ', sup_problem_cost,
              '  Actual Cost:   ', actual_cost, '  r:   ', r)
        #plt.clf()
    #plt.clf()
    obs_trajs = np.insert(obs_trajs, [agent_index], X[0:2, :], 0)

    out_list = []
    out_list.append(X)
    out_list.append(u)
    out_list.append(obs_trajs)

    q.put(out_list)
    return out_list


# Input

# x_ini=np.array(([[0], [0]]))


Ts = 0.1  # Sampling time
T_end = 20  # End time
T = int(T_end / Ts + 1)  # Total time steps
num_agent = 1

########### Main
if __name__ == "__main__":
    ### Global constants
    num_obs = 4
    num_agent = 1  # number of agent
    R = np.array(([2, 2, 2, 2]))  # obs r
    m = 2
    n = 4
    #### Initialization
    obs_trajs = np.zeros((num_obs, 2, T))
    for t in range(T):
        obs_trajs[:, :, t] = np.array([
            [0 - 0.0 * t, 0 - 0.0 * t],
            [0 - 0.0 * t, 5 - 0.0 * t],
            [0 - 0.0 * t, 10 - 0.0 * t],
            [0 + 0.0 * t, 15 + 0.0 * t]
        ])
    x_ini_all = np.array(([[0, 0],
                           [0, 5],
                           [0, 10],
                           [0, 15]]))
    x_des_all = np.array(([[20, 15],
                           [20, 10],
                           [20, 5],
                           [20, 0]]))

    X_traj_all_0 = np.zeros((4, m, T))
    X_traj_all_1 = np.zeros((4, m, T))
    X_traj_all_2 = np.zeros((4, m, T))
    X_traj_all_3 = np.zeros((4, m, T))
    for t in range(T):
        X_traj_all_0[:, 0:2, t] = x_ini_all
        X_traj_all_1[:, 0:2, t] = x_ini_all
        X_traj_all_2[:, 0:2, t] = x_ini_all
        X_traj_all_3[:, 0:2, t] = x_ini_all
    mp.set_start_method('spawn')

    for iter in range(20):

        x_ini_0 = np.array([x_ini_all[0, :]]).T
        x_des_0 = np.array([x_des_all[0, :]]).T
        if iter == 0:
            [X, u] = ini_traj(x_ini_0, x_des_0, T)
        else:
            X = out0[0]
            u = out0[1]
            X_traj_all_0[:, 0:2, :] = out0[2]
            X_traj_all_1[:, 0:2, :] = out1[2]
            X_traj_all_2[:, 0:2, :] = out2[2]
            X_traj_all_3[:, 0:2, :] = out3[2]
            X_traj_all_ture = np.zeros((4, m, T))
            X_traj_all_ture[0, 0:2, :] = out0[2][0]
            X_traj_all_ture[1, 0:2, :] = out1[2][1]
            X_traj_all_ture[2, 0:2, :] = out2[2][2]
            X_traj_all_ture[3, 0:2, :] = out3[2][3]

        input_list0 = [0]
        input_list0.append(R)
        input_list0.append(x_des_0)
        if iter ==0:
            input_list0.append(X_traj_all_0)
        else:
            input_list0.append(X_traj_all_ture)
        input_list0.append(X)
        input_list0.append(u)

        x_ini_1 = np.array([x_ini_all[1, :]]).T
        x_des_1 = np.array([x_des_all[1, :]]).T
        [X, u] = ini_traj(x_ini_1, x_des_1, T)
        input_list1 = [1]
        input_list1.append(R)
        input_list1.append(x_des_1)
        if iter ==0:
            input_list1.append(X_traj_all_1)
        else:
            input_list1.append(X_traj_all_ture)
        input_list1.append(X)
        input_list1.append(u)

        x_ini_2 = np.array([x_ini_all[2, :]]).T
        x_des_2 = np.array([x_des_all[2, :]]).T
        [X, u] = ini_traj(x_ini_2, x_des_2, T)
        input_list2 = [2]
        input_list2.append(R)
        input_list2.append(x_des_2)
        if iter ==0:
            input_list2.append(X_traj_all_2)
        else:
            input_list2.append(X_traj_all_ture)
        input_list2.append(X)
        input_list2.append(u)

        x_ini_3 = np.array([x_ini_all[3, :]]).T
        x_des_3 = np.array([x_des_all[3, :]]).T
        [X, u] = ini_traj(x_ini_3, x_des_3, T)
        input_list3 = [3]
        input_list3.append(R)
        input_list3.append(x_des_3)
        if iter == 0:
            input_list3.append(X_traj_all_3)
        else:
            input_list3.append(X_traj_all_ture)
        input_list3.append(X)
        input_list3.append(u)

        #### SCvx Computation
        q0 = mp.Queue()
        q1 = mp.Queue()
        q2 = mp.Queue()
        q3 = mp.Queue()
        p0 = Process(target=traj_gen, args=(input_list0, q0))
        p1 = Process(target=traj_gen, args=(input_list1, q1))
        p2 = Process(target=traj_gen, args=(input_list2, q2))
        p3 = Process(target=traj_gen, args=(input_list3, q3))
        p0.start()
        p1.start()
        p2.start()
        p3.start()
        out0 = q0.get()
        out1 = q1.get()
        out2 = q2.get()
        out3 = q3.get()

        p0.join()
        p1.join()
        p2.join()
        p3.join()

        theta = np.linspace(0, 2 * np.pi, 201)
        if iter >=1:
            for t in range(int(T)):
                for k in range(4):
                    x_theta = R[k]/2 * np.cos(theta)
                    y_theta = R[k]/2 * np.sin(theta)
                    X_traj_1 = np.zeros((4, m, T))
                    X_traj_1[:, 0:2, :] = out0[2]
                    plt.plot(X_traj_1[k, 0, t] + x_theta, X_traj_1[k, 1, t] + y_theta)
                    plt.xlim(-5, 20)
                    plt.ylim(-5, 20)
                plt.pause(0.001)
                plt.clf()
        if iter >=1:
            for t in range(int(T)):
                for k in range(4):
                    x_theta = R[k]/2 * np.cos(theta)
                    y_theta = R[k]/2 * np.sin(theta)
                    plt.plot(X_traj_all_ture[k, 0, t] + x_theta, X_traj_all_ture[k, 1, t] + y_theta)
                    plt.xlim(-5, 20)
                    plt.ylim(-5, 20)
                plt.pause(0.001)
                plt.clf()


        ######### SCvx end
    # x_ini_i = np.array([x_ini_all[i, :]]).T
    # x_des_i = np.array([x_des_all[i, :]]).T
    # [X, u] = ini_traj(x_ini_i, x_des_i, T)
    # agent_index = i
    # input_list = [agent_index]
    # input_list.append(R)
    # input_list.append(x_des_i)
    # input_list.append(X_traj_all_i)
    # input_list.append(X)
    # input_list.append(u)
    # q = mp.Queue()
    # p = Process(target=traj_gen, args=(input_list, q))
    # p.start()
    # out = q.get()
    # X = out[0]
    # u = out[1]
    # p.join()

    # Final trajectory plot
    theta = np.linspace(0, 2 * np.pi, 201)
    for t in range(int(T)):
        plt.plot(X[0:1, :t], X[1:2, :t], 'r.', markersize=2)
        plt.xlim(-5, 20)
        plt.ylim(-5, 20)
        for k in range(num_obs):
            x_theta = R[k] * np.cos(theta)
            y_theta = R[k] * np.sin(theta)
            plt.plot(obs_trajs[k, 0, t] + x_theta, obs_trajs[k, 1, t] + y_theta)
        plt.pause(0.001)
        plt.clf()
