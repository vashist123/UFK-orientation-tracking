import numpy as np
from scipy import io
from quaternion import Quaternion
import math
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import os


# data files are numbered on the server.
# for exmaple imuRaw1.mat, imuRaw2.mat and so on.
# write a function that takes in an input number (1 through 6)
# reads in the corresponding imu Data, and estimates
# roll pitch and yaw using an unscented kalman filter


def get_W(P, Q):
    S = np.sqrt(2 * P.shape[0]) * np.linalg.cholesky(P + Q)
    W = np.zeros((6, 12))
    W[:, :6] = S
    W[:, 6:] = -S

    return W


def get_X(prev_state, W):
    X = np.zeros((7, 12))
    quat_prev = Quaternion()
    quat_prev.q = prev_state[:4]
    quat_noise = Quaternion()

    for i in range(W.shape[1]):
        quat_noise.from_axis_angle(W[:3, i])
        X[:4, i] = (quat_prev * quat_noise).q
        X[4:, i] = prev_state[4:] + W[3:, i]

    return X


def process_model(X, delta_t):
    Y = X
    Quat_del = Quaternion()
    quat_init = Quaternion()

    for i in range(X.shape[1]):
        omg = X[4:, i]
        Quat_del.from_axis_angle(omg * delta_t)
        quat_init.q = X[:4, i]
        Y[:4, i] = (Quat_del * quat_init).q

    return Y


def measurement_model(Y):
    Z = np.zeros((6, 12))
    g_quat = Quaternion(scalar=0, vec=[0, 0, 9.81])
    quat = Quaternion()

    for i in range(Y.shape[1]):
        quat.q = Y[:4, i]
        Z[:3, i] = (quat.inv() * (g_quat * quat)).q[1:]
        Z[3:, i] = Y[4:, i]

    return Z


def compute_mean(Y, prev_estimate):
    mean = np.zeros(7)
    err_vecs = np.zeros((3, Y.shape[1]))
    q = Quaternion()
    q.q = prev_estimate[:4]
    qi = Quaternion()
    Quat_part = Y[:4, :]
    Omg_part = Y[4:, :]
    e = Quaternion()

    for _ in range(5):
        for j in range(Y.shape[1]):
            qi.q = Quat_part[:, j]
            err_q = qi * q.inv()
            err_vecs[:, j] = err_q.axis_angle()
        err_mean = np.mean(err_vecs, axis=1)
        e.from_axis_angle(err_mean)
        q.q = (e * q).q
        if np.linalg.norm(err_mean) < 0.001:
            break

    omg_mean = np.mean(Omg_part, axis=1)

    mean[:4] = q.q
    mean[4:] = omg_mean

    return mean, err_vecs


def compute_z_mean(Z):
    mean = np.mean(Z, axis=1)
    return mean


def get_W_dash(Y, err_vecs, mean_minus):
    W_dash = np.zeros((6, 12))
    W_dash[:3, :] = err_vecs
    for i in range(Y.shape[1]):
        W_dash[3:, i] = Y[4:, i] - mean_minus[4:]

    return W_dash


def get_Z_dash(Z, zk_minus):
    Z_dash = np.zeros((6, 12))

    for i in range(Z.shape[1]):
        Z_dash[:, i] = Z[:, i] - zk_minus
    return Z_dash


def get_covariance(vect):
    cov = np.zeros((vect.shape[0], vect.shape[0]))

    for i in range(vect.shape[1]):
        cov += vect[:, i].reshape(vect.shape[0], 1) @ vect[:, i].reshape(1, vect.shape[0])
    cov = cov / (vect.shape[1])
    return cov


def get_cross_covariance(vect, vect2):
    cov = np.zeros((vect.shape[0], vect.shape[0]))
    for i in range(vect.shape[1]):
        cov += vect[:, i].reshape(vect.shape[0], 1) @ vect2[:, i].reshape(1, vect2.shape[0])
    cov = cov / (vect.shape[1])
    return cov


def get_new_estimate(mean_minus, K, nu_k):
    gain = np.dot(K, nu_k)
    new_mean = np.zeros_like(mean_minus)

    old = Quaternion()
    old.q = mean_minus[:4]
    update = Quaternion()
    update.from_axis_angle(gain[:3])
    new_mean[:4] = (update * old).q
    new_mean[4:] = mean_minus[4:] + gain[3:]

    return new_mean


def estimate_rot(data_num=1):
    # loading the data from the IMU and Vicon

    imu = io.loadmat(os.path.join(os.path.dirname(__file__), "imu/imuRaw" + str(data_num) + ".mat"))
    imu = io.loadmat('imu/imuRaw' + str(data_num) + '.mat')
    vicon = io.loadmat('vicon/viconRot' + str(data_num) + '.mat')
    accel = imu['vals'][0:3, :].astype('int')
    gyro = imu['vals'][3:6, :].astype('int')
    T = np.shape(imu['ts'])[1]
    imu_time = imu['ts']
    time = vicon['ts']
    rots = vicon['rots']

    m = Quaternion()
    quats_x = []
    rot_arr = []
    for i in range(len(time[0])):
        r = Rotation.from_matrix(rots[:, :, i]).as_euler('xyz', degrees=False)
        rot_arr.append(r)
        m.from_rotm(rots[:, :, i])
        quats_x.append(m.q[3])

    time = np.array(time)
    rotationss = np.array(rot_arr)

    # Processing the raw data of the IMU

    true_accel = (accel - 506) * 3300 * 9.81 / (1023 * 320)
    test = (gyro[1, :] - 373.8) / (3.3 / 2.78) * np.pi / 180
    test2 = (gyro[2, :] - 375.8) / (3.3 / 2.78) * np.pi / 180
    test3 = (gyro[0, :] - 369.8) * 1.3 / (3.3 / 2.78) * np.pi / 180

    x_angle = 0
    y_angle = 0
    z_angle = 0
    xs = []
    ys = []
    zs = []
    for i in range(len(imu_time[0, :]) - 1):
        x_angle += test[i] * (imu_time[0, i + 1] - imu_time[0, i])
        xs.append(x_angle)
        y_angle += test2[i] * (imu_time[0, i + 1] - imu_time[0, i])
        ys.append(y_angle)
        z_angle += test3[i] * (imu_time[0, i + 1] - imu_time[0, i])
        zs.append(z_angle)

    x_accel = -1 * true_accel[0, :]
    y_accel = -1 * true_accel[1, :]
    z_accel = true_accel[2, :]
    measurements_cal = np.stack((x_accel, y_accel, z_accel, test, test2, test3))

    pitch_angs = []
    roll_angs = []

    for i in range(len(imu_time[0])):
        pitch_ang = math.atan2(-1 * x_accel[i], np.sqrt(y_accel[i] ** 2 + z_accel[i] ** 2))
        roll_ang = math.atan2(y_accel[i], z_accel[i])
        pitch_angs.append(pitch_ang)
        roll_angs.append(roll_ang)

    init_state = np.zeros((7))
    P = 1 * np.identity(6)
    Q = 500 * np.identity(6)
    R = 100 * np.identity(6)
    init_omegas = np.array([0, 0, 0])
    init_quat = Quaternion()
    init_quat.q = np.array([1, 0, 0, 0])
    init_state[:4] = init_quat.q
    init_state[4:] = init_omegas
    state_estimate = init_state

    phi = []
    theta = []
    psi = []

    quat_for_plot = Quaternion()

    for i in range(len(imu_time[0, :])):
        if i == len(imu_time[0, :]) - 1:
            delta_t = imu_time[0, -1] - imu_time[0, -2]
        else:
            delta_t = imu_time[0, i + 1] - imu_time[0, i]

        W = get_W(P, Q)
        X = get_X(state_estimate, W)
        Y = process_model(X, delta_t)
        mean_minus, err_vecs = compute_mean(Y, state_estimate)
        W_dash = get_W_dash(Y, err_vecs, mean_minus)
        Pk_minus = get_covariance(W_dash)
        Z = measurement_model(Y)
        zk_minus = compute_z_mean(Z)
        Z_dash = get_Z_dash(Z, zk_minus)
        nu_k = measurements_cal[:, i] - zk_minus
        P_zz = get_covariance(Z_dash)
        P_vv = P_zz + R
        P_xz = get_cross_covariance(W_dash, Z_dash)
        K = np.dot(P_xz, np.linalg.inv(P_vv))
        state_estimate = get_new_estimate(mean_minus, K, nu_k)
        P = Pk_minus - np.dot(K, np.dot(P_vv, K.T))
        quat_for_plot.q = state_estimate[:4]
        euls = quat_for_plot.euler_angles()
        phi.append(euls[0])
        theta.append(euls[1])
        psi.append(euls[2])

    phi = np.asarray(phi)
    theta = np.asarray(theta)
    psi = np.asarray(psi)

    # Comparing the ground truth orientation and orientation produced by the Unscented Kalman Filter

    (fig, axes) = plt.subplots(nrows=3, ncols=1, sharex=True, num='Orientation given by UKF')
    ax = axes[0]
    ax.plot(time[0, :], rotationss[:, 0], 'r', imu_time[0, :], phi, 'b')
    ax.legend(("vicon", "imu"))
    ax.set_ylabel('orientation x')
    ax.grid('major')
    ax = axes[1]
    ax.plot(time[0, :], rotationss[:, 1], 'r', imu_time[0, :], theta, 'b')
    ax.legend(("vicon", "imu"))
    ax.set_ylabel('orientation y')
    ax.grid('major')
    ax = axes[2]
    ax.plot(time[0, :], rotationss[:, 2], 'r', imu_time[0, :], psi, 'b')
    ax.legend(("vicon", "imu"))
    ax.set_ylabel('orientation z')
    ax.grid('major')
    plt.show()

    # True orientation of the drone given by the Vicon

    (fig, axes) = plt.subplots(nrows=3, ncols=1, sharex=True, num='Ground Truth Orientation vs Time')
    ax = axes[0]
    ax.plot(time[0, :], rotationss[:, 0], 'r')
    ax.set_ylabel('rotation x')
    ax.grid('major')
    ax = axes[1]
    ax.plot(time[0, :], rotationss[:, 1], 'r')
    ax.set_ylabel('rotation y')
    ax.grid('major')
    ax = axes[2]
    ax.plot(time[0, :], rotationss[:, 2], 'r')
    ax.set_ylabel('rotation z')
    ax.grid('major')
    plt.show()

    # Orientation as tracked only by the gyroscope

    (fig, axes) = plt.subplots(nrows=3, ncols=1, sharex=True, num='Orientation given by Gyroscope')
    ax = axes[0]
    ax.plot(time[0, :], rotationss[:, 0], 'r', imu_time[0, :-1], xs, 'b')
    ax.legend(("vicon", "imu"))
    ax.set_ylabel('orientation x')
    ax.grid('major')
    ax = axes[1]
    ax.plot(time[0, :], rotationss[:, 1], 'r', imu_time[0, :-1], ys, 'b')
    ax.legend(("vicon", "imu"))
    ax.set_ylabel('orientation y')
    ax.grid('major')
    ax = axes[2]
    ax.plot(time[0, :], rotationss[:, 2], 'r', imu_time[0, :-1], zs, 'b')
    ax.legend(("vicon", "imu"))
    ax.set_ylabel('orientation z')
    ax.grid('major')
    plt.show()

    return imu_time, phi, theta, psi


if __name__ == '__main__':
    imu_time, phi, theta, psi = estimate_rot(1)
