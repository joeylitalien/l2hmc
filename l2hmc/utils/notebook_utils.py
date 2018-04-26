import tensorflow as tf
import numpy as np
from dynamics import Dynamics
from sampler import propose
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def plot_grid(S, width=8):
    sheet_width = width
    plt.figure(figsize=(12, 12))
    for i in xrange(S.shape[0]):
        plt.subplot(sheet_width, sheet_width, i + 1)
        plt.imshow(S[i], cmap='gray')
        plt.grid('off')
        plt.axis('off')

def plot_line(S):
    sheet_width = S.shape[0]
    plt.figure(figsize=(16, 3))
    for i in xrange(S.shape[0]):
        plt.subplot(1, sheet_width, i + 1)
        plt.imshow(S[i], cmap='gray')
        plt.grid('off')
        plt.axis('off')

def get_hmc_samples(x_dim, eps, energy_function, sess, T=10, steps=200, samples=None):
    hmc_dynamics = Dynamics(x_dim, energy_function, T=T, eps=eps, hmc=True)
    hmc_x = tf.placeholder(tf.float32, shape=(None, x_dim))
    Lx, _, px, hmc_MH = propose(hmc_x, hmc_dynamics, do_mh_step=True)

    if samples is None:
        samples = gaussian.get_samples(n=200)

    final_samples = []

    for t in range(steps):
        final_samples.append(np.copy(samples))
        Lx_, px_, samples = sess.run([Lx, px, hmc_MH[0]], {hmc_x: samples})

    return np.array(final_samples)

def plot_gaussian_contours(mus, covs, colors=['blue', 'red'], spacing=5,
        x_lims=[-4,4], y_lims=[-3,3], res=100):

    X = np.linspace(x_lims[0], x_lims[1], res)
    Y = np.linspace(y_lims[0], y_lims[1], res)
    X, Y = np.meshgrid(X, Y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    for i in range(len(mus)):
        mu = mus[i]
        cov = covs[i]
        F = multivariate_normal(mu, cov)
        Z = F.pdf(pos)
        plt.contour(X, Y, Z, spacing, colors=colors[0])

    return plt

    #gaussian_1 = plt.contour(X, Y, Z_1, 5, colors=main)
    #gaussian_2 = plt.contour(X, Y, Z_2, 5, colors=main)
    #plt.plot(HMC_samples_1[:50, 1, 0], HMC_samples_1[:50, 1, 1], color=secondary, marker='o', alpha=0.75)
    #plt.axis('equal')
    #plt.show()

"""
from scipy.stats import multivariate_normal
N = 100
X = np.linspace(-3, 3, N)
Y = np.linspace(-2, 2, N)
X, Y = np.meshgrid(X, Y)
mu_1 = np.array([-2, 0])
mu_2 = np.array([2, 0])
cov_1 = 0.1*np.eye(2)
cov_2 = 0.1*np.eye(2)
F_1 = multivariate_normal(mu_1, cov_1)
F_2 = multivariate_normal(mu_2, cov_2)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y
Z_1 = F_1.pdf(pos)
Z_2 = F_2.pdf(pos)
gaussian_1 = plt.contour(X, Y, Z_1, 5, colors=main)
gaussian_2 = plt.contour(X, Y, Z_2, 5, colors=main)
"""
