import numpy as np
import matplotlib.pyplot as plt

class osc_network:
   def __init__(self, N, K):
      """
         N - number of oscillators
         ph - osciallator phases
         w - phase velocities
         Kmat = symmetric matrix of coupling strengths. Can also interpret as weighted adjacency matrix
      """
      self.N = N
      self.ph = 2.0*np.pi*np.random.uniform(size=N)
      self.ws = np.random.normal(size=N, scale=1.0)
      self.Kmat = K*np.ones(shape=(N,N))/N

   #return vector of phase time derivatives
   def vel_vec(self, ths):
      return self.ws + np.sum(np.multiply(self.Kmat, np.sin(np.subtract.outer(ths, ths).T)), axis=1)

   def huen_step(self, dt=0.0025):
      tmp = np.mod(self.ph + dt*self.vel_vec(self.ph), 2.0*np.pi)
      self.ph = np.mod(self.ph + 0.5*dt*(self.vel_vec(self.ph) + self.vel_vec(tmp)), 2.0*np.pi)
    
   def get_trajectory(self, T=4000):
      self.trajectory = [self.ph]
      for i in np.arange(T-1):
         self.huen_step()
         self.trajectory.append(self.ph)
      self.trajectory = np.asarray(self.trajectory)
    
   def time_avg_order_params(self):
      #only take last 1000 timesteps of trajectories
      self.get_trajectory()
      trj = self.trajectory[-2000:,:]
      m = np.mean(np.absolute(np.sum(np.exp(1j*trj), axis=1)/self.N))

      c = 1e6; uids = np.triu_indices(self.N,k=1)
      q_list = []
      for t in trj:
         vel = self.vel_vec(t)
         q_sum = np.sum(np.exp(-c*np.square(np.subtract.outer(vel, vel)))[uids])
         q_list.append(2*q_sum/(self.N*(self.N-1)))
      q = np.mean(q_list)
      return(m,q)

   def quench_avg_order_params(self):
      m_all = []; q_all = [];

      for i in np.arange(250):
         #reset quenched vars and IC
         self.ph = 2.0*np.pi*np.random.uniform(size=self.N)
         self.ws = np.random.normal(size=self.N, scale=1.0)
         m,q = self.time_avg_order_params()
         m_all.append(m); q_all.append(q)
      return(m_all, q_all)
