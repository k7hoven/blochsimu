# -*- coding: utf-8 -*-

"""
Created on Wed Nov 19 19:18:07 2014

Module for simulating the expectation of a spin-1/2 ensemble using the Bloch
equation (currently non-dissipative),

dM/dt = gamma M x B.

@author: Koos Zevenhoven
"""
from __future__ import unicode_literals, division
import numpy as np

gamma_H = 42.58e6 * 2 * np.pi


def simulate_step(M, B, dt, gamma = gamma_H, out = None):
    """Simulate one time step of the Bloch equation.
    
    Currently a second-order approximation of the Bloch solution is used,
    assuming constant B over the simulation step.
    
    Arguments:
        M: magnetization vector(s) last dimension must have size 3
        B: magnetic field vector(s) last dimension must have size 3
            The best thing would be to have the average value over [t, t+dt]
        dt: time step to simulate over
        gamma: gyromagnetic ratio
        out: if given, the result will be computed into this array
    
    Return:
        magnetization at t + dt
    """
    M, B = map(lambda A: np.asanyarray(A, dtype=float), (M, B))
    assert B.shape[-1] == 3
    assert M.shape[-1] == 3    
    
    M_shape = M.shape
    B = B.reshape((-1,3))
    M = M.reshape((-1,3))
    
    assert B.shape[0] == 1 or B.shape[0] == M.shape[0]
    
    # A = np.array([[0.0, Bz, -By], [-Bz, 0.0, Bx], [By, -Bx, 0.0]])*gamma*dt
    # A**2 ~ I + A + A**2/2
    # Calculate P = A + A**2/2
    P = np.empty((M.shape[0], 3, 3))
    
    B = B * gamma * dt # NOTE! B is not really B anymore!
    x, y, z = 0, 1, 2    
    B2p2 = B**2/2
    Bxyp2 = (B[:,x] * B[:,y])/2
    Byzp2 = (B[:,y] * B[:,z])/2
    Bzxp2 = (B[:,z] * B[:,x])/2
        
    P[:,0,0] = - B2p2[:,z] - B2p2[:,y]
    P[:,0,1] = B[:,z] + Bxyp2
    P[:,0,2] = Bzxp2 - B[:,y]
    P[:,1,0] = Bxyp2 - B[:,z]
    P[:,1,1] = - B2p2[:,x] - B2p2[:,z]
    P[:,1,2] = B[:,x] + Byzp2
    P[:,2,0] = Bzxp2 + B[:,y]
    P[:,2,1] = Byzp2 - B[:,x]
    P[:,2,2] = - B2p2[:,x] - B2p2[:,y]

    if out is None:
        out = np.empty(M.shape)

    kwargs = {"out": out.reshape(M.shape)}
        
    out[:] = M + np.einsum('...ij,...j->...i', P, M, **kwargs)

    return out.reshape(M_shape)

del simulate_step # was an old test function, only keeping code for reference

class Simulator(object):
    """Describes an object for simulating the Bloch equation."""
    
    def __init__(self, M0, t0 = 0, gamma = gamma_H, n_steps = None):
        """Initialize simulator with given initial magnetization.
        
        Arguments:
            M: Initial magnetization vector(s); last dimension must have 
                size 3.
            t0: Initial value of time.
            gamma: Gyromagnetic ratio.
            n_steps: Number of steps to preallocate for (None: forget 
                trajectory)
        """
        self._M0 = np.asanyarray(M0, dtype = float)
        assert self._M0.shape[-1] == 3
        self._t0 = float(t0)
        self._M = self._M0.copy().reshape((-1,3))
        self._tmp1 = self._M.copy()
        self._tmp2 = self._M.copy()
        self._tmps1 = np.empty(self._M.shape[:-1]) # temp scalar buffer
        self._tmps2 = self._tmps1.copy()
        self._tmpm = np.empty(self._M.shape + (3,)) # temp matrix buffer  
        self._isotrop = np.eye(3)
        self._diag_ind = np.array([0,4,8])
        self.gamma = gamma
        self.t = self._t0
        self._previousB = 0
        self._n_steps = n_steps
        self._step_i = 0
        self._P = np.empty((self._M.shape[0], 3, 3)) #space for matrices
        self._genM = np.array([[[[0, 0, 0],
                                 [0, 0, -1],
                                 [0, 1, 0]],
                                [[0, 0, 1],
                                 [0, 0, 0],
                                 [-1, 0, 0]],
                                [[0, -1, 0],
                                 [1, 0, 0],
                                 [0, 0, 0]]]])
        self._genM = np.transpose(self._genM, axes = (0,2,1,3))
        

        if n_steps is not None:
            self.M_traj = np.empty((self._n_steps + 1,) + self._M0.shape)
            self.t_traj = np.empty((self._n_steps + 1,))
            self.t_traj[0] = self.t
            self.M_traj[0,:] = self._M0
    
    @property
    def M(self):
        return self._M.reshape(self._M0.shape)
        
    def simulate_step(self, B, dt, T1T2 = None):
        """Simulate one time step of the Bloch equation.
        
        Currently a second-order approximation of the Bloch solution is used,
        assuming constant B over the simulation step.
        
        Arguments:
            B: Magnetic field vector(s); last dimension must have size 3.
                The best thing would be to have the average value 
                over [t, t+dt]. This should be a B corresponding to each
                magnetization (i.e. shape same as that of M0).
            dt: time step to simulate over
            T1T2: T1T2[0] = <T1 value(s)> and T1T2[1] = <T2 value(s)>
        Return:
            magnetization at t + dt
        """
        B = np.asanyarray(B, dtype=float)
        assert B.shape[-1] == 3
        B = B.reshape((-1,3))
        M = self._M
        assert B.shape[0] == M.shape[0]
        
        self._step_i += 1
        self.t += dt
        gdt = self.gamma * dt  

        # A = np.array([[0.0, Bz, -By], 
        #               [-Bz, 0.0, Bx], 
        #               [By, -Bx, 0.0]])*gamma*dt
        # A**2 ~ I + A + A**2/2
        # Calculate A + A**2/2
         
        #This is faster (than below), but needs the funny self._genM matrix
        P = self._P
        M = self._M
        np.einsum("...ijk,...k->...ij", self._genM, B, out = P)
        # Add relaxation if present
        if T1T2 is not None:
            tmps1 = self._tmps1
            tmps2 = self._tmps2
            R1 = 1 / (T1T2[0] * self.gamma)
            R2 = 1 / (T1T2[1] * self.gamma)            
            isotrop = self._isotrop
            isotrop.flat[self._diag_ind] = R2
            P -= isotrop
            np.einsum("...i,...i->...", B, B, out = tmps1)
            np.divide(R2 - R1, tmps1, out = tmps2)
            P += np.einsum("...i,...j,...->...ij", B, B, tmps2, 
                           out = self._tmpm)
        
        # Second-order approximation of time step
        tmp1 = self._tmp1
        np.einsum("...ij,...j->...i", P, M, out = tmp1)
        tmp1 *= gdt
        M += tmp1
        tmp1 *= gdt/2
        np.einsum("...ij,...j->...i", P, tmp1, out = self._tmp2)
        M += self._tmp2
        
        if self._n_steps is not None:
            self.M_traj[self._step_i,:] = M
            self.t_traj[self._step_i] = self.t
        self._previousB = B
        return M.reshape(self._M0.shape)

#    def simulate(self, B_t, t_stop, dt = None):
#        """Simulate until a given time."""
#        raise NotImplementedError()
#        if dt is None:
#            B = B_t(t)
#            dB = B - self._previousB
#            # Figure out what time step to use
#            dt_larmor = 0.05/(self.gamma * np.sqrt(B_t(np.array(self.t))**2))
#            #dt_field_change = 0.01/ (dB/dt/B)
#            #dt_relaxation = T2 * 0.01
#            dt = dt_larmor # or min of different options
    
    

if __name__ == "__main__":
    M = np.array([1., 0, 0])
    B = np.array([0.0, 0.0, 50e-6])
    N = 10**5
    
    sim = Simulator(M, n_steps = 100000)
    
    for i in xrange(sim._n_steps):
        sim.simulate_step(B, 2e-5)
    
    
#    Ms = np.empty((N,3))
#    oM = np.empty((3,))
#    for i in xrange(N):
#        simulate_step(M, B, 1e-5, out = oM)
#        Ms[i,:] = oM
#        M[:] = oM
#    t = np.arange(N) * 1e-5
    
    
#plt.plot(t, Ms[:,0])


# SCRAP collected from above (old, slower versions etc.)
#        if False: # Much faster version in "else"
#            x, y, z = 0, 1, 2
#            gdt2p2 = gdt**2 / 2
#            B2p2 = B**2 * gdt2p2
#            Brotp2 = B * np.roll(B, 1, axis = 1) * gdt2p2
#            Bxyp2 = Brotp2[:,0]
#            Byzp2 = Brotp2[:,1]
#            Bzxp2 = Brotp2[:,2]
#            
#            P = self._P
#            P[:,0,0] = - B2p2[:,z] - B2p2[:,y]
#            P[:,0,1] = B[:,z] * gdt + Bxyp2
#            P[:,0,2] = Bzxp2 - B[:,y] * gdt
#            P[:,1,0] = Bxyp2 - B[:,z] * gdt
#            P[:,1,1] = - B2p2[:,x] - B2p2[:,z]
#            P[:,1,2] = B[:,x] * gdt + Byzp2
#            P[:,2,0] = Bzxp2 + B[:,y] * gdt 
#            P[:,2,1] = Byzp2 - B[:,x] * gdt
#            P[:,2,2] = - B2p2[:,x] - B2p2[:,y]
#            self._tmp1[:] = self._M
#            self._M += np.einsum('...ij,...j->...i', P, self._tmp1, 
#                                 out = self._tmp2)
#        elif False: #This version avoids unnecessary allocations
#            P = self._P
#            P[:,0,0] = 0
#            P[:,0,1] = B[:,2]  #Bz
#            P[:,0,2] = -B[:,1] #By
#            P[:,1,0] = -B[:,2] #Bz
#            P[:,1,1] = 0
#            P[:,1,2] = B[:,0]  #Bx
#            P[:,2,0] = B[:,1]  #By
#            P[:,2,1] = -B[:,0] #Bx
#            P[:,2,2] = 0
#            
#            np.einsum("...ij,...j->...i", P, self._M, out = self._tmp1)
#            self._tmp1 *= gdt
#            self._M += self._tmp1
#            self._tmp1 *= gdt/2
#            np.einsum("...ij,...j->...i", P, self._tmp1, out = self._tmp2)
#            self._M += self._tmp2

        
