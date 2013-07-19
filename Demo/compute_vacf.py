'''Compute the velocity autocorrelation function C_v(t) for a
   Lennard Jones system at T=1.46 and rho=0.4. Results can be compared
   to Fig. 1 of Levesque D. and Verlet L. Phys. Rev. A. 2514. 201. (1970)
'''

from rtchemstats.tcf import VelocityAutocorrelationComputer
import matplotlib.pyplot as plt

from util import get_equilibrated_simulation, dt, N

sim = get_equilibrated_simulation(rho=0.85, T=0.76)

# Compute C_v(t)
cycle_rate = 1
cv_comp = VelocityAutocorrelationComputer(window_size=250, N_particles=N,
                                          analyze_rate=dt * cycle_rate)
for i in xrange(2000):
    if not i%10:
        print 'compute cycle i=%04d H=%.4f ' % (i, sim.evaluate_hamiltonian())
    sim.cycle(cycle_rate)
    cv_comp.analyze_velocities(sim.config.calculate_velocities())
Cvt = cv_comp.get_accumulated()

# Plot C_v(t)
plt.figure(3)
plt.clf()
plt.plot(Cvt.t / 0.128, Cvt.Cv / Cvt.Cv[0])
plt.xlabel(r'Time, $t$ ($0.128 \tau$)')
plt.ylabel(r'$C_v(t) / C_v(0)$')
plt.axhline(0, color='k')
plt.xlim(0, 35)
plt.subplots_adjust(left=0.18, bottom=0.18)
plt.draw()
plt.show()
plt.savefig('vacf.eps')
