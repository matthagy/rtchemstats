'''Compute the pair distribution function g(r) for a Lennard Jones
   system at T=1.46 and rho=0.4. Results can be compared to Fig. 1
   of Verlet L. Phys. Rev. 165. 201. (1968)
'''

from rtchemstats.pair1d import StaticIsotropicPairCorrelationComputer
import matplotlib.pyplot as plt

from util import get_equilibrated_simulation

sim = get_equilibrated_simulation(rho=0.4, T=1.46)

# Compute g(r)
gr_comp = StaticIsotropicPairCorrelationComputer(dr=0.01,
                                                 r_max=sim.config.box_size / 2)
for i in xrange(2000):
    if not i%10:
        print 'compute cycle i=%04d H=%.4f ' % (i, sim.evaluate_hamiltonian())
    sim.cycle(10)
    gr_comp.accumulate_positions(sim.config.positions,
                                 sim.config.box_size)
gr = gr_comp.get_accumulated()

# Plot g(r)
plt.figure(1)
plt.clf()
plt.plot(gr.r, gr.g)
plt.xlabel(r'Pair Separation, $r$ ($\sigma$)')
plt.ylabel(r'Pair Distribution, $g(r)$')
plt.xlim(0.8, 2.8)

plt.draw()
plt.show()

plt.savefig('gr.eps')
