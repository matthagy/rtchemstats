
from pyljfluid.components import LJForceField, Config, MDSimulator

N = 1000
forcefield = LJForceField(sigma=1.0, epsilon=1.0, r_cutoff=2.5)
r_neighbor_skin = 1.0 * forcefield.sigma
mass = 48 * forcefield.epsilon * forcefield.sigma ** -2
dt = 0.032

def get_equilibrated_simulation(rho, T):
    config0 = Config.create(N=N, rho=rho, dt=dt, sigma=forcefield.sigma, T=T, mass=mass)
    sim = MDSimulator(config0, forcefield, mass=mass, r_skin=r_neighbor_skin)
    for i in xrange(500):
        if not i%10:
            sim.config.randomize_velocities(T=T, mass=mass)
        sim.cycle(50)
        U = sim.evaluate_potential()
        print 'equilibrate cycle i=%03d U=%.3f' % (i, U)
    return sim

