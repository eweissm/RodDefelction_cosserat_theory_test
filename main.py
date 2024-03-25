import numpy as np
import matplotlib.pyplot as plt
from elastica.rod.cosserat_rod import CosseratRod
from elastica.boundary_conditions import OneEndFixedBC
from elastica.external_forces import UniformTorques

from elastica.dissipation import AnalyticalLinearDamper
from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.timestepper import integrate
from elastica.external_forces import EndpointForces

from elastica.modules import (
    BaseSystemCollection,
    Connections,
    Constraints,
    Forcing,
    CallBacks,
    Damping
)


final_time = 4   # seconds


class SystemSimulator(
    BaseSystemCollection,
    Constraints, # Enabled to use boundary conditions 'OneEndFixedBC'
    Forcing,     # Enabled to use forcing 'GravityForces'
    Connections, # Enabled to use FixedJoint
    CallBacks,   # Enabled to use callback
    Damping,     # Enabled to use damping models on systems.
):
    pass

SystemSim = SystemSimulator()


# create a rod
direction = np.array([0.0, 0.0, 1.0])
normal = np.array([0.0, 1.0, 0.0])
rod1 = CosseratRod.straight_rod(
    n_elements=20,                                # number of elements
    start=np.array([0.0, 0.0, 0.0]),              # Starting position of first node in rod
    direction=direction,                          # Direction the rod extends
    normal=normal,                                # normal vector of rod
    base_length=0.3,                              # original length of rod (m)
    base_radius=.0005,                            # original radius of rod (m)
    density=8730.0,                                 # density of rod (kg/m^3)
    youngs_modulus=100.0e9,                         # Elastic Modulus (Pa)
    shear_modulus=40.0e9,                           # Shear Modulus (Pa)
)

# Add rod to SystemSimulator
SystemSim.append(rod1)


#define BC's
SystemSim.constrain(rod1).using(
    OneEndFixedBC,                  # Displacement BC being applied
    constrained_position_idx=(0,),  # Node number to apply BC
    constrained_director_idx=(0,)   # Element number to apply BC
)

#apply loads
# direction = np.array([0.0, 1.0, 0.0])
# torque = np.array([.03, 0, 0.0])
# SystemSim.add_forcing_to(rod1).using(
#     UniformTorques,
#     torque,
#     direction
# )
origin_force = np.array([0.0, 0.0, 0.0])
end_force = np.array([0.0,0.0, 0.030])
SystemSim.add_forcing_to(rod1).using(
    EndpointForces,                 # Traction BC being applied
    origin_force,                   # Force vector applied at first node
    end_force,                      # Force vector applied at last node
    ramp_up_time=1.0   # Ramp up time
)

#apply damping for numeric stability
nu = 1e-3   # Damping constant of the rod
dt = 1e-5   # Time-step of simulation in seconds

SystemSim.dampen(rod1).using(
    AnalyticalLinearDamper,
    damping_constant = nu,
    time_step = dt
)

SystemSim.finalize()

timestepper = PositionVerlet()
total_steps = int(final_time / dt)
integrate(timestepper, SystemSim, final_time, total_steps)

print(rod1.position_collection)
plt.plot(rod1.position_collection[2][:])
plt.show()