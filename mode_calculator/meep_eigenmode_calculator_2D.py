import meep as mp
import numpy as np
import matplotlib.pyplot as plt


def find_mode_profile(simulation_domain, resolution, width, epsilon, wavelength, offset=0,
                      mode=1, until=200):
    centre = 0

    if offset == 0:
        centre = 0
    elif offset > 0:
        centre = offset + width / 2
    elif offset < 0:
        centre = offset - width / 2
    parity = mp.EVEN_Y
    if mode % 2 == 0:
        parity = mp.ODD_Y
    cell = mp.Vector3(*simulation_domain)
    w = width
    geometry = [mp.Block(mp.Vector3(w, mp.inf, mp.inf),
                         center=mp.Vector3(centre),
                         material=mp.Medium(epsilon=epsilon[1]))]
    pml_layers = [mp.PML(1.0)]

    fsrc = 1 / wavelength

    eigen = True

    if eigen:
        sources = [mp.EigenModeSource(src=mp.ContinuousSource(fsrc),
                                      center=mp.Vector3(centre),
                                      size=mp.Vector3(x=3 * w),
                                      direction=mp.NO_DIRECTION,
                                      eig_kpoint=mp.Vector3(y=1),
                                      eig_band=mode,
                                      eig_parity=parity,
                                      eig_match_freq=True)]
    else:
        sources = [mp.Source(src=mp.ContinuousSource(fsrc),
                             center=mp.Vector3(centre),
                             size=mp.Vector3(y=w),
                             component=mp.Ez)]

    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=sources,
                        resolution=resolution)

    sim.run(until=until)

    eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)

    ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)

    ind_max = np.argmax(ez_data[ez_data.shape[0] // 2 + int(centre*resolution)])
    plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary', origin='lower')
    plt.imshow(ez_data.transpose(), interpolation='spline36', cmap='RdBu', alpha=0.9, origin='lower')
    plt.axvline(x=ez_data.shape[0] // 2 + int(centre*resolution))
    plt.colorbar()
    # plt.axis('off')
    plt.savefig("./meep.png")
    plt.close()
    plt.plot(ez_data[:, ind_max])
    plt.savefig("./meep_field.png")
    plt.close()
    return ez_data[:, ind_max]

