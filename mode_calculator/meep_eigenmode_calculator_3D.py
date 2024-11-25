import meep as mp
import numpy as np
import matplotlib.pyplot as plt


def find_mode_profile(simulation_domain, resolution, width, epsilon, wavelength, offset_x=0, offset_y=0,
                      mode=1, until=20):
    parity = mp.EVEN_Y
    if mode % 2 == 0:
        parity = mp.ODD_Y
    cell = mp.Vector3(simulation_domain[0], simulation_domain[1], 3)
    geometry = [mp.Block(mp.Vector3(width[0], width[1], mp.inf),
                         center=mp.Vector3(offset_x, offset_y),
                         material=mp.Medium(epsilon=epsilon[1]))]
    pml_layers = [mp.PML(0.5)]

    fsrc = 1 / wavelength

    sources = [mp.EigenModeSource(src=mp.GaussianSource(fsrc, fwidth=0.1*fsrc),
                                  center=mp.Vector3(offset_x, offset_y),
                                  size=mp.Vector3(x=width[0], y=width[1]),
                                  direction=mp.Z,
                                  eig_kpoint=mp.Vector3(z=1),
                                  eig_band=mode,
                                  eig_parity=parity,
                                  eig_match_freq=True)]

    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=sources,
                        resolution=resolution)

    sim.run(until=until)

    eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)

    ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ex)

    s = eps_data.shape

    ind_max = np.argmax(ez_data[ez_data.shape[0] // 2, ez_data.shape[1] // 2])
    plt.imshow(eps_data[s[0] // 2].transpose(), interpolation='spline36', cmap='binary')
    plt.imshow(ez_data[s[0] // 2].transpose(), interpolation='spline36', cmap='RdBu', alpha=0.9)
    # plt.axis('off')
    plt.savefig("./meep_yz.png")
    plt.close()
    plt.imshow(eps_data[:, s[1] // 2].transpose(), interpolation='spline36', cmap='binary')
    plt.imshow(ez_data[:, s[1] // 2].transpose(), interpolation='spline36', cmap='RdBu', alpha=0.9)
    # plt.axis('off')
    plt.savefig("./meep_xz.png")
    plt.close()
    plt.imshow(eps_data[:, :, ind_max].transpose(), interpolation='spline36', cmap='binary')
    plt.imshow(ez_data[:, :, ind_max].transpose(), interpolation='spline36', cmap='RdBu', alpha=0.9)
    # plt.axis('off')
    plt.savefig("./meep_xy.png")
    plt.close()
    return ez_data[..., ind_max]
