import matplotlib.pyplot as plt
import jax.numpy as jnp
import h5py


def plot_maker():
    def save_plot(E, epsr, loss_hist, j):
        for i in range(E[0][0].shape[0]):
            plt.imshow(epsr[0][i].T, origin='lower', cmap='gray')
            plt.imshow(jnp.abs(E[0][0][i].T), origin='lower', cmap="magma", alpha=0.8,)
            cbar = plt.colorbar()
            cbar.set_label("|$E_z$|")
            plt.xlabel(r"y in $\mu$m")
            plt.ylabel(r"z in $\mu$m")
            plt.savefig(f"./Problems/cross/results/field_high_{j}_{i:03}.png")
            plt.close()

            plt.imshow(epsr[0][i].T, origin='lower', cmap='gray')
            plt.imshow(jnp.abs(E[1][0][i].T), origin='lower', cmap="magma", alpha=0.8,)
            cbar = plt.colorbar()
            cbar.set_label("|$E_z$|")
            plt.xlabel(r"y in $\mu$m")
            plt.ylabel(r"z in $\mu$m")
            plt.savefig(f"./Problems/cross/results/field_low_{j}_{i:03}.png")
            plt.close()

            plt.imshow(epsr[0][i].T, origin='lower', cmap='gray')
            cbar = plt.colorbar()
            cbar.set_label(r"$\epsilon_r$")
            plt.xlabel(r"x in $\mu$m")
            plt.ylabel(r"y in $\mu$m")
            plt.savefig(f"./Problems/cross/results/eps_{j}_{i:03}.png")
            plt.close()




        plt.plot(loss_hist)
        # plt.yscale('log')
        plt.grid(visible=True, which="minor", linewidth=0.2)
        plt.grid(visible=True, which="major")
        plt.savefig(f"./Problems/cross/results/history_{j}.png")
        plt.close()

    return save_plot

def data_maker():
    # don't ask, it's for avoiding opening and closing the same file with parallel executions on the same machine >:(
    name = {"a": 1, "b": 2, "c": 3, "d": 4}
    def save_data(E, T, epsr, loss_hist, i):
        with h5py.File(f"./Problems/cross/results/data_{i}.h5", "w") as name[i]:
            grp = name[i].create_group("lens_3D")
            grp.create_dataset("rho", data=epsr[0])
            grp.create_dataset("E_high", data=jnp.abs(E[0][0]))
            grp.create_dataset("E_low", data=jnp.abs(E[1][0]))
            grp.create_dataset("T_m", data=jnp.abs(T[0]))
            grp.create_dataset("T_v", data=jnp.abs(T[1]))
            grp.create_dataset("loss", data=loss_hist)
            name[i].close()
    return save_data
