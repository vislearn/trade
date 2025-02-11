
import numpy as np
import torch
import matplotlib.pyplot as plt
import mdtraj as md
from matplotlib.colors import LogNorm
from .data import get_loader
from matplotlib.ticker import FuncFormatter
from itertools import product
from math import floor
from functools import partial
import seaborn as sns

@torch.no_grad()
def create_plots(samples, flow, parameter_value, parameter_name, parameter_reference_value, writer, dataset_hparams, step=0):
    name = dataset_hparams["name"]
    try:
        dataset = get_loader(name)(**{parameter_name:parameter_value})
        reference_data = torch.from_numpy(dataset.coordinates.reshape(-1, dataset.dim))
        if hasattr(dataset, "conditions"):
            reference_conditions = torch.from_numpy(dataset.conditions)
        else:
            reference_conditions = None
    except (FileNotFoundError, ValueError):
        dataset = get_loader(name)(**{parameter_name:parameter_value, "read":False})
        reference_data = None
        reference_conditions = None
    system = dataset.system
    target_energy = partial(dataset.get_energy_model().energy, **{parameter_name:parameter_value})

    if name == "ala2":
        fig, ax = plot_ala2(samples, system, target_energy, reference_data)
        plot_name = "Alanine Dipeptide"
    elif name.startswith("double_well"):
        if dataset.dim == 2:
            fig, ax = plot_double_well_2d(samples, system, target_energy, reference_data)
        else:
            fig, ax = plot_nd_histogram(samples, system, target_energy, reference_data)
        plot_name = f"Double well {dataset.dim}D"
    elif name.startswith("multi_well"):
        fig, ax = plot_nd_histogram(samples, system, target_energy, reference_data)
        plot_name = f"Multi well {dataset.dim}D"    
    elif name.lower().startswith("gmm"):
        if dataset.dim == 2:
            fig, ax = plot_gmm_2d(samples, system, target_energy, reference_data, partial(flow.energy, parameter=parameter_value/parameter_reference_value))
            plot_name = "Gaussian Mixture Model 2D"
        else:
            fig, ax = plot_nd_histogram(samples, system, target_energy, reference_data)
            plot_name = f"Gaussian Mixture Model {dataset.dim}D"
    elif name.startswith("two_moons"):
        # Load different reference data
        plot_data = np.load(f"data/two_moons_target_obs_{parameter_value:.1f}.npz")
        reference_data = torch.from_numpy(plot_data["coordinates"])
        reference_conditions = torch.from_numpy(plot_data["conditions"])
        fig, ax = plot_two_moons(flow, reference_data, reference_conditions, parameter_value/parameter_reference_value)
        plot_name = "Two moons"
    else:
        raise ValueError(f"Unknown dataset {name}")
    
    writer.add_figure(f"{plot_name} {parameter_name}={parameter_value}", fig, step)
    plt.close(fig)


def plot_ala2(samples, system, target_energy, reference_data=None):
    has_reference = reference_data is not None
    fig, ax = plt.subplots(1, 3 + has_reference, figsize=(5*(3 + has_reference), 5))
    plot_energies(ax[2+has_reference], samples, target_energy, reference_data)
    ax[-1].set_title("Energy distribution")

    samples = samples.cpu().detach().numpy()
    if has_reference:
        reference_data = reference_data.cpu().detach().numpy()
        vmin, vmax = plot_phi_psi(ax[1], reference_data, system)
        ax[1].set_title("Ramachandran plot (MD)")
    else:
        vmin, vmax = None, None
    plot_phi_psi(ax[0], samples, system, vmin=vmin, vmax=vmax)
    plot_phi(ax[1+has_reference], samples, system, reference_data)
    ax[0].set_title("Ramachandran plot (BG)")
    plt.tight_layout()
    return fig, ax

def plot_phi(ax, trajectory, system, reference_data=None):
    if not isinstance(trajectory, md.Trajectory):
        trajectory = md.Trajectory(
            xyz=trajectory.reshape(-1, 22, 3), 
            topology=system.mdtraj_topology
        )
    phi, _ = system.compute_phi_psi(trajectory)
    p_phi = np.histogram(phi, bins=100, range=(-np.pi, np.pi), density=True)
    ax.plot(p_phi[1][:-1], p_phi[0], label="BG")

    if reference_data is not None:
        reference_data = md.Trajectory(
            xyz=reference_data.reshape(-1, 22, 3), 
            topology=system.mdtraj_topology
        )
        phi_ref, _ = system.compute_phi_psi(reference_data)
        p_phi_ref = np.histogram(phi_ref, bins=100, range=(-np.pi, np.pi), density=True)
        ax.plot(p_phi_ref[1][:-1], p_phi_ref[0], label="MD")

    ax.set_xlim(-np.pi, np.pi)
    ax.set_xlabel("$\phi$")
    ax.set_ylabel("$p(\phi)$")

def plot_nd_histogram(samples, system, target_energy, reference_data=None):
    has_reference = reference_data is not None
    n_plots = samples.shape[-1] + 1
    fig, ax = plt.subplots(floor(n_plots/5)+1, n_plots if n_plots < 5 else 5, figsize=((5*n_plots) if n_plots < 5 else 25, 5*(floor(n_plots/5)+1)))
    if len(ax.shape) == 1:
        ax = ax.reshape(1, -1)
    plot_energies(ax[floor((n_plots-1)/5), (n_plots-1)%5], samples, target_energy, reference_data)
    ax[floor((n_plots-1)/5), (n_plots-1)%5].set_title("Energy distribution")

    samples = samples.cpu().detach().numpy()
    if has_reference:
        reference_data = reference_data.cpu().detach().numpy()
    for i in range(samples.shape[-1]):
        ax[floor(i/5), i%5].hist(samples[:, i], bins=30, density=True, alpha=0.5, label="BG", range=(-2.2, 2.2))
        if has_reference:
            ax[floor(i/5), i%5].hist(reference_data[:, i], bins=30, density=True, alpha=0.5, label="MD", range=(-2.2, 2.2))
        ax[floor(i/5), i%5].set_xlabel(f"Dimension {i}")
        ax[floor(i/5), i%5].set_xlim(-2.2, 2.2)
    ax[floor((n_plots-2)/5), (n_plots-2)%5].legend()
    for i in range(n_plots, ax.shape[0]*ax.shape[1]):
        ax[floor(i/5), i%5].axis("off")
    plt.tight_layout()
    return fig, ax

def plot_double_well_2d(samples, system, target_energy, reference_data=None):
    has_reference = reference_data is not None
    fig, ax = plt.subplots(1, 2 + has_reference, figsize=(5*(2 + has_reference), 5))
    plot_energies(ax[1+has_reference], samples, target_energy, reference_data)
    ax[1+has_reference].set_title("Energy distribution")
    
    samples = samples.cpu().detach().numpy()
    if has_reference:
        reference_data = reference_data.cpu().detach().numpy()
    ax[0].hist2d(samples[:, 0], samples[:, 1], 50, norm=LogNorm(), range=[[-2.2, 2.2], [-4, 4]])
    ax[0].set_title("Double well (BG)")

    if has_reference:
        ax[1].hist2d(reference_data[:, 0], reference_data[:, 1], 50, norm=LogNorm(), range=[[-2.2, 2.2], [-4, 4]])
        ax[1].set_title("Double well (MD)")
    plt.tight_layout()
    return fig, ax

def plot_two_moons(flow, reference_data, reference_conditions, parameter):
    has_reference = reference_data is not None
    fig, ax = plt.subplots(1, 1 + has_reference, figsize=(5*(1 + has_reference), 5))
    target_observation = torch.zeros(10000, 2, device=list(flow.parameters())[0].device)
    samples = flow.sample(10000, c=[target_observation], parameter=parameter)
    if has_reference:
        ind_close = torch.where(torch.norm(reference_conditions - target_observation[:1].to(reference_conditions.device), dim=1) < 0.001)[0]
        data_close = reference_data[ind_close].cpu().detach().numpy()
    samples = samples.cpu().detach().numpy()

    # Plot for NF samples
    sns.kdeplot(x=samples[:, 0], y=samples[:, 1], ax=ax[0], fill=True, cmap='Blues', thresh=0.1)
    ax[0].scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.3, c='blue', edgecolor='none')  # Reduced size and added transparency
    ax[0].set_title("Two moons (NF)")
    ax[0].grid(True)

    if has_reference:
        # Adjusted KDE with smaller bandwidth for few points
        sns.kdeplot(x=data_close[:, 0], y=data_close[:, 1], ax=ax[1], fill=True, cmap='Reds', bw_adjust=0.5, thresh=0.1)
        
        # Emphasize the actual reference points
        ax[1].scatter(data_close[:, 0], data_close[:, 1], s=30, alpha=0.8, c='red', edgecolor='black', label='Reference points')  # Larger and more prominent points
        ax[1].legend()
        
        # Add a rug plot to show exact positions of few points
        sns.rugplot(x=data_close[:, 0], y=data_close[:, 1], ax=ax[1], height=0.05, color='black')
        
        ax[1].set_title("Two moons (GT)")
        ax[1].grid(True)

    plt.tight_layout()
    return fig, ax

def plot_gmm_2d(samples, system, target_energy, reference_data=None, energy_fnc=None):
    has_reference = reference_data is not None
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    plot_energies(ax[-1], samples, target_energy, reference_data)
    ax[-1].set_title("Energy distribution")

    # Determine the contour levels using target_energy
    x_points_1d = torch.linspace(-56, 56, 200)
    x_points = torch.tensor(list(product(x_points_1d, x_points_1d)))

    # Calculate the log probability values using target_energy
    log_p_x_target = - target_energy(x_points.to(samples.device)).cpu().detach()
    log_p_x_target = torch.clamp_min(log_p_x_target, -1000).reshape(200, 200)

    # Create grids for plotting
    x_grid = x_points[:, 0].reshape((200, 200)).numpy()
    y_grid = x_points[:, 1].reshape((200, 200)).numpy()

    # Determine the levels from the log probability values
    levels = np.linspace(log_p_x_target.min(), log_p_x_target.max(), 80)

    # Plot the contours for both energy functions
    for i, energy in enumerate((energy_fnc, target_energy)):
        if energy is not None:
            log_p_x = - energy(x_points.to(samples.device)).cpu().detach()
            log_p_x = torch.clamp_min(log_p_x, -1000).reshape(200, 200)
            ax[i].contour(x_grid, y_grid, log_p_x, levels=levels)

    
    samples = samples.cpu().detach().numpy()
    if has_reference:
        reference_data = reference_data.cpu().detach().numpy()

    ax[0].scatter(samples[:, 0], samples[:, 1])
    ax[0].set_title("GMM (BG)")
    ax[0].set_xlim(-56, 56)
    ax[0].set_ylim(-56, 56)

    if has_reference:
        ax[1].scatter(reference_data[:, 0], reference_data[:, 1])
        ax[1].set_title("GMM (MD)")
    plt.tight_layout()
    return fig, ax

def plot_phi_psi(ax, trajectory, system, vmin=None, vmax=None):
    if not isinstance(trajectory, md.Trajectory):
        trajectory = md.Trajectory(
            xyz=trajectory.reshape(-1, 22, 3), 
            topology=system.mdtraj_topology
        )
    phi, psi = system.compute_phi_psi(trajectory)
    hist = ax.hist2d(phi, psi, 50, norm=LogNorm(vmin=vmin, vmax=vmax), density=True)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel("$\phi$")
    ax.set_ylabel("$\psi$")
    
    return hist[-1].get_clim()

def plot_energies(ax, samples, target_energy, test_data):
    sample_energies = target_energy(samples).cpu().detach().numpy()
    if test_data is not None:
        md_energies = target_energy(test_data[:len(samples)]).cpu().detach().numpy()
    else:
        md_energies = sample_energies
    full_range = np.nanmin([np.nanmin(sample_energies), np.nanmin(md_energies)]) , np.nanmax([np.nanmax(sample_energies), np.nanmax(md_energies)])
    cut = max(np.nanpercentile(sample_energies, 80), np.nanmax(md_energies))
    plot_range = (full_range[0] - 0.1*(cut - full_range[0]), cut)

    ax.set_xlabel("Energy   [$k_B T$]")
    # y-axis on the right
    ax2 = plt.twinx(ax)
    ax.get_yaxis().set_visible(False)
    
    # This adjusts the counts to be comparable, even if different number of samples fall into the range
    count_sample_energies = np.sum(np.logical_and(sample_energies < plot_range[1], sample_energies > plot_range[0]))
    count_md_energies = np.sum(np.logical_and(md_energies < plot_range[1], md_energies > plot_range[0]))
    weights = np.ones_like(sample_energies)*count_md_energies/count_sample_energies


    ax2.hist(sample_energies, range=plot_range, bins=40, weights=weights, density=False, label="BG", alpha=0.5)
    if test_data is not None:
        ax2.hist(md_energies, range=plot_range, bins=40, density=False, label="MD", alpha=0.5)
    ax2.set_ylabel(f"Count   [#Samples / {len(samples)}]")
    ax2.legend()

@torch.no_grad()
def plot_minor_mode(parameter_samples, writer, name="ala2", step=0):
    minor_mode = []

    for i, (parameter, samples) in enumerate(parameter_samples):
        try:
            dataset = get_loader(name)(parameter=parameter)
            reference_data = torch.from_numpy(dataset.coordinates.reshape(-1, dataset.dim))
        except ValueError:
            continue
        system = dataset.system
        if not isinstance(samples, md.Trajectory):
            samples = md.Trajectory(
                xyz=samples.cpu().detach().numpy().reshape(-1, 22, 3), 
                topology=system.mdtraj_topology
            )

        if not isinstance(reference_data, md.Trajectory):
            reference_data = md.Trajectory(
                xyz=reference_data.cpu().detach().numpy().reshape(-1, 22, 3), 
                topology=system.mdtraj_topology
            )

        phi_sampled, _ = system.compute_phi_psi(samples)
        phi_reference, _ = system.compute_phi_psi(reference_data)

        minor_mode.append(np.mean(np.logical_and(phi_sampled > 0, phi_sampled < 2.5))/np.mean(np.logical_and(phi_reference > 0, phi_reference < 2.5)))

    fig = plt.figure(figsize=(5, 5), dpi=150)
    plt.plot([t for t, _ in parameter_samples], minor_mode)
    plt.axhline(1, color="black", linestyle="--")
    plt.xlabel("Temperature")
    plt.ylabel(r"Minor mode $\frac{\phi > 0}{\phi_{MD} > 0}$")
    plt.title("Minor mode occupancy")
    writer.add_figure(f"Minor mode", fig, step)
    plt.close(fig)

def plot_latent_space(latent, writer, step=0):
    fig = plt.figure(figsize=(5, 5), dpi=150)
    latent = latent.flatten().numpy()
    plt.hist(latent, bins=50, density=True, range=(-5, 5))
    plt.title("Latent space")
    plt.xlim(-5, 5)
    x = np.linspace(-5, 5, 100)
    plt.plot(x, np.exp(-x**2/2)/np.sqrt(2*np.pi), label="Normal distribution")
    writer.add_figure(f"Latent space", fig, step)
    plt.close(fig)

def plot_consistency_check(kl_divs, parameters, writer, step=0):
    fig = plt.figure(figsize=(6, 5), dpi=150)
    plt.axhline(0, color="black", linestyle="--")
    plt.plot(parameters, kl_divs)
    plt.xlabel("Temperature")
    plt.ylabel("KL divergence")
    plt.title("Consistency check")
    plt.yscale("symlog")
    plt.tight_layout()
    def custom_format(x, pos):
        return '{:.1e}'.format(x)
    plt.ylim(-1e-3, max(max(kl_divs)*1.1, 1))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(custom_format))

    writer.add_figure(f"Consistency check", fig, step)
    plt.close(fig)
