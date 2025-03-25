# Imports   
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import das4whales as dw
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from scipy.interpolate import RegularGridInterpolator
import cmocean.cm as cmo

def plot_associated(peaks, longi_offset, associated_list, localizations, cable_pos, dist, dx, c0, fs):
    fig = plt.figure(figsize=(20,8))

    # Plot the time picks with colored associated ones
    plt.subplot(1, 2, 1)
    plt.scatter(peaks[1][:] / fs, (longi_offset + peaks[0][:]) * dx * 1e-3, label='LF', s=0.5, alpha=0.2, color='tab:grey')
    for i, select in enumerate(associated_list):
        plt.scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3, label='LF', s=0.5)
    plt.xlim(0, 60)
    plt.xlabel('Time [s]')
    plt.ylabel('Distance [km]')

    # Plot the time picks with the the predicted hyperbola
    plt.subplot(1, 2, 2)
    for i, select in enumerate(associated_list):
        plt.scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3, label='LF', s=0.5)
        plt.plot(dw.loc.calc_arrival_times(localizations[i][-1], cable_pos, localizations[i][:3], c0), dist/1e3, color='tab:grey', ls='-', lw=2, alpha=0.7)
        # plt.plot(select[1][:] / fs, dw.loc.calc_arrival_times(0, cable_pos, alt_localizations[i][:3], c0), color='tab:orange', ls='-', lw=1)
    plt.grid(linestyle='--', alpha=0.6)
    plt.xlabel('Time [s]')
    plt.ylabel('Distance [km]')
    return fig


def compute_kde(delayed_picks, t_kde, bin_width):
    """Computes the KDE of the delayed picks.

    Parameters
    ----------
    delayed_picks : array-like
        Delayed picks array.
    t_kde : array-like
        Time grid for the KDE.
    bin_width : float
        Bin width for the KDE.

    Returns
    -------
    array-like
        KDE density values.  
    
    """

    # kde = gaussian_kde(delayed_picks, bw_method=bin_width / (np.max(t_kde) - np.min(t_kde)))
    kde = KernelDensity(kernel="epanechnikov", bandwidth=bin_width, algorithm='ball_tree')
    kde.fit(delayed_picks[:, None]) # Reshape to (n_samples, 1)
    log_dens = kde.score_samples(t_kde[:, np.newaxis]) # Evaluate on grid
    return np.exp(log_dens) # Convert log-density to normal density


def compute_selected_picks(peaks, hyperbola, dt_sel, fs):
    """Selects picks that are closest to the hyperbola within a given time window."""
    selected_picks = ([], [])
    for i, idx in enumerate(peaks[1]):
        dist_idx = peaks[0][i]
        pick_time = idx / fs

        if hyperbola[dist_idx] - dt_sel < pick_time < hyperbola[dist_idx] + dt_sel:
            if dist_idx in selected_picks[0]:
                existing_idx = selected_picks[0].index(dist_idx)
                if abs(hyperbola[dist_idx] - pick_time) < abs(hyperbola[dist_idx] - selected_picks[1][existing_idx] / fs):
                    selected_picks[1][existing_idx] = idx  # Replace with closer pick
            else:
                selected_picks[0].append(dist_idx)
                selected_picks[1].append(idx)
    
    return np.array(selected_picks[0]), np.array(selected_picks[1])


def compute_curvature(w_times, w_distances):
    """Computes curvature using second derivatives."""
    ddx = np.diff(w_times)
    ddy = np.diff(w_distances)
    ddx2 = np.diff(ddx)
    ddy2 = np.diff(ddy)
    curvature = np.abs(ddx2 * ddy[1:] - ddx[1:] * ddy2) / (ddx[1:]**2 + ddy[1:]**2)**(3/2)
    return np.mean(curvature)


def associate_picks(kde, t_grid, longi_offset, up_peaks, arr_tg, dx, c0, w_eval, dt_sel, fs, cable_pos, associated_list, used_hyperbolas, rejected_list, rejected_hyperbolas):
    """Associates picks with hyperbolas and updates the picks list."""
    # Find the maximum of the KDE
    max_kde_idx = np.argmax(kde)
    imax, tmax = np.unravel_index(max_kde_idx, kde.shape)
    max_time = t_grid[imax, tmax].item()
    # Select the picks that are within the 1.4 s window of the hyperbola
    hyperbola = max_time + arr_tg[imax, :] # Theoretical arrival times for the selected hyperbola
    idx_dist, idx_time = compute_selected_picks(up_peaks, hyperbola, dt_sel, fs) # Select the picks around the hyperbola within +/- dt_sel

    times = idx_time / fs
    distances = (longi_offset + idx_dist) * dx * 1e-3

    window_mask = (times > np.min(times)) & (times < np.min(times) + w_eval)
    w_times = times[window_mask]
    w_distances = distances[window_mask]

    # Calulate least squares fit
    idxmin_t = np.argmin(idx_time)
    apex_loc = cable_pos[:, 0][idx_dist[idxmin_t]]
    Ti = idx_time / fs
    Nbiter = 20
    # Initial guess (apex_loc, mean_y, -30m, min(Ti))
    n_init = [apex_loc, np.mean(cable_pos[:,1]), -40, np.min(Ti)]

    # Solve the least squares problem
    n, residuals = dw.loc.solve_lq(Ti, cable_pos[idx_dist], c0, Nbiter, fix_z=True, ninit=n_init, residuals=True)
    # rms residual
    rms = np.sqrt(np.mean(residuals[window_mask]**2))
    
    if rms < .5:
        # Compute the residual cumsum from the minimum time, in positive and negative directions
        #TODO: change variable names
        left_cs = np.cumsum(abs(residuals[idxmin_t::-1])) # negative direction
        right_cs = np.cumsum(abs(residuals[idxmin_t:])) # positive direction
        mod_cs = np.concatenate((left_cs[::-1], right_cs[1:]))

        mask_resi = mod_cs < 1500 # Mask the residuals that are below the threshold, key parameter

        associated_list.append(np.asarray((idx_dist[mask_resi], idx_time[mask_resi])))
        used_hyperbolas.append(arr_tg[imax, :])
        arr_tg[imax, :] = dw.loc.calc_arrival_times(0, cable_pos, n[:3], c0)

        # Remove selected picks from updated picks
        # Create a boolean mask that starts by marking every column as True (to keep)
        mask = np.ones(up_peaks.shape[1], dtype=bool)
        for d, t in zip(idx_dist[mask_resi], idx_time[mask_resi]):   # For each pair to remove, update the mask
            mask &= ~((up_peaks[0, :] == d) & (up_peaks[1, :] == t))
        # Apply the mask only once to filter the columns
        up_peaks = up_peaks[:, mask]

    # if compute_curvature(w_times, w_distances) < 1000:
    #     associated_list.append(np.asarray((sidx_dist, sidx_time)))
    #     used_hyperbolas.append(arr_tg[imax, :])

    else:
        # Add the rejected hyperbola to the list
        rejected_list.append(np.asarray((idx_dist, idx_time)))
        rejected_hyperbolas.append(arr_tg[imax, :])
        # Remove the hyperbola from the list
        arr_tg = np.delete(arr_tg, imax, axis=0)

    return up_peaks, arr_tg, associated_list, used_hyperbolas, rejected_list, rejected_hyperbolas


def plot_reject_pick(peaks, longi_offset, dist, dx, associated_list, rejected_list, rejected_hyperbolas, fs):
    # Plot the selected picks alongside the original picks
    fig = plt.figure(figsize=(20,8))
    plt.subplot(2, 2, 1)
    plt.scatter(peaks[1][:] / fs, (longi_offset + peaks[0][:]) * dx * 1e-3, label='HF', s=0.5)
    plt.xlabel('Time [s]')
    plt.ylabel('Distance [km]')
    plt.subplot(2, 2, 2)
    for select in associated_list:
        plt.scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3, label='LF', s=0.5)
    plt.xlabel('Time [s]') 
    # Plot the deleted hyperbolas
    plt.subplot(2, 2, 3)
    for hyp in rejected_hyperbolas:
        plt.plot(hyp, dist/1e3, label='Rejected hyperbola')
    plt.xlabel('Time [s]')
    plt.ylabel('Distance [km]')
    # plot the rejected picks
    plt.subplot(2, 2, 4)
    for select in rejected_list:
        plt.scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3, label='LF', s=0.5)
    plt.xlabel('Time [s]')
    return fig


def plot_pick_analysis(associated_list, fs, dx, longi_offset, cable_pos, dist, window_size=5, mu_ref=None, sigma_ref=None):
        """
        Create detailed plots of seismic picks with continuity analysis and a normalized curvature score.
        
        Parameters:
        -----------
        associated_list : list
            List of tuples containing pick coordinates and times
        fs : float
            Sampling frequency
        dx : float
            Spatial sampling interval
        longi_offset : float
            Longitudinal offset value
        window_size : float, optional
            Size of analysis window in seconds (default: 5)
        mu_ref : float, optional
            Reference mean curvature for normalization (default: computed from data)
        sigma_ref : float, optional
            Reference standard deviation of curvature for normalization (default: computed from data)
        
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The created figure object
        """
        
        fig = plt.figure(figsize=(24, 8))
        
        curvature_means = []
        curvature_stds = []
        
        for i, select in enumerate(associated_list):
            times = select[1][:] / fs
            distances = (longi_offset + select[0][:]) * dx * 1e-3
            
            ax = plt.subplot(1, 2*len(associated_list), (i + 1) * 2 - 1)
            if i == 0:
                ax.set_ylabel('Distance [km]')
            ax.scatter(times, distances, label='All Picks', s=0.5, color='gray', alpha=0.5)
            
            window_mask = (times > np.min(times)) & (times < np.min(times) + window_size)
            window_times = times[window_mask]
            window_distances = distances[window_mask]
            
            ax.plot(window_times, window_distances, 
                    label='Windowed Picks', 
                    lw=2, 
                    color='tab:red', 
                    alpha=0.6)
            # Calulate least squares fit
            idxmin_t = np.argmin(select[1][:])
            apex_loc = cable_pos[:, 0][select[0][idxmin_t]]
            Ti = select[1][:] / fs
            Nbiter = 20

            # Initial guess (apex_loc, mean_y, -30m, min(Ti))
            n_init = [apex_loc, np.mean(cable_pos[:,1]), -40, np.min(Ti)]

            # Solve the least squares problem
            n, residuals = dw.loc.solve_lq(Ti, cable_pos[select[0][:]], c0, Nbiter, fix_z=True, ninit=n_init, residuals=True)
            loc_hyerbola = dw.loc.calc_arrival_times(n[-1], cable_pos, n[:3], c0)
            test = np.cumsum(abs(residuals))
            # rms residual
            rms = np.sqrt(np.mean(residuals[window_mask]**2))
            # rms *= 1e4

            left_cs = np.cumsum(abs(residuals[idxmin_t::-1]))
            right_cs = np.cumsum(abs(residuals[idxmin_t:]))
            mod_cs = np.concatenate((left_cs[::-1], right_cs[1:]))

            mask_resi = mod_cs < 1500
            # plot indexes for which only the cumulative sum is less than 1000
            ax.scatter(select[1][mask_resi] / fs, (longi_offset + select[0][mask_resi]) * dx * 1e-3, label='HF', s=1, color='tab:blue')
            ax.plot(loc_hyerbola, dist/1e3, label='Hyperbola', color='tab:green', alpha=0.5)

            # Plot residuals
            # ax.plot(abs(residuals), distances, label='Residuals', color='tab:orange', alpha=0.5)
            # ax.plot(abs(residuals[window_mask]), window_distances, label='Windowed Residuals', color='tab:blue', alpha=0.5)
            # ax.plot(np.cumsum(residuals), distances, label='Cumulative Residuals', color='tab:green', alpha=0.5)
            

            # Calculate curvature
            ddx = np.diff(window_times)
            ddy = np.diff(window_distances)
            ddx2 = np.diff(ddx)
            ddy2 = np.diff(ddy)
            curvature = np.abs(ddx2 * ddy[1:] - ddx[1:] * ddy2) / (ddx[1:]**2 + ddy[1:]**2)**(3/2)
            # curvature = curvature[curvature > 10e-10]
            curvature_mean = np.mean(curvature)

            ax.set_title(f"Pick Analysis\n"
                            f"$\\mu_k$ = {compute_curvature(window_times, window_distances):.2f}\n"
                            f"$\\mu_r$ = {np.mean(abs(residuals[window_mask])):.2f}\n"
                            f"$RMS$ = {rms:.2f}\n",
                            fontsize=10)
            ax.set_xlabel('Time [s]')
            
            ax = plt.subplot(1, 2*len(associated_list), (i + 2) * 2 - 2)
            ax.plot(mod_cs, distances, label='Modified Cumulative Residuals', color='tab:purple', alpha=0.5)
            ax.set_xlabel('Cumulative Residuals')
            
        plt.tight_layout()
        return fig


def loc_from_picks(associated_list, cable_pos, c0, fs):
    localizations = []
    alt_localizations = []

    for select in associated_list:
        idxmin_t = np.argmin(select[1][:])
        apex_loc = cable_pos[:, 0][select[0][idxmin_t]]
        Ti = select[1][:] / fs
        Nbiter = 20

        # Initial guess (apex_loc, mean_y, -30m, min(Ti))
        n_init = [apex_loc, np.mean(cable_pos[:,1]), -40, np.min(Ti)]
        print(f'Initial guess: {n_init[0]:.2f} m, {n_init[1]:.2f} m, {n_init[2]:.2f} m, {n_init[3]:.2f} s')
        # Solve the least squares problem
        n = dw.loc.solve_lq(Ti, cable_pos[select[0][:]], c0, Nbiter, fix_z=True, ninit=n_init)
        nalt = dw.loc.solve_lq(Ti, cable_pos[select[0][:]], c0, Nbiter-1, fix_z=True, ninit=n_init)

        localizations.append(n)
        alt_localizations.append(nalt)

    return localizations, alt_localizations


def main(n_ds_path, s_ds_path):
    # Load the peak indexes and the metadata
    n_ds = xr.load_dataset(n_ds_path) 
    s_ds = xr.load_dataset(s_ds_path)

    # Constants from the metadata
    fs = n_ds.attrs['fs']
    dx = n_ds.attrs['dx']
    nnx = n_ds.attrs['data_shape'][0]
    snx = s_ds.attrs['data_shape'][0]
    n_selected_channels_m = n_ds.attrs['selected_channels_m']
    s_selected_channels_m = s_ds.attrs['selected_channels_m']
    fileBeginTimeUTC = n_ds.attrs['fileBeginTimeUTC']
        
    # Constants management
    c0 = 1480
    n_selected_channels = dw.data_handle.get_selected_channels(n_selected_channels_m, dx)
    s_selected_channels = dw.data_handle.get_selected_channels(s_selected_channels_m, dx)
    n_begin_chan = n_selected_channels[0]
    n_end_chan = n_selected_channels[1]
    n_longi_offset = n_selected_channels[0] // n_selected_channels[2]
    s_begin_chan = s_selected_channels[0]
    s_end_chan = s_selected_channels[1]
    s_longi_offset = s_selected_channels[0] // s_selected_channels[2]
    n_dist = (np.arange(nnx) * n_selected_channels[2] + n_selected_channels[0]) * dx
    s_dist = (np.arange(snx) * s_selected_channels[2] + s_selected_channels[0]) * dx
    dx = dx * n_selected_channels[2]

    # load the peak indexes - North cable
    npeakshf = n_ds["peaks_indexes_tp_HF"].values  # Extract as NumPy array
    npeakslf = n_ds["peaks_indexes_tp_LF"].values

    # load the peak indexes - South cable
    speakshf = s_ds["peaks_indexes_tp_HF"].values
    speakslf = s_ds["peaks_indexes_tp_LF"].values

    # Choose to work on the HF or LF calls
    n_peaks = npeakshf
    s_peaks = speakshf

    # Import the cable location
    df_north = pd.read_csv('data/north_DAS_multicoord.csv')
    df_south = pd.read_csv('data/south_DAS_multicoord.csv')

    # Extract the part of the dataframe used for the time picking process
    idx_shift0 = int(n_begin_chan - df_north["chan_idx"].iloc[0]) # Shift between the cable locations (starting at the beach) and the channel locations
    idx_shiftn = int(n_end_chan - df_north["chan_idx"].iloc[-1])

    df_north_used = df_north.iloc[idx_shift0:idx_shiftn:n_selected_channels[2]][:nnx]

    idx_shift0 = int(s_begin_chan - df_south["chan_idx"].iloc[0]) # Shift between the cable locations (starting at the beach) and the channel locations
    idx_shiftn = int(s_end_chan - df_south["chan_idx"].iloc[-1])

    df_south_used = df_south.iloc[idx_shift0:idx_shiftn:s_selected_channels[2]][:snx]

    # Import the bathymetry data
    bathy, xlon, ylat = dw.map.load_bathymetry('data/GMRT_OOI_RCA_Cables.grd')
    print(f'Origin of the corrdinates. Latitude = {ylat[0]}, Longitude = {xlon[-1]}')

    utm_x0, utm_y0 = dw.map.latlon_to_utm(xlon[0], ylat[0])
    utm_xf, utm_yf = dw.map.latlon_to_utm(xlon[-1], ylat[-1])

    # Change the reference point to the last point
    x0, y0 = utm_xf - utm_x0, utm_y0 - utm_y0
    xf, yf = utm_xf - utm_xf, utm_yf - utm_y0
    print(xf, yf)
    # # Create vectors of coordinates
    utm_x = np.linspace(utm_x0, utm_xf, len(xlon))
    utm_y = np.linspace(utm_y0, utm_yf, len(ylat))
    x = np.linspace(x0, xf, len(xlon))
    y = np.linspace(y0, yf, len(ylat))

    # dw.map.plot_cables2D(df_north, df_south, bathy, xlon, ylat)

    # Cable geometry (make it correspond to x,y,z = cable_pos[:, 0], cable_pos[:, 1], cable_pos[:, 2])
    n_cable_pos = np.zeros((len(df_north_used), 3))
    s_cable_pos = np.zeros((len(df_south_used), 3))

    n_cable_pos[:, 0] = df_north_used['x']
    n_cable_pos[:, 1] = df_north_used['y']
    n_cable_pos[:, 2] = df_north_used['depth']

    s_cable_pos[:, 0] = df_south_used['x']
    s_cable_pos[:, 1] = df_south_used['y']
    s_cable_pos[:, 2] = df_south_used['depth']

    # Create a grid of coordinates, choosing the spacing of the grid
    dx_grid = 2000 # [m]
    dy_grid = 2000 # [m]
    xg, yg = np.meshgrid(np.arange(xf, x0, dx_grid), np.arange(y0, yf, dy_grid))

    ti = 0
    zg = -40

    interpolator = RegularGridInterpolator((x, y),  bathy.T)
    bathy_interp = interpolator((xg, yg))

    # Remove points if the ocean depth is too shallow (i.e., less than -25 m)
    mask = bathy_interp < -25
    # Compute arrival times only for valid grid points
    # Flatten the grid points
    xg, yg = xg[mask], yg[mask]

    # Compute KDEs for all delayed picks
    Nkde = 300
    bin_width = 1
    kde_hf = np.empty((xg.shape[0], Nkde))
    n_shape_x = xg.shape[0]
    s_shape_x = xg.shape[0]
    dt_sel = 1.4 # [s] Selected time "distance" from the theoretical arrival time
    w_eval = 5 # [s] Width of the evaluation window for curvature estimation
    # Set the number of iterations for testing
    iterations = 4

    # Initialize the max_kde variable to enter the loop
    n_associated_list = []
    n_used_hyperbolas = []
    n_rejected_list = []
    n_rejected_hyperbolas = []

    s_associated_list = []
    s_used_hyperbolas = []
    s_rejected_list = []
    s_rejected_hyperbolas = []

    n_up_peaks = np.copy(n_peaks)
    s_up_peaks = np.copy(s_peaks)
    n_arr_tg = dw.loc.calc_arrival_times(ti, n_cable_pos, (xg, yg, zg), c0)
    s_arr_tg = dw.loc.calc_arrival_times(ti, s_cable_pos, (xg, yg, zg), c0)

    pbar = tqdm(range(iterations), desc="Associated calls: 0")

    # Start the loop that runs for a fixed number of iterations
    for iteration in pbar:
        # Precompute the time indices
        n_idx_times = np.array(n_up_peaks[1]) / fs # Update with the remaining peaks
        s_idx_times = np.array(s_up_peaks[1]) / fs # Update with the remaining peaks

        # Make a delayed picks array for all the grid points
        # Broadcast the time indices delayed by the theoretical arrival times for the grid points
        n_delayed_picks_hf = n_idx_times[None, :] - n_arr_tg[:, n_up_peaks[0]]
        s_delayed_picks_hf = s_idx_times[None, :] - s_arr_tg[:, s_up_peaks[0]]

        # Generate a time grid for each grid point by linearly spacing Nkde points 
        # between the minimum and maximum delayed pick times. Transpose to ensure 
        # the shape (shape_x, Nkde) (shape_x, Nkde) delayed_picks shape for consistency with KDE computation.
        n_t_grid = np.linspace(np.min(n_delayed_picks_hf, axis=1), np.max(n_delayed_picks_hf, axis=1), Nkde).T
        s_t_grid = np.linspace(np.min(s_delayed_picks_hf, axis=1), np.max(s_delayed_picks_hf, axis=1), Nkde).T

        # Parallelized KDE computation
        n_kde_hf = np.array(Parallel(n_jobs=-1)(
            delayed(compute_kde)(n_delayed_picks_hf[i, :], n_t_grid[i, :], bin_width) 
            for i in range(n_shape_x)
        ))

        s_kde_hf = np.array(Parallel(n_jobs=-1)(
            delayed(compute_kde)(s_delayed_picks_hf[i, :], s_t_grid[i, :], bin_width)
            for i in range(s_shape_x)
        ))


        n_up_peaks, n_arr_tg, n_associated_list, n_used_hyperbolas, n_rejected_list, n_rejected_hyperbolas = associate_picks(n_kde_hf, n_t_grid, n_longi_offset, n_up_peaks, n_arr_tg, dx, c0, w_eval, dt_sel, fs, n_cable_pos, n_associated_list, n_used_hyperbolas, n_rejected_list, n_rejected_hyperbolas)
        n_shape_x = n_arr_tg.shape[0] 
        
        s_up_peaks, s_arr_tg, s_associated_list, s_used_hyperbolas, s_rejected_list, s_rejected_hyperbolas = associate_picks(s_kde_hf, s_t_grid, s_longi_offset, s_up_peaks, s_arr_tg, dx, c0, w_eval, dt_sel, fs, s_cable_pos, s_associated_list, s_used_hyperbolas, s_rejected_list, s_rejected_hyperbolas)
        s_shape_x = s_arr_tg.shape[0]
        
        pbar.set_description(f"Associated calls: {len(n_associated_list) + len(s_associated_list)}")


    print(f"Test completed with {iterations} iterations.")

    fig = plot_reject_pick(n_peaks, n_longi_offset, n_dist, dx, n_associated_list, n_rejected_list, n_rejected_hyperbolas, fs)
    fig.savefig(f'figs/rej_associated_calls_north_{fileBeginTimeUTC}.png')
    # clear the figure
    plt.clf()

    fig = plot_reject_pick(s_peaks, s_longi_offset, s_dist, dx, s_associated_list, s_rejected_list, s_rejected_hyperbolas, fs)
    fig.savefig(f'figs/rej_associated_calls_south_{fileBeginTimeUTC}.png')
    plt.clf()

    # Example usage:
    # fig = plot_pick_analysis(n_associated_list, fs, dx, n_longi_offset, n_cable_pos, n_dist)
    # fig = plot_pick_analysis(s_associated_list, fs, dx, s_longi_offset, s_cable_pos, s_dist)

    # Localize using the selected picks
    n_localizations, n_alt_localizations = loc_from_picks(n_associated_list, n_cable_pos, c0, fs)
    s_localizations, s_alt_localizations = loc_from_picks(s_associated_list, s_cable_pos, c0, fs)

    fig = plot_associated(n_peaks, n_longi_offset, n_associated_list, n_localizations, n_cable_pos, n_dist, dx, c0, fs)
    fig.savefig(f'figs/associated_calls_north_{fileBeginTimeUTC}.png')
    fig = plot_associated(s_peaks, s_longi_offset, s_associated_list, s_localizations, s_cable_pos, s_dist, dx, c0, fs)
    fig.savefig(f'figs/associated_calls_south_{fileBeginTimeUTC}.png')

    # Create two list of coordinates, for ponts every 10 km along the cables, the spatial resolution is 2m 
    opticald_n = []
    opticald_s = []

    for i in range(int(10000/2), len(df_north), int(10000/2)):
        opticald_n.append((df_north['x'][i], df_north['y'][i]))

    for i in range(int(10000/2), len(df_south), int(10000/2)):
        opticald_s.append((df_south['x'][i], df_south['y'][i]))

    # Plot the grid points on the map
    colors_undersea = cmo.deep_r(np.linspace(0, 1, 256)) # blue colors for under the sea
    colors_land = np.array([[0.5, 0.5, 0.5, 1]])  # Solid gray for above sea level

    # Combine the color maps
    all_colors = np.vstack((colors_undersea, colors_land))
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', all_colors)

    extent = [x[0], x[-1], y[0], y[-1]]

    # Set the light source
    ls = LightSource(azdeg=350, altdeg=45)

    # Plot the location of the apex
    plt.figure(figsize=(14, 7))
    ax = plt.gca()
    # Plot the bathymetry relief in background
    rgb = ls.shade(bathy, cmap=custom_cmap, vert_exag=0.1, blend_mode='overlay', vmin=np.min(bathy), vmax=0)
    plot = ax.imshow(rgb, extent=extent, aspect='equal', origin='lower', vmin=np.min(bathy), vmax=0)
    # Plot the cable location in 2D
    ax.plot(df_north['x'], df_north['y'], 'tab:red', label='North cable')
    ax.plot(df_south['x'], df_south['y'], 'tab:orange', label='South cable')
    # ax.plot(cable_pos[j_hf_call[i]][:,0], cable_pos[j_hf_call[i]][:,1], 'tab:green', label='used_cable')

    # Add dashed contours at selected depths with annotations
    depth_levels = [-1500, -1000, -600, -250, -80]

    contour_dashed = ax.contour(bathy, levels=depth_levels, colors='k', linestyles='--', extent=extent, alpha=0.6)
    ax.clabel(contour_dashed, fmt='%d m', inline=True)

    # Plot points along the cable every 10 km in terms of optical distance
    for i, point in enumerate(opticald_n, start=1):
        ax.plot(point[0], point[1], '.', color='k')
        ax.annotate(f'{i*10}', (point[0], point[1]), textcoords='offset points', xytext=(5, 8), ha='center', fontsize=12)

    for i, point in enumerate(opticald_s, start=1):
        ax.plot(point[0], point[1], '.', color='k')
        ax.annotate(f'{i*10}', (point[0], point[1]), textcoords='offset points', xytext=(5, -15), ha='center', fontsize=12)


    for i, loc in enumerate(n_localizations):
        # Put label only for the first point
        if i == 0:
            ax.plot(loc[0], loc[1], 'o',  c='tab:purple', lw=4, label='Localized call - north')
        else:
            ax.plot(loc[0], loc[1], 'o', c='tab:purple', lw=4)
    for i, loc in enumerate(s_localizations):
        # Put label only for the first point
        if i == 0:
            ax.plot(loc[0], loc[1], 'o', c='tab:green', label='Localized call - south', lw=4)
        else:
            ax.plot(loc[0], loc[1], 'o', c='tab:green', lw=4)

    # Use a proxy artist for the color bar
    im = ax.imshow(bathy, cmap=custom_cmap, extent=extent, aspect='equal', origin='lower', vmin=np.min(bathy), vmax=0)
    # Calculate width of image over height
    im_ratio = bathy.shape[1] / bathy.shape[0]
    plt.colorbar(im, ax=ax, label='Depth [m]', pad=0.02, orientation='vertical', aspect=25, fraction=0.0195)
    im.remove()
    # Set the labels
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    # plt.xlim(40000, 34000)
    # plt.ylim(15000, 25000)
    plt.legend(loc='upper left')
    plt.grid(linestyle='--', alpha=0.6, color='k')
    plt.tight_layout()
    plt.savefig(f'figs/localized_calls_{fileBeginTimeUTC}.png')

    print('File processed and figures saved.')

    return


if __name__ == "__main__":
    nds_path = 'out/peaks_indexes_tp_North_2021-11-04_02:00:02_ipi2_th_4.nc'
    sds_path = 'out/peaks_indexes_tp_South_2021-11-04_02:00:02_ipi2_th_4.nc'
    main(nds_path, sds_path)

