import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import argparse

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cmap import Colormap

from gmr.gmr_memory_dataset import GMRMemoryDataset
from gmr.model import GMRMemoryModelDualHead
from gmr.utils import *

def gmr_combine_all_predictions(labels, shape):
    """_summary_,

    Args:
        labels (list): List of all used dataset labels.
        shape (str): Selected shape to draw.

    Returns:
        list: Predictions of all samples in all datasets.
        list: True value of all samples in all datasets.
        list: Encoded label values (from 0 to 7).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    phase = None

    model_phase1 = GMRMemoryModelDualHead(output_dim=2)
    model_phase1.load_state_dict(torch.load('phase1.pth', map_location=device))
    model_phase2 = GMRMemoryModelDualHead(output_dim=3)
    model_phase2.load_state_dict(torch.load('phase2.pth', map_location=device))
    
    all_predictions = []
    all_original_targets = []
    encoded_labels = []  # For creating colors
    for label_idx, label in enumerate(labels):
        if 't' not in label:
            # phase 1
            phase = 1
            model = model_phase1
            continue
        else:
            # phase 2
            phase = 2
            model = model_phase2

        dataset = GMRMemoryDataset(label=label,
                                   run_augment=False,
                                   num_samples=1)
        original_data = torch.tensor(dataset.signal_interp, dtype=torch.float32)
        original_targets = dataset.accumulation_time

        dataloader = DataLoader(original_data, batch_size=64, shuffle=False)

        model.to(device)
        model.eval()

        pred_times = []
        pred_presences = []
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                features = batch.to(device)             # Expected shape: [B, 6, 8]
                outputs = model(features)               # Output shape: [B, 2, num_shape, 1]

                pred_time = outputs[:, 0, :, :]         # [B, num_shape, 1]
                pred_presence = outputs[:, 1, :, :]     # [B, num_shape, 1]
                pred_presence = torch.where(pred_presence < 0, -1, 1)

                pred_times.append(pred_time.squeeze(-1).cpu().numpy())
                pred_presences.append(pred_presence.squeeze(-1).cpu().numpy())
                predictions.append((pred_time * pred_presence).squeeze(-1).cpu().numpy())

        pred_times = np.concatenate(pred_times, axis=0) # [N, num_shape]
        pred_presences = np.concatenate(pred_presences, axis=0)  # [N, num_shape]
        predictions = np.concatenate(predictions, axis=0)  # [N, num_shape]

        num_samples = original_data.shape[0]

        if phase == 1:
            # change phase 1 predictions from [N, 2] to [N, 3]
            predictions = np.concatenate((np.zeros((num_samples, 1)), predictions), axis=1)
            # change phase 1 triangle targets to all zeros
            for row in range(num_samples):
                original_targets[row, 0] = 0

        if shape == 't':
            all_predictions.append(predictions[:, 0])
            all_original_targets.append(original_targets[:, 0])
        elif shape == 'c':
            all_predictions.append(predictions[:, 1])
            all_original_targets.append(original_targets[:, 1])
        elif shape == 's':
            all_predictions.append(predictions[:, 2])
            all_original_targets.append(original_targets[:, 2])
        elif shape == 'a':
            all_predictions.append(predictions)
            all_original_targets.append(original_targets)
        encoded_labels.append(label_idx)

    return all_predictions, all_original_targets, encoded_labels

def gmr_calculate_interval_r2(all_pred, all_true, interval=0.05):
    """
    Calculates the sum of squared residuals (ss_res) and the total sum of squares (ss_tot)
    for predictions and targets falling into windowed intervals over the range [-1.5, 1.5].

    Args:
        all_pred (list of np.ndarray): List of 6 numpy arrays, each of shape [num_samples, num_shapes],
            containing prediction values.
        all_true (list of np.ndarray): List of 6 numpy arrays, each of shape [num_samples, num_shapes],
            containing true target values.
        interval (float, optional): The length of each interval/window over the range [-1.5, 1.5]. Defaults to 0.05.

    Returns:
        mse_list (list): List of mean squared error of predictions for each interval.
        var_list (list): List of mean variance of true values for each interval.
    """
    all_pred_values = np.concatenate([arr.flatten() for arr in all_pred])
    all_true_values = np.concatenate([arr.flatten() for arr in all_true])


    bins = np.arange(-1.5, 1.5 + interval, interval)

    mse_list = []
    var_list = []

    # For each interval/bin, compute the sum of squared residuals and total sum of squares
    for i in range(len(bins) - 1):
        lower = bins[i]
        upper = bins[i + 1]
 
        if i == len(bins) - 2:
            # For the last interval, include the upper bound
            mask = (all_true_values >= lower) & (all_true_values <= upper)
        else:
            mask = (all_true_values >= lower) & (all_true_values < upper)

        num_interval_samples = np.sum(mask)
        if num_interval_samples > 0:
            # If there are any samples in this interval, calculate the metrics
            y_true_bin = all_true_values[mask]
            y_pred_bin = all_pred_values[mask]
            mse = np.sum((y_true_bin - y_pred_bin) ** 2) / num_interval_samples
            var = np.sum((y_true_bin - np.mean(y_true_bin)) ** 2) / num_interval_samples
        else:
            # If no data points fall in this interval, mark as NaN
            mse = np.nan
            var = np.nan
        
        mse_list.append(mse)
        var_list.append(var)

    return mse_list, var_list

def gmr_calculate_interval_per_group_r2(all_pred, all_true, interval=0.05):
    """
    Calculates the mean squared error (mse) and the variance (var) for all three shapes
    for predictions and targets falling into windowed intervals over the range [-1.5, 1.5].
    Only called when selected shape is 'a'.

    Args:
        all_pred (list of np.ndarray): List of 6 numpy arrays, each of shape [num_samples, num_shapes],
            containing prediction values.
        all_true (list of np.ndarray): List of 6 numpy arrays, each of shape [num_samples, num_shapes],
            containing true target values.
        interval (float, optional): The length of each interval/window over the range [-1.5, 1.5]. Defaults to 0.05.
        split_group (bool, optional): Decides if the mse and var for each group is calculated separately, only work
            if selected_shape is 'a'.

    Returns:
        mse_list (list): List of mean squared error of predictions for each interval.
        var_list (list): List of mean variance of true values for each interval.
    """

    all_pred_values_t = np.concatenate([arr[:, 0] for arr in all_pred])
    all_true_values_t = np.concatenate([arr[:, 0] for arr in all_true])

    all_pred_values_s = np.concatenate([arr[:, 1] for arr in all_pred])
    all_true_values_s = np.concatenate([arr[:, 1] for arr in all_true])

    all_pred_values_c = np.concatenate([arr[:, 2] for arr in all_pred])
    all_true_values_c = np.concatenate([arr[:, 2] for arr in all_true])

    bins = np.arange(-1.5, 1.5 + interval, interval)

    mse_list_t = []
    var_list_t = []

    mse_list_s = []
    var_list_s = []
    
    mse_list_c = []
    var_list_c = []

    # For each interval/bin, compute the sum of squared residuals and total sum of squares
    for i in range(len(bins) - 1):
        lower = bins[i]
        upper = bins[i + 1]
 
        if i == len(bins) - 2:
            # For the last interval, include the upper bound
            mask_t = (all_true_values_t >= lower) & (all_true_values_t <= upper)
            mask_s = (all_true_values_s >= lower) & (all_true_values_s <= upper)
            mask_c = (all_true_values_c >= lower) & (all_true_values_c <= upper)
        else:
            mask_t = (all_true_values_t >= lower) & (all_true_values_t < upper)
            mask_s = (all_true_values_s >= lower) & (all_true_values_s < upper)
            mask_c = (all_true_values_c >= lower) & (all_true_values_c < upper)

        num_interval_samples_t = np.sum(mask_t)
        if num_interval_samples_t > 0:
            # If there are any samples in this interval, calculate the metrics
            y_true_bin = all_true_values_t[mask_t]
            y_pred_bin = all_pred_values_t[mask_t]
            mse_t = np.sum((y_true_bin - y_pred_bin) ** 2) / num_interval_samples_t
            var_t = np.sum((y_true_bin - np.mean(y_true_bin)) ** 2) / num_interval_samples_t
        else:
            # If no data points fall in this interval, mark as NaN
            mse_t = np.nan
            var_t = np.nan

        mse_list_t.append(mse_t)
        var_list_t.append(var_t)
        
        num_interval_samples_s = np.sum(mask_s)
        if num_interval_samples_s > 0:
            # If there are any samples in this interval, calculate the metrics
            y_true_bin = all_true_values_s[mask_s]
            y_pred_bin = all_pred_values_s[mask_s]
            mse_s = np.sum((y_true_bin - y_pred_bin) ** 2) / num_interval_samples_s
            var_s = np.sum((y_true_bin - np.mean(y_true_bin)) ** 2) / num_interval_samples_s
        else:
            # If no data points fall in this interval, mark as NaN
            mse_s = np.nan
            var_s = np.nan
        
        mse_list_s.append(mse_s)
        var_list_s.append(var_s)

        num_interval_samples_c = np.sum(mask_c)
        if num_interval_samples_c > 0:
            # If there are any samples in this interval, calculate the metrics
            y_true_bin = all_true_values_c[mask_c]
            y_pred_bin = all_pred_values_c[mask_c]
            mse_c = np.sum((y_true_bin - y_pred_bin) ** 2) / num_interval_samples_c
            var_c = np.sum((y_true_bin - np.mean(y_true_bin)) ** 2) / num_interval_samples_c
        else:
            # If no data points fall in this interval, mark as NaN
            mse_c = np.nan
            var_c = np.nan
        
        mse_list_c.append(mse_c)
        var_list_c.append(var_c)

    return {
        'mse_list_t': mse_list_t,
        'var_list_t': var_list_t,
        'mse_list_s': mse_list_s,
        'var_list_s': var_list_s,
        'mse_list_c': mse_list_c,
        'var_list_c': var_list_c
    }
             

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model performance.')
    parser.add_argument('-s', '--shape', type=str, choices=['t', 's', 'c', 'a'],
                        help='Selected shape to draw, can be \'t\'riangle, \'s\'quare, \'c\'ircle, or \'a\'ll.')
    args = parser.parse_args()
    selected_shape = args.shape

    selected_all_shapes = False
    if selected_shape == 'a':
        selected_all_shapes = True

    labels = list(KEY_FRAMES_DICT.keys())
    
    # Acquire all predictions and true values for the selected label group and shape
    # If selected_shape is not 'a', all_pred and all_true will be lists of [num_samples, 1] arrays
    # If selected shape is 'a', all_pred and all_true will be lists of [num_samples, 3] arrays
    all_pred, all_true, encoded_labels = gmr_combine_all_predictions(labels, selected_shape)

    interval = 0.1
    mse_list, var_list = gmr_calculate_interval_r2(all_pred, all_true, interval)
    if selected_all_shapes:
        # If selected to plot for all shapes, need to also find the mse and var for each shape
        per_group_stats = gmr_calculate_interval_per_group_r2(all_pred, all_true, interval)

    if selected_shape == 't':
        color_map = Colormap('colorbrewer:blues')
    elif selected_shape == 's':
        color_map = Colormap('colorbrewer:greens')
    elif selected_shape == 'c':
        color_map = Colormap('colorbrewer:reds')
    elif selected_all_shapes:
        shape_colors = {'tri': (0.4431, 0.5843, 0.7725, 0.4),
                        'sqr': (0.0039, 0.5176, 0.3098, 0.4), 
                        'cir': (0.9137, 0.1294, 0.1725, 0.4)}
        color_map = Colormap('colorbrewer:greys')  # A general color for all shapes

    # Use a color gradient for different labels (rounds of tests)
    colors = color_map(np.linspace(0.2, 0.8, len(labels)))

    fig, ax = plt.subplots(figsize=(8, 8))
    for group_idx, (group_pred, group_target) in enumerate(zip(all_pred, all_true)):
        group = encoded_labels[group_idx]
        color = colors[group, :]
        ax.plot(group_target, group_pred,
                marker='o', ls='None', alpha=0.2,
                ms=2, mew=1, mec='None', mfc=color)

    data_range = [-1, 1]
    ax.plot([data_range[0], data_range[1]], [data_range[0], data_range[1]], ls='--', c='#b57979')
    ax.hlines(0, data_range[0], data_range[1], colors='k', linestyles='--')
    ax.vlines(0, data_range[0], data_range[1], colors='k', linestyles='--')

    bins = np.arange(-1.5, 1.5 + interval, interval)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    rmse = np.array([np.sqrt(mse) if not np.isnan(mse) else np.nan for mse in mse_list])
    std_true = np.array([np.sqrt(var) if not np.isnan(var) else np.nan for var in var_list])
    ax.fill_between(bin_centers, bin_centers - rmse, bin_centers + rmse,
                     color='#e3b87f', alpha=0.3)
    ax.fill_betweenx(bin_centers, bin_centers - std_true, bin_centers + std_true,
                      color='#576fa0', alpha=0.5)

    ax.set_xlim([data_range[0], data_range[1]])
    ax.set_ylim([data_range[0], data_range[1]])
    ax.set_aspect('equal', 'box')

    # When plotting for all shapes, add three inset axes for the per-shape interval metrics.
    if selected_all_shapes:
        # Extract per-shape metrics.
        # Triangle
        mse_list_t = per_group_stats['mse_list_t']
        var_list_t = per_group_stats['var_list_t']
        rmse_t = np.array([np.sqrt(mse) if not np.isnan(mse) else np.nan for mse in mse_list_t])
        std_t = np.array([np.sqrt(var) if not np.isnan(var) else np.nan for var in var_list_t])
        # Square
        mse_list_s = per_group_stats['mse_list_s']
        var_list_s = per_group_stats['var_list_s']
        rmse_s = np.array([np.sqrt(mse) if not np.isnan(mse) else np.nan for mse in mse_list_s])
        std_s = np.array([np.sqrt(var) if not np.isnan(var) else np.nan for var in var_list_s])
        # Circle
        mse_list_c = per_group_stats['mse_list_c']
        var_list_c = per_group_stats['var_list_c']
        rmse_c = np.array([np.sqrt(mse) if not np.isnan(mse) else np.nan for mse in mse_list_c])
        std_c = np.array([np.sqrt(var) if not np.isnan(var) else np.nan for var in var_list_c])

        # Use the full interval width for the bars.
        bar_width = interval * 0.8

        # Inset for STD curves at the top margin.
        ax_std = inset_axes(ax, width="100%", height="40%", loc='lower center',
                            bbox_to_anchor=(0, 1.01, 1, 0.4),
                            bbox_transform=ax.transAxes, borderpad=0)
        # For each shape, plot vertical bars with symmetric extension.
        ax_std.bar(bin_centers, std_t, width=bar_width, bottom=-std_t/2,
                   edgecolor=shape_colors['tri'], facecolor='none', lw=1.5)
        ax_std.bar(bin_centers, std_s, width=bar_width, bottom=-std_s/2,
                   edgecolor=shape_colors['sqr'], facecolor='none', lw=1.5)
        ax_std.bar(bin_centers, std_c, width=bar_width, bottom=-std_c/2,
                   edgecolor=shape_colors['cir'], facecolor='none', lw=1.5)
        ax_std.axhline(0, color='black', lw=1)
        ax_std.set_xlim(data_range)
        ax_std.set_xticklabels([])
        ax_std.set_yticklabels([])
        # Set symmetric y-limits using the maximum STD value.
        max_std = np.nanmax(np.concatenate([std_t, std_s, std_c]))
        ax_std.set_ylim(-max_std, max_std)
        for spine in ax_std.spines.values():
            spine.set_visible(False)
        plt.tick_params(left=False, bottom=False)

        # Inset axis for RMSE bars at the right margin.
        ax_rmse = inset_axes(ax, width="40%", height="100%", loc='center left',
                             bbox_to_anchor=(1.05, 0, 0.4, 1),
                             bbox_transform=ax.transAxes, borderpad=0)
        # For each shape, plot horizontal bars with symmetric extension.
        ax_rmse.barh(bin_centers, rmse_t, height=bar_width, left=-rmse_t/2,
                     edgecolor=shape_colors['tri'], facecolor='none', lw=1.5)
        ax_rmse.barh(bin_centers, rmse_s, height=bar_width, left=-rmse_s/2,
                     edgecolor=shape_colors['sqr'], facecolor='none', lw=1.5)
        ax_rmse.barh(bin_centers, rmse_c, height=bar_width, left=-rmse_c/2,
                     edgecolor=shape_colors['cir'], facecolor='none', lw=1.5)
        ax_rmse.axvline(0, color='black', lw=1)
        ax_rmse.set_ylim(data_range)
        ax_rmse.set_xticklabels([])
        ax_rmse.set_yticklabels([])
        # Set symmetric x-limits using the maximum RMSE value.
        max_rmse = np.nanmax(np.concatenate([rmse_t, rmse_s, rmse_c]))
        # Remove the box (spines) from the inset.
        for spine in ax_rmse.spines.values():
            spine.set_visible(False)
        plt.tick_params(left=False, bottom=False)

    plt.subplots_adjust(top=0.85, right=0.85)
    plt.savefig(f'results/performances/{selected_shape}.svg', bbox_inches='tight', dpi=600, transparent=True)
    plt.show()