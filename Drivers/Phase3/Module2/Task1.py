#%% Importing Libreries
from pathlib import Path
import pandas as pd
import pyvista as pv
import cv2 as cv
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, binary_fill_holes
from tqdm import tqdm
import pickle
from termcolor import colored
#%% Defining Subroutines
def triangulationEngine(pts1, pts2, calib):
    ''' Triangola punti 2D stereo in coordinate 3D spaziali rimuovendo la distorsione radiale e tangenziale. '''
    if len(pts1) == 0 or len(pts2) == 0:
        return np.empty((0, 3))
    pts1_f32 = np.asarray(pts1, dtype=np.float32).reshape(-1, 1, 2)
    pts2_f32 = np.asarray(pts2, dtype=np.float32).reshape(-1, 1, 2)
    udist1 = cv.undistortPoints(pts1_f32, calib['K1'], calib['D1'], P = calib['K1']) 
    udist2 = cv.undistortPoints(pts2_f32, calib['K2'], calib['D2'], P = calib['K2'])
    P1 = calib['K1'] @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = calib['K2'] @ np.hstack((calib['R'], calib['T']))
    points_4d_h = cv.triangulatePoints(
        P1, P2, udist1.reshape(-1, 2).T, udist2.reshape(-1, 2).T)
    return (points_4d_h[:3] / points_4d_h[3]).T
def surfaceCarver(masks, calib, pose, occupancyLimit):
    ''' Carve the Propeller Surface '''
    filled = []
    for i, mask in enumerate(masks):
        imgH, imgW = mask.shape
        ys, xs = np.where(mask > 0)
        points = np.stack([xs, ys], axis=-1)  # (N, 2)
        zero = np.zeros_like(mask, dtype=np.uint8)
        if points.size > 0:
            K = calib.get(f"K{i+1}")
            D = calib.get(f"D{i+1}")
            if K is None or D is None or K.size == 0:
                raise ValueError(f"Missing Calibration {i+1}")
            undistorted = cv.undistortPoints(
                points.astype(np.float32).reshape(-1, 1, 2),
                K, D, None, K
            ).reshape(-1, 2)
            undistorted = np.round(undistorted).astype(int)
            valid = (
                (undistorted[:, 0] >= 0) & (undistorted[:, 0] < imgW) &
                (undistorted[:, 1] >= 0) & (undistorted[:, 1] < imgH))
            undistorted = undistorted[valid]
            zero[undistorted[:, 1], undistorted[:, 0]] = 1
        poseH = np.hstack([pose, np.ones((pose.shape[0], 1))])
        if i == 0:
            P = calib['K1'] @ np.hstack((np.eye(3), np.zeros((3, 1))))
        else: 
            P = calib['K2'] @ np.hstack((calib['R'], calib['T']))
        if P is None or P.size == 0:
            raise ValueError(f"Missing Matrix P{i+1}")
        uvs = P @ poseH.T
        uvs /= uvs[2, :]
        uvs = np.round(uvs).astype(int)
        x_good = (uvs[0, :] >= 0) & (uvs[0, :] < imgW)
        y_good = (uvs[1, :] >= 0) & (uvs[1, :] < imgH)
        good = x_good & y_good
        indices = np.where(good)[0]
        fill = np.zeros(uvs.shape[1])
        sub_uvs = uvs[:2, indices]
        fill[indices] = zero[sub_uvs[1, :], sub_uvs[0, :]]
        filled.append(fill) 
    filled = np.vstack(filled)
    occupancy = np.sum(filled, axis=0)
    good_mask = occupancy >= occupancyLimit
    good_indices = np.where(good_mask)[0]
    poseH = np.hstack([pose, np.ones((pose.shape[0], 1))])
    good_points = poseH[good_indices, :][:, :3]
    return good_indices, good_points
def validatePoints(pts, refPts):
    """ Filtra i punti esterni mantenendo solo quelli "davanti" alla pose. """
    if len(pts) == 0 or len(refPts) == 0:
        return np.empty((0, 3))
    planar_axes = [0, 1] 
    ref_2d = refPts[:, planar_axes]
    ref_z = refPts[:, 2]
    target_2d = pts[:, planar_axes]
    target_z = pts[:, 2]
    z_surface_at_target = griddata(ref_2d, ref_z, target_2d, method='linear')
    valid_mask = ~np.isnan(z_surface_at_target)
    front_mask = target_z < z_surface_at_target
    return pts[valid_mask & front_mask]
def removeOutliers(pts, k=10, std_multiplier=1.5):
    """
    Filtra gli outlier statistici basandosi sulla distanza media dai k-vicini (SOR).
    """
    if len(pts) < k + 1:
        return pts
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=k+1)
    mean_dists = np.mean(dists[:, 1:], axis=1)
    global_mean = np.mean(mean_dists)
    global_std = np.std(mean_dists)
    threshold = global_mean + std_multiplier * global_std
    filtered_pts = pts[mean_dists < threshold]
    if len(filtered_pts) < 6:
        return pts
    return filtered_pts
def generateVolume(upperSurf, lowerSurf, resolution=1.0, smooth_sigma=1.5, margin_multiplier=2.5, taper_px=3.0, min_thickness=0.05, spline_smoothing=0.1):
    """
    Genera il volume lavorando su isole indipendenti, scartando i punti di controllo 
    geometricamente rumorosi per impedire avvallamenti anomali.
    """
    if len(upperSurf) < 4 or len(lowerSurf) < 4:  
        return 0.0, None, None
    pad = resolution * (margin_multiplier + taper_px + 2)
    min_x, max_x = np.min(lowerSurf[:, 0]) - pad, np.max(lowerSurf[:, 0]) + pad
    min_y, max_y = np.min(lowerSurf[:, 1]) - pad, np.max(lowerSurf[:, 1]) + pad
    if max_x - min_x < resolution or max_y - min_y < resolution:
        return 0.0, None, None
    grid_x, grid_y = np.mgrid[min_x:max_x:resolution, min_y:max_y:resolution]
    nx, ny = grid_x.shape
    lower_z_lin = griddata(lowerSurf[:, :2], lowerSurf[:, 2], (grid_x, grid_y), method='linear')
    lower_z_near = griddata(lowerSurf[:, :2], lowerSurf[:, 2], (grid_x, grid_y), method='nearest')
    lower_z_grid = np.where(np.isnan(lower_z_lin), lower_z_near, lower_z_lin)
    tree = cKDTree(lowerSurf[:, :2])
    grid_pts_2d = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    dists, _ = tree.query(grid_pts_2d)
    valid_mask_global = (dists.reshape(nx, ny) <= resolution * margin_multiplier)
    valid_mask_global = binary_fill_holes(valid_mask_global)
    if not np.any(valid_mask_global):
        return 0.0, None, None
    mask_uint8_global = (valid_mask_global * 255).astype(np.uint8)
    num_labels, labels = cv.connectedComponents(mask_uint8_global)
    dz_final_global = np.zeros_like(lower_z_grid)
    idx_x_up_all = np.clip(np.round((upperSurf[:, 0] - min_x) / resolution).astype(int), 0, nx - 1)
    idx_y_up_all = np.clip(np.round((upperSurf[:, 1] - min_y) / resolution).astype(int), 0, ny - 1)
    base_z_for_all_upper = lower_z_grid[idx_x_up_all, idx_y_up_all]
    global_dz = upperSurf[:, 2] - base_z_for_all_upper
    direction = 1 if np.nanmean(global_dz) >= 0 else -1
    for label in range(1, num_labels):
        loc_mask = (labels == label)
        kernel_dilate = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        loc_mask_dilated = cv.dilate(loc_mask.astype(np.uint8), kernel_dilate) > 0
        island_pts_mask = loc_mask_dilated[idx_x_up_all, idx_y_up_all]
        upper_island = upperSurf[island_pts_mask]
        if len(upper_island) < 1:
            continue
        island_idx_x = idx_x_up_all[island_pts_mask]
        island_idx_y = idx_y_up_all[island_pts_mask]
        base_z_isl = lower_z_grid[island_idx_x, island_idx_y]
        dz_raw = upper_island[:, 2] - base_z_isl
        if direction == 1:
            valid_ctrl = dz_raw >= min_thickness
        else:
            valid_ctrl = dz_raw <= -min_thickness
        if np.sum(valid_ctrl) >= 3:
            upper_island = upper_island[valid_ctrl]
            dz_raw = dz_raw[valid_ctrl]
        else:
            if direction == 1:
                dz_raw = np.clip(dz_raw, min_thickness, None)
            else:
                dz_raw = np.clip(dz_raw, None, -min_thickness)
        if len(upper_island) > 500:
            idx_sub = np.random.choice(len(upper_island), 500, replace=False)
            upper_island = upper_island[idx_sub]
            dz_raw = dz_raw[idx_sub]
        xs, ys = np.where(loc_mask)
        pad_loc = int(taper_px + 5)
        x0, x1 = max(0, np.min(xs) - pad_loc), min(nx, np.max(xs) + pad_loc + 1)
        y0, y1 = max(0, np.min(ys) - pad_loc), min(ny, np.max(ys) + pad_loc + 1)
        crop_grid_x = grid_x[x0:x1, y0:y1]
        crop_grid_y = grid_y[x0:x1, y0:y1]
        crop_loc_mask = loc_mask[x0:x1, y0:y1]
        train_pts = upper_island[:, :2]
        if len(train_pts) >= 4:
            dz_lin = griddata(train_pts, dz_raw, (crop_grid_x, crop_grid_y), method='linear')
            dz_near = griddata(train_pts, dz_raw, (crop_grid_x, crop_grid_y), method='nearest')
            dz_extrap = np.where(np.isnan(dz_lin), dz_near, dz_lin)
        else:
            dz_extrap = np.full_like(crop_grid_x, np.mean(dz_raw))
        effective_sigma = smooth_sigma * (1.0 + spline_smoothing * 5.0)
        if effective_sigma > 0:
            dz_extrap = gaussian_filter(dz_extrap, sigma=effective_sigma)
        if direction == 1:
            dz_extrap = np.clip(dz_extrap, min_thickness, None)
        else:
            dz_extrap = np.clip(dz_extrap, None, -min_thickness)
        crop_mask_uint8 = (crop_loc_mask * 255).astype(np.uint8)
        dist_tf = cv.distanceTransform(crop_mask_uint8, cv.DIST_L2, 5)
        t_linear = np.clip(dist_tf / max(1.0, taper_px), 0.0, 1.0)
        t_weight = t_linear * t_linear * t_linear * (t_linear * (t_linear * 6.0 - 15.0) + 10.0)
        dz_cropped_final = dz_extrap * t_weight
        dz_final_global[x0:x1, y0:y1] = np.where(crop_loc_mask, dz_cropped_final, dz_final_global[x0:x1, y0:y1])
    upper_z_grid = lower_z_grid + dz_final_global
    height_diff = np.abs(upper_z_grid[valid_mask_global] - lower_z_grid[valid_mask_global])
    cell_area = resolution ** 2
    volume = np.sum(height_diff) * cell_area
    valid_x = grid_x[valid_mask_global]
    valid_y = grid_y[valid_mask_global]
    valid_z = upper_z_grid[valid_mask_global]
    upper_surface_pts = np.column_stack((valid_x, valid_y, valid_z))
    mean_lower_z = np.nanmean(lower_z_grid)
    lower_z_safe = np.where(np.isnan(lower_z_grid), mean_lower_z, lower_z_grid)
    safe_lower_z_mesh = np.where(valid_mask_global, lower_z_grid, lower_z_safe)
    safe_upper_z_mesh = np.where(valid_mask_global, upper_z_grid, lower_z_safe)
    X_mesh = np.dstack((grid_x, grid_x))
    Y_mesh = np.dstack((grid_y, grid_y))
    Z_mesh = np.dstack((safe_lower_z_mesh, safe_upper_z_mesh))
    grid = pv.StructuredGrid()
    grid.dimensions = (nx, ny, 2) 
    grid.points = np.column_stack((
        X_mesh.ravel('F'),
        Y_mesh.ravel('F'),
        Z_mesh.ravel('F')))
    valid_3d = np.dstack((valid_mask_global, valid_mask_global))
    grid.point_data['valid_area'] = valid_3d.ravel('F').astype(int)
    volume_mesh = grid.threshold(0.5, scalars='valid_area')
    return volume, volume_mesh, upper_surface_pts
def dataCollector(masks, pts, calib, pose, areas, settings):
    mask1, mask2 = masks
    pts1, pts2 = pts
    pts3d = triangulationEngine(pts1, pts2, calib)
    valid3d = validatePoints(pts3d, pose)
    valid3d = removeOutliers(valid3d, k=settings.Filters.k, std_multiplier=settings.Filters.std_multiplier)
    id, inPts = surfaceCarver([mask1, mask2], calib, pose, settings.occupancyLimit)
    area = sum(areas[id]) 
    if hasattr(settings, 'Volume'):
        sigma_val = getattr(settings.Volume, 'smooth_sigma', 1.5)
        margin_val = getattr(settings.Volume, 'margin_multiplier', 2.5)
        taper_val = getattr(settings.Volume, 'taper_px', 3.0)
        min_thick_val = getattr(settings.Volume, 'min_thickness', 0.05)
        spline_val = getattr(settings.Volume, 'spline_smoothing', 0.1)
    else:
        sigma_val = 1.5
        margin_val = 2.5
        taper_val = 3.0
        min_thick_val = 0.05
        spline_val = 0.1
    volume, mesh, exPts = generateVolume(
        valid3d, inPts, settings.resolution, 
        smooth_sigma=sigma_val, margin_multiplier=margin_val, 
        taper_px=taper_val, min_thickness=min_thick_val, spline_smoothing=spline_val)
    return [inPts, exPts, valid3d, mesh, area, volume]

#%% Defining Main Function
def main(Config):       
    task_conf = Config.Packages.Drivers.Phases.Phase3.Modules.Module2.Tasks.Task1
    if task_conf.General.Activation is True:
        print('.... Task1:', colored('Running ℹ️', 'cyan'))
        main_root = Path(Config.Paths.mainRooot)
        poseRoot = (
            main_root / 
            Config.Paths.DataRoots.ResourcesRoot / 
            Config.Paths.DataRoots.StreamRoot / 
            Config.Paths.DataRoots.CaseStudyRoot() /
            Config.Packages.Drivers.__name__ / 
            Config.Packages.Drivers.Phases.Phase2.__name__ / 
            Config.Packages.Drivers.Phases.Phase2.Modules.Module2.__name__ / 
            Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task4.__name__ / 
            Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task4.MetaData.OutputExt)
        objRoot = (
            main_root / 
            Config.Paths.DataRoots.ResourcesRoot / 
            Config.Paths.DataRoots.StreamRoot / 
            Config.Paths.DataRoots.CaseStudyRoot() /
            Config.Packages.Drivers.__name__ / 
            Config.Packages.Drivers.Phases.Phase3.__name__ / 
            Config.Packages.Drivers.Phases.Phase3.Modules.Module1.__name__ / 
            Config.Packages.Drivers.Phases.Phase3.Modules.Module1.Tasks.Task2.__name__ )
        calibRoot = (
            main_root /
            Config.Paths.DataRoots.ResourcesRoot /
            Config.Paths.DataRoots.StreamRoot / 
            Config.Paths.DataRoots.CaseStudyRoot() /
            Config.Packages.Drivers.__name__ / 
            Config.Packages.Drivers.Phases.Phase1.__name__ / 
            Config.Packages.Drivers.Phases.Phase1.Modules.Module3.__name__ /
            Config.Packages.Drivers.Phases.Phase1.Modules.Module3.Tasks.Task2.__name__ / 
            Config.Packages.Drivers.Phases.Phase1.Modules.Module3.Tasks.Task2.MetaData.OutputExt)
        dstRoot = (
            main_root /
            Config.Paths.DataRoots.ResourcesRoot /
            Config.Paths.DataRoots.StreamRoot / 
            Config.Paths.DataRoots.CaseStudyRoot() /
            Config.Packages.Drivers.__name__ / 
            Config.Packages.Drivers.Phases.Phase3.__name__ / 
            Config.Packages.Drivers.Phases.Phase3.Modules.Module2.__name__ /
            Config.Packages.Drivers.Phases.Phase3.Modules.Module2.Tasks.Task1.__name__ )
        settings = task_conf.Settings
        period = Config.Settings.Acquisition.PPR
        calib = pd.read_pickle(calibRoot)[settings.Calib.Dataset][settings.Calib.Pair][settings.Calib.Model]
        poses, areas = [pd.read_pickle(poseRoot)[k ] for k in ['points', 'areas']]
        files = sorted(f.name for f in objRoot.iterdir() if f.is_file())
        cavityData, cloudData = dict(), dict()
        try:
            for _, name in enumerate(tqdm(files, total=len(files), desc=colored(f'Surface Carving 🚀', 'magenta'), ncols=100)): 
                key = ''.join(c for c in name if c.isdigit())
                phase = int(key) % period 
                pose = poses[phase]
                cavityData[key] = pd.DataFrame(
                    columns=['Inner', 'Outer', 'Control', 'Mesh', 'Area', 'Volume'])
                cloudData[key] = pd.DataFrame(
                    columns=['Inner', 'Outer', 'Control', 'Mesh', 'Area', 'Volume'])
                objs = pd.read_pickle((objRoot / name))
                for i, (_, obj) in enumerate(objs.iterrows()):
                    cavity1, cloud1, cavPts1, clPts1 = obj[objs.columns[0]]
                    cavity2, cloud2, cavPts2, clPts2 = obj[objs.columns[1]]
                    cavity = dataCollector((cavity1, cavity2), (cavPts1, cavPts2), calib, pose, areas, settings)
                    cloud = dataCollector((cloud1, cloud2), (clPts1, clPts2), calib, pose, areas, settings)
                    cavityData[key].loc[i] = cavity
                    cloudData[key].loc[i] = cloud
            pickle.dump(cavityData, open((dstRoot / task_conf.MetaData.CavityExt), 'wb'))
            pickle.dump(cloudData, open((dstRoot / task_conf.MetaData.CloudExt), 'wb'))
        except Exception as e:
            print('.... Task1:', colored(f'Error: {e} ❌', 'red'))
            raise e
    elif task_conf.General.Activation is False:
        print('.... Task1:', colored('Offline ⚠️', 'yellow'))
    else:
        raise ValueError('Please Set the Task1 Switch (on/off) ❌')
    return