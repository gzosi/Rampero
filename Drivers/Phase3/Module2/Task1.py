#%% Importing Libreries
from pathlib import Path
import pandas as pd
import pyvista as pv
import cv2 as cv
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
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
        ys, xs = np.where(mask == True)
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
    """
    Filtra i punti esterni mantenendo solo quelli "davanti" alla pose.
    """
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
def smoothSurface(pts, grid_x, grid_y):
    """
    Approssima i punti sparsi con una superficie matematica (quadratica o piana).
    Questo garantisce una superficie perfettamente liscia, immune ai singoli punti rumorosi.
    """
    X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
    if len(pts) >= 6:
        A = np.c_[X**2, Y**2, X*Y, X, Y, np.ones(X.shape)]
        C, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
        Z_grid = C[0]*grid_x**2 + C[1]*grid_y**2 + C[2]*grid_x*grid_y + C[3]*grid_x + C[4]*grid_y + C[5]
    else:
        A = np.c_[X, Y, np.ones(X.shape)]
        C, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
        Z_grid = C[0]*grid_x + C[1]*grid_y + C[2]
    return Z_grid
def generateVolume(upperSurf, lowerSurf, resolution=1.0):
    """
    Genera il volume calcolato tramite Fit Polinomiale e crea una Voxel Mesh 3D solida.
    """
    if len(upperSurf) == 0 or len(lowerSurf) == 0:  
        print("\nAttenzione: Una delle nuvole di punti è vuota.")
        return 0.0, None, None
    min_x, max_x = np.min(lowerSurf[:, 0]), np.max(lowerSurf[:, 0])
    min_y, max_y = np.min(lowerSurf[:, 1]), np.max(lowerSurf[:, 1])
    grid_x, grid_y = np.mgrid[min_x:max_x:resolution, min_y:max_y:resolution]
    lower_z_grid = griddata(lowerSurf[:, :2], lowerSurf[:, 2], (grid_x, grid_y), method='linear')
    nx, ny = grid_x.shape
    mask_2d = np.zeros((nx, ny), dtype=np.uint8)
    idx_x = np.clip(np.round((lowerSurf[:, 0] - min_x) / resolution).astype(int), 0, nx - 1)
    idx_y = np.clip(np.round((lowerSurf[:, 1] - min_y) / resolution).astype(int), 0, ny - 1)
    mask_2d[idx_x, idx_y] = 255
    tree = cKDTree(lowerSurf[:, :2])
    dists, _ = tree.query(lowerSurf[:, :2], k=2)
    typical_spacing = np.percentile(dists[:, 1], 95)
    pixel_spacing = int(np.ceil(typical_spacing / resolution))
    k_size = max(3, pixel_spacing * 2 + 1)
    k_size = min(k_size, 31)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_size, k_size))
    mask_closed = cv.morphologyEx(mask_2d, cv.MORPH_CLOSE, kernel)
    contours, _ = cv.findContours(mask_closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    valid_mask = np.zeros((nx, ny), dtype=bool)
    if contours:
        largest_contour = max(contours, key=cv.contourArea)
        mask_filled = np.zeros((nx, ny), dtype=np.uint8)
        cv.drawContours(mask_filled, [largest_contour], -1, 255, thickness=cv.FILLED)
        valid_mask = mask_filled > 0
    valid_mask = valid_mask & ~np.isnan(lower_z_grid)
    upper_z_grid = smoothSurface(upperSurf, grid_x, grid_y)
    height_diff = np.abs(upper_z_grid[valid_mask] - lower_z_grid[valid_mask])
    cell_area = resolution ** 2
    volume = np.sum(height_diff) * cell_area
    valid_x = grid_x[valid_mask]
    valid_y = grid_y[valid_mask]
    valid_z = upper_z_grid[valid_mask]
    upper_surface_pts = np.column_stack((valid_x, valid_y, valid_z))
    safe_lower_z = np.where(valid_mask, lower_z_grid, upper_z_grid)
    X_mesh = np.dstack((grid_x, grid_x))
    Y_mesh = np.dstack((grid_y, grid_y))
    Z_mesh = np.dstack((safe_lower_z, upper_z_grid))
    grid = pv.StructuredGrid()
    nx, ny = grid_x.shape
    grid.dimensions = (nx, ny, 2) 
    grid.points = np.column_stack((
        X_mesh.ravel('F'),
        Y_mesh.ravel('F'),
        Z_mesh.ravel('F')))
    valid_3d = np.dstack((valid_mask, valid_mask))
    grid.point_data['valid_area'] = valid_3d.ravel('F').astype(int)
    volume_mesh = grid.threshold(0.5, scalars='valid_area')
    return volume, volume_mesh, upper_surface_pts
def dataCollector(masks, pts, calib, pose, areas, settings):
    mask1, mask2 = masks
    pts1, pts2 = pts
    pts3d = triangulationEngine(pts1, pts2, calib)
    valid3d = validatePoints(pts3d, pose)
    id, inPts = surfaceCarver([mask1, mask2], calib, pose, settings.occupancyLimit)
    area = sum(areas[id]) 
    volume, mesh, exPts = generateVolume(valid3d, inPts, settings.resolution)
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
            Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task3.__name__ / 
            Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task3.MetaData.OutputExt)
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
        period = Config.Settings.Acquisition.PPR / Config.Settings.Acquisition.Blades
        calib = pd.read_pickle(calibRoot)[settings.Calib.Dataset][settings.Calib.Pair][settings.Calib.Model]
        poses = pd.read_pickle(poseRoot)
        files = sorted(f.name for f in objRoot.iterdir() if f.is_file())
        cavityData, cloudData = dict(), dict()
        try:
            for _, name in enumerate(tqdm(files, total=len(files), desc=colored(f'Surface Carving 🚀', 'magenta'), ncols=100)):  
                key = ''.join(c for c in name if c.isdigit())
                phase = int(key) % period 
                pose, areas = poses[phase]
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