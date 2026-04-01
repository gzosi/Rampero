#%% Importing Libreries
from pathlib import Path
import numpy as np
import cv2 as cv
import h5py
import torch
from segment_anything_hq import SamPredictor, sam_model_registry
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm
import pickle
from termcolor import colored
#%% Defining Subroutines
def getROI(param, idx, shape):
    ''' Define a dynamic and periodi ROI '''
    indices = sorted(param.keys())
    if not indices:
        return np.zeros(shape[:2], dtype=np.uint8), None
    if idx <= indices[0]:
        target_pts = np.array(param[indices[0]], dtype=np.float32)
    elif idx >= indices[-1]:
        target_pts = np.array(param[indices[-1]], dtype=np.float32)
    else:
        target_pts = None
        for i in range(len(indices) - 1):
            t0 = indices[i]
            t1 = indices[i+1]
            if t0 <= idx <= t1:
                alpha = (idx - t0) / (t1 - t0)
                p0 = np.array(param[t0], dtype=np.float32)
                p1 = np.array(param[t1], dtype=np.float32)
                target_pts = p0 + (p1 - p0) * alpha
                break
    mask = np.zeros(shape[:2], dtype=np.uint8)
    int_pts = target_pts.astype(np.int32)
    cv.fillPoly(mask, [int_pts], 255)
    return mask, int_pts
def getSmartPrompts(Config, fg, roi):
    ''' Trova Box e Punti analizzando la texture (varianza locale) per superare i limiti del colore/luce '''
    settings = Config.Packages.Drivers.Phases.Phase3.Modules.Module1.Tasks.Task1.Settings
    if getattr(settings.SmartPrompt, 'ApplyCLAHE', False):
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        fg_proc = clahe.apply(fg)
    else:
        fg_proc = fg
    fg_f = fg_proc.astype(np.float32)
    k_size = settings.SmartPrompt.TextureKernel
    mean_sq = cv.blur(fg_f ** 2, (k_size, k_size))
    sq_mean = cv.blur(fg_f, (k_size, k_size)) ** 2
    variance = np.maximum(mean_sq - sq_mean, 0)
    std_dev = np.sqrt(variance)
    std_dev_norm = cv.normalize(std_dev, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    if roi is not None:
        std_dev_norm = cv.bitwise_and(std_dev_norm, std_dev_norm, mask=roi)
    _, thresh = cv.threshold(std_dev_norm, settings.SmartPrompt.TextureThreshold, 255, cv.THRESH_BINARY)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, settings.SmartPrompt.MorphKernel)
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    prompts = []
    pad = settings.SmartPrompt.BoxPadding
    area_min = settings.SmartPrompt.AreaMin
    h, w = fg.shape[:2]
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area >= area_min:
            x, y, bw, bh = cv.boundingRect(cnt)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + bw + pad)
            y2 = min(h, y + bh + pad)
            M = cv.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + bw//2, y + bh//2
            bg_pt1 = [max(0, x1 - 5), max(0, y1 - 5)]
            bg_pt2 = [min(w-1, x2 + 5), min(h-1, y2 + 5)]
            prompts.append({
                'box': [x1, y1, x2, y2],
                'points': [[cx, cy], bg_pt1, bg_pt2],
                'labels': [1, 0, 0] 
            })
    return prompts
def segmentEngine(fg, predictor, prompts):
    masksFg = []
    predictor.set_image(cv.cvtColor(fg, cv.COLOR_GRAY2BGR))  
    for prompt in prompts:
        input_box = np.array(prompt['box'])
        input_points = np.array(prompt['points'])
        input_labels = np.array(prompt['labels'])
        mask_fg, _, _ = predictor.predict(
            point_coords=input_points, 
            point_labels=input_labels, 
            box=input_box[None, :],
            multimask_output=True)
        masksFg.extend(list(mask_fg))
    return masksFg
def getMetadata(masks, fg):
    ''' Estrae metadati (Area, Intensità Media, Deviazione Standard) per raggruppamento, senza filtrare/scartare nulla. '''
    if len(masks) == 0: return np.array([])
    masksArray = np.asarray(masks)
    areas = masksArray.sum(axis=(1, 2))
    safeAreas = np.where(areas == 0, 1e-8, areas)
    fg_float = fg.astype(np.float32)
    avg = np.tensordot(masksArray, fg_float, axes=([1, 2], [0, 1])) / safeAreas
    avg2 = np.tensordot(masksArray, np.square(fg_float), axes=([1, 2], [0, 1])) / safeAreas
    stds = np.sqrt(np.maximum(0, avg2 - np.square(avg)))
    stats = np.stack([areas, avg, stds], axis=1)
    maxVals = stats.max(axis=0) + 1e-8
    metadata = stats / maxVals
    return metadata
def groupMasks(Config, masks, stats):
    contained = Config.Packages.Drivers.Phases.Phase3.Modules.Module1.Tasks.Task1.Settings.Group.Contained 
    smaller = Config.Packages.Drivers.Phases.Phase3.Modules.Module1.Tasks.Task1.Settings.Group.Smaller
    similar = Config.Packages.Drivers.Phases.Phase3.Modules.Module1.Tasks.Task1.Settings.Group.Similar
    if len(masks) == 0: return []
    N = len(masks)
    statsArr = np.array(stats) 
    masksFlat = np.stack([m.ravel() for m in masks]).astype(np.float32)
    intersection = np.dot(masksFlat, masksFlat.T)
    areas = np.diag(intersection).copy()
    safeAreas = np.where(areas == 0, 1, areas)
    containment = intersection / safeAreas[:, None]
    toRemove = np.zeros(N, dtype=bool)
    for i in range(N):
        isContained = (containment[i, :] > contained)
        isSmaller = (areas[i] / safeAreas < smaller)
        if np.any(isContained & isSmaller & (np.arange(N) != i)):
            toRemove[i] = True
    survivors = np.where(~toRemove)[0]
    if len(survivors) == 0: return []
    union = areas[:, None] + areas[None, :] - intersection
    iou = intersection / np.where(union == 0, 1, union)
    toMerge = (iou[survivors][:, survivors] > similar)
    nComp, labels = connected_components(csr_matrix(toMerge), directed=False)
    mergedMasks, mergedStats = [], []
    for c in range(nComp):
        inCluster = survivors[labels == c]
        if len(inCluster) > 1:
            clusterMasks = [masks[i] for i in inCluster]
            mergedMasks.append(np.logical_or.reduce(clusterMasks).astype(np.uint8))
            avg_stats = np.mean(statsArr[inCluster], axis=0)
            mergedStats.append(avg_stats)
        else:
            idx = inCluster[0]
            mergedMasks.append(masks[idx])
            mergedStats.append(statsArr[idx])
    mergedStats = np.array(mergedStats)
    finalFlat = np.stack([m.ravel() for m in mergedMasks]).astype(np.float32)
    finalInter = np.dot(finalFlat, finalFlat.T)
    finalAreas = np.diag(finalInter)
    finalUnion = finalAreas[:, None] + finalAreas[None, :] - finalInter
    finalIoU = finalInter / np.where(finalUnion == 0, 1, finalUnion)
    n_groups, group_labels = connected_components(csr_matrix(finalIoU > 0.1), directed=False)
    groups = []
    for g in range(n_groups):
        indices = np.where(group_labels == g)[0]    
        current_group = []
        for i in indices:       
            current_group.append((mergedMasks[i], mergedStats[i]))
        groups.append(current_group)
    return groups
def maskCollapse(Config, groups, shape):
    h, w = shape
    conf = Config.Packages.Drivers.Phases.Phase3.Modules.Module1.Tasks.Task1.Settings.Collapse
    weights = np.array(conf.Weights)
    percentile = conf.Percentile
    best_candidates = []
    for group in groups:
        if not group: 
            continue
        if len(group) == 1:
            mask, metadata = group[0]
            score = np.dot(metadata, weights)
            best_candidates.append((mask, score))
            continue
        metadata_matrix = np.array([item[1] for item in group])
        scores = np.dot(metadata_matrix, weights)
        idx = np.argmax(scores)
        best_candidates.append((group[idx][0], scores[idx]))
    if not best_candidates:
        return np.zeros((h, w), dtype=np.uint8), [np.zeros((h, w), dtype=np.uint8)]
    all_scores = np.array([c[1] for c in best_candidates])
    threshold = np.percentile(all_scores, percentile)
    maksList = [c[0] for c in best_candidates if c[1] >= threshold]
    if maksList:
        collapse = np.logical_or.reduce(maksList).astype(np.uint8)
        return collapse, maksList
    else:
        return np.zeros((h, w), dtype=np.uint8), [np.zeros((h, w), dtype=np.uint8)]
def cloudSegmenter(Config, img, mask):
    conf = Config.Packages.Drivers.Phases.Phase3.Modules.Module1.Tasks.Task1.Settings.Cloud
    if np.count_nonzero(mask) == 0:
        return np.zeros_like(mask), 0
    blurred = cv.GaussianBlur(img, tuple(conf.BlurKernel), 0) 
    masked_pixels = blurred[mask > 0].reshape(-1, 1) 
    val, _ = cv.threshold(masked_pixels, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    relaxed_val = val * conf.Relaxation 
    _, binary = cv.threshold(blurred, relaxed_val, 255, cv.THRESH_BINARY)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, tuple(conf.DilateKernel))
    expanded_mask = cv.dilate(mask, kernel, iterations=conf.DilateIter) 
    region = cv.bitwise_and(binary, binary, mask=expanded_mask)
    n_labels, labels = cv.connectedComponents(region)
    final_cloud = np.zeros_like(mask)
    for i in range(1, n_labels):
        comp_mask = np.uint8(labels == i) * 255
        if cv.countNonZero(cv.bitwise_and(comp_mask, mask)) > 0:
            final_cloud = cv.bitwise_or(final_cloud, comp_mask)
    return final_cloud, relaxed_val
#%% Defining Main Function
def main(Config):       
    task_conf = Config.Packages.Drivers.Phases.Phase3.Modules.Module1.Tasks.Task1
    if task_conf.General.Activation is True:
        print('.... Task1:', colored('Running ℹ️', 'cyan'))
        main_root = Path(Config.Paths.mainRooot)
        srcRoot = (main_root /
            Config.Paths.DataRoots.ResourcesRoot /
            Config.Paths.DataRoots.StreamRoot / 
            Config.Paths.DataRoots.CaseStudyRoot() /
            Config.Packages.Drivers.__name__ /
            Config.Packages.Drivers.Phases.Phase0.__name__ /
            Config.Packages.Drivers.Phases.Phase0.Modules.Module2.__name__ /
            Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task1.__name__ /
            Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task1.MetaData.OutputName)
        dstRoot = (
            main_root /
            Config.Paths.DataRoots.ResourcesRoot /
            Config.Paths.DataRoots.StreamRoot / 
            Config.Paths.DataRoots.CaseStudyRoot() /
            Config.Packages.Drivers.__name__ /
            Config.Packages.Drivers.Phases.Phase3.__name__ /
            Config.Packages.Drivers.Phases.Phase3.Modules.Module1.__name__ /
            Config.Packages.Drivers.Phases.Phase3.Modules.Module1.Tasks.Task1.__name__)
        settings = task_conf.Settings
        ppr = Config.Settings.Acquisition.PPR
        blades = Config.Settings.Acquisition.Blades
        period = ppr / blades 
        bounds = settings.Bounds
        model = sam_model_registry[
            task_conf.Settings.Segmenter.Name](
                checkpoint = (main_root /
            Config.Paths.DataRoots.ResourcesRoot /
            Config.Paths.DataRoots.DependeciesRoot / 
            task_conf.Settings.Segmenter.Model /
            task_conf.Settings.Segmenter.Checkpoint))
        model.to(device="cuda" if torch.cuda.is_available() else "cpu")
        predictor = SamPredictor(model)
        try:
            with h5py.File(srcRoot, 'r') as f:
                cameras = list(f.keys())
                foreGroups = {camera:
                    f[camera][settings.Src.Database][settings.Src.Dataset][settings.Src.Foreground]
                    for camera in cameras}
                backGroups = {camera:
                    f[camera][settings.Src.Database][settings.Src.Dataset][settings.Src.Background]
                    for camera in cameras}
                proces = [k for k in foreGroups[cameras[0]].keys()
                    if any(lower <= (int(k) % period) <= upper for lower, upper in bounds)]
                for key in tqdm(proces, total=len(proces), desc=colored(f'Cavitation Analysis 🚀', 'magenta'), ncols=100):
                    phase = int(key) % period
                    data = dict()
                    for camera in cameras:
                        try:    
                            fgRaw = foreGroups[camera][key][:].astype(np.uint8)
                            roi, _ = getROI(
                                getattr(task_conf.Settings.DynamicROI, camera), phase, fgRaw.shape)
                            prompts = getSmartPrompts(Config, fgRaw, roi)
                            if len(prompts) > 0:
                                masks = segmentEngine(fgRaw, predictor, prompts)
                                stats = getMetadata(masks, fgRaw)
                                groups = groupMasks(Config, masks, stats)
                                collapse, cavities = maskCollapse(Config, groups, fgRaw.shape)
                                cloud, _ = cloudSegmenter(Config, fgRaw, collapse)
                            else:
                                cavities = [np.zeros_like(fgRaw)]
                                collapse = np.zeros_like(fgRaw)
                                cloud = np.zeros_like(fgRaw)
                            data[camera] = [cavities, cloud, roi]    
                        except Exception as e:
                            print(f"Errore al frame {key}: {e}")
                            data[camera] = [[np.zeros_like(fgRaw)], np.zeros_like(fgRaw), np.zeros_like(fgRaw)]
                    dst = (dstRoot / f'{task_conf.MetaData.OutputName}_{key}{task_conf.MetaData.OutputExt}')
                    pickle.dump(data, open(dst, 'wb'))
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception as e:
            print('.... Task1:', colored(f'Error: {e} ❌', 'red'))
            raise e
    elif task_conf.General.Activation is False:
        print('.... Task1:', colored('Offline ⚠️', 'yellow'))
    else:
        raise ValueError('Please Set the Task1 Switch (on/off) ❌')
    return