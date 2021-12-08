import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.measure import EllipseModel


def normalize_point_spread(points):
    distances = np.array([
        np.linalg.norm(points[i-1] - points[i])
        for i in range(1, points.shape[0]) # first and last point are the same
    ])
    min_distance = distances.min()
    while distances.max() > min_distance + (min_distance / 4):
        new_points = []
        for i, distance in enumerate(distances):
            new_points.append(points[i])
            if distance > min_distance + (min_distance / 4):
                new_point = points[i:i+2].mean(axis=0)
                new_points.append(new_point)
        new_points.append(points[-1]) # complete the ring
        points = np.array(new_points)
        distances = np.array([
            np.linalg.norm(points[i-1] - points[i])
            for i in range(1, points.shape[0]) # first and last point are the same
        ])
    return points


def center_points(points):
    return points - points.mean(axis=0), points.mean(axis=0)


class LogMeanNormalizedEllipseResiduals(object):
    NAME = 'log_mean_noramlized_ellipse_residuals'

    def __init__(self, override=False, **config):
        self.features_dir = config['features_dir']
        self.file_path = os.path.join(self.features_dir, f'{self.NAME}.csv')
        self.override = override
        if self.override or not os.path.exists(self.file_path):
            self.computed_keys = set()
        else:
            self.computed_keys = set(pd.read_csv(self.file_path)['key'])

    def build_features(self, obj_iterator):
        keys = set()
        features = []
        for key, shape in tqdm(obj_iterator):
            if key in self.computed_keys:
                continue
            # center and normalize the points on the boundary
            normed_points = center_points(normalize_point_spread(np.array(shape.boundary.coords)))[0]
            # compute the distances from center for each of the points
            distances = np.linalg.norm(normed_points, axis=1)
            # build an ellipse
            model = EllipseModel()
            model.estimate(normed_points)
            # compute the residuals
            residuals = model.residuals(normed_points)
            # compute the mean normalized residual
            mean = np.mean(residuals / distances)
            features.append({
                'key': key,
                'log_mean_noramlized_ellipse_residual': np.log(mean)
            })
        new_feature_df = pd.DataFrame(features)
        if os.path.exists(self.file_path) and not self.override:
            old_feature_df = pd.read_csv(self.file_path)
            new_feature_df = pd.concat([old_feature_df, new_feature_df])
        new_feature_df.to_csv(self.file_path, index=False)
        self.computed_keys |= keys

        
