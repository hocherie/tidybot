# Author: Jimmy Wu
# Date: February 2023

import argparse
import math
import time
from multiprocessing import Process
import cv2 as cv
import numpy as np
import constants
import utils
from camera_client import CameraClient
from camera_server import CameraServer
from constants import CAMERA_SERIALS
from publisher import Publisher

class Detector:
    def __init__(self, placement, serial, port):
        assert placement in {'top', 'bottom', 'top_only', 'bottom_only'}
        self.placement = placement

        # Camera
        self.camera_center, self.camera_corners = utils.get_camera_alignment_params(serial)
        self.camera_client = CameraClient(port)

        # Aruco marker detection
        cv.setNumThreads(4)  # Based on 12 CPUs
        self.marker_dict = cv.aruco.getPredefinedDictionary(constants.MARKER_DICT_ID)
        self.marker_dict.bytesList = self.marker_dict.bytesList[constants.MARKER_IDS]

        # Reduce false positives
        self.detector_params = cv.aruco.DetectorParameters()
        self.detector_params.minCornerDistanceRate = 0.2  # Marker detections should be fronto-parallel
        self.detector_params.adaptiveThreshWinSizeMin = 23  # Use single thresholding step with window size 23 since all markers are the same size

        # Pose transformation
        self.transformation_matrix = self.compute_transformation_matrix(np.array(self.camera_corners, dtype=np.float32))
        self.height_ratio = (constants.CAMERA_HEIGHT - constants.ROBOT_HEIGHT) / constants.CAMERA_HEIGHT
        self.position_offset = constants.ROBOT_DIAG / 2 - constants.MARKER_PARAMS['sticker_length'] / math.sqrt(2)
        self.marker_template = self._compute_marker_template()

    def compute_transformation_matrix(self, src_points):
        if self.placement == 'top':
            # Top left, top right, bottom right, bottom left
            dst_points = np.array([
                [-(constants.FLOOR_WIDTH / 2), constants.FLOOR_LENGTH / 2],
                [constants.FLOOR_WIDTH / 2, constants.FLOOR_LENGTH / 2],
                [constants.FLOOR_WIDTH / 2, 0],
                [-(constants.FLOOR_WIDTH / 2), 0],
            ], dtype=np.float32)
        elif self.placement == 'bottom':
            dst_points = np.array([
                [-(constants.FLOOR_WIDTH / 2), 0],
                [constants.FLOOR_WIDTH / 2, 0],
                [constants.FLOOR_WIDTH / 2, -(constants.FLOOR_LENGTH / 2)],
                [-(constants.FLOOR_WIDTH / 2), -(constants.FLOOR_LENGTH / 2)],
            ], dtype=np.float32)
        elif self.placement == 'top_only':
            dst_points = np.array([
                [-(constants.FLOOR_WIDTH / 2), constants.FLOOR_LENGTH / 4],
                [constants.FLOOR_WIDTH / 2, constants.FLOOR_LENGTH / 4],
                [constants.FLOOR_WIDTH / 2, -(constants.FLOOR_LENGTH / 4)],
                [-(constants.FLOOR_WIDTH / 2), -(constants.FLOOR_LENGTH / 4)],
            ], dtype=np.float32)
        elif self.placement == 'bottom_only':
            dst_points = np.array([
                [-(constants.FLOOR_WIDTH / 2), constants.FLOOR_LENGTH / 2],
                [constants.FLOOR_WIDTH / 2, constants.FLOOR_LENGTH / 2],
                [constants.FLOOR_WIDTH / 2, -(constants.FLOOR_LENGTH / 2)],
                [-(constants.FLOOR_WIDTH / 2), -(constants.FLOOR_LENGTH / 2)],
            ], dtype=np.float32)

        return cv.getPerspectiveTransform(src_points, dst_points).astype(np.float32)

    def _compute_marker_template(self):
        center_to_marker = np.radians([135, 45, -45, -135], dtype=np.float64)
        template = np.column_stack([
            self.position_offset * np.cos(center_to_marker),
            self.position_offset * np.sin(center_to_marker),
        ])
        return template

    def _rigid_body_fit(self, template_points, observed_points):
        # Closed-form least-squares rigid body fit (Kabsch algorithm)
        # Solves for optimal rotation + translation: observed = R @ template + t
        centroid_template = template_points.mean(axis=0)
        centroid_observed = observed_points.mean(axis=0)
        P = template_points - centroid_template
        Q = observed_points - centroid_observed
        H = P.T @ Q
        U, _, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        S = np.array([[1, 0], [0, d]])
        R = Vt.T @ S @ U.T
        t = centroid_observed - R @ centroid_template
        heading = math.atan2(R[1, 0], R[0, 0])
        return t[0], t[1], heading

    def get_poses_from_markers(self, corners, indices, debug=False):
        data = {'poses': {}, 'single_marker_robots': set()}
        if indices is None:
            return data

        # Convert marker corners from pixel coordinates to real-world coordinates
        corners = np.concatenate(corners, axis=1).squeeze(0)
        camera_center = np.array(self.camera_center, dtype=np.float32)
        corners = camera_center + self.height_ratio * (corners - camera_center)
        corners = np.c_[corners, np.ones(corners.shape[0], dtype=np.float32)]
        corners = corners @ self.transformation_matrix.T
        corners = (corners[:, :2] / corners[:, 2:]).reshape(-1, 4, 2)

        # Compute marker positions
        centers = corners.mean(axis=1)

        # Compute marker headings for single-marker fallback
        diffs = (corners - centers.reshape(-1, 1, 2)).reshape(-1, 2)
        angles = np.arctan2(diffs[:, 1], diffs[:, 0]).reshape(-1, 4) + np.radians([-135, -45, 45, 135], dtype=np.float32)
        angles1 = np.mod(angles + math.pi, 2 * math.pi) - math.pi
        angles2 = np.mod(angles, 2 * math.pi)
        headings = np.where(
            angles1.std(axis=1) < angles2.std(axis=1),
            angles1.mean(axis=1),
            np.mod(angles2.mean(axis=1) + math.pi, 2 * math.pi) - math.pi)

        # Compute robot poses
        indices = indices.squeeze(1)
        robot_indices = np.floor_divide(indices, 4)
        for robot_idx in np.unique(robot_indices):
            robot_idx = robot_idx.item()
            robot_mask = robot_indices == robot_idx
            indices_robot = np.mod(indices[robot_mask], 4)
            centers_robot = centers[robot_mask]

            single_marker = robot_mask.sum() == 1
            if single_marker:
                heading = headings[robot_mask].item()
                angle = heading + np.radians([-45, -135, 135, 45], dtype=np.float32)[indices_robot[0]]
                position = centers_robot[0].copy()
                position[0] += self.position_offset * np.cos(angle)
                position[1] += self.position_offset * np.sin(angle)
            else:
                # Rigid-body fit of known marker template to observed positions
                template_subset = self.marker_template[indices_robot]
                observed = centers_robot.astype(np.float64)
                x, y, heading = self._rigid_body_fit(template_subset, observed)
                position = np.array([x, y], dtype=np.float32)

            data['poses'][robot_idx] = (position[0], position[1], heading)
            if single_marker:
                data['single_marker_robots'].add(robot_idx)

        if debug:
            positions_debug = centers.copy()
            for i, idx in enumerate(indices):
                robot_idx = idx // 4
                if robot_idx in data['poses']:
                    px, py, _ = data['poses'][robot_idx]
                    positions_debug[i] = [px, py]
            data['debug_data'] = list(zip(indices.tolist(), centers.tolist(), positions_debug.tolist()))

        return data

    def get_poses(self, debug=False):
        image = self.camera_client.get_image()

        # Detect markers
        corners, indices, _ = cv.aruco.detectMarkers(image, self.marker_dict, parameters=self.detector_params)

        if debug:
            image_copy = image.copy()  # 0.2 ms
            if indices is not None:
                cv.aruco.drawDetectedMarkers(image_copy, corners, indices)
            cv.imshow(f'Detections ({self.placement})', image_copy)  # 0.3 ms

        return self.get_poses_from_markers(corners, indices, debug=debug)

class MarkerDetectorServer(Publisher):
    def __init__(self, hostname='localhost', port=6002, top_only=False, bottom_only=False, debug=False):
        super().__init__(hostname=hostname, port=port)
        self.debug = debug
        if bottom_only:
            self.detectors = [Detector('bottom_only', CAMERA_SERIALS[1], 6001)]
        elif top_only:
            self.detectors = [Detector('top_only', CAMERA_SERIALS[0], 6000)]
        else:
            self.detectors = [Detector('top', CAMERA_SERIALS[0], 6000), Detector('bottom', CAMERA_SERIALS[1], 6001)]

    def get_data(self):
        data = {'poses': {}}
        if self.debug:
            data['debug_data'] = []
        for detector in self.detectors:
            new_data = detector.get_poses(debug=self.debug)
            for robot_idx, pose in new_data['poses'].items():
                if robot_idx in data['poses'] and robot_idx in new_data['single_marker_robots']:
                    continue  # Single marker pose estimates are not as reliable
                data['poses'][robot_idx] = pose  # Bottom detector takes precedence
            if 'debug_data' in new_data:
                data['debug_data'].extend(new_data['debug_data'])
        if self.debug:
            cv.waitKey(1)
        return data

    def clean_up(self):
        for detector in self.detectors:
            detector.camera_client.close()
        cv.destroyAllWindows()

def main(args):
    # Start camera servers
    def start_camera_server(serial, port):
        CameraServer(serial, port=port).run()
    if args.bottom_only:
        cameras = [(CAMERA_SERIALS[1], 6001)]
    elif args.top_only:
        cameras = [(CAMERA_SERIALS[0], 6000)]
    else:
        cameras = [(CAMERA_SERIALS[0], 6000), (CAMERA_SERIALS[1], 6001)]
    for serial, port in cameras:
        Process(target=start_camera_server, args=(serial, port), daemon=True).start()

    # Wait for camera servers to be ready
    time.sleep(1.5)

    # Start marker detector server
    MarkerDetectorServer(top_only=args.top_only, bottom_only=args.bottom_only, debug=args.debug).run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--top-only', action='store_true')
    parser.add_argument('--bottom-only', action='store_true')
    parser.add_argument('--debug', action='store_true')
    main(parser.parse_args())
