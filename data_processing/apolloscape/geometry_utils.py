import numpy as np
from scipy.spatial.transform import Rotation as R

def get_homo_pose(pose_array):
    translation = pose_array[:3]
    rotation = R.from_quat(pose_array[3:])
    # convert translation and rotation to homogenous matrix
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation.as_matrix()
    pose[:3, 3] = translation
    return pose

def points_in_box(bbox, points: np.ndarray, scale_factor=1.0):
    """
    Check if points are inside a 3D bounding box.

    Args:
        bbox (tuple): A tuple containing the center coordinates, length, width, height, and yaw angle of the bounding box.
        points (np.ndarray): A numpy array of shape (N, 3) containing the 3D points to check.
        scale_factor (float, optional): A scaling factor to adjust the size of the bounding box. Defaults to 1.0.

    Returns:
        np.ndarray: A boolean numpy array of shape (N,) indicating whether each point is inside the bounding box.
    """
    # Unpack the parameters of bbox
    cx, cy, cz, l, w, h, yaw = bbox

    # Scale the size of the box
    l *= scale_factor
    w *= scale_factor
    h *= scale_factor

    # Move the points to the origin
    points_shifted = points - np.array([cx, cy, cz])

    # Create the rotation matrix
    rot_matrix = np.array([[np.cos(-yaw), -np.sin(-yaw), 0],
                           [np.sin(-yaw),  np.cos(-yaw), 0],
                           [0,             0,             1]])

    # Rotate the points
    points_rotated = points_shifted.dot(rot_matrix.T)

    # Check if the points are inside the box
    in_box = (
        (np.abs(points_rotated[:, 0]) < l/2) &
        (np.abs(points_rotated[:, 1]) < w/2) &
        (np.abs(points_rotated[:, 2]) < h/2)
    )

    return in_box


def transform_points_from_src_to_dest(src_box: tuple, dest_box: tuple, points: np.ndarray) -> np.ndarray:
    """
    Transforms a set of points from a source bounding box to a destination bounding box.

    Args:
        src_box (tuple): A tuple containing the parameters of the source bounding box in the format
            (cx, cy, cz, l, w, h, yaw).
        dest_box (tuple): A tuple containing the parameters of the destination bounding box in the format
            (cx, cy, cz, l, w, h, yaw).
        points (np.ndarray): A numpy array of shape (N, 3) containing the points to be transformed.

    Returns:
        np.ndarray: A numpy array of shape (N, 3) containing the transformed points.
    """
    # Unpack the parameters of the source and destination bounding boxes
    src_cx, src_cy, src_cz, src_l, src_w, src_h, src_yaw = src_box
    dest_cx, dest_cy, dest_cz, dest_l, dest_w, dest_h, dest_yaw = dest_box

    # Calculate the rotation from the source bounding box to the destination bounding box
    yaw_diff = dest_yaw - src_yaw
    rotation_matrix = np.array([[np.cos(yaw_diff), -np.sin(yaw_diff), 0],
                                [np.sin(yaw_diff),  np.cos(yaw_diff), 0],
                                [0,                 0,                 1]])

    # Calculate the translation from the source bounding box to the destination bounding box
    translation_vector = np.array([dest_cx - src_cx, dest_cy - src_cy, dest_cz - src_cz])

    # Rotate and translate the points from the source bounding box to the destination bounding box
    points_transformed = points.dot(rotation_matrix.T) + translation_vector

    return points_transformed


def apply_lidar_pose_to_boxes(boxes: np.ndarray, src_pose: np.ndarray, dest_pose: np.ndarray) -> np.ndarray:
    """
    Applies the transformation from the source pose to the destination pose to the given boxes.
    
    Args:
    - boxes: A numpy array of shape (N, 7) representing N boxes in the format (x, y, z, w, l, h, yaw).
    - src_pose: A numpy array of shape (7,) representing the source pose in the format (x, y, z, qx, qy, qz, qw).
    - dest_pose: A numpy array of shape (7,) representing the destination pose in the format (x, y, z, qx, qy, qz, qw).
    
    Returns:
    - A numpy array of shape (N, 7) representing the transformed boxes in the format (x, y, z, w, l, h, yaw).
    """
        
    # First, extract translation and rotation from the source pose
    src_rotation = R.from_quat(src_pose[3:])
    src_translation = np.array(src_pose[:3])

    # Extract translation and rotation from the destination pose
    dest_rotation = R.from_quat(dest_pose[3:])
    dest_translation = np.array(dest_pose[:3])

    # Convert boxes to numpy array if it is not already
    if not isinstance(boxes, np.ndarray):
        boxes = np.array(boxes)

    # Apply rotation and translation to the box centers
    box_centers = boxes[:, :3]
    box_centers = src_rotation.apply(box_centers) + src_translation
    box_centers = dest_rotation.inv().apply(box_centers.copy() - dest_translation)
    boxes[:, :3] = box_centers

    # Apply rotation to the box yaws
    box_yaws = boxes[:, -1]
    for i, yaw in enumerate(box_yaws):
        # Convert the yaw angle to a quaternion
        yaw_quaternion = R.from_euler('z', yaw).as_quat()
        
        # Combine the rotation quaternion from the source pose with the yaw quaternion from the box
        # and then apply the combined rotation to the destination pose
        combined_rotation = dest_rotation.inv() * src_rotation * R.from_quat(yaw_quaternion)
        
        # Convert the combined rotation quaternion back to Euler angles to get the updated yaw angle
        boxes[i, -1] = combined_rotation.as_euler('zyx')[0]

    return boxes



def transform_points_from_src_to_dest(src_box, dest_box, points: np.ndarray):
    """
    Transforms a set of 3D points from the coordinate system of a source bounding box to the coordinate system of a destination bounding box.

    Args:
        src_box (tuple): A tuple containing the parameters of the source bounding box in the following order: (center_x, center_y, center_z, length, width, height, yaw).
        dest_box (tuple): A tuple containing the parameters of the destination bounding box in the following order: (center_x, center_y, center_z, length, width, height, yaw).
        points (numpy.ndarray): A numpy array of shape (N, 3) containing the 3D points to be transformed.

    Returns:
        numpy.ndarray: A numpy array of shape (N, 3) containing the transformed 3D points.
    """
    # Unpack the parameters of src_box and dest_box
    src_cx, src_cy, src_cz, src_l, src_w, src_h, src_yaw = src_box
    dest_cx, dest_cy, dest_cz, dest_l, dest_w, dest_h, dest_yaw = dest_box

    # Calculate the rotation from src_box to dest_box
    yaw_diff = dest_yaw - src_yaw
    rotation_matrix = np.array([[np.cos(yaw_diff), -np.sin(yaw_diff), 0],
                                [np.sin(yaw_diff),  np.cos(yaw_diff), 0],
                                [0,                 0,                 1]])

    # Calculate the translation from src_box to dest_box
    translation_vector = np.array([dest_cx - src_cx, dest_cy - src_cy, dest_cz - src_cz])

    # Move points to the origin of src_box coordinate system
    points[:,:3] = points[:,:3] - np.array([src_cx, src_cy, src_cz])

    # Rotate points
    points[:,:3] = points[:,:3].dot(rotation_matrix.T)

    # Move points back to the global coordinate system
    points[:,:3] = points[:,:3] + np.array([src_cx, src_cy, src_cz])

    # Translate points
    points[:,:3] = points[:,:3] + translation_vector

    return points


    


