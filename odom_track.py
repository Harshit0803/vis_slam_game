import numpy as np 
import cv2
from pathlib import Path
import matplotlib.cm as cm
import sys
import torch
from lightglue import LightGlue, SuperPoint, DISK, ALIKED
import lightWork

# Set floating-point precision for matrix multiplication
torch.set_float32_matmul_precision("medium")

# Check if CUDA (GPU) is available and use it, otherwise default to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize SuperPoint feature extractor and LightGlue matcher with specific settings
extractor = SuperPoint(max_num_keypoints=300, nms_radius=2).eval().cuda()
matcher = LightGlue(features="superpoint").eval().cuda()
# extractor.compile(mode="reduce-overhead")

# Define constants for different stages in visual odometry
STAGE_FIRST_FRAME = 0   
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 3000  # Minimum number of features required for reliable processing

def vo_initialization(cam_width, cam_height, cam_fx, cam_fy, cam_cx, cam_cy, cam_k1, cam_k2, cam_k3, cam_p1, cam_p2):
    """
    Initialize the state of the visual odometry system.
    Sets up camera parameters and initializes variables for tracking and matching features.
    """
    vo_state = {
        'frame_stage': 0,
        'cam': {
            'width': cam_width,
            'height': cam_height,
            'fx': cam_fx,
            'fy': cam_fy,
            'cx': cam_cx,
            'cy': cam_cy,
            'k1': cam_k1,
            'k2': cam_k2,
            'k3': cam_k3,
            'p1': cam_p1,
            'p2': cam_p2
        },
        'new_frame': None,
        'last_frame': None,
        'cur_R': None,
        'cur_t': None,
        'cur_Normal': None,
        'px_ref': None,
        'px_cur': None,
        'focal': cam_fx,
        'pp': (cam_cx, cam_cy),
        'trueX': 0,
        'trueY': 0,
        'trueZ': 0,
        'detector': cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True),
        'prev_normal': None,
        'K': np.array([[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]]),
        'P': np.concatenate((np.array([[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]]), np.zeros((3, 1))), axis=1),
        'frame_R': None,
        'frame_T': None
    }
    return vo_state

def rot_yaw(R,yaw):
    """
    Generate a yaw rotation matrix for 3D rotations around the Z-axis.
    """
    R[0,0] = np.cos(yaw)
    R[0,1] = -np.sin(yaw)
    R[0,2] = 0
    R[1,0] = np.sin(yaw)
    R[1,1] = np.cos(yaw)
    R[1,2] = 0
    R[2,0] = 0
    R[2,1] = 0
    R[2,2] = 1
    return R

def transformation_mat(R, t):
    """
    Create a 4x4 transformation matrix from rotation and translation components.
    Used for 3D pose transformations.
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.reshape(t,(3,))
    return T

def rel_scale_posi_z(vo_state,R, t,q1,q2):
    """
    Determine the sum of positive Z coordinates and relative scale between two views.
    Used for evaluating and selecting the best pose estimation.
    """
    # Combine the rotation and translation to form a complete transformation matrix
    T = transformation_mat(R, t)

    # Compute the projection matrix for the current camera pose
    P = np.matmul(np.concatenate((vo_state['K'], np.zeros((3, 1))), axis=1), T)

    # Use triangulation to find the position of keypoints in 3D space
    hom_Q1 = cv2.triangulatePoints(vo_state['P'], P, q1.T, q2.T)

    # Apply the transformation to see how these points map into the second camera's coordinate system
    hom_Q2 = np.matmul(T, hom_Q1)

    # Convert points from homogeneous to Euclidean coordinates for both views
    uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
    uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

    # Count the number of 3D points that have a positive depth in both camera views
    sum_of_pos_z_Q1 = np.sum(uhom_Q1[2, :] > 0)
    sum_of_pos_z_Q2 = np.sum(uhom_Q2[2, :] > 0)

    # Return the total count of keypoints with positive depth and a constant scale factor
    return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, 1

def cam_pose_recov(vo_state, H, q1, q2):
	"""
    Recover the camera pose using a homography matrix.
    Decomposes homography into possible rotations and translations.
    """
    # Decompose the homography matrix to extract possible rotation matrices, translation vectors, and normal vectors
	retval, R, t, normal = cv2.decomposeHomographyMat(H, vo_state['K'])
	
    # If there's only one solution, use it directly
	if(len(R) == 1):
		R = np.array(R)
		t = np.array(t)
		normal = np.array(normal)
		R = np.reshape(R[0], (3, 3))
		t = np.reshape(t[0], (3, 1))
		normal = np.reshape(normal[0], (3, 1))
		return R, t, normal, vo_state
	else:
        # For multiple solutions, evaluate each to find the most appropriate one
		z_sums = []
		relative_scales = []
		
        # Assess each rotation and translation pair
		for i in range(len(R)):
			Ri, ti = R[i], t[i]
			z_sum, scale = rel_scale_posi_z(vo_state,Ri, ti,q1,q2)
			z_sums.append(z_sum)
			relative_scales.append(scale)

        # Select the solution with the maximum sum of positive Z coordinates
		right_pair_idx = np.argmax(z_sums)
		right_pair = R[right_pair_idx], t[right_pair_idx], normal[right_pair_idx]
		R1, t, normal1 = right_pair

		# Reshape for consistency
		R1 = np.array(R1).reshape((3, 3))
		t = np.array(t).reshape((3, 1))
		normal1 = np.array(normal1).reshape((3, 1))

		return R1, t, normal1, vo_state

def cam_pose_recov_homography(vo_state, H):
    """
    Alternative method to recover the camera pose from a homography matrix.
    """
    # Construct the intrinsic camera matrix from the visual odometry state
    intrinsic_matrix = np.array([[vo_state['focal'], 0, vo_state['pp'][0]],
                                [0, vo_state['focal'], vo_state['pp'][1]],
                                [0, 0, 1]])

    # Decompose the homography matrix to find possible camera poses
    retval, R, t, normal = cv2.decomposeHomographyMat(H, intrinsic_matrix)

    # Select the first solution from the possible solutions given by decomposition
    # The selection criteria or additional checks for the right solution can be implemented here
    R = R[0]
    t = t[0]

    # Naively select the first solution from the decomposition
    R = np.array(R).reshape((3, 3))
    t = np.array(t).reshape((3, 1))

    # Return the rotation matrix, translation vector, and updated state
    return R, t, vo_state

def initial_frame_PROCESSING(vo_state):
    """
    Detect and initialize features in the first frame.
    Establishes the initial set of keypoints for tracking.
    """
    # Use the specified feature detector to find keypoints in the first frame
    vo_state['px_ref'] = vo_state['detector'].detect(vo_state['new_frame'])

    # Convert the list of keypoints to a numpy array for easier processing
    vo_state['px_ref'] = np.array([x.pt for x in vo_state['px_ref']], dtype=np.float32)

    # Update the state to indicate that the system is ready to process the second frame
    vo_state['frame_stage'] = STAGE_SECOND_FRAME  # Replace STAGE_SECOND_FRAME with the actual value or constant

    # Return the updated visual odometry state
    return vo_state

def second_frame_PROCESSING(vo_state, STAGE_DEFAULT_FRAME):
    """
    Process the second frame to establish initial motion estimation.
    Extracts keypoints and matches them with the first frame.
    """
    l, c = lightWork.light(vo_state['last_frame'], vo_state['new_frame'])

    # Swap the variables to align with the expected naming convention
    l, c = c, l
    
    # Check if any points are matched, return current state if none are found
    if l is None:
        return vo_state
    
    # Update the visual odometry state with the matched keypoints
    vo_state['px_ref'], vo_state['px_cur'] = l, c

    # Optionally, compute and store the error between matched points for analysis or filtering
    error = []
    for i in range(vo_state['px_ref'].shape[0]):
        error.append((vo_state['px_ref'][i][0] - vo_state['px_cur'][i][0])**2 + (vo_state['px_ref'][i][1] - vo_state['px_cur'][i][1])**2)

    # Initialize an arbitrary rotation matrix for demonstration (45 degrees rotation around the X-axis)
    angle = -np.pi / 4  # 45 degrees
    vo_state['cur_R'] = np.array([[np.cos(angle), 0, np.sin(angle)],
                                  [0, 1, 0],
                                  [-np.sin(angle), 0, np.cos(angle)]])

    # Set the processing stage to handle default (subsequent) frames
    vo_state['frame_stage'] = STAGE_DEFAULT_FRAME

    # Update the reference keypoints to the currently matched points for the next frame processing
    vo_state['px_ref'] = vo_state['px_cur']

    return vo_state

def final_PROCESSING(vo_state, kMinNumFeature, absolute_scale=2):
    """
    Core processing for each new frame in the visual odometry sequence.
    Tracks feature points, updates poses, and manages keypoint lifecycle.
    """
    # Extract and match keypoints between the last and the current frame
    l, c = lightWork.light(vo_state['last_frame'], vo_state['new_frame'])
    l, c = c, l  # Swap to maintain naming convention
    
    # Terminate processing if no keypoints are matched
    if l is None:
        return vo_state
    
    # Update the visual odometry state with current and reference keypoints
    vo_state['px_ref'], vo_state['px_cur'] = l, c

    # Calculate the error (distance) between matched points
    error = [(vo_state['px_ref'][i][0] - vo_state['px_cur'][i][0])**2 + (vo_state['px_ref'][i][1] - vo_state['px_cur'][i][1])**2 for i in range(vo_state['px_ref'].shape[0])]

    # Compute a homography matrix between the matched points
    H, mask = cv2.findHomography(vo_state['px_cur'], vo_state['px_ref'], method=cv2.RANSAC, confidence=0.995)
    
    # Recover the camera pose from the homography matrix
    R, t, vo_state['cur_Normal'],vo_state = cam_pose_recov(vo_state, H, vo_state['px_ref'], vo_state['px_cur'])  # recover_pose needs to be defined or adjusted accordingly

    # If the translation magnitude is significant, refine the pose using the essential matrix
    if np.linalg.norm(t) > 0.06:
        E, mask = cv2.findEssentialMat(vo_state['px_cur'], vo_state['px_cur'], focal=vo_state['focal'], pp=vo_state['pp'], method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv2.recoverPose(E, vo_state['px_cur'], vo_state['px_ref'], focal=vo_state['focal'], pp=vo_state['pp'])

    # Store the rotation and translation
    vo_state['frame_R'] = R
    vo_state['frame_T'] = t

    # Update rotation and translation based on the absolute scale
    if absolute_scale > 0.01:
        if vo_state['cur_t'] is None:
            vo_state['cur_t'] = absolute_scale * t
        else:
            vo_state['cur_t'] = vo_state['cur_t'] + absolute_scale * vo_state['cur_R'].dot(t)
            vo_state['cur_R'] = R.dot(vo_state['cur_R'])

    # Detect new keypoints if below the minimum threshold
    if vo_state['px_ref'].shape[0] < kMinNumFeature:
        vo_state['px_cur'] = vo_state['detector'].detect(vo_state['new_frame'])
        vo_state['px_cur'] = np.array([x.pt for x in vo_state['px_cur']], dtype=np.float32)
    
    # Update reference keypoints for the next frame
    vo_state['px_ref'] = vo_state['px_cur']

    return vo_state

def frame_update_step(vo_state, img, kMinNumFeature, STAGE_DEFAULT_FRAME, STAGE_SECOND_FRAME, STAGE_FIRST_FRAME):
    """
    Update the visual odometry system with a new frame.
    Drives the entire visual odometry process by handling each new frame.
    """
    # Ensure the provided frame is in the expected format and dimensions
    assert(img.ndim == 2 and img.shape[0] == vo_state['cam']['height'] and img.shape[1] == vo_state['cam']['width']), "Frame: provided image has not the same size as the camera model or image is not grayscale"

    # Set the current frame in the visual odometry state
    vo_state['new_frame'] = img

    # Determine the current processing stage and execute the corresponding function
    if vo_state['frame_stage'] == STAGE_DEFAULT_FRAME:
        vo_state = final_PROCESSING(vo_state, kMinNumFeature)
    elif vo_state['frame_stage'] == STAGE_SECOND_FRAME:
        vo_state = second_frame_PROCESSING(vo_state, STAGE_DEFAULT_FRAME)
    elif vo_state['frame_stage'] == STAGE_FIRST_FRAME:
        vo_state = initial_frame_PROCESSING(vo_state)

    # Update the last processed frame in the state for reference in the next iteration
    vo_state['last_frame'] = vo_state['new_frame']

    return vo_state
