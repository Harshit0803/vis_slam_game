import numpy as np 
import cv2
# Importing necessary libraries for visual odometry
import time
import odom_track

# Setting up initial frame index for image processing
img_id = 0

# Establishing variables for plotting the trajectory
# Coordinates for previous trajectory point
prev_draw_x, prev_draw_y = 290, 90
traj_points = []

# Initial direction vector for navigation
prev_direction = np.array([20, 0, 20]).T

# Index for visual place recognition
VPRIndex = 1

# Defining global variables for SLAM implementation
# List of locations for navigation targets
target_locations = []
navigate = False

# Trajectory towards the target locations
target_traj = []

# Camera intrinsic parameters
cam_width = 320
cam_height = 240
cam_fx = 92
cam_fy = 92
cam_cx = 160
cam_cy = 120

# Distortion coefficients
cam_k1 = 0
cam_k2 = 0
cam_k3 = 0
cam_p1 = 0
cam_p2 = 0

# Defining constants for different stages in visual odometry
STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2

# Minimum number of features for tracking
kMinNumFeature = 3000

# Initializing visual odometry state
vo_state = odom_track.vo_initialization(320, 240, 92, 92, 160, 120, 0, 0, 0, 0, 0)

def reset(new_target_locations):
    global vo_state, target_locations, navigate
    # Reinitializing visual odometry to default state
    vo_state['cur_R'] = np.array([[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]])
    angle = -np.pi / 4
    vo_state['cur_R'][0][0] = np.cos(angle)
    vo_state['cur_R'][2][0] = -np.sin(angle)
    vo_state['cur_R'][0][2] = np.sin(angle)
    vo_state['cur_R'][2][2] = np.cos(angle)
    vo_state['cur_R'][1][1] = 1

    # Setting initial translation vector to origin
    vo_state['cur_t'][0],vo_state['cur_t'][1],vo_state['cur_t'][2]=0,0,0

    # Updating target locations and enabling navigation
    target_locations = new_target_locations
    navigate = True

def getOdometryFromOpticalFlow(data):
    global img_id, prev_draw_x, prev_draw_y, traj_points, prev_direction, VPRIndex, target_locations, navigate, target_traj
    
    # Converting input data to compatible image format
    # Incrementing frame count
    # Starting timer for performance tracking

    img = data  # Data is assumed to be in a suitable format

    # Incrementing image ID for frame tracking
    img_id += 1

    # Timing the visual odometry update process
    start_time = time.time()
    # Updating visual odometry with new image frame
    _ = odom_track.frame_update_step(vo_state, img, kMinNumFeature, STAGE_DEFAULT_FRAME, STAGE_SECOND_FRAME, STAGE_FIRST_FRAME)
    # Calculating time taken for update
    delta_time = time.time() - start_time

    # Extracting current camera pose
    cur_t = vo_state['cur_t']
    cur_R = vo_state['cur_R']

    # Initializing direction vector for visualization
    dir = np.array([20, 0, 20]).T
    if cur_R is not None:
        # Updating direction vector based on camera rotation
        dir = cur_R @ dir
    # Store the direction for future reference
    prev_direction = dir

    # Using estimated translation after a certain number of frames
    if img_id > 15:
        x, y, z = cur_t[0], cur_t[1], cur_t[2]
    else:
        x, y, z = 0., 0., 0.
        # Return an identity matrix and a zero vector if not enough frames have passed
        return np.eye(3), np.zeros((3,))

    # Calculating trajectory drawing coordinates
    draw_x, draw_y = int(x) + 290, int(z) + 290

    # Preparing image for trajectory visualization
    traj = np.full((720, 720, 3), [0, 165, 255], dtype=np.uint8)

    # Marking current camera position on trajectory
    cv2.circle(traj, (draw_x, draw_y), 1, (img_id * 255 / 4540, 255 - img_id * 255 / 4540, 0), 1)

    # Setting scale for direction arrow
    arrow_scale = 1

    # Determining arrow end point
    end_point_x = draw_x - int(dir[0] * arrow_scale)
    end_point_y = draw_y - int(dir[2] * arrow_scale)
    # Drawing arrowed line to indicate camera orientation
    cv2.arrowedLine(traj, (draw_x, draw_y), (end_point_x, end_point_y), (0, 0, 255), thickness=2)

    # Updating trajectory points and drawing them
    if (draw_x, draw_y) != (prev_draw_x, prev_draw_y):
        if not navigate:
            traj_points.append([draw_x, draw_y])
        else:
            target_traj.append([draw_x, draw_y])
        # Update the previous position
        prev_draw_x, prev_draw_y = draw_x, draw_y

    # Visualizing entire trajectory
    for i in range(1, len(traj_points)):
        cv2.line(traj, (traj_points[i - 1][0], traj_points[i - 1][1]), (traj_points[i][0], traj_points[i][1]), (255, 125, 0), 2)

    # Visualizing target trajectory during navigation
    for i in range(1, len(target_traj)):
        cv2.line(traj, (target_traj[i - 1][0], target_traj[i - 1][1]), (target_traj[i][0], target_traj[i][1]), (204, 255, 255), 2)

    # Displaying target locations on trajectory
    for target in target_locations:
        cv2.circle(traj, (int(target[0]), int(target[1])), 1, (0, 0, 255), 5)

    # Format the text with coordinates on separate lines
    text = "Pose:\nx=%2fm\ny=%2fm\nz=%2fm" % (x, y, z)

    # Split the text into separate lines
    lines = text.split('\n')

    # Starting position for the text
    text_x, text_y = 20, 40

    # Use a monospace font
    font = cv2.FONT_HERSHEY_COMPLEX

    # Loop through the lines and put them on the image
    for line in lines:
        cv2.putText(traj, line, (text_x, text_y), font, 1, (255, 255, 255), 1, 8)
        # Move to the next line position
        text_y += 30  # Adjust the value to change spacing between lines


    # Show the trajectory image with the current camera location and trajectory
    cv2.imshow('Trajectory', traj)
    # Wait for 1 ms to allow for the image to be displayed properly
    cv2.waitKey(1)

    # Return the drawing coordinates for potential further use
    return draw_x, draw_y
