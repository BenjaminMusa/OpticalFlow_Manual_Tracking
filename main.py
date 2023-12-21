import cv2
import numpy as np
from imutils.video import VideoStream
import imutils
import imageio
from moviepy.editor import ImageSequenceClip
import os

cap = cv2.VideoCapture('highway.mp4')

ok, frame = cap.read()
#------------------Manually Select Object to Track--------------------------------------
# # Initialize the video stream
vs = VideoStream(src=0).start()
# bbox = cv2.selectROI(frame)
# generate initial corners of detected object
# set limit, minimum distance in pixels and quality of object corner to be tracked
# parameters_shitomasi = dict(maxCorners=100, qualityLevel=0.3, minDistance=7)
# # convert to grayscale
frame_gray_init = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# # Use Shi-Tomasi to detect object corners / edges from initial frame
# edges = cv2.goodFeaturesToTrack(frame_gray_init, mask = None, **parameters_shitomasi)
# # create a black canvas the size of the initial frame
canvas = np.zeros_like(frame)
# # create random colours for visualization for all 100 max corners for RGB channels
# colours = np.random.randint(0, 255, (100, 3))
# # set min size of tracked object, e.g. 15x15px
parameter_lucas_kanade = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# define function to manually select object to track
def select_point(event, x, y, flags, params):
    global point, selected_point, old_points
    # record coordinates of mouse click
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        selected_point = True
        old_points = np.array([[x, y]], dtype=np.float32)


# associate select function with window Selector
cv2.namedWindow('Optical Flow')
cv2.setMouseCallback('Optical Flow', select_point)

# initialize variables updated by function
selected_point = False
point = ()
old_points = ([[]])

height, width, layers = frame.shape

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as per your requirement
video = cv2.VideoWriter('output_video.mp4', fourcc, 30, (width, height))
# loop through the remaining frames of the video
# and apply algorithm to track selected objects
frame_folder = 'frames'
i = 0
while True:
    # get next frame
    ok, frame = cap.read()
    # covert to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    i = i + 1
    if selected_point is True:
        cv2.circle(frame, point, 5, (0, 0, 255), 2)
        # update object corners by comparing with found edges in initial frame
        new_points, status, errors = cv2.calcOpticalFlowPyrLK(frame_gray_init, frame_gray, old_points, None,
                                                         **parameter_lucas_kanade)

        # overwrite initial frame with current before restarting the loop
        frame_gray_init = frame_gray.copy()
        # update to new edges before restarting the loop
        old_points = new_points

        x, y = new_points.ravel()
        j, k = old_points.ravel()

        # draw line between old and new corner point with random colour
        canvas = cv2.line(canvas, (int(x), int(y)), (int(j), int(k)), (0, 255, 0), 3)
        # draw circle around new position
        frame = cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    result = cv2.add(frame, canvas)
    # Save the frame
    frame_path = os.path.join(frame_folder, f'frame_{i + 1:03d}.png')
    print(frame_path)
    cv2.imwrite(frame_path, result)

    # video.write(result)
    cv2.imshow('Optical Flow', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Get the list of frames in the directory
frames = [f for f in os.listdir(frame_folder) if f.endswith('.png')]  # Assuming frames are in JPEG format

# Sort the frames by filename (if necessary)
frames.sort()

# Create a list to store the frames as image paths
image_paths = [os.path.join(frame_folder, frame) for frame in frames]

# Create a video from the frames
clip = ImageSequenceClip(image_paths, fps=30)  # Change the frame rate as per your requirement

# Save the video to a file
output_video_path = 'output_video.mp4'  # Change the output file name as per your requirement
clip.write_videofile(output_video_path, codec='libx264')  # You can change the codec as per your requirement
#----------------------Dense Optical Flow---------------------------

#
# # create canvas to paint on
# hsv_canvas = np.zeros_like(frame)
# # set saturation value (position 2 in HSV space) to 255
# hsv_canvas[..., 1] = 255
#
# while True:
#     # get next frame
#     ok, frame = cap.read()
#     if not ok:
#         print("[ERROR] reached end of file")
#         break
#     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     frame_gray_init = frame_gray.copy()
#     # compare initial frame with current frame
#     flow = cv2.calcOpticalFlowFarneback(frame_gray_init, frame_gray, None, 0.5, 3, 15, 3, 5, 1.1, 0)
#     # get x and y coordinates
#     magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
#     # set hue of HSV canvas (position 1)
#     hsv_canvas[..., 0] = angle*(180/(np.pi/2))
#     # set pixel intensity value (position 3
#     hsv_canvas[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
#
#     frame_rgb = cv2.cvtColor(hsv_canvas, cv2.COLOR_HSV2BGR)
#
#     # optional recording result/mask
#     # video_output.write(frame_rgb)
#
#     cv2.imshow('Optical Flow (dense)', frame_rgb)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
#     # set initial frame to current frame
#     frame_gray_init = frame_gray