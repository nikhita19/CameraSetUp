from calib import calib_video
import json
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
from geometry import get_angle, convert_frames_to_video

print(cv2.__version__)
video_name = "vid_left.mp4"

# Run calibration (Diamond Space - note used for this)
# calib_video(video_name, debug=False, out_path='output_calib_vid')

# Get 2 VPs from calibration file
# (refer notebook on ransac method for vanishing points)
with open('correct_vp.txt', 'r+') as file:
    structure = json.load(file)
    camera_calibration = structure['camera_calibration']

# Compute all Vps for test video
from geometry import computeCameraCalibration
vp1, vp2, vp3, pp , roadPlane, focal = computeCameraCalibration(camera_calibration["vp1"],
                                                  camera_calibration["vp2"],
                                                  camera_calibration["pp"])


# Read video and get count of total frames
cap = cv2.VideoCapture(video_name)
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

# Show all VPs on videoframe
# ret, frame = cap.read()
# Save first frame
# cv2.imwrite("first_vid_main.png", frame)
# plt.plot([pp[0],vp1[0]], [pp[1],vp1[1]], 'r-', 6)
# plt.plot([pp[0],vp2[0]],[pp[1], vp2[1]], 'g-', 6)
# plt.plot([pp[0],vp3[0]], [pp[1],vp3[1]], 'b-', 6)
# plt.imshow(frame)
# plt.show()

# Background Subtraction for Vehicle Detection
object_detector = cv2.createBackgroundSubtractorMOG2()

# Loop over all frames to find angle between VP and midpoint of detected vehicles
frame_count = 0
frame_array = []
angles = []
while(frame_count < total_frames-200):
    frame_count +=1
    ret, frame = cap.read()
    height, width,_ = frame.shape
    # Background subtraction
    mask = object_detector.apply(frame)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    # detect vehicles
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        # Calculate area of contour
        area = cv2.contourArea(cnt)
        if area > 5000 and area < 8000:
            # cv2.drawContours(frame, [cnt], -1, (0,255,0),2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x,y),(x+w, y+h), (0,255,0),2)
            # detections.append([x, y, w, h])
            c = [x+w/2, y+h/2]
            theta = str(round(get_angle(vp1[0:2], c[0:2] , vp2[0:2]),3))
            angles.append(theta)
            text = "Angle = " + theta
            cv2.putText(frame, text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.line(frame, (int(vp1[0]), int(vp1[1])), (int(c[0]), int(c[1])), (0,0,255), 3)
            cv2.line(frame, (int(vp2[0]), int(vp2[1])), (int(c[0]), int(c[1])), (0,255,0), 3)
            # frame = cv2.resize(frame, [])
    cv2.imshow("frame1", frame)
    frame_array.append(frame)
    key = cv2.waitKey(1)
    if key == 27:
        break



# Print results and save into video and text file
print(len(frame_array))
print(min(angles))
print(max(angles))
# json_structure = {'video': [str(video_name)], 'angles': angles}
# with open('testvid_left.txt', 'w') as file:
#     json.dump(json_structure, file)
convert_frames_to_video(frame_array, 'sample.mp4')
cap.release()
cv2.destroyAllWindows()


