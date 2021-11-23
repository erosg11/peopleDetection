import cv2
import numpy as np

from tools import iter_cel_phone_frames
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
corners = np.array([[[0, 0], [0, 0], [0, 0], [0, 0]]], dtype=np.int32)
torax_mask = np.zeros((480, 640, 3), dtype=np.uint8)
mean_torax_color = np.zeros((480, 640, 3), dtype=np.uint8)

with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        enable_segmentation=True,
        smooth_segmentation=True,
) as pose:
    for frame in iter_cel_phone_frames():
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        cv2.imshow('MediaPipe Pose', image)
        res = cv2.bitwise_and(frame, frame, mask=(results.segmentation_mask * 255).astype('uint8'))
        cv2.imshow('Masked', res)
        frame2 = frame.copy()
        for i, idx in enumerate([12, 11, 23, 24]):
            landmark = results.pose_landmarks.landmark[idx]
            x = int(landmark.x * frame2.shape[1])
            y = int(landmark.y * frame2.shape[0])
            cv2.circle(frame2, (x, y), radius=1, color=(225, 0, 100), thickness=1)
            corners[0][i] = [x, y]
        cv2.imshow('Pontos', frame2)
        torax_mask.fill(0)
        ignore_mask_color = (255, ) * image.shape[2]
        cv2.fillConvexPoly(torax_mask, corners, ignore_mask_color)
        masked_image = cv2.bitwise_and(frame, torax_mask)
        cv2.imshow('Torax', masked_image)
        mean_color = cv2.mean(frame, mask=torax_mask[:,:,2])
        mean_torax_color[:, :, 0].fill(mean_color[0])
        mean_torax_color[:, :, 1].fill(mean_color[1])
        mean_torax_color[:, :, 2].fill(mean_color[2])
        cv2.imshow('Mean Color', mean_torax_color)
