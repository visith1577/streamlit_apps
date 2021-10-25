import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from matplotlib import gridspec
import gradio as gr
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation


def get_frames():
    cap = cv2.VideoCapture(1)
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as model:
        while cap.isOpened():
            ret, frame = cap.read()

            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = model.process(frame)
            frame.flags.writeable = True

            cv2.imshow('Selfie seg', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

    get_plots(frame, res.segmentation_mas)

    background = np.zeros(frame.shape, dtype=np.uint8)
    mask = np.stack((res.segmentation_mask,) * 3, axis=-1) > 0.5

    segmented_image = np.where(mask, frame, background)

    get_plots(res.segmentation_mask, segmented_image)

    blurred_image = np.where(mask, frame, cv2.blur(frame, (40, 40)))

    get_plots(res.segmentation_mask, blurred_image)


def get_plots(img1, img2):
    plt.figure(figsize=(15, 15))
    grid = gridspec.GridSpec(1, 2)

    plt.figure(figsize=(15, 15))
    grid = gridspec.GridSpec(1, 2)

    ax0 = plt.subplot(grid[0])
    ax1 = plt.subplot(grid[1])

    ax0.imshow(img1)
    ax1.imshow(img2)
    plt.show()


def segment(image):
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as model:
        res = model.process(image)
        mask = np.stack((res.segmentation_mask, )*3, axis=-1) > 0.5
        return np.where(mask, image, cv2.blur(image, (30, 30)))

webcam = gr.inputs.Image(shape=(640, 480), source='webcam')
webapp = gr.interface.Interface(fn=segment, inputs=webcam,outputs="image")
webapp.launch()