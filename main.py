# -*- encoding: utf-8 -*-

import vehicle_detection
import cv2
from moviepy.editor import VideoFileClip

pipeline = vehicle_detection.constructor_pipeline_from_classifier("clf.p")

clip_name = "project_video"
clip_input_path = "./{}.mp4".format(clip_name)
clip_output_path = "./{}_output.mp4".format(clip_name)

clip = VideoFileClip(clip_input_path)
out_clip = clip.fl_image(
    lambda img: cv2.cvtColor(pipeline.search_cars(cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                                  region_of_interest=((0, 400), (1280, 656))),
                             cv2.COLOR_BGR2RGB))
# out_clip.write_videofile(clip_output_path, audio=False)
out_clip.write_images_sequence("debug2/frame_%04d.jpg")
