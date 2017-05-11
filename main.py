# -*- encoding: utf-8 -*-

import vehicle_detection
import cv2
from moviepy.editor import VideoFileClip

pipeline = vehicle_detection.constructor_pipeline_from_classifier("clf.p")

clip_name = "test_video"
clip_input_path = "./{}.mp4".format(clip_name)
clip_output_path = "./{}_output.mp4".format(clip_name)

clip = VideoFileClip(clip_input_path)
out_clip = clip.fl_image(
    lambda img: pipeline.search_cars(cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                     region_of_interest=((0, 400), (1280, 656))))
out_clip.write_videofile(clip_output_path, audio=False)
# clip.write_images_sequence("debug/frame_%03d.jpg")
