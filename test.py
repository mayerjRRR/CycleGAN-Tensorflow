import glob
import cv2
import numpy as np
import os

from src.inference_machine import InferenceMachine
from src.utils import argument_parser
from src.utils.frame_video_writer import FrameVideoWriter
from src.utils.image_utils import load_float_image, save_float_image, is_image, is_video, \
    iterate_all_video_float_frames, float_to_unit8, is_directory, uint8_to_float
from src.utils.utils import get_latest_model, get_subdirs
import tensorflow as tf

def main():
    args, unparsed = argument_parser.get_inference_parser().parse_known_args()
    forwards, input, model_dir, output, model_super_dir, width, height, unet, no_temp = parse_arguments(args)

    if model_dir is None and model_super_dir is None:
        model_dir = get_latest_model()
    if not model_super_dir is None:
        model_directories = get_subdirs(model_super_dir)
        for model in model_directories:
            _, model_name = os.path.split(model)
            output_head, output_tail = os.path.split(output)
            current_output = os.path.join(output_head, model_name+"_"+output_tail)
            print(current_output)
            process_input(args, forwards, input, model, current_output, width, height, unet, no_temp)
            tf.reset_default_graph()
    else:
        process_input(args, forwards, input, model_dir, output, width, height, unet, no_temp)


def process_input(args, forwards, input, model_dir, output, width, height, unet, no_temp):
    if is_video(input):
        process_video(input, output, forwards, model_dir, args.with_old, width, height, unet, no_temp)
    elif is_image(input):
        process_single_image(input, output, forwards, model_dir, width, height, unet, no_temp)
    elif is_directory(input):
        process_image_directory(input, output, forwards, model_dir, args.with_old, width, height, unet, no_temp)
    else:
        print("Input must be either image, video, or directory containing images!")


def parse_arguments(args):
    input = args.input
    output = args.output
    forwards = args.forwards
    model_dir = args.model_dir
    model_super_dir = args.model_super_dir
    width = args.width
    height = args.height
    unet = args.unet
    no_temp = args.no_temp
    return forwards, input, model_dir, output, model_super_dir, width, height, unet, no_temp


def process_image_directory(input, output, forwards, model_dir, with_old, width, height, unet, no_temp):
    print("Opening input directory...")
    all_frames = glob.glob(input + "*.jpg")
    if len(all_frames) == 0:
        all_frames = glob.glob(input + "*.png")
    all_frames.sort(key=str.lower)

    frame_rate = 30
    if width is None or height is None:
        height, width =  cv2.imread(all_frames[0]).shape[0:2]
    inference_height, inference_width = compute_inference_resolution(height, width)

    print("Setting up Inference Machine...")
    inference_machine = InferenceMachine(inference_height, inference_width, model_dir, unet, no_temp)

    print("Creating output video file...")
    video_writer = create_video_writer(frame_rate, height, output, width * 2 if with_old else width)

    for frame_path in all_frames:
        frame = load_float_image(frame_path)
        process_and_store_frame(frame, inference_machine, video_writer, forwards, height, width,
                                         inference_height, inference_width, with_old)
    print("Done.")

    video_writer.release()


# TODO: Make class with state
def process_video(input, output, forwards, model_dir, with_old, width, height, unet, no_temp):
    print("Opening input video file...")
    video_capture = cv2.VideoCapture(input)

    frame_rate, height, inference_height, inference_width, width = compute_video_parameters(video_capture, width, height)

    print("Setting up Inference Machine...")
    inference_machine = InferenceMachine(inference_height, inference_width, model_dir, unet, no_temp)

    print("Creating output video file...")
    video_writer = create_video_writer(frame_rate, height, output, width * 2 if with_old else width)

    for frame in iterate_all_video_float_frames(input):
        cv2.imshow("raw",frame)
        frame = uint8_to_float(frame)
        process_and_store_frame(frame, inference_machine, video_writer, forwards, height, width,
                                inference_height, inference_width, with_old)
    print("Done.")

    video_capture.release()
    video_writer.release()


def create_video_writer(frame_rate, height, output, width):
    _, output_format = os.path.splitext(output)
    if output_format == ".mp4":
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output, fourcc, frame_rate, (width, height))
        return video_writer
    else:
        return FrameVideoWriter(output)


def compute_video_parameters(video_capture, width, height):
    if height is None:
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width is None:
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
    inference_height, inference_width = compute_inference_resolution(height, width)
    return frame_rate, height, inference_height, inference_width, width


def compute_inference_resolution(height, width):
    if not (height % 8 == 0 and width % 8 == 0):
        inference_height, inference_width = ((height // 8) + 1) * 8, ((width // 8) + 1) * 8
    else:
        inference_height, inference_width = height, width
    return inference_height, inference_width


def process_and_store_frame(frame, inference_machine, video_writer, forwards, height, width,
                            inference_height, inference_width, with_old):
    resized_frame = cv2.resize(frame, (inference_width, inference_height))
    result = inference_machine.recurrent_inference(resized_frame, forwards)
    resized_result = cv2.resize(result, (width, height))
    uint8_result = float_to_unit8(resized_result)
    if with_old:
        uint8_result = np.concatenate([cv2.resize(float_to_unit8(frame), (width, height)), uint8_result], axis=1)
    bgr_result = cv2.cvtColor(uint8_result, cv2.COLOR_RGB2BGR)

    cv2.imshow("Output Frames", bgr_result)
    cv2.waitKey(1)
    video_writer.write(bgr_result)


# TODO: Make class
def process_single_image(input, output, forwards, model_dir, width, height, unet, no_temp):
    print("Opening input image...")
    input_image = load_float_image(input)

    if width is None or height is None:
        height, width, _ = input_image.shape
    inference_height, inference_width = compute_inference_resolution(height, width)

    print("Setting up Inference Machine...")
    inference_machine = InferenceMachine(inference_height, inference_width, model_dir, unet, no_temp)

    result = process_single_float_image(input_image, inference_machine, forwards, height, width, inference_height,
                                        inference_width)

    save_float_image(result, output)
    print("Done.")


def process_single_float_image(input_image, inference_machine, forwards, height, width, inference_height,
                               inference_width):
    input_image = cv2.resize(input_image, (inference_width, inference_height))
    result = inference_machine.recurrent_inference(input_image, forwards)
    result = cv2.resize(result, (width, height))
    return result


if __name__ == "__main__":
    main()
