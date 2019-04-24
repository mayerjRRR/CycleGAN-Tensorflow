import glob
import cv2
import numpy as np

from src.inference_machine import InferenceMachine
from src.utils import argument_parser
from src.utils.image_utils import load_float_image, save_float_image, is_image, is_video, \
    iterate_all_video_float_frames, float_to_unit8, is_directory, uint8_to_float
from src.utils.utils import get_latest_model


def main():
    args, unparsed = argument_parser.get_inference_parser().parse_known_args()
    forwards, input, model_dir, output = parse_arguments(args)

    if model_dir is None:
        model_dir = get_latest_model()

    if is_video(input):
        process_video(input, output, forwards, model_dir)
    elif is_image(input):
        process_single_image(input, output, forwards, model_dir)
    elif is_directory(input):
        process_image_directory(input, output, forwards, model_dir)
    else:
        print("Input must be either image, video, or directory containing images!")


def parse_arguments(args):
    input = args.input
    output = args.output
    forwards = args.forwards
    model_dir = args.model_dir
    return forwards, input, model_dir, output


def process_image_directory(input, output, forwards, model_dir):
    print("Opening input directory...")
    all_frames = glob.glob(input + "*.jpg")
    all_frames.sort(key=str.lower)

    frame_rate = 30
    height, width = cv2.imread(all_frames[0]).shape[0:2]
    inference_height, inference_width = compute_inference_resolution(height, width)

    print("Setting up Inference Machine...")
    inference_machine = InferenceMachine(inference_height, inference_width, model_dir)

    print("Creating output video file...")
    video_writer = create_video_writer(frame_rate, height, output, width * 2)

    frame_counter = 1
    for frame_path in all_frames:
        frame = cv2.imread(frame_path, 1)

        rgb_frame = uint8_to_float(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        print(f"Processing Frame {frame_counter}...", sep=' ', end='\r', flush=True)
        frame_counter += 1
        process_and_store_frame_with_old(rgb_frame, inference_machine, video_writer, forwards, height, width,
                                         inference_height, inference_width)
    print("Done.")

    video_writer.release()


# TODO: Make class with state
def process_video(input, output, forwards, model_dir):
    print("Opening input video file...")
    video_capture = cv2.VideoCapture(input)

    frame_rate, height, inference_height, inference_width, width = compute_video_parameters(video_capture)

    print("Setting up Inference Machine...")
    inference_machine = InferenceMachine(inference_height, inference_width, model_dir)

    print("Creating output video file...")
    video_writer = create_video_writer(frame_rate, height, output, width)

    for frame in iterate_all_video_float_frames(input):
        process_and_store_frame(frame, inference_machine, video_writer, forwards, height, width,
                                inference_height, inference_width)
    print("Done.")

    video_capture.release()
    video_writer.release()


def create_video_writer(frame_rate, height, output, width):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output, fourcc, frame_rate, (width, height))
    return video_writer


def compute_video_parameters(video_capture):
    height, width, frame_rate = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(
        video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), video_capture.get(cv2.CAP_PROP_FPS)
    inference_height, inference_width = compute_inference_resolution(height, width)
    return frame_rate, height, inference_height, inference_width, width


def compute_inference_resolution(height, width):
    if height % 8 == 0 and width % 8 == 0:
        inference_height, inference_width = ((height // 8) + 1) * 8, ((width // 8) + 1) * 8
    else:
        inference_height, inference_width = height, width
    return inference_height, inference_width


def process_and_store_frame(frame, inference_machine, video_writer, forwards, height, width,
                            inference_height, inference_width):
    resized_frame = cv2.resize(frame, (inference_width, inference_height))
    result = inference_machine.recurrent_inference(resized_frame, forwards)
    resized_result = cv2.resize(result, (width, height))
    uint8_result = float_to_unit8(resized_result)
    bgr_result = cv2.cvtColor(uint8_result, cv2.COLOR_RGB2BGR)
    video_writer.write(bgr_result)


def process_and_store_frame_with_old(frame, inference_machine, video_writer, forwards, height, width,
                                     inference_height, inference_width):
    resized_frame = cv2.resize(frame, (inference_width, inference_height))
    result = inference_machine.recurrent_inference(resized_frame, forwards)
    resized_result = cv2.resize(result, (width, height))
    uint8_result = float_to_unit8(resized_result)
    uint8_result = np.concatenate([float_to_unit8(frame), uint8_result], axis=1)
    bgr_result = cv2.cvtColor(uint8_result, cv2.COLOR_RGB2BGR)
    video_writer.write(bgr_result)


# TODO: Make class
def process_single_image(input, output, forwards, model_dir):
    print("Opening input image...")
    input_image = load_float_image(input)

    height, width, _ = input_image.shape
    inference_height, inference_width = compute_inference_resolution(height, width)

    print("Setting up Inference Machine...")
    inference_machine = InferenceMachine(inference_height, inference_width, model_dir)

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
