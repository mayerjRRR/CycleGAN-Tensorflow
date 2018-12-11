from src.inference_machine import InferenceMachine
from src.utils import argument_parser
import cv2
from src.utils.image_utils import load_float_image, save_float_image, is_image, is_video, load_all_video_float_frames, save_frames_to_video
from src.utils.utils import get_latest_model
#TODO: Support Video Inference
#TODO; Support Image Directory Inference
#TODO: Support inference on lastest model directory

def run(args):
    forwards, input, model_dir, output = parse_arguments(args)

    if model_dir is None:
        model_dir = get_latest_model()

    if is_video(input):
        process_video(input, output, forwards, model_dir)
    elif is_image(input):
        process_single_image(input, output, forwards, model_dir)


def parse_arguments(args):
    input = args.input
    output = args.output
    forwards = args.forwards
    model_dir = args.model_dir
    return forwards, input, model_dir, output


def process_video(input, output, forwards, model_dir):
    frames = load_all_video_float_frames(input)


    height, width, _ = frames[0].shape

    resized_frames = []
    for frame in frames:
        resized_frames.append(cv2.resize(frame, (height, height)))

    inference_machine = InferenceMachine(height, width, model_dir)

    result_frames = inference_machine.multi_frame_inference(frames, forwards)

    save_frames_to_video(result_frames, output)


def process_single_image(input, output, forwards, model_dir):

    input_image = load_float_image(input)

    #determine resolution
    height, width, _ = input_image.shape
    #TODO: Support Make work for non power of two resolutions
    if height % 8 == 0 and width % 8 == 0:
        good_height, good_width = ((height//8)+1)*8, ((width//8)+1)*8
    else:
        good_height, good_width = height, width
    input_image = cv2.resize(input_image, (good_width, good_height))

    inference_machine = InferenceMachine(good_height, good_width, model_dir)

    result = inference_machine.single_image_inference(input_image, forwards)

    result = cv2.resize(result, (width, height))

    save_float_image(result, output)

def main():
    args, unparsed = argument_parser.get_inference_parser().parse_known_args()
    run(args)


if __name__ == "__main__":
    main()