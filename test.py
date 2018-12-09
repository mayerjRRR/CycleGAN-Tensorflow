from src.inference_machine import InferenceMachine
from src.utils import argument_parser
import cv2
from src.utils.image_utils import load_float_image, save_float_image, is_image, is_video, load_all_video_float_frames, save_frames_to_video

#TODO: Support Video Inference
#TODO; Support Image Directory Inference
#TODO: Support inference on lastest model directory

def run(args):
    forwards, input, model_dir, output = parse_arguments(args)

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

    result_frames = inference_machine.multi_frame_inference(resized_frames)

    save_frames_to_video(result_frames, output)


def process_single_image(input, output, forwards, model_dir):

    input_image = load_float_image(input)

    #determine resolution
    height, width, _ = input_image.shape
    #TODO: Support Make work for non power of two resolutions
    #input_image = cv2.resize(input_image, (height, height))

    inference_machine = InferenceMachine(height, width, model_dir)

    if forwards:
        result = inference_machine.forward_inference(input_image)
    else:
        result = inference_machine.backward_inference(input_image)

    save_float_image(result, output)

def main():
    args, unparsed = argument_parser.get_inference_parser().parse_known_args()
    run(args)


if __name__ == "__main__":
    main()