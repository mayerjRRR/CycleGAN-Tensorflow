from src.inference_machine import InferenceMachine
from src.utils import argument_parser
import cv2
from src.utils.image_utils import load_float_image, save_float_image, is_image, is_video, load_all_video_float_frames, save_frames_to_video, iterate_all_video_float_frames, float_to_unit8
from src.utils.utils import get_latest_model

#TODO; Support Image Directory Inference

def run(args):
    forwards, input, model_dir, output = parse_arguments(args)

    if model_dir is None:
        model_dir = get_latest_model()

    if is_video(input):
        process_video2(input, output, forwards, model_dir)
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

    if height % 8 == 0 and width % 8 == 0:
        good_height, good_width = ((height//8)+1)*8, ((width//8)+1)*8
    else:
        good_height, good_width = height, width

    good_height, good_width = 720, 1280

    resized_frames = []
    for frame in frames:
        resized_frames.append(cv2.resize(frame, (good_width, good_height)))

    inference_machine = InferenceMachine(good_height, good_width, model_dir)

    result_frames = inference_machine.multi_frame_inference(resized_frames, forwards)

    save_frames_to_video(result_frames, output)


def process_video2(input, output, forwards, model_dir):

    print("Opening input video file...")
    video_capture = cv2.VideoCapture(input)

    height, width, frame_rate = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),int( video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), video_capture.get(cv2.CAP_PROP_FPS)
    if height % 8 == 0 and width % 8 == 0:
        good_height, good_width =((height // 8) + 1) * 8, ((width // 8) + 1) * 8
    else:
        good_height, good_width = height, width
    inference_machine = InferenceMachine(good_height, good_width, model_dir)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    print("Opening input video file...")
    video_writer = cv2.VideoWriter(output, fourcc, frame_rate, (width, height))

    frame_counter = 1
    for frame in iterate_all_video_float_frames(input):
        print(f"Processing Frame {frame_counter}...", sep=' ', end='\r', flush=True)
        frame_counter+=1
        resized_frame = cv2.resize(frame, (good_width, good_height))

        result = inference_machine.single_image_inference(resized_frame, forwards)
        result = cv2.resize(result, (width, height))
        uint8_frame = float_to_unit8(result)
        bgr_frame = cv2.cvtColor(uint8_frame, cv2.COLOR_RGB2BGR)
        video_writer.write(bgr_frame)

    print("Done.")
    video_capture.release()
    video_writer.release()



def process_single_image(input, output, forwards, model_dir):

    input_image = load_float_image(input)

    #determine resolution
    height, width, _ = input_image.shape
    #TODO: Make work for non power of two resolutions
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