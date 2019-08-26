import glob
import time


from src.inference_machine import InferenceMachine
from src.utils.image_utils import load_float_image


def benchmark(model_path, resolution=512):
    image_dir = "results\\smoke_real\\seqA\\"
    all_frames = glob.glob(image_dir + "*.jpg")
    if len(all_frames) == 0:
        all_frames = glob.glob(image_dir + "*.png")
    all_frames.sort(key=str.lower)

    print(len(all_frames))

    frames = []
    for frame_path in all_frames:
        frames.append(load_float_image(frame_path))


    model_path = "logs/faces_75k_6_full_new_balancing_ObamaTrump_2019-08-01_19-59-52_0-1"
    inference_machine = InferenceMachine(resolution,resolution,model_path, False, False)


    iterations = 500

    start = time.time()
    for i in range(iterations):
        inference_machine.recurrent_inference(frames[(i%len(frames))], True)
    end = time.time()
    print ((end-start)/500.0*1000)

benchmark("yiff")