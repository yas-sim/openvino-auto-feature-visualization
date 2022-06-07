import sys, os
import time
import threading

import numpy as np
import cv2

from openvino.preprocess import PrePostProcessor, ResizeAlgorithm
from openvino.runtime import Core, Layout, Type, InferRequest, AsyncInferQueue


model_name = './public/googlenet-v1-tf/FP16/googlenet-v1-tf.xml'

timing_data = []

class display_thread(threading.Thread):
    width = 1920
    height = 80
    exit_flag = False
    interval = 300
    niter = 100
    lock = None

    def run(self):
        global timing_data

        total_time = self.niter * self.interval + 2000 # ms
        coeff_t2p = display_thread.width / total_time  # coefficient to convert from time (ms) to pixel

        window_name = 'Inference Latency'
        cv2.namedWindow(window_name)
        canvas = np.zeros((display_thread.height, display_thread.width, 3), dtype=np.uint8)
        while display_thread.exit_flag == False:
            while len(timing_data)>0:
                display_thread.lock.acquire()
                submit_time, result_time = timing_data.pop()
                display_thread.lock.release()
                submit_pos = int((submit_time*1000) * coeff_t2p)
                result_pos = int((result_time*1000) * coeff_t2p)
                submit_point = (submit_pos, 20)
                result_point = (result_pos, 60)
                cv2.line(canvas, submit_point, result_point, (255,255,0), 1)
                cv2.drawMarker(canvas, submit_point, (0,255,0), markerType=cv2.MARKER_TRIANGLE_UP, markerSize=8, thickness=1)
                cv2.drawMarker(canvas, result_point, (0,255,0), markerType=cv2.MARKER_TRIANGLE_UP, markerSize=8, thickness=1)
            cv2.imshow(window_name, canvas)
            cv2.waitKey(1)
            time.sleep(1/30)
        print('Hit any key')
        cv2.waitKey(0)
        print('Thread exited')


def callback_func(ireq: InferRequest, user_data: tuple):
    global timing_data
    result_time = time.time()
    niter , start_time, submit_time, lock = user_data
    submit_time = submit_time - start_time
    result_time = result_time - start_time
    lock.acquire()
    timing_data.append((submit_time, result_time))
    lock.release()
    #print('Result #{} latency {} (ms)'.format(niter, result_time - submit_time), flush=True)
    #print('{}'.format(time.time()-submit_time), flush=True)
    #res = next(iter(ireq.results.values())).ravel()


def main():
    global timing_submit, timing_result
    niter = 40
    interval = 30

    core = Core()
    model = core.read_model(model_name)

    input_name  = model.input().get_any_name()			# port index # can be omitted in case the model has only single input 
    input_shape = model.input().get_shape()
    output_name = model.output().get_any_name()
    output_shape = model.output().get_shape()
    print(input_name, input_shape)
    print(output_name, output_shape)

    cfg = {}
    cfg['CACHE_DIR'] = './cache'
    cfg['PERFORMANCE_HINT'] = ['THROUGHPUT', 'LATENCY'][1]
    cfg['ALLOW_AUTO_BATCHING'] = 'YES'      # default=YES
    #cfg['AUTO_BATCH_TIMEOUT']= '100'        # (milli-second) default=1000
    #cfg["CPU_THREADS_NUM"] = "2"
    cfg['PERFORMANCE_HINT_NUM_REQUESTS'] = '4'
    device = ['GPU', 'AUTO', 'AUTO:GPU', 'CPU', 'BATCH:GPU(8)'][0]
    compiled_model = core.compile_model(model, device, cfg)
    print('Device:', device)
    print('Config:', cfg)
    print('niter:', niter, ', interval:', interval, 'ms')
    opt_nireq = compiled_model.get_property('OPTIMAL_NUMBER_OF_INFER_REQUESTS')
    print('OPTIMAL_NUMBER_OF_INFER_REQUESTS', opt_nireq)

    num_ireq = [4, 16, opt_nireq][1]
    iqueue = AsyncInferQueue(compiled_model, num_ireq)
    print('Number of infer requests:', num_ireq)
    iqueue.set_callback(callback_func)

    dummy_input = np.zeros(input_shape, dtype=np.uint8)

    lock = threading.Lock()
    disp_thread = display_thread()
    display_thread.exit_flag = False
    display_thread.lock = lock
    display_thread.interval = interval
    display_thread.niter = niter
    disp_thread.start()

    start_time = time.time()
    for n in range(niter):
        iqueue.start_async({0: dummy_input}, (n, start_time, time.time(), lock))  # the 2nd parameter is a user defined parameter (arbitrary)
        key = cv2.waitKey(interval)
        if key == 27:
            break

    iqueue.wait_all()
    time.sleep(3)
    display_thread.exit_flag = True
    disp_thread.join()

if __name__ == '__main__':
    sys.exit(main())
