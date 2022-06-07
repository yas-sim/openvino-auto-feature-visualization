import sys, os
import time

import matplotlib.pyplot as plt
import numpy as np
import cv2

from openvino.preprocess import PrePostProcessor, ResizeAlgorithm
from openvino.runtime import Core, Layout, Type, InferRequest, AsyncInferQueue

latency_data = []

model_name = './public/googlenet-v1-tf/FP16/googlenet-v1-tf.xml'

def callback_func(ireq: InferRequest, user_data: tuple):
    global latency_data
    result_time = time.time()
    niter , start_time, submit_time = user_data
    submit_time = submit_time - start_time
    result_time = result_time - start_time
    latency = result_time - submit_time
    latency_data.append(latency)


def main():
    global latency_data
    latency_data = []
    niter = 2000
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
    #cfg['CACHE_DIR'] = './cache'
    cfg['PERFORMANCE_HINT'] = ['THROUGHPUT', 'LATENCY'][0]
    #cfg['ALLOW_AUTO_BATCHING'] = 'NO'      # default=YES
    #cfg['AUTO_BATCH_TIMEOUT']= '1000'        # (milli-second) default=1000
    device = ['GPU', 'AUTO', 'AUTO:GPU', 'CPU'][1]
    compiled_model = core.compile_model(model, device, cfg)
    print('Device:', device)
    print('Config:', cfg)
    print('niter:', niter, 'interval:', interval)
    opt_nireq = compiled_model.get_property('OPTIMAL_NUMBER_OF_INFER_REQUESTS')
    print('OPTIMAL_NUMBER_OF_INFER_REQUESTS', opt_nireq)

    num_ireq = [0, 16, opt_nireq][1]
    #iqueue = AsyncInferQueue(compiled_model, num_ireq)
    #iqueue = AsyncInferQueue(compiled_model, 16)
    iqueue = AsyncInferQueue(compiled_model)
    print('Number of infer requests:', num_ireq)
    iqueue.set_callback(callback_func)

    dummy_input = np.zeros(input_shape, dtype=np.uint8)

    start_time = time.time()
    for n in range(niter):
        iqueue.start_async({0: dummy_input}, (n, start_time, time.time()))  # the 2nd parameter is a user defined parameter (arbitrary)
        key = cv2.waitKey(interval)
        if key == 27:
            break

    iqueue.wait_all()
    time.sleep(3)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    xaxis = np.array(range(len(latency_data)))
    yaxis = np.array(latency_data)
    #print(xaxis, yaxis)
    ax.grid(True)
    ax.set_xlabel('Inference Iteration ({}ms/inf)'.format(interval))
    ax.set_ylabel('Latency (sec)')
    ax.set_title('Inference Latency')
    ax.plot(xaxis, yaxis)
    plt.show()

if __name__ == '__main__':
    sys.exit(main())
