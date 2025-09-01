# Copyright (c) 2025, Cauchy WuChao, D-Robotics.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import argparse
import logging
import time

import cv2
import numpy as np
from server_client import Server
from libpyCauchyKesai import CauchyKesai

# server.py
from flask import Flask, request, Response
import numpy as np
import pickle
import zlib
import time

global siglip
siglip = None

app = Flask(__name__)


@app.route('/process', methods=['POST'])
def handle_request():
    global siglip
    begin_time = time.time()
    compressed_data = request.data
    data_bytes = compressed_data
    imgs_list = pickle.loads(data_bytes)
    print("\033[1;31m" + f"pickle.loads time = {1000*(time.time() - begin_time):.2f} ms" + "\033[0m")

    if not isinstance(imgs_list, np.ndarray) or imgs_list.shape != (3, 480, 640, 3):
        return "Invalid input format", 400
    
    begin_time = time.time()
    last_hidden_states = []
    for i in range(3):
        # Pre Process
        img_resized = cv2.resize(imgs_list[i], (384, 288))
        img_padded = cv2.copyMakeBorder( img_resized, 48, 48, 0, 0, cv2.BORDER_CONSTANT, value=(127, 127, 127))
        input_tensor = np.expand_dims(img_padded.transpose((2, 0, 1)), axis=0).astype(np.float32) / 127.5 - 1.0
        # Forward
        siglip.start([input_tensor.copy()], task_id=i)
    for i in range(3):
        last_hidden_states.append(siglip.wait(task_id=i)[0][0])
    result = np.stack(last_hidden_states, axis=0)
    print("\033[1;31m" + f"process_data time = {1000*(time.time() - begin_time):.2f} ms" + "\033[0m")
    
    begin_time = time.time()
    result_bytes = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
    compressed_result = result_bytes #zlib.compress(result_bytes)
    print("\033[1;31m" + f"pickle.dumps time = {1000*(time.time() - begin_time):.2f} ms" + "\033[0m")

    begin_time = time.time()
    r = Response(compressed_result, mimetype='application/octet-stream')
    print("\033[1;31m" + f"Response time = {1000*(time.time() - begin_time):.2f} ms" + "\033[0m")
    
    return r
    

logging.basicConfig(
    level = logging.DEBUG,
    format = '[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger("RDK_RDT")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bpu_siglip_path', type=str, default="BPU_RDT_Policy/bpu_siglip_so400m_patch14_nashm_384x384_featuremaps.hbm", help="")
    parser.add_argument('--port', type=int, default=5000, help="")

    opt = parser.parse_args()
    logger.info(opt)
    
    global siglip
    if not os.path.exists(opt.bpu_siglip_path):
        logger.error("Please Download bpu_siglip_so400m_patch14_nashm_384x384_featuremaps.hbm")
        logger.info("command: wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/RoboticsDiffusionTransformers/bpu_siglip_so400m_patch14_nashm_384x384_featuremaps.hbm")
        exit()
    logger.info("Loading bpu_siglip_so400m_patch14_nashm_384x384_featuremaps.hbm ... (Please wait for 20 seconds.)")
    siglip = CauchyKesai(opt.bpu_siglip_path, n_task=3)
    siglip.s()

    app.run(host='0.0.0.0', port=5000, threaded=False)

if __name__ == "__main__":
    main()
