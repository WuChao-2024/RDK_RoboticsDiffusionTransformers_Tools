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
import time
from collections import deque

import numpy as np
import torch
import cv2

from pathlib import Path
from policy.RDT.models.multimodal_encoder.t5_encoder import T5Embedder
from .server_client import Server


class RDKS100_RDT_Server:
    def __init__(self):
        self.server = Server(host='0.0.0.0', port=50023)

        # obs
        self.observation_window = None
        # language embedding
        GPU = 0
        device = torch.device(f"cuda:{GPU}")
        t5_path = os.path.join(Path(__file__).parent.parent, "weights/RDT/t5-v1_1-xxl")
        print(f"{t5_path = }")
        text_embedder = T5Embedder(
            from_pretrained=t5_path,
            model_max_length=1024,
            device=device,
            use_offload_folder=None,
        )
        self.tokenizer, self.text_encoder = text_embedder.tokenizer, text_embedder.model
        self.text_encoder.eval()

        self.rdt_step = 32
        self.img_size = (640, 480)

    def update_observation_window(self, img_arr, state):
        # JPEG transformation
        # Align with training
        def jpeg_mapping(img):
            if img is None:
                return None
            img = cv2.imencode(".jpg", img)[1].tobytes()
            img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
            return img
        def resize_img(img, size):
            return cv2.resize(img, size)

        if self.observation_window is None:
            self.observation_window = deque(maxlen=2)

            # Append the first dummy image
            self.observation_window.append({
                "qpos": None,
                "images": {
                    "1": None,
                    "2": None,
                    "3": None,
                },
            })

        img_front, img_right, img_left, puppet_arm = (
            img_arr[0],
            img_arr[1],
            img_arr[2],
            state,
        )
        # img resize
        img_front = resize_img(img_front, self.img_size)
        img_left = resize_img(img_left, self.img_size)
        img_right = resize_img(img_right, self.img_size)
        # img jprg encoding
        img_front = jpeg_mapping(img_front)
        img_left = jpeg_mapping(img_left)
        img_right = jpeg_mapping(img_right)

        qpos = np.array(puppet_arm)
        qpos = torch.from_numpy(qpos).float()
        self.observation_window.append({
            "qpos": qpos,
            "images": {
                "1": img_front,
                "2": img_right,
                "3": img_left,
            },
        })


    def reset_obsrvationwindows(self):
        self.observation_window = None
        print("successfully unset obs and language intruction")

    def set_language_instruction(self, language_instruction):
        device = next(self.text_encoder.parameters()).device
        with torch.no_grad():
            tokens = self.tokenizer(
                language_instruction,
                return_tensors="pt",
                padding="longest",
                truncation=True,
            )["input_ids"].to(device)
            tokens = tokens.view(1, -1)
            output = self.text_encoder(tokens)
            lang_embeddings  = output.last_hidden_state.float().contiguous().cpu().detach().numpy()
            del tokens, output
            torch.cuda.empty_cache()
            self.server.send({
                'flag': 'set_lang_condition',
                'instruction': lang_embeddings
            })
            print(self.server.receive())
        print(f"successfully set instruction: {language_instruction}")

    def get_action(self, img_arr=None, state=None):
        assert (img_arr is None) ^ (state is None) == False, "input error"
        if (img_arr is not None) and (state is not None):
            self.update_observation_window(img_arr, state)

        image_arrs_ = [
            self.observation_window[-2]["images"]["1"],
            self.observation_window[-2]["images"]["2"],
            self.observation_window[-2]["images"]["3"],
            self.observation_window[-1]["images"]["1"],
            self.observation_window[-1]["images"]["2"],
            self.observation_window[-1]["images"]["3"],
        ]

        image_arrs = []
        for img in image_arrs_:
            if img is None:
                image_arrs.append(np.ones((480, 640, 3), dtype=np.uint8) * 127)
            else:
                image_arrs.append(img)


        # get last qpos in shape [14, ]
        proprio = self.observation_window[-1]["qpos"]
        # unsqueeze to [1, 14]
        proprio = proprio.numpy()

        begin_time = time.time()
        self.server.send({
                'flag': 'step',
                'imgs_0': image_arrs[0],
                'imgs_1': image_arrs[1],
                'imgs_2': image_arrs[2],
                'imgs_3': image_arrs[3],
                'imgs_4': image_arrs[4],
                'imgs_5': image_arrs[5],
                'joints': proprio
            })

        actions = self.server.receive()["actions"]

        print(f"RDT with Server Client: Cost {(1000*(time.time() - begin_time)):.1f} ms")

        return actions



def encode_obs(observation):  # Post-Process Observation
    observation["agent_pos"] = observation["joint_action"]["vector"]
    return observation

def get_model(usr_args):
    return RDKS100_RDT_Server()

def eval(TASK_ENV, model, observation):

    obs = encode_obs(observation)
    instruction = TASK_ENV.get_instruction()
    input_rgb_arr, input_state = [
        obs["observation"]["head_camera"]["rgb"],
        obs["observation"]["right_camera"]["rgb"],
        obs["observation"]["left_camera"]["rgb"],
    ], obs["agent_pos"]  # TODO

    if (model.observation_window
            is None):  # Force an update of the observation at the first frame to avoid an empty observation window
        model.set_language_instruction(instruction)
        model.update_observation_window(input_rgb_arr, input_state)

    actions = model.get_action()[:model.rdt_step, :] 
    for action in actions:  # Execute each step of the action
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        input_rgb_arr, input_state = [
            obs["observation"]["head_camera"]["rgb"],
            obs["observation"]["right_camera"]["rgb"],
            obs["observation"]["left_camera"]["rgb"],
        ], obs["agent_pos"]  # TODO
        model.update_observation_window(input_rgb_arr, input_state)  # Update Observation


def reset_model(model):
    model.reset_obsrvationwindows()
