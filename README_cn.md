
[English](./README.md) | 简体中文

```
GitHub: https://github.com/WuChao-2024/RDK_RoboticsDiffusionTransformers_Tools
FeiShu Document: https://horizonrobotics.feishu.cn/docx/IRVLdawAUoUAoUxNJv7cGlM4nHg
D-Robotics Developer Community NodeHub CN: 
D-Robotics Developer Community NodeHub EN: 
```
```
Contributors: 
- Cauchy @吴超
- SkyXZ @熊旗
```

## Introduction
本文介绍了如何基于RoboTwin2仿真环境和RDK S100系列机器人开发套件复现170M参数的RDT(Robotics Diffusion Transformers)端到端VLA机器人灵巧操作类大模型。

[Video bilibili](https://www.bilibili.com/video/BV17zh9zLEXd)

[Video Youtube](https://youtu.be/jYbSUjFiFik?si=UPY1CCWs7IwCdFyZ)

![](sources/imgs/hardware_in_loop.png)

## 方案优势亮点
1. **全流程方案**：包含数据采集，模型训练，GPU验证，模型导出，模型编译，BPU验证的全流程步骤。以RoboTwin2仿真环境实现全套RDT方案，去除了硬件实机复现的复杂度，同时也以配置较差的8卡A100服务器来实现这个方案，方便复现。当真机出现数值不一致，行为不一致，数据集结构对不上等问题时，也可以使用这个方便复现的纯软件的全流程方案逐一对比。
2. **BPU项目高度封装**：一个Python程序帮你准备好各种ONNX模型，各种单输入多输入的校准数据，yaml文件，config文件，bash编译脚本，您导出后，只需要将准备好的所有文件挂载进工具链Docker，即可获得所有的BPU模型及各种在RDK S100P上运行所需要的文件和数据。
3. **硬件在环**：如果您已经在工控机或者NVIDIA Jetson等设备上复现了RDT，BPU的RDT模型使用标准RoboTwin的Server Client形式封装，支持通过http请求调用，您只需要接入一根网线，即可快速验证BPU加速下的RDT模型的算法效果，避免更改硬件连接的工期。
暂时无法在飞书文档外展示此内容

## 性能数据
1. 所有的RDK S100P均为最佳状态: CPU: 6 x A78AE @ 2.0 GHz, BPU: 1 x Nash-m @ 1.5 GHz, 128 TOPS @ int8
2. 测试时为本地300组数据循环回灌，所有S100P处于循环工作持续负载状态。实际工作中，action chunks下发时不推理，S100P为静息状态，仅在端到端推理时是表格描述时的负载状态。
3. CPU占用和CPU内存占用使用htop命令查看，BPU占用、ION内存占用、DDR带宽占用使用hrt_ucp_monitor -d 3000 命令查看。
4. 端到端延迟是6张图像观测数据和所有关节角度数据输入模型，到最终得到RDT模型预测的未来64个时间步的所有关节角度的过程的时间，端到端相似度是使用完整BPU模型推理的结果与GPU模型推理的结果计算对比得到，数据输入经过了所有的BPU模型的计算，对最终RDT模型预测的未来64个时间步的数据计算余弦相似度。


|  | 单S100P推理 <br/> (RDT - 170M) | 双S100P推理(主) <br/> (RDT - 170M) | 双S100P推理(从) <br/> (RDT - 170M) |
|--|--|--|--|
| 端到端延迟 | 2.160 秒 | 1.630 秒 | - |
|CPU占用 (最大100%)|8%|12 %| 2 %|
|CPU 内存占用 |0.6 GB| 0.7 GB |0.1 GB |
|BPU占用 (最大100%) |100% |100%|40 %|
|ION 内存占用|1.4GB|1.5 GB|0.9 GB|
|DDR 带宽占用|30GB / s|31 GB / s|14 GB / s|
|端到端余弦相似度|0.9998|0.9998|-|

## 成功率数据
由于RoboTwin2的ENV中，环境状态、语言指令都是有随机因素在程序中，虽然已经固定了随机种子，但是我们还是发现在多次评测的过程中，成功率会有一些波动，这里记录的是多次评测的成功率的最大值。

| Task Name | Success Rate <br/> (GPU) | Success Rate <br/> (BPU)| BPU / GPU |
|---|---|---|---|
| blocks_ranking_rgb    | 0.14 | 0.07 |  50.00 % |
| click_bell            | 0.61 | 0.36 |  59.02 % |
| dump_bin_bigbin       | 0.85 | 0.79 |  92.94 % | 
| move_stapler_pad      | 0.19 | 0.10 |  52.63 % |
| place_a2b_left        | 0.26 | 0.20 |  76.92 % |
| place_cans_plasticbox | 0.13 | 0.15 | 115.38 % |
| place_container_plate | 0.81 | 0.93 | 114.81 % | 
| place_object_stand    | 0.41 | 0.42 | 102.44 % |
| put_bottles_dustbin   | 0.53 | 0.31 |  58.49 % |
| shake_bottle          | 0.98 | 0.96 |  97.96 % |


## 环境安装

### 开发机
开发机环境及配置参考
```
Ubuntu 22.04, Python 3.10.12, CUDA 12.4, OpenExplore 3.2.0
CPU: Intel(R) Xeon(R) Platinum 8350C, 52 Cores 112 Threads
GPU: 8 × NVIDIA A100-SXM4-40GB
RAM: 1024 GB | Disk: 963 TB
```
1. RoboTwin-2.0环境安装：https://robotwin-platform.github.io/doc/usage/robotwin-install.html
2. RoboTwin-2.0的RDT环境安装：https://robotwin-platform.github.io/doc/usage/RDT.html
3. RDK S100 OpenExplore 工具链获取：https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview

### 开发板
开发板环境及配置参考  (可选双S100P协同推理)
```
RDK OS 4.0.2-Beta Based on Ubuntu 22.04, Python 3.10.12, OpenExplore 3.2.0
CPU: 6 × A78AE @ 2.0GHz
DDR: 12GB LPDDR5 @ MT/s, 96-bit
BPU: 1 × Nash-m @ 1.5GHz, 128 TOPs @ int8
GPU: 1 × Arm Mali-G78AE,  100 GFLOPS @ FP32
MCU: 4 × Arm Cortex-R52+ @ 1.2GHz
```
1. 在板端编译为RDT具身模型设计的高性能BPU Python接口libpyCaychyKesai.so，对应的commit id为: 6db363bf6a2ef77faa4aeafc378ebf7bc045020b，文档链接参考：https://github.com/WuChao-2024/pyCauchyKesai/blob/6db363bf6a2ef77faa4aeafc378ebf7bc045020b/README_cn.md
步骤包括：从OpenExplore包中获取对应版本的UCP的动态库和头文件，替换板端自带的动态库和头文件，然后编译接口，将libpyCauchyKesai.so动态库移动到Python解释器能找到的引入位置，然后验证.
```
$ python3 -c "from libpyCauchyKesai import __version__ ;print(__version__)"
[UCP]: log level = 3
[UCP]: UCP version = 3.7.4
[VP]: log level = 3
[DNN]: log level = 3
[HPL]: log level = 3
[UCPT]: log level = 6
0.0.7
```

2. 参考RDT_RoboticsDiffusionTransformers_Tools仓库的中的requirements，安装板端的推理环境。
板端依赖文件打包下载，使用文档参考相关链接。

## 一、数据采集 (Data Collect)

### Clean 场景采集example

我们以常见的Adjust Bottle任务为例，从文档中对这个任务的说明可以看到Aloha-AgileX机械臂的成功率大概在93%的样子因此我们这次收集数据选择使用这款双臂平台，由于我们首先收集的是干净环境，因此所有的domain_randomization配置我们都关掉，所以我们收集数据使用的config如下：
```
render_freq: 0
episode_num: 300
use_seed: false
save_freq: 15
embodiment:
- aloha-agilex
language_num: 100
domain_randomization:
  random_background: false
  cluttered_table: false
  clean_background_rate: 1
  random_head_camera_dis: 0
  random_table_height: 0
  random_light: false
  crazy_random_light_rate: 0
camera:
  head_camera_type: D435
  wrist_camera_type: D435
  collect_head_camera: true
  collect_wrist_camera: true
data_type:
  rgb: true
  third_view: false
  depth: false
  pointcloud: false
  observer: false
  endpose: false
  qpos: true
  mesh_segmentation: false
  actor_segmentation: false
pcd_down_sample_num: 1024
pcd_crop: true
save_path: ./clean_data
clear_cache_freq: 5
collect_data: true
eval_video_log: true
```
我们首先进入 RoboTwin 项目的根目录，并在 task_config 目录下创建一个新的 YAML 配置文件，用于保存上述数据采集的配置参数，为了便于识别与管理，我们将其命名为RDKS100_Clean，接着我们在项目的根目录运行如下命令即可开启数据采集：
```
conda activate RoboTwin
bash collect_data.sh adjust_bottle RDKS100_Clean 0
```
出现如下日志即代表数据收集正常运行ing
```
(RoboTwin) qi.xiong@instance-qih2207m:~/DualArm/RoboTwin$ bash collect_data.sh adjust_bottle demo_clean 4
Render Well
============= Config =============

Messy Table: False
Random Background: False
Random Light: False
Random Table Height: 0
Random Head Camera Distance: 0
Head Camera Config: D435, True
Wrist Camera Config: D435, True
Embodiment Config: aloha-agilex

==================================
Task Name: adjust_bottle
[Start Seed and Pre Motion Data Collection]
simulate data episode 0 success! (seed = 0)
```

接着我们便可以看到在我们项目根目录会有我们的收集的数据，具体文件结构如下：
```
(RoboTwin) qi.xiong@instance-qih2207m:~/DualArm/RoboTwin/clean_data/adjust_bottle$ tree -L 2
.
└── RDKS100_Clean
    ├── data             # 轨迹数据文件夹
    ├── instructions     # 任务指令文件夹
    ├── scene_info.json  # 场景信息文件
    ├── seed.txt         # 成功种子文件
    ├── _traj_data       # 原始轨迹数据缓存
    └── video            # 数据采集可视化视频文件夹
5 directories, 2 files
```
除adjust_bottle之外，还可以参考RoboTwin的文档，运行其他的命令，采集不同场景不同任务的数据。


### RoboTwin to RDT 训练数据转换
接着我们对我们采集到的数据进行转换，以适配RDT所需的训练数据格式HDF5，RoboTwin已经提供了这部分的转换代码，我们只需要运行以下命令即可，其中task_name是需要转换的任务名称，对应上面我们收集的任务名称；task_config对应我们上面的数据收集的配置文件；expert_data_num为我们需要导出的数据数量，我们收集了300个数据那便可以选择导出300条也可以根据实际情况导出100条；gpu_id则是数据转换部分我们选择使用的GPU-ID
```
# 处理数据，将获取的专家数据转换为RDT训练数据格式
cd policy/RDT/
mkdir processed_data && mkdir training_data
bash process_data_rdt.sh ${task_name} ${task_config} ${expert_data_num} ${gpu_id}
```

然后我们可以生成RDT训练的参数配置文件，使用如下命令即可,model_name为我们希望给自己模型命的名，其生成的配置文件将会保存在./robotwin/policy/RDT/model_config/中，在其中可以配置我们的训练参数，如train_batch_size、sample_batch_size、cuda_visible_device等，具体的配置介绍如下：
```
cd policy/RDT
bash generate.sh ${model_name}
```

```
# 模型配置名称，用于标识当前训练配置
model: RDT_S100_Clean
# 训练数据路径，指向包含HDF5格式轨迹数据的目录
data_path: training_data/RDT_S100_Clean
# 模型检查点保存路径，训练过程中会定期保存模型权重
checkpoint_path: checkpoints/RDT_S100_Clean
# 预训练模型路径，使用RDT-1B作为基础模型进行微调
pretrained_model_name_or_path: ../weights/RDT/rdt-1b
# 使用的GPU-ID
cuda_visible_device: '0,1,2,3,4,5,6,7'
# 训练批次大小，每个GPU处理的样本数量=train_batch_size//cuda_visible_device
train_batch_size: 16
# 测试批次大小
sample_batch_size: 32
# 最大训练步数，总共训练20000步
max_train_steps: 20000
# 检查点保存周期，每2500步保存一次模型权重
checkpointing_period: 2500
# 采样周期，每100步进行一次轨迹采样以评估模型性能
sample_period: 100
# 检查点总数限制，最多保存40个检查点文件
checkpoints_total_limit: 40
# 学习率，设置为0.0001用于模型微调
learning_rate: 0.0001
# 数据加载器工作进程数，使用8个进程并行加载数据
dataloader_num_workers: 8
# 状态噪声信噪比，添加40dB的噪声到状态数据以增强鲁棒性
state_noise_snr: 40
# 梯度累积步数，每1步进行一次梯度更新
gradient_accumulation_steps: 1
```
接着我们便可以将我们需要使用的训练数据从processed_data复制进training_data/${model_name}目录下了，有多少个任务的数据直接复制进去即可
```
training_data/${model_name}
├── ${task_1}
│   ├── episode_0
|   |   |── episode_0.hdf5
|   |   |-- instructions
|   │   │   ├── lang_embed_0.pt
|   │   │   ├── ...
├── ${task_2}
│   ├── ...
├── ...
```

## 二、模型训练 (Train) 
在完成依赖安装之后我们还需要下载RDT的预训练模型，这部分比较简单，按照以下步骤执行即可，如果你的服务器或者电脑没有配置代理，可能下载速度会很慢，这时只需在终端中输入以下即可自动HuggingFaces的镜像：
```
export HF_ENDPOINT=https://hf-mirror.com
```
接着依次执行以下命令即可完成预训练模型的下载：
```
# step1:进入RoboTwin的policy根目录并创建RDT权重路径
mkdir -p policy/weights/RDT && cd policy/weights/RDT
# step2:依次下载1b的模型即可
huggingface-cli download google/t5-v1_1-xxl --local-dir t5-v1_1-xxl
huggingface-cli download google/siglip-so400m-patch14-384 --local-dir siglip-so400m-patch14-384
huggingface-cli download robotics-diffusion-transformer/rdt-170m --local-dir rdt-170m
```
RoboTwin默认集成的的RDT以及RDT原始仓库默认配置是1B的DiT模型，如果需要训练RDT的170M版本的话需要做些修改，我们首先需要修改模型的参数配置，我们进入RoboTwin/policy/RDT/configs路径找到base.yaml文件170M的模型的lang_token_dim、depth及hidden_size是1B参数量的一半，因此我们对model类的RDT做出如下修改：
```
rdt:
    # 1B: num_head 32 hidden_size 2048
    # 170M: num_head 32 hidden_size 1024 depth 14
    hidden_size: 1024
    depth: 14
    num_heads: 32
    cond_pos_embed_type: multimodal 
```

完成了模型配置的修改之后我们便开始生成我们的训练配置.

```
cd policy/RDT
bash generate.sh ${model_name}
```
接着我们将需要的训练数据从processed_data复制进training_data/${model_name}目录下了，有多少个任务的数据直接复制进去即可，然后我们进入生成的配置文件修改一下预训练模型的路径，修改为我们上面下载的RDT-170M即可，接着我们修改一下我们的训练batch和GPU-ID即可：
```
# Generated on 2025-08-03 01:35:52
model: RDT170M_10Tasks
data_path: training_data/RDT170M_10Tasks
checkpoint_path: checkpoints/RDT170M_10Tasks
pretrained_model_name_or_path: ../weights/RDT/rdt-170m
cuda_visible_device: '4,5,6,7'
train_batch_size: 16
sample_batch_size: 32
max_train_steps: 20000
checkpointing_period: 2500
sample_period: 100
checkpoints_total_limit: 40
learning_rate: 0.0001
dataloader_num_workers: 8
state_noise_snr: 40
gradient_accumulation_steps: 1
```
配置完成后我们便可以输入如命令开始微调：
```
bash finetune.sh ${model_name}
```

### 三、GPU验证 (GPU Eval)
在开发机的GPU上进行精度评测比较简单，RoboTwin已经帮我们做好适配啦，我们只需要输入如下命令即可在仿真环境自动开始Eval啦，具体命令如下，task_name为我们需要评测的任务名称；task_config为我们对仿真环境的配置，和上面的采集数据的时候配置的文件一致，我们可以直接使用数据采集的配置文件也可以自定义一个新的用于评测的仿真环境配置文件；model_name则为我们上面自己命名的模型名字；checkpoint_id则为我们需要使用的检查点，这里只需输入具体的id就好啦；seed则为评测时使用的种子；gpu_id则是评测时使用的GPU-ID：
```
bash eval.sh ${task_name} ${task_config} ${model_name} ${checkpoint_id} ${seed} ${gpu_id}
```

运行成功之后，RoboTwin会自动评测配置文件中设定的次数并统计平均成功率, 评测结束后，可以在`eval_result/adjust_bottle/RDT/demo_clean/RDT170M_10Tasks/2025-08-26_10:18:01/_result.txt`类似目录的文件中，找到本轮任务评测的成功率。
完整的精度和性能数据参考本文最前方Introduction章节。


## 四、模型导出 (ONNX Export) 
在RoBoTwin的目录运行export_all.py一键导出脚本，这个脚本会帮你完成若干个ONNX模型的导出，还会帮你准备校准数据，编译需要的yam配置文件，json配置文件，bash编译脚本，还会帮你准备来自训练集的语言指令及其嵌入张量。
这个脚本在GitHub仓库路径为: `https://github.com/WuChao-2024/RDK_RoboticsDiffusionTransformers_Tools/blob/develop/export_all.py`
在程序的main()函数中，有一些可以配置的项目。
```
parser.add_argument('--export_path', type=str, default="rdt_export_ws", help="")
parser.add_argument('--config_path', type=str, default="policy/RDT/configs/base.yaml", help="")
parser.add_argument('--pretrained_vision_encoder', type=str, default="policy/weights/RDT/siglip-so400m-patch14-384", help="")
parser.add_argument('--pretrained_model', type=str, default="policy/BPU_RDT/checkpoints/RDT170M_10Tasks/checkpoint-22500/pytorch_model/mp_rank_00_model_states.pt", help="")
parser.add_argument('--train_data', type=str, default="policy/RDT/processed_data", help="")
parser.add_argument('--num_samples', type=int, default=100, help="")
parser.add_argument('--jobs', type=int, default=8, help="")
parser.add_argument('--optimized_level', type=str, default="O2", help="")
parser.add_argument('--ctrl_freq', type=int, default=25, help="")
parser.add_argument('--left_arm_dim', type=int, default=6, help="")
parser.add_argument('--right_arm_dim', type=int, default=6, help="")
parser.add_argument('--cal_data_device', type=str, default='cuda', help="")
```
其中：
1. --export_path: 导出的所有产物的文件夹，也是Model Compile时挂载进Docker的文件夹，这个文件夹的包含以下内容。
```
  a. build_all.sh脚本，在Docker内启动BPU模型编译的脚本。
  b. DiT_WorkSpace和img_adaptor_WorkSpace文件夹，被build_all.sh脚本调用，挂载进Docker的环境中，生成这两个部分的BPU模型。
  c. BPU_RDT_Policy文件夹，预先生成的最终BPU产物的文件夹，此时除了存放在BPU模型中的权重，还有一些零散的模型权重，CPU推理已经足够快速，所以选择直接使用ONNXRuntime来推理。
  d. instructions文件夹，存放着语言指令和语言指令的嵌入。语言指令由RoboTwin环境中预制的模板生成，语言指令嵌入由t5模型生成。这些在实际使用中，选择一个效果较好的指令作为Prompt即可。但是请注意，在RoboTwin的BPU Eval环节，我们使用的是随机的语言指令生成一眼指令嵌入来评测，避免语言指令完全相同，成功率直接100%或者0%的情况。
  e. test_data文件夹，保存了端到端的输入和输出数据，用于对比板端的BPU模型端到端输入和输出时的数据一致性。
```
```
.
├── build_all.sh
├── DiT_WorkSpace
│   ├── build.sh
│   ├── config.yaml
│   ├── quant_config.json
│   ├── rdt_dit.onnx
│   └── rdt_dit_calibration
│       ├── freq
│       ├── img_c
│       ├── lang_c
│       ├── lang_mask
│       ├── t
│       └── x
├── img_adaptor_WorkSpace
│   ├── build.sh
│   ├── config.yaml
│   ├── rdt_image_adaptor.onnx
│   └── rdt_image_adaptor_calibration
├── BPU_RDT_Policy
│   ├── base.yaml
│   ├── rdt_lang_adaptor.onnx
│   ├── rdt_state_adaptor_1x1x256.onnx
│   └── rdt_state_adaptor_1x64x256.onnx
├── instructions
│   ├── adjust_bottle-demo_clean-300
│   │   ├── Grab_the_medium-sized_green_bottle_and_lift_it_upright__.pt
│   │   ├── Lift_the_red-capped_bottle_off_the_table_and_hold_it_upright.__.pt
│   │   └── Position_the_narrow-necked_bottle_head-up_and_lift_it__.pt
│   ├── place_dual_shoes-demo_clean-300
│   │   └── Grab_both_the_red_sneaker,_ensure_tips_left,_place_in_the_smooth_bright_orange_box.__.pt
│   └── place_empty_cup-demo_clean-300
│       ├── Use_an_arm_to_place_the_cup_with_smooth_plastic_surface_on_the_round_coaster_with_light_streaks__.pt
│       └── Use_the_arm_to_position_the_smooth_blue_drinking_cup_onto_the_grainy_texture_coaster.__.pt
└── test_data
    ├── ... ...
    ├── 9_actions.npy
    ├── 9_cam_high_0.npy
    ├── 9_cam_high_1.npy
    ├── 9_cam_left_wrist_0.npy
    ├── 9_cam_left_wrist_1.npy
    ├── 9_cam_right_wrist_0.npy
    ├── 9_cam_right_wrist_1.npy
    ├── 9_joints.npy
    └── 9_lang_embeddings.pt
```
2. config_path: RDT模型结构的配置文件，这个是RDT模型训练时就已经准备好的配置文件，与训练保持一致即可，程序内部调用这个模型的目的是加载一个与训练时完全一致的nn.Module类型的PyTorch模型用于前向传播保存校准数据和导出ONNX模型。
3. pretrained_vision_encoder: 预训练的视觉编码器权重的路径，这里是huggingface的SigLIP模型的路径。
4. pretrained_model：训练时会生成很多类型的RDT权重，这里我们只需要mp_rank_00_model_states.pt的权重即可。
5. train_data: 训练时的数据目录，这里用于生成校准数据，程序会遍历其中的每个文件夹，也就是对应每一种任务类型，然后随机提取校准数据。
```
--train_data exampe:
.
├── adjust_bottle-demo_clean-300
│   ├── episode_0
│   │   ├── episode_0.hdf5
│   │   └── instructions
│   ├── episode_1
│   │   ├── episode_1.hdf5
│   │   └── instructions
...
├── place_dual_shoes-demo_clean-300
│   ├── episode_0
│   │   ├── episode_0.hdf5
│   │   └── instructions
...
└── place_empty_cup-demo_clean-300
    ├── episode_0
    │   ├── episode_0.hdf5
    │   └── instructions
...
```
6. num_samples: 每种任务类型随机提取的校准数据数量，默认值100。
7. jobs: 工具链编译模型时的线程数量，默认值8 。
8. optimized_level：工具链编译模型时的优化等级，最高O2，默认O2 。
9. ctrl_freq, left_arm_dim, right_arm_dim: 模型相关的配置项目，与训练时保持一致。
10. cal_data_device: 用于选择生成校准数据时的设备，默认cuda，即使用GPU生成校准数据。


## 五、模型编译 (Model Compile)@吴超 
在这里，我们使用算法工具链标准交付的docker环境，来完成BPU模型的量化和编译，docker的挂载命令参考以下代码块。
```
[sudo] docker run [--gpus all] -it -v <BPU_Work_Space>:/open_explorer REPOSITORY:TAG
```
其中：
- sudo可选，有时候我们的docker安装选项不同，所以需要根据自己的实际情况来选择是否sudo.
- --gpus all是挂载 GPU Docker 所需要的参数，如果是 CPU Docker，则不需要此参数。在X5的PTQ方案中，使用CPU Docker即可，仅仅QAT方案会使用到GPU Docker。在S100的PTQ方案中，GPU Docker会使用CUDA去加速前向传播calibrate阶段，不过，CPU Docker也是完全可以使用PTQ方案的。
- <BPU_Work_Space>需要替换为您想挂载进Docker的路径，这里为您导出的BPU工作目录的路径，注意，需要使用绝对路径。
- REPOSITORY:TAG需要根据您下载的Docker容器名称和版本号，可以使用docker的images命令来确认。
注：
这里只是提供参考的挂载方式，实际上您可以使用任何您喜欢的方式来使用Docker软件，如果您有其他疑问，请参考Docker的官方文档。

然后运行一键编译脚本，编译时间与校准数据集数量和电脑配置有关，GPU加速编译，300条校准数据，大约需要20分钟。
```
bash build_all.sh
```
运行结束后，BPU_RDT_Policy目录会生成RDK S100 BPU 部署所需要的所有产物
```
BPU_RDT_Policy
.
|-- base.yaml
|-- rdt_dit.hbm
|-- rdt_img_adaptor.hbm
|-- rdt_lang_adaptor.onnx
|-- rdt_state_adaptor_1x1x256.onnx
`-- rdt_state_adaptor_1x64x256.onnx
```

## 六、S100 BPU 验证 (BPU Eval)
在RoboTwin的工作目录中，在policy/RDKS100_RDT目录中，准备好GitHub仓库中的这些文件，这其实就是新建一个Policy。这些文件在GitHub仓库对应的网址为：`https://github.com/WuChao-2024/RDK_RoboticsDiffusionTransformers_Tools/tree/develop/RDKS100_RDT`
```
.
├── __init__.py
├── deploy_policy.py
├── deploy_policy.yml
├── eval.sh
└── server_client.py
```
在板端，除了将Model Compile阶段编译好的模型文件夹放到板端，还需要准备好以下工作目录，同时按照本文档的环节安装章节安装好开发板环节。这些文件在GitHub仓库对应的网址为：`https://github.com/WuChao-2024/RDK_RoboticsDiffusionTransformers_Tools/tree/develop/board`
```
.
|-- BPU_RDT_Policy     # BPU Weights
|-- BPU_RDT_Policy.py  # BPU Model
|-- client.py
|-- friend_server.py
|-- server_client.py
`-- requirements.txt
```

### 单S100推理
在RoboTwin环境的deploy_policy.py文件中，我们指定服务器开放的端口，其他默认即可。
```
class RDKS100_RDT_Server:
    def __init__(self):
        self.server = Server(host='0.0.0.0', port=50023)
在S100的client.py文件中，我们将IP修改为服务器的IP，端口修改为服务器的端口。其他的参数与GPU精度评测时保持一致即可。
    parser.add_argument('--bpu_rdt_path', type=str, default='./BPU_RDT_Policy/', help='') 
    # example: $ tree BPU_RDT_Policy
    # .
    # |-- base.yaml
    # |-- bpu_siglip_so400m_patch14_nashm_384x384_featuremaps.hbm
    # |-- rdt_dit.hbm
    # |-- rdt_img_adaptor.hbm
    # |-- rdt_lang_adaptor.onnx
    # |-- rdt_state_adaptor_1x1x256.onnx
    # `-- rdt_state_adaptor_1x64x256.onnx
    parser.add_argument('--host', type=str, default='10.112.20.37', help='')
    parser.add_argument('--port', type=int, default=50023, help='')
    parser.add_argument('--ctrl_freq', type=int, default=25, help="")
    parser.add_argument('--left_arm_dim', type=int, default=6, help="")
    parser.add_argument('--right_arm_dim', type=int, default=6, help="")

```
然后按照顺序开启评测和板子client端，其余与GPU精度评测时一致。

```
# RoboTwin
cd policy/RDKS100_RDT
bash eval.sh place_cans_plasticbox aloha-agilex-m0_b0_l0_h0_c0_D435 1 0

# RDK S100
python3 client.py
```

### 双S100协同推理
在第二块S100上，运行friend_server.py程序，第一块S100的client.py程序中，修改模型实例化方式。
其中，SERVER_URL修改为第二块S100的IP地址，其余与单S100推理一致。
```
# bpu_model = BPU_RDT_Policy(opt.bpu_rdt_path, config_base_yaml, SERVER_URL = None)
bpu_model = BPU_RDT_Policy(opt.bpu_rdt_path, config_base_yaml, SERVER_URL = 'http://10.64.60.208:5000/process')
```
到这里就完成了所有步骤了，精度和性能数据参考本文最前方Introduction章节。

