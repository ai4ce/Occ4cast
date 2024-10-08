# Occ4cast: LiDAR-based 4D Occupancy Completion and Forecasting

<a href='https://arxiv.org/abs/2310.11239'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a> <a href='https://ai4ce.github.io/Occ4cast/'><img src='https://img.shields.io/badge/Project-website-green'></a> <a href='https://huggingface.co/datasets/ai4ce/OCFBench'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue'></a>

[Xinhao Liu](https://gaaaavin.github.io/)\*,
[Moonjun Gong](https://moonjungong.github.io/)\*, 
[Qi Fang](https://scholar.google.com/citations?user=LIuiQlkAAAAJ),
[Haoyu Xie](.),
[Yiming Li](https://roboticsyimingli.github.io/),
[Hang Zhao](https://hangzhaomit.github.io/), 
[Chen Feng](https://engineering.nyu.edu/faculty/chen-feng)

![](./src/teaser.gif)

# News
* [2024/09]: OCFBench-Waymo is now available on Hugging Face.
* [2024/07]: Occ4cast is accepted to [IROS 2024](https://iros2024-abudhabi.org/).
* [2023/10]: OCFBech-nuScenes is available on [Hugging Face](https://huggingface.co/datasets/ai4ce/OCFBench/tree/main/OCFBench-nuScenes).
* [2023/10]: The preprint version is available on [arXiv](https://arxiv.org/abs/2310.11239). The OCFBench dataset is available on [Hugging Face](https://huggingface.co/datasets/ai4ce/OCFBench).

# Abstract
Scene completion and forecasting are two popular perception problems in research for mobile agents like autonomous vehicles. Existing approaches treat the two problems in isolation, resulting in a separate perception of the two aspects. In this paper, we introduce a novel LiDAR perception task of Occupancy Completion and Forecasting (OCF) in the context of autonomous driving to unify these aspects into a cohesive framework. This task requires new algorithms to address three challenges altogether: (1) sparse-to-dense reconstruction, (2) partial-to-complete hallucination, and (3) 3D-to-4D prediction. To enable supervision and evaluation, we curate a large-scale dataset termed OCFBench from public autonomous driving datasets. We analyze the performance of closely related existing baseline models and our own ones on our dataset. We envision that this research will inspire and call for further investigation in this evolving and crucial area of 4D perception.

# Getting Started
## Installation
The code is tested with Python 3.9, Pytorch 2.0.1, and CUDA 11.8. Please install dependencies by
```
conda create -n occ4cast python=3.9
conda activate occ4cast
pip install -r requirements.txt
```

## OCFBench
Please refer to our [Hugging Face](https://huggingface.co/datasets/ai4ce/Occ4D) page for documentation and download.

## Training
To train the model, please modify the configurations in `baselines/run_train.sh` and run
```
cd baselines
bash run_train.sh
```

# TODO
- [x] Add Waymo dataset.
- [x] Add nuScenes dataset.



# Related Projects
* [Point Cloud Forecasting as a Proxy for 4D Occupancy Forecasting](https://github.com/tarashakhurana/4d-occ-forecasting), CVPR 2023
* [SSCBench: Monocular 3D Semantic Scene Completion Benchmark in Street Views](https://github.com/ai4ce/SSCBench), arXiv 2023
* [Occ3D: A Large-Scale 3D Occupancy Prediction Benchmark for Autonomous Driving](https://github.com/Tsinghua-MARS-Lab/Occ3D), arXiv 2023

# Citation
If you find our work useful in your research, please consider citing:
```
@article{Liu2023occ4cast,
      title={LiDAR-based 4D Occupancy Completion and Forecasting}, 
      author={Xinhao Liu and Moonjun Gong and Qi Fang and Haoyu Xie and Yiming Li and Hang Zhao and Chen Feng},
      journal={arXiv preprint arXiv:2310.11239},
      year={2023}
}
```

# Star history

[![Star History Chart](https://api.star-history.com/svg?repos=ai4ce/occ4cast&type=Date)](https://star-history.com/#ai4ce/occ4cast&Date)
