# 细粒度片段级中文仇恨言论识别研究
---
本项目为DUT未来技术学院nlp课程安排内容，实验训练了 20 epoches ，用时3.5小时左右。

采用“hfl/chinese-roberta-wwm-ext”作为基础的预训练语言模型。该模型基于Transformer架构，并在大规模中文语料库上进行了预训练，使其能够捕获丰富的中文语言特征和上下文语义信息。

硬件配置使用RTX4060Ti cuda11.8

本实验环境安装包以及依赖如下：
name: nlp2
channels:
  - defaults
  - https://repo.anaconda.com/pkgs/main
  - https://repo.anaconda.com/pkgs/r
  - https://repo.anaconda.com/pkgs/msys2
dependencies:
  - bzip2=1.0.8=h2bbff1b_6
  - ca-certificates=2025.2.25=haa95532_0
  - expat=2.7.1=h8ddb27b_0
  - libffi=3.4.4=hd77b12b_1
  - openssl=3.0.16=h3f729d1_0
  - pip=25.1=pyhc872135_2
  - python=3.9.23=h716150d_0
  - setuptools=78.1.1=py39haa95532_0
  - sqlite=3.45.3=h2bbff1b_0
  - tk=8.6.14=h5e9d12e_1
  - tzdata=2025b=h04d1e81_0
  - vc=14.42=haa95532_5
  - vs2015_runtime=14.42.34433=hbfb602d_5
  - wheel=0.45.1=py39haa95532_0
  - xz=5.6.4=h4754444_1
  - zlib=1.2.13=h8cc25b3_1
  - pip:
      - accelerate==1.7.0
      - certifi==2025.4.26
      - charset-normalizer==3.4.2
      - colorama==0.4.6
      - filelock==3.18.0
      - fsspec==2025.5.1
      - graphviz==0.20.3
      - hf-xet==1.1.3
      - huggingface-hub==0.32.4
      - idna==3.10
      - jinja2==3.1.6
      - joblib==1.5.1
      - markupsafe==3.0.2
      - mpmath==1.3.0
      - networkx==3.2.1
      - numpy==2.0.2
      - packaging==25.0
      - pillow==11.0.0
      - pytorch-crf==0.7.2
      - pyyaml==6.0.2
      - regex==2024.11.6
      - requests==2.32.4
      - safetensors==0.5.3
      - scikit-learn==1.6.1
      - scipy==1.13.1
      - sympy==1.14.0
      - threadpoolctl==3.6.0
      - tokenizers==0.21.1
      - torch==2.7.1+cu118
      - torchaudio==2.7.1+cu118
      - torchvision==0.22.1+cu118
      - tqdm==4.67.1
      - transformers==4.52.4
      - typing-extensions==4.14.0
      - urllib3==2.4.0

感谢杨老师教诲，学生受益匪浅。
