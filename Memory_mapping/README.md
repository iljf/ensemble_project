# Augmented memory sample efficient reinforcement learning
해당 프로젝트에서는 sample efficient 한 알고리즘을 적용하여 학습시 agent 가 환경과 상호작용하는 것을 최소화한 상태에서 효율적인 정책을 학습할 수 있도록 하는 것을 목표
- Growing When Required (GWR) 네트워크를 활용한 Map based memory 구현
- 진행중인 개인연구(https://github.com/iljf/ensemble_project/tree/main/Ensemble/rainbow) 구현한 Map based memroy 적용하여 학습시 메모리 사이즈 비교 분석
- Memory 구조비교
  - Prioritized experience replay : 기존 연구에서 사용하던 메모리 방식으로 수집된 sample transition들을 segment tree 구조로 학습 sample의 중요도를 계산하여 높은 Td-error를 우선적으로 sampling
  - GWR replay : 해당 프로젝트에서 적용한 방식으로 Graph network 구조를 가지고 agent의 state space 를 node 로 transition 을 edge로 표현하여 현재 input state와 유사항 node 들을 병합하여 메모리 사이즈 감소

## GWR memory
Map-based Experience Replay: A Memory-Efficient Solution to Catastrophic Forgetting in Reinforcement Learning (https://arxiv.org/abs/2305.02054)

## Requirements
- [atari-py](https://github.com/openai/atari-py)
- [PyTorch](http://pytorch.org/)

run `conda env create -f environment.yml` and use `conda activate GWR`
  
Atari 2700 environment을 사용 [`atari-py` ROMs folder](https://github.com/openai/atari-py/tree/master/atari_py/atari_roms)

## Environmental settings
Both Prioiritized expereince replay and GWR replay are tested with [DQN](https://arxiv.org/abs/1312.5602) using Atari 2700 [road_runner](https://ale.farama.org/environments/road_runner/) envrionment


```
python GWR_main.py --target-update 16000 \
                   --T-max 100000 \
                   --learn-start 10000 \
                   --memory-capacity 100000 \
                   --architecture canonical \
                   --hidden-size 512 \
                   --learning-rate 0.0001 \
                   --evaluation-interval 1000 \
                   --activation-threshold 0.9 \
                   --habituation-threshold 0.85 \
                   --block-id 0
```

memory is pre-stacked with 10000 samples before training starts both for PER and GWR

## Experiments
  
PER 과 GWR 의 학습 시 메모리 사이즈 비교
- activiation-threshold = [0.68, 0.7, 0.83 0.9]
- habituation-threshold = [0.45, 0.65, 0.8, 0.85]

![gwr_memory_size](img src="https://github.com/user-attachments/assets/e70f83e6-f7a4-49a2-8a0b-9ac272dafb2b" width="200" height="200"/)
![gwr_reward](https://github.com/user-attachments/assets/5e6f728d-873b-482e-9a6d-9948796d9438)
