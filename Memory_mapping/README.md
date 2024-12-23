# Augmented memory sample efficient reinforcement learning
해당 프로젝트에서는 sample efficient 한 알고리즘을 적용하여 학습시 agent 가 환경과 상호작용하는 것을 최소화한 상태에서 효율적인 정책을 학습할 수 있도록 하는 것을 목표
- Growing When Required (GWR) 네트워크를 활용한 Map based memory 구현
- 진행중인 개인연구(https://github.com/iljf/ensemble_project/tree/main/Ensemble/rainbow) 구현한 Map based memroy 적용하여 학습시 메모리 사이즈 비교 분석
- Memory 구조비교
  - Prioritized experience replay : 기존 연구에서 사용하던 메모리 방식으로 수집된 sample transition들을 segment tree 구조로 학습 sample의 중요도를 계산하여 높은 Td-error를 우선적으로 sampling
  - GWR replay : 해당 프로젝트에서 적용한 방식으로 Graph network 구조를 가지고 agent의 state space 를 node 로 transition 을 edge로 표현하여 현재 input state와 유사항 node 들을 병합하여 메모리 사이즈 감소

## GWR memory
Map-based Experience Replay: A Memory-Efficient Solution to Catastrophic Forgetting in Reinforcement Learning (https://arxiv.org/abs/2305.02054)

- GWR mermoy는 agent의 state space를 graph node로 mapping 하고 action 을 통해 edge로 연결하여 행동패턴 반영
- BMU(Best matching unit)는 state space를 관리하며 현재 state가 메모리에서 기존 node와 얼마나 유사한지 결정
  - euclidean distance를 계산하여 현재 state와 BMU의 거리가 특정 임계값 이하면 기존 노드와 병합하고 거리가 임계값보다 크다면 새로운 node 생성함으로 메모리를 관리
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

![gwr-mem-reward](https://github.com/user-attachments/assets/39c93416-ff05-4303-8b96-f246fb3cbfec)
  
- AT(action threshold) 가 높을 수록 다양한 행동 패턴이 메모리에 계속 저장 될 가능성이 높음으로 메모리 구조의 action edge가 증가한다
- As AT increases, more diverse data are stored in the data leading to improvements in performance
- HT(habituation-threshold) 가 낮을 수록 기존 node 와의 유사도가 비슷하다고 간주되어 자주 node 끼리의 병합이 일어나 메모리 사이즈 감소
- Low HT can cause important sample loss due to nodes being merged frequently
- Above statements are what 'I was expecting' but the results using DQN was quite different
- The original paper tested AT and HT seperatly but the experiment I performed made synergy between AT,HT; making low HT to use more memory than high HT

Overall, GWR were able to use less than 80%(?) of the memory duing training compare to PER since GWR continously merge new nodes created by BMU with existing nodes using distance in map space; and PER keep stacking transitions untill it reaches the capacity of the memory.
  
![per-gwr](https://github.com/user-attachments/assets/f3192f8e-999c-495f-98e6-b9bfe89fd1a5)

## Conclusion
Come to think of it, the map-based memory (graph network) has a strong side in continous task since the action determines the edges and discrete tasks has only 18 actions max.
(I think this is the reason why so many nodes are merged)
- DQN, road runner 환경에서 실험을 했을때 total time step 을 100k, 기존 메모리 stacking 을 10k만 진행한 점에서 만족할만한 비교결과를 얻을 수는 없었다 (GWR memory로 실험을 하는 단계에서 graph network를 통해서 새로운 노드들을 생성하고 map sapce에 저장함으로 time consuming wise PER 보다 AT,HT 설정값에 따라 5-10배는 더 오래걸렸음)
- 하지만 실험결과에서 일단은 performance 차이가 dramatic 하게 나지 않는 다는점, training 이전 memory stacking 을 80k 나 100k 를 진행하였으면 memory 증가 폭으로 예측하는데 PER 과 GWR memory 간의 메모리 사이즈 차이가 더 줄어들 것이라고 예상된다
