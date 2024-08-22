models=("DQNV" "DDQN" "DuelingDQN" "NoisyDQN" "DistributionalDQN")

for model_name in "${models[@]}"; do
    python main_sche_2.py --game $1 --model_name $model_name --seed 122 --target-update 2000 --T-max 500000 --learn-start 20000 --memory-capacity 500000 --replay-frequency 4 --multi-step 3 --architecture data-efficient --hidden-size 256 --learning-rate 0.0001 --evaluation-interval 1000 --id Diverse_rainbow
done