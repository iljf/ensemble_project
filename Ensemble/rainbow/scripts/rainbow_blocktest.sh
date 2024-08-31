models=("DQNV" "DDQN" "DuelingDQN" "NoisyDQN" "DistributionalDQN")

for model_name in "${models[@]}"; do
    python main.py --block-id 0 --game $1 --model_name $model_name --seed 122 --target-update 2000 --T-max 100000 --learn-start 20000 --memory-capacity 500000 --replay-frequency 1 --multi-step 20 --architecture data-efficient --hidden-size 256 --learning-rate 0.0001 --evaluation-interval 1000 --id $2
done