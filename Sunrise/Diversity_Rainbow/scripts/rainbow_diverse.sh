
#"crazy_climber" "jamesbond" "road_runner" "frostbite" "kangaroo" "chopper_command" "bank_heist"
model=("DDQN" "DuelingDQN" "DistributionalDQN")

for model_name in "${model[@]}"; do
    python main_sche.py --model_name $model --seed 122 --target-update 2000 --T-max 500000 --learn-start 1600 --memory-capacity 500000 --replay-frequency 1 --multi-step 20 --architecture data-efficient --hidden-size 256 --learning-rate 0.0001 --evaluation-interval 1000 --id eff_rainbow
done