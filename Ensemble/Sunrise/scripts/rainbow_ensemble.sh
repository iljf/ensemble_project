
#"crazy_climber" "jamesbond" "road_runner" "frostbite" "kangaroo" "chopper_command" "bank_heist"
games=("road_runner" "frostbite" "kangaroo" "chopper_command")

for game in "${games[@]}"; do
    python main.py --game $game --seed 123 --target-update 2000 --T-max 500000 --learn-start 1600 --memory-capacity 500000 --replay-frequency 1 --multi-step 20 --architecture data-efficient --hidden-size 256 --learning-rate 0.0001 --evaluation-interval 1000 --id eff_rainbow
done