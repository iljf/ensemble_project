for seed in 122; do
    python sunrise_scheduler.py --game $1 --num-ensemble 5 --seed $seed --target-update 2000 --T-max 500000 --learn-start 1600 --memory-capacity 500000 --replay-frequency 1 --multi-step 20 --architecture data-efficient --hidden-size 256 --learning-rate 0.0001 --evaluation-interval 1000 --id sigmoid_test_P --beta-mean $2 --temperature $3 --ucb-infer $4
done
