beta_means=(0.5 1)
temperatures=(10 40)
ucb_infers=(1 10)

for beta_mean in "${beta_means[@]}"; do
    for temperature in "${temperatures[@]}"; do
        for ucb_infer in "${ucb_infers[@]}"; do
            python sunrise.py --game $1 --num-ensemble 5 --seed 123 \
                              --target-update 2000 --T-max 500000 --learn-start 1600 \
                              --memory-capacity 500000 --replay-frequency 1 --multi-step 20 \
                              --architecture data-efficient --hidden-size 256 --learning-rate 0.0001 \
                              --evaluation-interval 1000 --id sunrise \
                              --beta-mean $beta_mean --temperature $temperature --ucb-infer $ucb_infer
        done
    done
done