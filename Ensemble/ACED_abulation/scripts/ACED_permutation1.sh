games=(
    "chopper_command" "crazy_climber" "demon_attack" "freeway" "frostbite" "gopher" "hero"
)

for env_game in "${games[@]}"; do
  case "$env_game" in
    "assault"|"alien"|"asterix"|"bank_heist"|"crazy_climber"|"frostbite"|"kangaroo"|"krull"|"qbert")
      iteration=2
      ;;
    "amidar"|"battle_zone"|"boxing"|"chopper_command"|"demon_attack"|"hero"|"seaquest")
      iteration=1
      ;;
    *)
      iteration=0
      ;;
  esac

  python main_permutation.py --game "$env_game" --iteration "$iteration"
done