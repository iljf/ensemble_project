import os
import numpy as np
import pandas as pd
from scipy import stats
envs = ['alien', 'amidar', 'assault', 'asterix', 'bank_heist', 'battle_zone', 'boxing', 'breakout', 'chopper_command', 'crazy_climber', 'demon_attack', 'freeway', 'frostbite', 'gopher', 'hero', 'jamesbond', 'kangaroo', 'krull', 'kung_fu_master', 'ms_pacman', 'pong', 'private_eye', 'qbert', 'road_runner', 'seaquest', 'up_n_down']

raw_dir = './raw'
csv_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
csv_files = sorted(csv_files)

print(csv_files)
# get folder names in ./reliability
reliability_dir = './reliability'
reliability_folders = [f for f in os.listdir(reliability_dir) if os.path.isdir(os.path.join(reliability_dir, f))]
reliability_folders = sorted(reliability_folders)

significant_envs = []
significant_ps = []
significant_rs = []

    
correl_mse  = []
correl_reliablity = []
correl_ratio = []

# amidar, bankheist, battlezone, crazy climber, frostbite, gopher, hero, jamesbond, kangaroo, krull, mspacman, qbert, roadrunner upndown
# alien, assault, boxing, breakout,freeway
# envs_better = ['alien', 'amidar', 'assault', 'bank_heist', 'battle_zone', 'boxing', 'breakout', 'crazy_climber', 
#                'freeway', 'frostbite', 'gopher', 'hero', 'jamesbond', 'kangaroo', 'krull', 'ms_pacman', 'qbert', 'road_runner', 'up_n_down']

# envs_better = ['amidar', 'bank_heist', 'battle_zone', 'crazy_climber', 
#                'frostbite', 'gopher', 'hero', 'jamesbond', 'kangaroo', 'krull', 'ms_pacman', 'qbert', 'road_runner', 'up_n_down']

# envs_better = ['alien', 'amidar', 'assault', 'bank_heist', 'battle_zone', 'crazy_climber', 
#                'frostbite', 'gopher', 'hero', 'ms_pacman',  'road_runner', 'up_n_down']

# envs_better = ['amidar', 'bank_heist', 'crazy_climber', 
#                'frostbite', 'gopher', 'ms_pacman']

topn = 2
match_count_reliability = 0
match_count_mse = 0
match_count_ratio = 0
match_count_random = 0
match_count_total = 0

match_count_reliability_30 = np.zeros(30)
match_count_mse_30 = np.zeros(30)
match_count_ratio_30 = np.zeros(30)
match_count_random_30 = np.zeros(30)
# seed everything
seed = 42
np.random.seed(seed)
pd.np.random.seed(seed)


# for each env, load csvs correspondingly 
for counter, env in enumerate(envs):
    # if not env in envs_better:
    #     print(f'Skipping environment: {env}')
    #     continue
    # else:
    #     print(f'Processing environment: {env}')


    print(f'Processing environment: {env}')
    print(f'CSV file: {csv_files[counter]}, Reliability file: {reliability_folders[counter]}/ACED_infer_AC_{env}.csv')
    raw_data = pd.read_csv(os.path.join(raw_dir, csv_files[counter]))
    rel_data = pd.read_csv(os.path.join(reliability_dir, reliability_folders[counter], f'ACED_infer_AC_{env}.csv'))
    
    # first, get model column with DQN from raw_data
    model_col = raw_data['model']
    # extract raw_data with DQN in model column
    dqn_col = model_col == 'DQN'
    ddqn_col = model_col == 'DDQN'
    noisy_col = model_col == 'NoisyDQN'
    dueling_col = model_col == 'DuelingDQN'
    dist_col = model_col == 'DistributionalDQN'
    
    dqn_data = raw_data[dqn_col]
    ddqn_data = raw_data[ddqn_col]
    noisy_data = raw_data[noisy_col]
    dueling_data = raw_data[dueling_col]
    dist_data = raw_data[dist_col]
    
    for block in range(5):
        if True:
            # dqn_data with block_id == block
            dqn_block_data = dqn_data[dqn_data['block_id'] == block].reward.values
            ddqn_block_data = ddqn_data[ddqn_data['block_id'] == block].reward.values
            noisy_block_data = noisy_data[noisy_data['block_id'] == block].reward.values
            dueling_block_data = dueling_data[dueling_data['block_id'] == block].reward.values
            dist_block_data = dist_data[dist_data['block_id'] == block].reward.values
            
            rel_block_data = rel_data[rel_data['block_id'] == block]
            rel_block_data_reward = rel_block_data['reward'].values
            rel_block_data_reliability = rel_block_data['reliability'].values
            rel_block_data_reliability = np.array([[float(x) for x in rel_block_data_reliability[c].strip('[]').split(',')] for c in range(len(rel_block_data_reliability))])
            rel_block_data_mse = rel_block_data['mse'].values
            rel_block_data_mse = np.array([[float(x) for x in rel_block_data_mse[c].strip('[]').split(',')] for c in range(len(rel_block_data_mse))])
            rel_block_data_ratio = rel_block_data_reliability / rel_block_data_mse
        
            agents_block_data = [dqn_block_data.mean(), 
                                ddqn_block_data.mean(),
                                noisy_block_data.mean(),
                                dueling_block_data.mean(),
                                dist_block_data.mean()]
            
            # if top3 of agents_block_data is in rel_block_data_*, then match_count += 1
            top_agent = np.argsort(agents_block_data)[-1]
            
            temp_rel = []
            temp_mse = []
            temp_ratio = []
            temp_random = []
            for i in range(30):
                random_topn_agents = np.random.choice(range(len(agents_block_data)), size=topn, replace=False)

                rel_block_data_reliability_i = rel_block_data_reliability[i]
                rel_block_data_mse_i = rel_block_data_mse[i]
                rel_block_data_ratio_i = rel_block_data_ratio[i]
                
                topn_agents_rel = np.argsort(rel_block_data_reliability_i)[-topn:]
                topn_agents_mse = np.argsort(rel_block_data_mse_i)[-topn:]
                topn_agents_ratio = np.argsort(rel_block_data_ratio_i)[-topn:]  
                
                if top_agent in topn_agents_rel:
                    temp_rel.append(1)
                else:
                    temp_rel.append(0)
                if top_agent in topn_agents_mse:
                    temp_mse.append(1)
                else:
                    temp_mse.append(0)
                if top_agent in topn_agents_ratio:
                    temp_ratio.append(1)
                else:
                    temp_ratio.append(0)
                if top_agent in random_topn_agents:
                    temp_random.append(1)
                else:
                    temp_random.append(0)
            
            match_count_reliability_30 += np.array(temp_rel)
            match_count_mse_30 += np.array(temp_mse)
            match_count_ratio_30 += np.array(temp_ratio)
            match_count_random_30 += np.array(temp_random)
            match_count_total += 1
                
                    
t_reliability, p_reliability = stats.ttest_rel(match_count_reliability_30, match_count_random_30)
t_mse, p_mse = stats.ttest_rel(match_count_mse_30, match_count_random_30)
t_ratio, p_ratio = stats.ttest_rel(match_count_ratio_30, match_count_random_30)

w, p = stats.wilcoxon(match_count_ratio_30/match_count_total, match_count_random_30/match_count_total)

# print mean and std of the match counts
print(f'Top 2 Ratio-based match count: {(match_count_ratio_30/match_count_total).mean():.2f} ± {(match_count_ratio_30/match_count_total).std():.2f}')
print(f'Top 2 Random match count: {(match_count_random_30/match_count_total).mean():.2f} ± {(match_count_random_30/match_count_total).std():.2f}')
#print statistics
print(f'Wilcoxon test: W = {w}, p = {p}')
                                   
                                   
# # print results
# print(f'Environment: {env}')
# # print everything
# print(f'Match count reliability: {match_count_reliability}/{match_count_total} ({match_count_reliability / match_count_total * 100:.2f}%)')
# print(f'Match count mse: {match_count_mse}/{match_count_total} ({match_count_mse / match_count_total * 100:.2f}%)')
# print(f'Match count ratio: {match_count_ratio}/{match_count_total} ({match_count_ratio / match_count_total * 100:.2f}%)')
# print(f'Match count random: {match_count_random}/{match_count_total} ({match_count_random / match_count_total * 100:.2f}%)')
# print(f'Total match count: {match_count_total}') 