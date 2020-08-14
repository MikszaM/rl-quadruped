import numpy as np
import csv

save_dir = "rl-quadruped/training/save/results/"

results = np.load(save_dir+"evaluations.npz")

for k in results.keys():
    print(k)
with open(save_dir+'result.csv', 'w', newline='') as csvfile:
    my_writer = csv.writer(csvfile, delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    my_writer.writerow(["Num steps","Episode legth","Reward"])
    for i in range(len(results['timesteps'])):
        output = f"Evaluation at {results['timesteps'][i]}: Reward: {results['results'][i].squeeze()}, Episode lengths: {results['ep_lengths'][i].squeeze()}"
        print(output)
        my_writer.writerow([results['timesteps'][i],results['ep_lengths'][i].squeeze(),results['results'][i].squeeze()])
    

    
