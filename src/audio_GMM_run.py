from audio_GMM import train_modelA, test_modelA
import matplotlib.pyplot as plt
import numpy as np

muf, covf, mug, covg, ws = train_modelA(['data/target_dev', 'data/target_train'],['data/non_target_train/', 'data/non_target_dev'])
# target =test_modelA(['data/target_dev', 'data/target_train'],muf,covf, mug, covg, ws)
# non_target = test_modelA(['data/non_target_dev/', 'data/non_target_train/'],muf,covf, mug, covg, ws)
eval_data = test_modelA(['data/target_train/'],muf,covf, mug, covg, ws)

print("Writing results to file")
file = open('../audio_GMM.txt', 'w')
for key, value in eval_data.items():
	file.write(key[:-4])
	file.write(' ')
	file.write(value[0])
	file.write(' ')
	file.write(int(value[1]))
	file.write("\n")
	print(key[:-4], value[0], int(value[1]))

file.close()
# print("Plotting")
# plt.figure()
# plt.scatter(np.vstack(eval_data.values())[:,0], np.zeros(len(eval_data)), label='eval')
# plt.scatter(np.vstack(non_target.values())[:,0], np.zeros(len(non_target)), label='non_target')
# plt.scatter(np.vstack(target.values())[:,0], np.zeros(len(target)), label='target')
# plt.legend()
# plt.show()
