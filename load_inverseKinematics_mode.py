import numpy as np

# Load the parameters
mode_structure = np.load('inverse_kinematics_mode_structure.npy', allow_pickle=True).item()
input_size = mode_structure['input_size']
hidden_sizes = mode_structure['hidden_sizes']
output_size = mode_structure['output_size']

print(f'Loaded input size: {input_size}')
print(f'Loaded hidden sizes: {hidden_sizes}')
print(f'Loaded output size: {output_size}')

from mlp_model import MLP
import torch


device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlp_model   = MLP(input_size, hidden_sizes, output_size).to(device)

# Load the learned model
mlp_model = MLP(input_size, hidden_sizes, output_size).to(device)
mlp_model.load_state_dict(torch.load('mlp_model_inverse_kinematics.pth', map_location=device))
mlp_model.eval()


# Test the model, manuallt given angles
theta1_series = np.linspace(-np.pi/3, np.pi/3, 100)
theta2_series = np.linspace(-np.pi/4, np.pi/4, 100)
theta3_series = np.linspace(np.pi/4, np.pi/2, 100)

from inverseKinematics import spider_forward_T

PosXYZ_list = []
for i in range(100):
    T30 = spider_forward_T([theta1_series[i], theta2_series[i], theta3_series[i]], [1.0, 1.0, 1.0])
    PosX= T30[0,3]
    PosY= T30[1,3]
    PosZ= T30[2,3]
    PosXYZ_list. append([PosX, PosY, PosZ])
            
PosXYZ_array = np.array(PosXYZ_list)
PosXYZ_tensor = torch.from_numpy(PosXYZ_array).float().to(device)


print('PosXYZ_array:', PosXYZ_array[0,:])

mlp_model_predictions = mlp_model(PosXYZ_tensor)
mlp_model_predictions = mlp_model_predictions.detach().cpu().numpy()

theta1_pre = mlp_model_predictions[:,0]
theta2_pre = mlp_model_predictions[:,1]
theta3_pre = mlp_model_predictions[:,2]


PosXYZ_pre_list = []
for i in range(100):
    T30 = spider_forward_T([theta1_pre[i], theta2_pre[i], theta3_pre[i]], [1.0, 1.0, 1.0])
    PosX= T30[0,3]
    PosY= T30[1,3]
    PosZ= T30[2,3]
    PosXYZ_pre_list.append([PosX, PosY, PosZ])
            
PosXYZ_pre_array = np.array(PosXYZ_pre_list)

import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle('Actual vs Predicted Joint Angles: the actual angles are manually selected, \n'+ \
             'and then fed into the forward kinematics model, such that we get the end point position, \n' +\
            'the position is subsequently fed into the mlp inverse kinematics model, \n' +\
                'such taht we get the predicted joint angles.', fontsize=10)

axs[0].plot(theta1_series)
axs[0].plot(theta1_pre)
axs[0].set_title('theta1 vs Predicted value')
axs[0].legend(['Actual', 'Predicted'], loc='upper left')

axs[1].plot(theta2_series)
axs[1].plot(theta2_pre)
axs[1].set_title('theta2 vs Predicted value')
axs[1].legend(['Actual', 'Predicted'], loc='upper left')

axs[2].plot(theta3_series)
axs[2].plot(theta3_pre)
axs[2].set_title('theta3 vs Predicted value')
axs[2].legend(['Actual', 'Predicted'], loc='upper left')

plt.tight_layout()

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle('Actual vs Predicted End Point Position', fontsize=10)

axs[0].plot(PosXYZ_array[:,0])
axs[0].plot(PosXYZ_pre_array[:,0])
axs[0].set_title('PosX vs Predicted value')
axs[0].legend(['Actual', 'Predicted'], loc='upper left')

axs[1].plot(PosXYZ_array[:,1])
axs[1].plot(PosXYZ_pre_array[:,1])
axs[1].set_title('PosY vs Predicted value')
axs[1].legend(['Actual', 'Predicted'], loc='upper left')

axs[2].plot(PosXYZ_array[:,2])
axs[2].plot(PosXYZ_pre_array[:,2])
axs[2].set_title('PosZ vs Predicted value')
axs[2].legend(['Actual', 'Predicted'], loc='upper left')

plt.tight_layout()


plt.show()


            

