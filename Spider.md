# MLP spider inverse kinematic model

To date (Jul 19 2024), there are five relevated files related to our application.
- [inverseKinematics.py](https://github.com/Chainplain/Naive_Spider/blob/master/inverseKinematics.py)
- [mlp_model.py](https://github.com/Chainplain/Naive_Spider/blob/master/mlp_model.py)
- [load_inverseKinematics_mode.py](https://github.com/Chainplain/Naive_Spider/blob/master/load_inverseKinematics_mode.py)
- mlp_model_inverse_kinematics.pth
- inverse_kinematics_mode_structure.npy

------------------------------------

## inverseKinematics.py
This is the training file.
The basic logic is we randomly sample between:
```python:
    theta1_bound = [-np.pi/3, np.pi/3]
    theta2_bound = [-np.pi/4, np.pi/4]
    theta3_bound = [0, np.pi/2]
```

The samples are like
```python
    # Sample 1000 sets of theta values uniformly from the bounds
    sample_num = 10000
    theta_samples = np.random.uniform(theta1_bound[0], theta1_bound[1], (sample_num, 1))
    theta2_samples = np.random.uniform(theta2_bound[0], theta2_bound[1], (sample_num, 1))
    theta3_samples = np.random.uniform(theta3_bound[0], theta3_bound[1], (sample_num, 1))
    theta_samples = np.concatenate((theta_samples, theta2_samples, theta3_samples), axis=1)
```
and we feed these angles into the forward_kinematic

```python
def spider_forward_T(theta_s, l_s):
...
return T30  # See complete code in file
```
After define the MLP model , we train the model batchwise:
```python
 batch_size = 128
 train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
```

The model parameters (weights)  are saved into  **mlp_model_inverse_kinematics.pth**.
The model structure (neuron arragement)  is saved into **inverse_kinematics_mode_structure.npy** .
```python
   # Save the learned model
    torch.save(mlp_model.state_dict(), 'mlp_model_inverse_kinematics.pth')

    # Load the learned model
    loaded_model = MLP(input_size, hidden_sizes, output_size).to(device)
    loaded_model.load_state_dict(torch.load('mlp_model_inverse_kinematics.pth', map_location=device))
    loaded_model.eval()

```

## load_inverseKinematics_mode.py
In order to see how to use the trained model, we provide this scipt.
First we load the model (MLP) back, and set the mode into evaluate mode:
``` python
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
```
Then we show how the comparisons between actual side and the predicted side are conducted.

**Manual defiend angle curve(Actual Angle)** =Forward=> **Actual Pos** =Inverse(MLP)=> **Predicted Angle** =Forward=> **Predicted Pos**
- Forward: ```spider_forward_T```
- Inverse: ```mlp_model```

Then we compare the angles:
![Figure_1](https://github.com/user-attachments/assets/82dfc9e2-5220-474f-a086-c228d51389e7)

and the poss:
![Figure_2](https://github.com/user-attachments/assets/988165f5-080b-4336-8e59-b5d886a751fb)

 
 
