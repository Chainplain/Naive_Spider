# Written by Chainplain Jul 15 2024

import numpy as np
from mlp_model import MLP

def spider_forward_T(theta_s, l_s):
    # Extract the angles from the input list
    theta1, theta2, theta3 = theta_s
    
    # Compute the cosine and sine of the angles
    c1 = np.cos(theta1)
    s1 = np.sin(theta1)
    c2 = np.cos(theta2)
    s2 = np.sin(theta2)
    c23 = np.cos(theta2 + theta3)
    s23 = np.sin(theta2 + theta3)
    
    # Extract the lengths from the input list
    l1, l2, l3 = l_s
    
    # Compute the elements of the transformation matrix T30
    T30 = np.array([
        [c1*c23, -c1*s23, -s1, l1*c1 + l2*c1*c2 + l3*c1*c23],
        [s1*c23, -s1*s23, c1, l1*s1 + l2*s1*c2 + l3*s1*c23],
        [-s23, -c23, 0, -l2*s2 - l3*s23],
        [0, 0, 0, 1]
    ])
    
    return T30


if __name__ == '__main__':
    theta_list = [0, 1, 0]  # Angles in radians
    l_list = [1.0, 1.0, 1.0]  # Lengths

    # theta bounds
    theta1_bound = [-np.pi/3, np.pi/3]
    theta2_bound = [-np.pi/4, np.pi/4]
    theta3_bound = [0, np.pi/2]

    # Parameters
    input_size = 3
    hidden_sizes = [100] * 4  # Example hidden layer sizes
    output_size = 3

    mode_structure = {'input_size': input_size,
                    'hidden_sizes': hidden_sizes,
                    'output_size': output_size}
    np.save('inverse_kinematics_mode_structure.npy', mode_structure)


    # Load the parameters
    mode_structure = np.load('inverse_kinematics_mode_structure.npy', allow_pickle=True).item()
    input_size = mode_structure['input_size']
    hidden_sizes = mode_structure['hidden_sizes']
    output_size = mode_structure['output_size']

    print(f'Loaded input size: {input_size}')
    print(f'Loaded hidden sizes: {hidden_sizes}')
    print(f'Loaded output size: {output_size}')

    # Sample 1000 sets of theta values uniformly from the bounds
    sample_num = 10000
    theta_samples = np.random.uniform(theta1_bound[0], theta1_bound[1], (sample_num, 1))
    theta2_samples = np.random.uniform(theta2_bound[0], theta2_bound[1], (sample_num, 1))
    theta3_samples = np.random.uniform(theta3_bound[0], theta3_bound[1], (sample_num, 1))
    theta_samples = np.concatenate((theta_samples, theta2_samples, theta3_samples), axis=1)


    # Sample inputs for testing
    theta_list = theta_samples
    PosXYZ_list = []
    for i in range(theta_samples.shape[0]):
        T30 = spider_forward_T(theta_list[i], l_list)
        PosX= T30[0,3]
        PosY= T30[1,3]
        PosZ= T30[2,3]
        PosXYZ_list. append([PosX, PosY, PosZ])
        
    PosXYZ_array = np.array(PosXYZ_list)
    # print('PosXYZ_array:\n', PosXYZ_array)

    import torch
    theta_samples_tensor = torch.from_numpy(theta_samples).float()
    PosXYZ_tensor = torch.from_numpy(PosXYZ_array).float()



    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp_model   = MLP(input_size, hidden_sizes, output_size).to(device)

    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    # Convert to TensorDataset
    train_dataset = TensorDataset( PosXYZ_tensor, theta_samples_tensor)
    # Inverse Kinematics: PosXYZ_tensor -> theta_samples_tensor
    # Forward Kinematics: theta_samples_tensor -> PosXYZ_tensor

    # Create DataLoader
    batch_size = 128
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Step 3: Define Loss and Optimizer
    import torch.nn as nn
    learning_rate = 2e-3
    criterion = nn.MSELoss()
    optimizer = optim.Adam(mlp_model.parameters(), lr = learning_rate)

    # Step 4: Train the Model
    num_epochs = 200

    for epoch in range(num_epochs):
        loss_all = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = mlp_model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_all = loss_all + loss.item()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_all * batch_size / sample_num:.8f}')
            
            
    # Save the learned model
    torch.save(mlp_model.state_dict(), 'mlp_model_inverse_kinematics.pth')

    # Load the learned model
    loaded_model = MLP(input_size, hidden_sizes, output_size).to(device)
    loaded_model.load_state_dict(torch.load('mlp_model_inverse_kinematics.pth', map_location=device))
    loaded_model.eval()


