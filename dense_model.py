import torch
import pandas as pd
from torch.utils.data import DataLoader
from data_preprocessing import IMUDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model_architecture import NNDataset, NNModel
import matplotlib.pyplot as plt
from scipy import signal
import signal_utils


def training_loop():
    # Model params
    batch_size = 16
    num_features = X.shape[0]
    
    # Training params
    epochs = 100
    learning_rate = 0.0001
    weight_decay = 1e-4

    # Instantiate model
    model = NNModel(num_features).to(device)
    model = model.float()

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()

    # Load train and val datasets
    train_generator = DataLoader(NNDataset(X_train, y_train), batch_size=batch_size)
    val_generator = DataLoader(NNDataset(X_val, y_val), batch_size=batch_size)

    # Track loss history with each epoch as one datapoint
    train_loss_history = []
    val_loss_history = []

    # Loop over epochs
    for epoch in range(epochs):
        # Train batch
        batch_train_loss_history = []
        
        

        for (batch_train_X, batch_train_y) in train_generator:
            print('shapex', batch_train_X.shape, 'shapey', batch_train_y.shape)
            # Zero out gradients every mini-batch
            optimizer.zero_grad()

            # Turn gradients on
            model.train()

            # Make predictions
            train_predictions = model(batch_train_X.float())
            print('preds', train_predictions)
            print('true', batch_train_y.float())
            
            # Calculate train loss
            train_loss = criterion(train_predictions, batch_train_y)
            batch_train_loss_history.append(float(train_loss.item()))

            # Back prop and optimizer step every batch
            train_loss.backward()
            optimizer.step()

        # Append average loss across batches
        # train_loss_history.append(sum(batch_train_loss_history) / len(batch_train_loss_history))
        train_loss_history.append(batch_train_loss_history[-1])

        # Validation batch NOTE: after back prop on training
        batch_val_loss_history = []
        
        for (batch_val_X, batch_val_y) in val_generator:
            # Evaluate model without updating gradients
            model.eval()
            with torch.no_grad():
                val_predictions = model(batch_val_X)
            model.train()

            # Calculate validation loss
            val_loss = criterion(val_predictions, batch_val_y)
            batch_val_loss_history.append(float(val_loss.item()))
        # val_loss_history.append(sum(batch_val_loss_history) / len(batch_val_loss_history))
        val_loss_history.append(batch_val_loss_history[-1])

    # torch.save(model, 'trained_model.pt')

    # Plot loss
    plt.figure()
    plt.title('MSE Loss')
    plt.plot(range(epochs), train_loss_history, label='train loss')
    plt.plot(range(epochs), val_loss_history, label='val loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':

    # Set device to enable GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('DEVICE:', device)

    # Read in data
    # path = 'merged_data.xlsx'
    # data = pd.read_excel(path)
    # data = data.iloc[:int(len(data)/20)]
    
    # path = 'ml_data_v2/Keller_Emily_Walking4.xlsx'
    # data = pd.read_excel(path, header=None)
    path = 'merged_data.xlsx'
    data = pd.read_excel(path)
    

    # Format dataset
    # imu = IMUDataset(data)
    # header = imu.grab_imu_header()
    # imu.header_from_dict(header)
    # data = imu.df.copy()

    # Get features and labels and convert to arrays
    feats_to_use = ['Grav1_0','Grav1_1','Grav1_2','Gyro1_0','Gyro1_1','Gyro1_2','Acc1_0','Acc1_1', 'Acc1_2']
    label = 'Angle_0'
    X = data[feats_to_use].to_numpy()
    y = data[label].to_numpy()

    # Split data into train, test, val
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, shuffle=False)

    # Fit scaler to training data
    scaler = StandardScaler()
    scaled_X_train = scaler.fit_transform(X_train)

    # Apply scaler to val and test data
    scaled_X_val, scaled_X_test = list(map(scaler.transform, [X_val, X_test]))
    
    plt.plot(range(len(scaled_X_train)), scaled_X_train)
    plt.show()

    # Apply lowpass filter to training, validation, and test sets
    lp_filter = signal.butter(N=1, Wn=0.2, btype='low', output='sos')
    filtered_X_train, filtered_X_val, filtered_X_test = list(map(lambda x: signal_utils.filter(lp_filter, x), 
                                                                 [scaled_X_train, scaled_X_val, scaled_X_test]))

    # plot data --> low pass filter
    print(X_train[:, 1])
    plt.plot(range(len(scaled_X_train)), filtered_X_train)
    plt.show()

    # first order, high pass 0.5 Hz cut-off
    # scipy.signal.butter
    # outliers

    # Cast to tensors
    X_train, X_val, X_test = [torch.tensor(_X, requires_grad=False, dtype=torch.float).to(device) for _X in
                              [filtered_X_train, filtered_X_val, filtered_X_test]]
    y_train, y_val, y_test = [torch.tensor(_y, requires_grad=False, dtype=torch.float).to(device) for _y in
                              [y_train, y_val, y_test]]

    training_loop()
