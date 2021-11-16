import pandas as pd
import os


class IMUDataset:
    def __init__(self, df=None, path=None):
        self.df = df
        self.path = path
        # self.df.dropna(axis=1, inplace=True)
        self.header = None
    
    def grab_imu_header(self):
        new_head = {}  # Initialize dict of new header values and number of associated columns
        count = 0  # Initialize associated columns count
        first = True  # Indicates first column

        # Loop through first row and test whether value is numeric. Append to new_head if non-numeric
        for value, col in zip(self.df.iloc[0], self.df.columns):
            try:
                int(value)
                count += 1  # Increase associated columns count

                # If value is last, append latest exception
                if value == self.df.iloc[0].tolist()[-1]:
                    new_head[future_col] = count

            except:
                print(self.df.columns)
                self.df.drop(columns=[col], axis=1, inplace=True)
                if not first:  # Avoid first because no future_col stored
                    new_head[future_col] = count
                else:
                    first = False
                count = 0  # Reset count
                future_col = value  # Future == current
        
        return new_head

    def header_from_dict(self, head_dict):
        new_head = []

        for key, value in head_dict.items():
            [new_head.append('{0}_{1}'.format(key.split(':')[0], i)) for i in range(value)]
        
        self.df.columns = new_head
        
    def multi_concat(self):
        concat_df = pd.DataFrame()
        for file in os.listdir(self.path):
            df = pd.read_excel(os.path.join(self.path, file), header=None)
            df.dropna(axis=1, inplace=True)
            concat_df = pd.concat((concat_df, df), axis=1)
            
        print(concat_df)


if __name__ == '__main__':
    path = 'train_data'
    # path = 'train_data/Keller_Emily_Walking4.xlsx'
    # data = pd.read_excel(path, header=None)
    imu = IMUDataset(path=path)
    imu.multi_concat()
    
    # header = imu.grab_imu_header()
    # imu.header_from_dict(header)