import pickle
import numpy as np
import os

def inspect_data():
    base_path = "/Users/gaoyucen/Library/Mobile Documents/com~apple~CloudDocs/2. 论文/期刊1. ICDM 24转期刊/MetaSTC-J/data/traffic_flow/1/20230306"
    file_path = os.path.join(base_path, "part-00000.pkl")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Data type: {type(data)}")
        if isinstance(data, np.ndarray):
            print(f"Data shape: {data.shape}")
            print(f"Data sample (first 5 items):\n{data.flatten()[:5]}")
        elif isinstance(data, list):
            print(f"Data length: {len(data)}")
            print(f"First item type: {type(data[0])}")
            if len(data) > 0 and isinstance(data[0], dict):
                print("First item keys and value types/shapes:")
                for k, v in data[0].items():
                    if isinstance(v, np.ndarray):
                        print(f"  {k}: {type(v)} - shape: {v.shape}")
                    elif isinstance(v, list):
                        print(f"  {k}: {type(v)} - len: {len(v)}")
                    else:
                        print(f"  {k}: {type(v)} - value: {v}")

        elif isinstance(data, dict):
            print(f"Keys: {data.keys()}")
        else:
            print(f"Data content: {data}")
            
    except Exception as e:
        print(f"Error loading pickle: {e}")

if __name__ == "__main__":
    inspect_data()
