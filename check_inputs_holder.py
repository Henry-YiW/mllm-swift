# check_inputs_holder.py
import pickle
import torch
import numpy as np
import sys
from datetime import datetime

class TeeStream:
    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2

    def write(self, data):
        self.stream1.write(data)
        self.stream2.write(data)
        self.stream1.flush()
        self.stream2.flush()

    def flush(self):
        self.stream1.flush()
        self.stream2.flush()

def load_pickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def tensor_to_list(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().tolist()
    return tensor

def compare_values(v1, v2, name=""):
    """Compare two values of any type and print differences."""
    print(f"\nComparing '{name}':")
    print(f"  Type in Transformers: {type(v1)}")
    print(f"  Type in Swift: {type(v2)}")
    
    # If both are tensors or one is tensor
    if isinstance(v1, torch.Tensor) or isinstance(v2, torch.Tensor):
        # Convert both to lists for comparison
        v1_list = tensor_to_list(v1)
        v2_list = tensor_to_list(v2)
        
        if v1_list != v2_list:
            print("  Values differ:")
            print(f"    Transformers: {v1_list}")
            print(f"    Swift: {v2_list}")
            # If both can be converted to numpy arrays, show detailed comparison
            try:
                v1_arr = np.array(v1_list)
                v2_arr = np.array(v2_list)
                print(f"    Shapes: {v1_arr.shape} vs {v2_arr.shape}")
                if v1_arr.shape == v2_arr.shape:
                    diff = np.abs(v1_arr - v2_arr)
                    print(f"    Max difference: {np.max(diff)}")
                    print(f"    Mean difference: {np.mean(diff)}")
                    print(f"    Number of different elements: {np.sum(diff != 0)}")
                    if np.sum(diff != 0) > 0:
                        diff_indices = np.where(diff != 0)
                        print("    First few differences (idx: transformers, swift):")
                        for idx in zip(*diff_indices)[:5]:
                            print(f"      {idx}: {v1_arr[idx]}, {v2_arr[idx]}")
            except Exception as e:
                print(f"    Could not perform numerical comparison: {e}")
        else:
            print("  Values match exactly")
    
    # If both are lists or tuples
    elif isinstance(v1, (list, tuple)) and isinstance(v2, (list, tuple)):
        if v1 != v2:
            print("  Values differ:")
            print(f"    Transformers: {v1}")
            print(f"    Swift: {v2}")
            print(f"    Lengths: {len(v1)} vs {len(v2)}")
            # If lengths match, show first few differences
            if len(v1) == len(v2):
                print("    First few differences (idx: transformers, swift):")
                diffs = [(i, x, y) for i, (x, y) in enumerate(zip(v1, v2)) if x != y][:5]
                for i, x, y in diffs:
                    print(f"      {i}: {x}, {y}")
        else:
            print("  Values match exactly")
    
    # If both are dictionaries
    elif isinstance(v1, dict) and isinstance(v2, dict):
        all_keys = set(v1.keys()) | set(v2.keys())
        if v1 != v2:
            print("  Dictionaries differ:")
            print(f"    Keys in Transformers: {list(v1.keys())}")
            print(f"    Keys in Swift: {list(v2.keys())}")
            for k in all_keys:
                if k not in v1:
                    print(f"    Key '{k}' only in Swift")
                elif k not in v2:
                    print(f"    Key '{k}' only in Transformers")
                elif v1[k] != v2[k]:
                    print(f"    Values differ for key '{k}':")
                    print(f"      Transformers: {v1[k]}")
                    print(f"      Swift: {v2[k]}")
        else:
            print("  Dictionaries match exactly")
    
    # For all other types
    else:
        if v1 != v2:
            print("  Values differ:")
            print(f"    Transformers: {v1}")
            print(f"    Swift: {v2}")
        else:
            print("  Values match exactly")

# Create a timestamp for the output file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f'comparison_report_{timestamp}.txt'

# Redirect output to both console and file
with open(output_file, 'w') as f:
    original_stdout = sys.stdout
    sys.stdout = TeeStream(original_stdout, f)

    try:
        # Load both pickle files
        # inputs_holder_transformers = load_pickle('hidden_states_list_transformers_llama.pkl')
        # inputs_holder_swift = load_pickle('hidden_states_list_swift_llama.pkl')
        inputs_holder_transformers = load_pickle('inputs_holder_transformers_llama.pkl')
        inputs_holder_swift = load_pickle('inputs_holder_swift_llama.pkl')

        print(f"\nComparison Report Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # Print type information
        print("\nType Information:")
        print("Transformers inputs_holder type:", type(inputs_holder_transformers))
        print("Swift inputs_holder type:", type(inputs_holder_swift))

        # If transformers version is a dictionary, compare all entries
        if isinstance(inputs_holder_transformers, dict):
            print("\nTransformers Keys:", list(inputs_holder_transformers.keys()))
            
            if isinstance(inputs_holder_swift, dict):
                print("Swift Keys:", list(inputs_holder_swift.keys()))
                
                # Compare all keys present in either dictionary
                all_keys = set(inputs_holder_transformers.keys()) | set(inputs_holder_swift.keys())
                
                print("\nComparing all entries:")
                for key in sorted(all_keys):
                    if key not in inputs_holder_transformers:
                        print(f"\nKey '{key}' only present in Swift inputs_holder")
                        value = inputs_holder_swift[key]
                        print(f"Value type: {type(value)}")
                        print(f"Value: {value}")
                    elif key not in inputs_holder_swift:
                        print(f"\nKey '{key}' only present in Transformers inputs_holder")
                        value = inputs_holder_transformers[key]
                        print(f"Value type: {type(value)}")
                        print(f"Value: {value}")
                    else:
                        compare_values(inputs_holder_transformers[key], 
                                     inputs_holder_swift[key], 
                                     key)
            
            elif isinstance(inputs_holder_swift, torch.Tensor):
                print("\nNote: Swift inputs_holder is a tensor while Transformers is a dictionary")
                print("Swift tensor shape:", inputs_holder_swift.shape)
                print("Swift tensor values:", tensor_to_list(inputs_holder_swift))
            
        # If both are tensors, compare them directly
        elif isinstance(inputs_holder_transformers, torch.Tensor) and isinstance(inputs_holder_swift, torch.Tensor):
            print("\nComparing tensors:")
            compare_values(inputs_holder_transformers, inputs_holder_swift, "Full tensor")
        
        else:
            print("\nInputs holders have different types:")
            print(f"Transformers: {type(inputs_holder_transformers)}")
            print(f"Swift: {type(inputs_holder_swift)}")
            print("\nTransformers value:", inputs_holder_transformers)
            print("Swift value:", inputs_holder_swift)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure both pickle files exist in the current directory:")
        print("  - inputs_holder_transformers_llama.pkl")
        print("  - inputs_holder_swift_llama.pkl")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Restore original stdout
        sys.stdout = original_stdout
        print(f"\nComparison report has been saved to: {output_file}")