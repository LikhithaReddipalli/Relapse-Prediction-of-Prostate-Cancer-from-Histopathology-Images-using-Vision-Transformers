import pandas as pd

# Replace 'file_path.pickle' with the path to your Pickle file
file_path = '/data/PANDA/code/survival_prediction/survival_prediction/.cache/.patch_size=256.fg_mask_threshold=0.1.label_mask_threshold=0.9.pkl'

file_path1 = '/data/PANDA/code/survival_prediction/survival_prediction/.cache/.patch_size=1024.fg_mask_threshold=0.1.label_mask_threshold=0.9.pkl'
# Load the Pickle file into a DataFrame
df = pd.read_pickle(file_path1)

# Now df is a DataFrame containing the data from the Pickle file

print(df)
print(df['label'].value_counts())