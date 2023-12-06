import cv2
import numpy as np
import pandas as pd
import os

import sqlite3
from sqlite3 import Error

import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def create_connection(path):
    try:
        connection = sqlite3.connect(path)
        print(f"Connection to SQLite DB at '{path}' successful")
    except Error as e:
        print(f"The error '{e}' occurred")
        raise e

    return connection


def get_db_table(table_name, db_path, keep_cols=None, index_col=None):

    db_con = create_connection(db_path)
    result = pd.read_sql_query(f"SELECT * FROM '{table_name}'", db_con)

    if keep_cols is not None:
        result = result[keep_cols]

    if index_col is not None:
        result = result.set_index(index_col)

    return result


# Load the npz file
with np.load('/data/PANDA/code/survival_prediction/Experiments/2023_09_14_11_32_36/results_test.npz') as data:
    
    # Get the list of array names in the npz file
    array_names = data.files
    
    # Create a dictionary to store the arrays
    arrays_dict = {name: data[name] for name in array_names}
    
# Convert the dictionary of arrays to a pandas DataFrame
df = pd.DataFrame(arrays_dict)
print(df)


with np.load('/data/PANDA/code/survival_prediction/Experiments/2023_09_28_12_15_50/results_test.npz') as data:
    
    # Get the list of array names in the npz file
    array_names = data.files
    
    # Create a dictionary to store the arrays
    arrays_dict = {name: data[name] for name in array_names}
    
# Convert the dictionary of arrays to a pandas DataFrame
df_h = pd.DataFrame(arrays_dict)
print(df_h)








df1  = get_db_table('uke.experiments.clas_60_months','/data/PANDA/code/survival_prediction/db_lr.sqlite')
 
df['filename_last_part'] = df['filenames'].apply(lambda x: os.path.basename(x))

df_h['filename_last_part'] = df_h['filenames'].apply(lambda x: os.path.basename(x))

df1['filename_last_part'] = df1['filepath'].str.split('/').str[-1].str.replace('_true.tif', '.tif')
df2 = df1[df1['split']=='test']
#print(df2)


df['filename_suffix'] = df['filename_last_part'].str.replace('.png', '')
df_h['filename_suffix'] = df_h['filename_last_part'].str.replace('.png', '')
df2['filename_suffix'] = df2['filename_last_part'].str.replace('.tif', '')

result_df = df.merge(df2[['filename_suffix', 'isup','has_event_before_60_months']], on='filename_suffix', how='left')
print(result_df.dropna())

result_df_h =  df_h.merge(df2[['filename_suffix', 'isup']], on='filename_suffix', how='left')

unique_isup = sorted(result_df['isup'].unique())


data = {
    'grading': [],
    'label_count': [],
    'label_count_re': [],
    'label_count_nore': [],
    'prediction_count': [],
    'prediction_count_re': [],
    'prediction_count_nore': [],
    'prediction_count_h' : []
}

# Get the unique ISUP grades in your dataframe
unique_grades = result_df['isup'].unique()

# Loop through each unique grade and calculate the necessary values
for grade in unique_grades:
    data['grading'].append(grade)
    
    # Get the total count of labels for the current grade
    label_count = len(result_df[result_df['isup'] == grade])
    label_count_re = len(result_df[(result_df['isup'] == grade) & (result_df['has_event_before_60_months'] == 1.0)])
    label_count_nore = len(result_df[(result_df['isup'] == grade) & (result_df['has_event_before_60_months'] == 0.0)])
    
    data['label_count'].append(label_count)
    data['label_count_re'].append(label_count_re)
    data['label_count_nore'].append(label_count_nore)
    
    # Get the count of correct predictions for the current grade
    correct_preds = len(result_df[(result_df['isup'] == grade) & (result_df['labels'] == result_df['predictions'])])
    data['prediction_count'].append(correct_preds)
    
    correct_preds_re = len(result_df[(result_df['isup'] == grade) & (result_df['has_event_before_60_months'] == 1.0) & (result_df['predictions'] == 1.0)])
    data['prediction_count_re'].append(correct_preds_re)   
    
    correct_preds_nore = len(result_df[(result_df['isup'] == grade) & (result_df['has_event_before_60_months'] == 0.0) & (result_df['predictions'] == 0.0)])
    data['prediction_count_nore'].append(correct_preds_nore) 
    
    
    
    
    correct_preds_h = len(result_df_h[(result_df_h['isup'] == grade) & (result_df_h['labels'] == result_df_h['predictions'])])
    data['prediction_count_h'].append(correct_preds_h)




result_df1 = pd.DataFrame(data)
result_df1 = result_df1.sort_values(by='grading')
result_df1['percentage'] = result_df1['prediction_count']/result_df1['label_count'] * 100


### for initial fig format
'''

# #plotting
fig, ax = plt.subplots(figsize=(10, 6))

width = 0.25  

# Create an array with the position of each bar along the x-axis
indices = list(range(len(result_df1['grading'])))

# Adjusting indices for spacing
spacing = 0.3  # Adjust the spacing value to your preference
adjusted_indices = [i * (width * 3 + spacing) for i in indices]

# For bar height, use the values of Y1 and Y2 columns
rects1 = ax.bar([i - width for i in adjusted_indices], result_df1['label_count'], width, label='Ground truth', color='darkseagreen')
rects2 = ax.bar(adjusted_indices, result_df1['prediction_count'], width, label='ViT + MIL', color='darksalmon')
rects3 = ax.bar([i + width for i in adjusted_indices], result_df1['prediction_count_h'], width, label='HViT', color='thistle')


# # Label the chart
ax.set_xlabel('ISUP Grades')
ax.set_ylabel('TMA spot count')
#ax.set_title('')
ax.set_xticks([i + width/2 for i in indices])
ax.set_xticklabels(result_df1['grading'])
ax.legend()

def display_values(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
        xy=(rect.get_x() + rect.get_width() / 2, height),
        xytext=(0, 3),  # 3 points vertical offset
        textcoords="offset points",
        ha='center', va='bottom')

display_values(rects1)
display_values(rects2)
display_values(rects3)

# # Show the chart
plt.tight_layout()
plt.savefig("fig.png")

'''
#percentage plot
'''plt.bar(result_df1['grading'], round(result_df1['percentage']))

for i, val in enumerate(round(result_df1['percentage'])):
    plt.text(i, val + 1, str(val), ha='center')

plt.xlabel('Grading')
plt.ylabel('Percentage')
plt.title('Grading')

plt.show()
plt.savefig("fig1.png")'''




### for gt, pred on top of each other
result_df2 = pd.DataFrame(data)
result_df2['grading'] = result_df2['grading'].astype(int)

# Sort the DataFrame based on the 'grading' column
result_df2 = result_df2.sort_values(by='grading')

# Reset the index of result_df2 after sorting
#result_df2 = result_df2.reset_index(drop=True)
#result_df2 = result_df2.sort_values(by='grading')

fig, ax = plt.subplots(figsize=(10, 6))

# Set width of the bars
width = 0.35

# Create an array with the position of each bar along the x-axis
indices = list(range(len(result_df2['grading'])))

# Plotting the bars
rects1 = ax.bar(indices, result_df2['label_count_re'], width, label='label_count_re', color='tomato', alpha=0.5)
rects2 = ax.bar(indices, result_df2['prediction_count_re'], width, label='correct_preds_re', color='red', alpha=0.5)

rects3 = ax.bar([i + width for i in indices], result_df2['label_count_nore'], width, label='label_count_nore', color='green', alpha=0.5)
rects4 = ax.bar([i + width for i in indices], result_df2['prediction_count_nore'], width, label='correct_preds_nore', color='darkgreen', alpha=0.5)



# Label the chart
ax.set_xlabel('Grades')
ax.set_ylabel('Counts')
ax.set_title('Bar chart of Label Counts and Correct Predictions by Grade')
ax.set_xticks([i + width/2 for i in indices])
ax.set_xticklabels(result_df2['grading'])
ax.legend()

def display_values(rects, offset=None):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
        xy=(rect.get_x() + rect.get_width() / 2, height + (0 if offset is None else offset.values[rects.index(rect)])),
        xytext=(0, 3),  # 3 points vertical offset
        textcoords="offset points",
        ha='center', va='bottom')


# Display values for all sets of bars
#display_values(rects1)
#display_values(rects2, result_df2['label_count_re'])  # offset by the height of the first bar
#display_values(rects3)
#display_values(rects4, result_df2['label_count_nore'])  # offset by the height of the first bar

# Show the chart
plt.tight_layout()
plt.savefig("fig_gt_pre_ol_1.png")
plt.show()









