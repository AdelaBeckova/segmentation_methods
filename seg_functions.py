# Decision tree plotting
from sklearn import tree # for text representation
import matplotlib.pyplot as plt
import pandas as pd

# BYTES TO STRING FUNCTION
def convert_bytes_to_str(x):
    return x.decode('utf-8') if isinstance(x, bytes) else x

# CHECK DISTRIBUTION OF A CATEGORICAL VARIABLE ON BOTH DATASETS
def distribution(var_name, data_mw, data_st):
    print(f"""\033[1mABSOLUTE\033[0m
    MW distribution: 
    {data_mw[var_name].value_counts().sort_index()} 
    ST distribution: 
    {data_st[var_name].value_counts().sort_index()}

    \033[1mRELATIVE\033[0m
    MW distribution: 
    {data_mw[var_name].value_counts(normalize = True).sort_index()}
    ST distribution: 
    {data_st[var_name].value_counts(normalize = True).sort_index()}""")

# VALUE COUNTS WHERE NA IS CONSIDERED A CATEGORY AS WELL
def value_counts_with_nan(series):
    # Use value_counts with dropna=False to include NaN as a category
    value_counts = series.value_counts(dropna=False)
    # Calculate the total count
    total_count = value_counts.sum()
    # Append the total count to the result
    value_counts['Total'] = total_count
    return value_counts

# DECISION TREE TEXT FORM
def tree_viz_text(model):
    text_rep = tree.export_text(model,
                                feature_names = list(model.feature_names_in_),
                                show_weights = True)
    print(text_rep)

# DECISION TREE GRAPHIC FORM
def tree_viz_plot(model, class_names = ["B","G"], w = 30, h = 20):
    fig = plt.figure(figsize=(w, h))
    _ = tree.plot_tree(model,
                    feature_names = list(model.feature_names_in_),
                    class_names = class_names,
                    filled = True)
    
# PRODUCTS COUNTS FORMAT
def format_num_cnt(value):
    if value >= 5:
        return 'More than 5'
    format_mapping = {
        0: '0',
        1: '1',
        2: '2',
        3: '3',
        4: '4'
    }
    return format_mapping.get(value, '0') 

# FEATURE IMPORTANCE
def feat_imp(model, X):
    fe_imp = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    })
    return fe_imp
