import streamlit as st
import random
import pandas as pd
from knnutils import parse_data, create_folds, create_train_test, build_knn, extract_minmax

nfeat = 8 # Number of features
col_names = ["cement","slag","ash","water","superplasticizer","coarse aggregate","fine aggregate","age"]

@st.cache(allow_output_mutation=True)
def load_data(fname):
    return parse_data(fname)

@st.cache(allow_output_mutation=True)
def load_minmax(data):
    return extract_minmax(data)

st.title("A simple implementation of a concrete compressive strength estimator")

info_txt = """
This app approximates concrete compressive strength (MPa) from a database of concrete characteristics. The $k$ Nearest Neighbor (kNN) algorithm is used to estimate the value.
A random point has been pre-selected from the data set. All inputs must be specified in the side bar.

The data set is comprised of 1,030 observations and each observation has 8 measurements. The data dictionary for this data set tells us the definitions of the individual variables (columns/indices):

| Index | Variable | Definition |
|-------|----------|------------|
| 0     | cement   | kg in a cubic meter mixture |
| 1     | slag     | kg in a cubic meter mixture |
| 2     | ash      | kg in a cubic meter mixture |
| 3     | water    | kg in a cubic meter mixture |
| 4     | superplasticizer | kg in a cubic meter mixture |
| 5     | coarse aggregate | kg in a cubic meter mixture |
| 6     | fine aggregate | kg in a cubic meter mixture |
| 7     | age | days |
| 8     | concrete compressive strength | MPa |

The target ("y") variable is at Index 8, concrete compressive strength in Mega Pascals.
"""
info_expander = st.expander("Information")
with info_expander:
    info_expander.markdown(info_txt)

data = load_data("concrete_compressive_strength.csv")
feat_minmax = load_minmax(data)

button_rand_inputs = st.sidebar.button("Pick Random Inputs")

# Put all model parameters in the sidebar
params = st.sidebar.form("Options")
params.header("Parameters")

if 'nneighbors' not in st.session_state:
    st.session_state['nneighbors'] = 3
nneighbors = params.number_input("Number of neighbors", min_value=1, max_value=len(data)-1, value=3, step=1)

# Randomly pick measurements
if ('x' not in st.session_state) or button_rand_inputs:
    x0 = data[random.randint(0,len(data)-1)]
    st.session_state['x'] = x0

# Store variables across sessions
#params_col1, params_col2 = params.columns(2)
x = [0.0]*nfeat
for i in range(nfeat):
    x[i] = params.number_input(f"feature {i}: {col_names[i]} [{feat_minmax[i][0]:.1f}, {feat_minmax[i][1]:.1f}]",
        min_value=feat_minmax[i][0], max_value=feat_minmax[i][1], value=st.session_state['x'][i])

# Store inputs and execute model
if params.form_submit_button("Predict target value"):
    st.session_state['x'] = x
    st.session_state['nneighbors'] = nneighbors

    # Predict target value
    knn = build_knn(data,idx_label=8)
    result = knn(st.session_state.nneighbors, st.session_state['x'] + [0.0])
    st.write(f"Predicted value: {result['y_pred']:.3f} +/- {result['y_err']:.3f} MPa")

    # Arrange data for display
    nn_data = [[result['y_pred'],0.0] + x]
    for p in result['nearest_neighbors']:
        obs = [p['y'],p['d']] + p['x']
        nn_data.append(obs)
    df = pd.DataFrame(nn_data, columns=['concrete compressive strength','distance metric']+col_names)
    st.write("Query point:")
    st.write(df.iloc[0,:])
    st.write("Nearest measurements to the query point:")
    st.write(df.iloc[1:,:])
