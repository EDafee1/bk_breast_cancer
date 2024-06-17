import streamlit as st
import pandas as pd
import joblib as jb
from sklearn.naive_bayes import GaussianNB

st.set_page_config(page_title='Breast Cancer Prediction', layout='wide')

st.title('Breast Cancer Prediction')
st.header('Input your variable and predict with Naive Bayes Algorithm.', divider='violet')

margin_left, col_left, margin_mid, col_right, margin_right = st.columns([0.2,5,1,5,0.2])

if 'result' not in st.session_state:
    st.session_state.result = 'Fill the column and click Predict Button.'

clumpthickness = col_left.slider('Clump Thickness', 1, 5, 10)
cellsize = col_left.slider('Cell Size', 1, 5, 10)
cellshape = col_left.slider('Cell Shape', 1, 5, 10)
marginaladhesion = col_left.slider('Marginal Adhesion', 1, 5, 10)
singlecellsize = col_right.slider('Single Ephetelial Cell Size', 1, 5, 10)
barenuclei = col_right.slider('Bare Nuclei', 1, 5, 10)
blandchromatin = col_right.slider('Bland Chromatin', 1, 5, 10)
normalnucleoli = col_right.slider('Normal Nucleoli', 1, 5, 10)

def predict():
    x = {
        'Clump_thickness' : [clumpthickness],
        'Uniformity_of_cell_size' : [cellsize],
        'Uniformity_of_cell_shape' : [cellshape],
        'Marginal_adhesion' : [marginaladhesion],
        'Single_epithelial_cell_size' : [singlecellsize],
        'Bare_nuclei' : [barenuclei],
        'Bland_chromatin' : [blandchromatin],
        'Normal_nucleoli' : [normalnucleoli]
    }
    x = pd.DataFrame(x)

    try:
        model = jb.load(f'.\\NB.pkl')

        pred = model.predict(x)
        pred = pred[0]

        if pred == 2:
            st.session_state.result = 'You are save from cancer'
        elif pred == 4:
            st.session_state.result = 'YOU HAVE CANCER!'
        else:
            st.session_state.result = 'Something Wrong'

    except:
        st.session_state.result = 'Something Wrong.'

col_left.button('Predict', on_click=predict)

col_left.write(st.session_state.result)