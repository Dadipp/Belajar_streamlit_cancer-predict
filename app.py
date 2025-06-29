import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np


def get_clean_data():
    data = pd.read_csv("data/data.csv")
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data


def add_sidebar():
    st.sidebar.header("Pengukuran Inti Sel")

    data = get_clean_data()

    slider_labels = [
        ("Radius (rata-rata)", "radius_mean"),
        ("Tekstur (rata-rata)", "texture_mean"),
        ("Perimeter (rata-rata)", "perimeter_mean"),
        ("Luas (rata-rata)", "area_mean"),
        ("Kelembutan (rata-rata)", "smoothness_mean"),
        ("Kekompakan (rata-rata)", "compactness_mean"),
        ("Kekonkavan (rata-rata)", "concavity_mean"),
        ("Titik Cekung (rata-rata)", "concave points_mean"),
        ("Simetri (rata-rata)", "symmetry_mean"),
        ("Dimensi Fraktal (rata-rata)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Tekstur (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Luas (se)", "area_se"),
        ("Kelembutan (se)", "smoothness_se"),
        ("Kekompakan (se)", "compactness_se"),
        ("Kekonkavan (se)", "concavity_se"),
        ("Titik Cekung (se)", "concave points_se"),
        ("Simetri (se)", "symmetry_se"),
        ("Dimensi Fraktal (se)", "fractal_dimension_se"),
        ("Radius (terburuk)", "radius_worst"),
        ("Tekstur (terburuk)", "texture_worst"),
        ("Perimeter (terburuk)", "perimeter_worst"),
        ("Luas (terburuk)", "area_worst"),
        ("Kelembutan (terburuk)", "smoothness_worst"),
        ("Kekompakan (terburuk)", "compactness_worst"),
        ("Kekonkavan (terburuk)", "concavity_worst"),
        ("Titik Cekung (terburuk)", "concave points_worst"),
        ("Simetri (terburuk)", "symmetry_worst"),
        ("Dimensi Fraktal (terburuk)", "fractal_dimension_worst"),
    ]

    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )

    return input_dict


def get_scaled_values(input_dict):
    data = get_clean_data()
    X = data.drop(['diagnosis'], axis=1)

    scaled_dict = {}
    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict


def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)

    categories = ['Radius', 'Tekstur', 'Perimeter', 'Luas',
                  'Kelembutan', 'Kekompakan',
                  'Kekonkavan', 'Titik Cekung',
                  'Simetri', 'Dimensi Fraktal']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Nilai Rata-rata'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
            input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
            input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
            input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Nilai Terburuk'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True
    )

    return fig


def add_predictions(input_data):
    model = pickle.load(open("models/model.pkl", "rb"))
    scaler = pickle.load(open("models/scaler.pkl", "rb"))

    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)

    st.subheader("Prediksi Kluster Sel")
    st.write("Hasil prediksi kluster sel adalah:")

    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Jinak</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'>Ganas</span>", unsafe_allow_html=True)

    st.write("Probabilitas jinak: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probabilitas ganas: ", model.predict_proba(input_array_scaled)[0][1])

    st.write("Aplikasi ini dapat membantu tenaga medis dalam melakukan diagnosis, namun tidak dapat menggantikan diagnosis dari tenaga profesional.")


def main():
    st.set_page_config(
        page_title="Prediksi Kanker Payudara",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    input_data = add_sidebar()

    with st.container():
        st.title("Prediksi Kanker Payudara")
        st.write("Hubungkan aplikasi ini dengan laboratorium sitologi Anda untuk membantu mendiagnosis kanker payudara dari sampel jaringan Anda. Aplikasi ini menggunakan model pembelajaran mesin untuk memprediksi apakah massa payudara bersifat jinak atau ganas berdasarkan pengukuran dari laboratorium. Anda juga dapat mengatur nilai-nilai pengukuran secara manual melalui slider di sidebar.")

    col1, col2 = st.columns([4, 1])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        add_predictions(input_data)


if __name__ == '__main__':
    main()
