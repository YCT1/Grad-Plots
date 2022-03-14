import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
import pickle
def main():
    st.header("Simulated Data Analysis")
    
    st.subheader("Created from Numpy")
    noise = st.slider("Noise", 0.001, 10., value=2.)
    source = np.random.normal(0,0.5, (500,35)).astype(np.float32)
    target = np.random.normal(0,0.5, (500,160)).astype(np.float32)

    source_ood = np.random.normal(0+(10-noise),0.5,(500,35)).astype(np.float32)
    target_ood = np.random.normal(0+(10-noise),0.5,(500,160)).astype(np.float32)

    source_all = np.concatenate((source,source_ood))
    target_all = np.concatenate((target,target_ood))

    # Prepara the data
    data_index = np.arange(0,source.shape[0])

    normal_index = np.random.choice(data_index,400,replace=False)
    h1_index, h2_index = normal_index[0:200],normal_index[200:400]

    h1_source,h1_target = source[h1_index], target[h1_index]
    h2_source,h2_target = source[h2_index], target[h2_index]

    ood_index = np.random.choice(data_index,200,replace=False)
    h3_source, h3_target = source_ood[ood_index], target_ood[ood_index]

    all_data_index = np.arange(0,source_all.shape[0])
    testing_index = np.random.choice(all_data_index,100,replace=False)
    testing_source, testing_target = source_all[testing_index], target_all[testing_index]

    pca = PCA(n_components=2)

    source_data = np.concatenate([h1_source,h2_source,h3_source,testing_source])

    pca.fit(source_data)


    h1_source_t = pca.transform(h1_source)
    h2_source_t = pca.transform(h2_source)
    h3_source_t = pca.transform(h3_source)
    testing_source_t = pca.transform(testing_source)

    fig1 = px.scatter(x=h1_source_t.T[0], y=h1_source_t.T[1],color_discrete_sequence=['blue'])
    fig2 = px.scatter(x=h2_source_t.T[0], y=h2_source_t.T[1],color_discrete_sequence=['green'])
    fig3 = px.scatter(x=h3_source_t.T[0], y=h3_source_t.T[1],color_discrete_sequence=['red'])
    fig4 = px.scatter(x=testing_source_t.T[0], y=testing_source_t.T[1],color_discrete_sequence=['gray'])

    fig5 = go.Figure(data=fig1.data + fig2.data + fig3.data+fig4.data)
    st.plotly_chart(fig5, use_container_width=True)


def main2():
    st.header("Aligment Analysis")

    pass

if __name__ == '__main__':
    st.title("Week 2 - Yekta Can Tursun")
    main()
    main2()
    pass