import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
import pickle
import pandas as pd
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


def main2(id, exp,fold):
    
    # Read from file

    st.subheader(f"Example {exp}")
    filehandler = open(f"Layer Results/Fold{fold}/id_{id}.obj", 'rb') 
    data = pickle.load(filehandler)
    filehandler.close()

    output, target, before = data
    output = output.detach().cpu().numpy()
    
    output_pd = pd.DataFrame( dict(Output=output.flatten()) )
    fig1 = px.histogram(output_pd,x="Output", color_discrete_sequence=['blue'],nbins=20,opacity=0.5, labels=["test"])

    before_pd = pd.DataFrame( dict(Before=before.flatten()) )
    fig2 = px.histogram(before_pd,x="Before" , color_discrete_sequence=['red'],nbins=20, opacity=0.5)

    target_pd = pd.DataFrame( dict(Target=target.flatten()) )
    fig3 = px.histogram(target_pd,x="Target" , color_discrete_sequence=['green'],nbins=20, opacity=0.5)

    fig = go.Figure(data=fig2.data + fig1.data + fig3.data)
    
    st.plotly_chart(fig, use_container_width=True)

    fig1 = px.scatter(x=before.flatten(), y=target.flatten(),color_discrete_sequence=['blue'])
    fig2 = px.scatter(x=output.flatten(), y=target.flatten(),color_discrete_sequence=['green'])

    fig = go.Figure(fig1.data + fig2.data)
    st.plotly_chart(fig, use_container_width=True)
    pass



def readHospital(name, h):
    address = f"Results/{name}_fold{1}/"
    h1_fold1 = pd.read_csv(address + f"h_{h}.csv", header=None)

    address = f"Results/{name}_fold{2}/"
    h1_fold2 = pd.read_csv(address + f"h_{h}.csv", header=None)

    h1 = pd.concat((h1_fold1,h1_fold2))
    return h1

def readHospital(names, h):
    result = pd.DataFrame()
    for name in names:
        address = f"Results/{name}_fold{1}/"
        h1_fold1 = pd.read_csv(address + f"h_{h}.csv", header=None)

        address = f"Results/{name}_fold{2}/"
        h1_fold2 = pd.read_csv(address + f"h_{h}.csv", header=None)

        h1 = pd.concat((h1_fold1,h1_fold2))
        result = pd.concat((result,h1))
    return result

def readAllHospital(name):
    result = []
    for i in range(1,4):
        result.append(readHospital(name,i))
    return result

def main3():
    showData = st.checkbox("Show Data")
    experiment_names = ["Almanak1","Almanak2","Bursa3","Bursa4"]

    h1 = readHospital(experiment_names, 1)
    h2 = readHospital(experiment_names, 1)
    h3 = readHospital(experiment_names, 1)

    text = []
    keys = ["Local","FedL", "FedL with Aligment", "Fed with Aligment All layer"]
    for i in range(4):
        for k in range(200):
            text.append(keys[i])

            pass

    
    fig = go.Figure()
    if showData:
        fig.add_trace(go.Box(x=text, y=h1[0],name="Hospital 1", boxpoints='all'))
        fig.add_trace(go.Box(x=text, y=h2[0],name="Hospital 2",boxpoints='all'))
        fig.add_trace(go.Box(x=text, y=h3[0],name="Hospital 3 OOD",boxpoints='all'))
    else:
        fig.add_trace(go.Box(x=text, y=h1[0],name="Hospital 1"))
        fig.add_trace(go.Box(x=text, y=h2[0],name="Hospital 2"))
        fig.add_trace(go.Box(x=text, y=h3[0],name="Hospital 3 OOD"))
    
    fig.update_layout(
    yaxis_title='MAE',
    boxmode='group' # group together boxes of the different traces for each value of x
        )
    st.plotly_chart(fig, use_container_width=True)
    pass

if __name__ == '__main__':
    st.title("Week 2 - Yekta Can Tursun")
    main()

    st.header("Aligment Analysis")
    main2(1,1,1)
    main2(1,2,2)


    st.header("Results")
    main3()

    pass