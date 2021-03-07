import streamlit as st
import pandas as pd
import numpy as numpy
import matplotlib.pyplot as plt
import urllib,pickle
# For demonstrate the ts data example
import seaborn as sns
import statsmodels.api as sm

def main():
    preface = st.markdown(read_preface())
    st.sidebar.title('Navigation')
    
    
    page = st.sidebar.selectbox('Functionality:',
                            ['Instruction','ML data quality check','Table Summary'])
    if page == 'Table Summary':
        preface.empty()
        table = st.sidebar.radio('Explore:',
                        ('Organization','Address','Organization_Address','Client_Hierarchy','tab5'))

        if table == 'Organization':
            plot_table_summary('Organization')
        
        elif table == 'Address':
            plot_table_summary('Address')
        
        elif table == 'Organization_Address':
            plot_table_summary('Organization_Address')
        
        elif table == 'Client_Hierarchy':
            write_df('Client_Hierarchy')
    
    elif page == 'ML data quality check':
        preface.empty()
        st.title('Input data prediction tool')
        st.subheader('Input area:')

        #User input area and we can define the input format and default values
        Organization_name = st.text_input('Format:xxxxxxxxx','Prospect33,llc')
        Address = st.text_input('Format:xxxxxxxxxxxx','1 Liberty St 3rd Floor, New York, NY 10006')

        # Here is a ts prediction example how we use trained model to show prediction
        st.subheader('Here is a example for a ts data prediction:')
        period = st.text_input('Please input the data time to predict(larger than 2014 in format yyyy-mm-dd)','2017-01-01')

        # Button, click button to start prediction
        if st.button('Prediction'):
            st.write('Predicting.......')
            run_prediction(Organization_name,Address,period)
            st.write('Done!')
            st.write('Output')

# Prediction model, the number of arguments should be two, name and address, but include additional one for example.
def run_prediction(org_name,addr,period):

    # Load model by pickle
    results = pickle.load(open('model.P','rb'))
    y = pd.read_pickle('df.pkl')
    try:
        pred = results.get_prediction(start=pd.to_datetime(period), dynamic=False)
        #pred_ci = pred.conf_int()
        fig,ax = plt.subplots()
        ax = y['2014':].plot(label='observed')
        
        pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
        plt.style.use('ggplot')
        ax.set_xlabel('Date')
        ax.set_ylabel('Furniture Sales')
        plt.legend()
        plt.show()
        
        st.pyplot(fig)
    except:
        st.write('Double check datetime input range and format')

# Show df summary table

def write_df(table):
    #df = pd.read_csv(table+'csv')
    df = pd.DataFrame(
        {
            'first column': [1, 2, 3, 4],
            'second column': [10, 20, 30, 40]
        })
    st.title('two different ways to show df')
    st.subheader('The first one')

    st.table(df)

    st.subheader('The second one')
    st.write(df)
    #st.line_chart(df)

# Plotting df

def plot_table_summary(table):
    table = 'tips'
    tips = sns.load_dataset(table)
    fig = sns.lmplot(x = 'total_bill',y ='tip',data = tips )
    plt.title('sample plot')
    plt.style.use('ggplot')
    plt.show()
    st.pyplot(fig)
    
@st.cache
def read_preface():
    url = 'https://raw.githubusercontent.com/Yangxiao2498/fl_streamlit/main/Preface.md'
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

main()
