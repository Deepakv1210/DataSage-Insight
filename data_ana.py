#Import required libraries
import os 
from Openai_api import apikey 

import streamlit as st
import pandas as pd
#import langchain
from langchain.llms import OpenAI
#from langchain.agents import create_pandas_dataframe_agent
#from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv, find_dotenv

os.environ['OPENAI_API_KEY'] = apikey
load_dotenv(find_dotenv())

st.title("DataSage Insight 🤖")
st.write("Hello, 👋 I am DataSage and I am here to help you with data science projects.")

with st.sidebar:
    st.write('*Please upload your CSV files to Analyze*')
    st.caption('''**Discover hidden gems in your data with our powerful analytics and visualization tools.
    No PhD in data science required! Our intuitive interface ensures that everyone can navigate and analyze data like a pro.**
    ''')

    st.divider()
    # with st.expander('Expander section'):
    #     st.write('Test')

    st.caption("<p style ='text-align:center'> Open for Everyone..🎁</p>",unsafe_allow_html=True )


if 'clicked' not in st.session_state:
    st.session_state.clicked ={1:False}
def clicked(button):
    st.session_state.clicked[button]= True
st.button("Let's get started!!", on_click = clicked, args=[1])
if st.session_state.clicked[1]:
    st.header('Data Analysis')
    st.subheader('Checking..')
    user_csv = st.file_uploader("Upload your file here", type="csv")
    if user_csv is not None:
        user_csv.seek(0)
        df = pd.read_csv(user_csv, low_memory=False)

        #llm model
        llm = OpenAI(temperature = 0)
        #Function sidebar
        @st.cache_data
        def steps():
            steps_eda = llm('What are the steps of Data Analysis')
            return steps_eda

        #Testing
        pandas_agent=create_pandas_dataframe_agent(llm,df,verbose=True)
        # q='What is this data about?'
        # ans=pandas_agent.run(q)
        # st.write(ans)

        @st.cache_data
        def function_agent():
            # st.write("**Data Overview**")
            # st.write("The first rows dataset look like this:")
            # st.write(df.head())
            # st.write("**Data Cleaning**")
            # columns_df = pandas_agent.run("What are the meaning of the columns?")
            # st.write(columns_df)
            # missing_values = pandas_agent.run("How many missing values does this dataframe have? Start the answer with 'There are'")
            # st.write(missing_values)
            # st.write("**Data Summarisation**")
            # st.write(df.describe())
            # analysis = pandas_agent.run("Categorize the data as positive, netutral or negative sentiment based on stars")
            # st.write(analysis)
            # conc = pandas_agent.run("So what can you conclude from this data?.")
            # st.write(conc)
            new_features = pandas_agent.run("What new features would be interesting to create? Just give some ideas.")
            st.write(new_features)
            return

        @st.cache_data
        def function_question_variable():
            st.bar_chart(df, y =[user_question_variable])
            # summary_statistics = pandas_agent.run(f"Give me a summary of the statistics of {user_question_variable}")
            # st.write(summary_statistics)
            # trends = pandas_agent.run(f"Analyse trends, seasonality, or cyclic patterns of {user_question_variable}")
            # st.write(trends)
            # missing_values = pandas_agent.run(f"Determine the extent of missing values of {user_question_variable}")
            # st.write(missing_values)
            return
        
        @st.cache_data
        def function_question_dataframe():
            dataframe_info = pandas_agent.run(user_question_dataframe)
            st.write(dataframe_info)
            return

        #Main

        st.header('Data analysis')
        st.subheader('General information about the dataset')

        with st.sidebar:
            with st.expander('Steps of Data Analysis'):
                st.write(steps())


        st.subheader('Parameter study')
        user_question_variable = st.text_input('What parameter are you interested in')
        if user_question_variable is not None and user_question_variable !="":
            function_question_variable()

            st.subheader('Further study')

        if user_question_variable:
            user_question_dataframe = st.text_input( "Is there anything else you would like to know about your dataframe?")
            if user_question_dataframe is not None and user_question_dataframe not in ("","no","No"):
                function_question_dataframe()
            if user_question_dataframe in ("no", "No"):
                st.write("")
