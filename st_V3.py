#!/usr/bin/env python3
__author__ = "Yang Xiao"

__editor__ = "John Onwuemeka"



import streamlit as st
import pandas as pd
import numpy as numpy
import urllib,pickle,json,os
#import seaborn as sns
import psycopg2
import unidecode,re,dateparser
from date_model import *
#from date_reformater import date_format
from geopy.geocoders import Nominatim
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
#from Generate_report import tables_df_dict,create_report

global conn

dd = dict()
dd['user'] = ''
dd['pwd'] = ''
dd['database'] = ''
dd['port'] = ''
dd['host'] = ''
if not os.path.exists('cred.json'):
    with open('cred.json','w') as json_file:
        json.dump(dd,json_file)




def save_sign_in_file(user,pwd,database,port,host):
    dictfile = dict()
    dictfile['user'] = user
    dictfile['pwd'] = pwd
    dictfile['database'] = database
    dictfile['port'] = port
    dictfile['host'] = host
    with open('cred.json', 'w') as json_file:
        json.dump(dictfile, json_file)
    return None

def open_sign_in_file():

    with open('cred.json') as json_file:
        cred = json.load(json_file)
    return cred['user'],cred['pwd'],cred['database'],cred['port'],cred['host']


def connect2GCP(database, user, pwd, host, port):

    conn = psycopg2.connect(database=database, user=user, password=pwd, host=host, port=port)

    return conn


def fetchtables(cur):
    sql = '''SELECT * FROM pg_catalog.pg_tables
            WHERE schemaname != 'pg_catalog' AND
            schemaname != 'information_schema';'''

    cur.execute(sql)
    result= cur.fetchall()

    return tuple(pd.DataFrame(result)[1])


def main():
    st.sidebar.title('Navigation')
    
    pwd = ''
    page = st.sidebar.selectbox('Option',['Preface','User Sign in','Data Quality Check','Table Summary'])
    if page == 'Preface':
        st.markdown(read_preface())
    elif page == 'User Sign in':

        host = st.text_input('Please enter Host:','35.224.194.176')
        port = st.text_input('Please enter Port:','5432')
        database = st.text_input('Please enter Database:','flc-12')
        user = st.text_input('Please enter username:','postgres')
        pwd = st.text_input('Please enter pwd:',type='password')
        if (pwd == 'Prospect33') & (user == 'postgres'):
            save_sign_in_file(user,pwd,database,port,host)
        else:
            st.subheader('Either user name or pwd is wrong, please enter again')

    user,pwd,database,port,host = open_sign_in_file()
    try:
        conn = connect2GCP(database,user,pwd,host,port)
        cur = conn.cursor()
    except:
        pass

    if pwd == 'Prospect33':
        if  page == 'Table Summary':
            st.subheader('Welcome!! ' + user)
            st.empty()
            table_page = st.sidebar.radio('Explore:',fetchtables(cur))
            st.subheader((str(table_page).upper()))
            st.subheader('Table summary')
            st.table(SummaryTables(table_page))
            st.subheader('Table top rows')
            headrows(conn,table_page)
            
        elif page == 'Data Quality Check':

            st.title('Tool list')
            tool = st.radio('Tool selection',('Data Quality','Info Entry'))
            if tool == 'Data Quality':
                data_quality_ui()
            elif tool == 'Info Entry':
                source_entry()
            else:
                st.subheader('Please choose a tool')


# Show df summary table
def data_quality_ui():
    st.subheader('Data Quality')
    option = st.radio('Please choose what type of data you input:',
                        ('Date','Name','Address'))
    if option == 'Date':
        time_input = st.text_input('Please enter date type info to check','10 Jul 2007')
        if st.button('Check'):
            with st.spinner('Transforming date with ML...'):
                time_output = date_format(time_input)
            try:
                time_output2 = dateparser.parser(time_input).strftime('%d-%m-%Y')

                st.write('Date Parser: {}'.format(time_output2))
            except:
                st.write('Could not transform entered date with Date Parser')
                pass
            st.write('ML: {}'.format(time_output))
    if option == 'Name':
        name_input = st.text_input('Please enter name info to check','Prospect33, llc')
        # name = cleaned(name_input)
        if st.button('Check'):
            st.write(name_input)
    if option  == 'Address':
        st.write('Please enter address here')
        address1 = st.text_input('Address','1 Liberty St',key='2')
        city = st.text_input('City','New York',key='3')
        state = st.text_input('State/Province','New York',key='4')
        zip_code = st.text_input('Zip/Postal Code','10016',key='5')    
        if st.button('Check'):
            exist_response,valid_response = AddressCheck(address1, city, state, zip_code,True,True)
            #st.write(exist_response,valid_response)
            if exist_response and valid_response:
                st.write('Address is valid and exist in our database. Address ID is %s' % exist_response)
            elif valid_response and not exist_response:
                st.write('Address is valid and but does not exist in our database.')
                

def source_entry():
    st.subheader('Source_check & entry')
    option = st.radio('Please choose what type of data you input:',
                        ('Name','Address'))
    if option == 'Name':
        name = st.text_input('Please enter name here','Prospect33, llc',key='1')
        name = cleaned(name)
        if st.button('Check Name'):
            exist_response = NameCheck(name)
            if exist_response:
                if len(exist_response) > 5:
                    st.write('Here are the top 5 records matching {}'.format(name))
                else:
                    st.write('Here are the top {} records matching {}'.format(len(exist_response),name))
                j = 0
                for Name in exist_response:
                    st.write(Name[0])
                    j+=1
                    if j==5:
                        break
            else:
                st.write('{} does not exist in our database'.format(name))
                if st.button('Add'):
                    st.write('Successfully added')
        
        # # st.write(name)
        # if st.button('Delete'):
        #     st.write('Record deleted!')
        # if st.button('Update'):
        #     st.text_input('Please enter name info to update:')
        #     st.button('Add')
        #     st.write('Successfully added')
    if option == 'Address':
        st.write('Please enter address here')
        address1 = st.text_input('Address','1 Liberty St',key='2')
        city = st.text_input('City','New York',key='3')
        state = st.text_input('State/Province','New York',key='4')
        zip_code = st.text_input('Zip/Postal Code','10016',key='5')
        if st.button('Check Address'):
            exist_response,valid_response = AddressCheck(address1, city, state, zip_code,True,True)
            if exist_response and valid_response:
                st.write('Address is valid and exist in our database. Address ID is %s' % exist_response)
                if st.button('Delete'):
                    st.write('Record deleted!')
                if st.button('Update'):
                    st.text_input('Please enter address info to update:')
                    if st.button('Add'):
                        st.write('Successfully added')
            elif valid_response and not exist_response:
                st.write('Address is valid and but does not exist in our database.')
                if st.button('Add record'):
                    st.text_input('Please enter address info to update:')
                    address1 = st.text_input('Address',key='2')
                    city = st.text_input('City',key='3')
                    state = st.text_input('State/Province',key='4')
                    zip_code = st.text_input('Zip/Postal Code',key='5')
                    if st.button('Add'):
                        st.write('Successfully added')
    #st.write('show the record')
    #st.write('if exist')
   
    # if st.button('Update'):
    #     st.text_input('Please enter name info to update:')
    #     st.text_input('Please enter address info to update:')
    # st.write('If it not exists')
    # if st.button('Add'):
    #     st.write('Added !')

def UserConfirmation():
    #TODO:
    # When user want to manipulate record, show confirmation button to confirm action.
    return None

def SummaryTables(tablename):

    return pd.read_excel('tables_report.xlsx',sheet_name=tablename)


def ManipulateRecord(mode):
    #TODO:
    #mode: Add, delete, update
    return None

def SearchRecord():
    #TODO:
    #Search record in GCP to find if it exist
    return None

def namecheck():
    #TODO:
    #Take organization name as input and predict its org type and reformat the name.
    return None

def AddressCheck(address1, city, state, zip_code,check_exist,check_valid):
    

    exist_response = None
    valid_response = None
    
    if check_exist is True:
        
        user,pwd,database,port,host = open_sign_in_file()
        conn = connect2GCP(database,user,pwd,host,port)
        cur = conn.cursor()
        
        sql = "Select ID, Address1, city, state, country, zip_code from Address \
        where Address1 = '%s'" % address1

        if city:
            sql = sql+ " and city = '%s'" % city
        if state:
            sql = sql+" and state = '%s'" % state
        if zip_code:
            sql = sql+" and zip_code = '%s'" % zip_code


        cur = conn.cursor()
        cur.execute(sql)
        
        try:
            exist_response = cur.fetchall()[0][0]
        except:
            pass
        conn.close()
    
    if check_valid is True:
        addr = address1 +' '+city+' '+state
        

        #user_agent="geoapiExercises",
        geoloc = Nominatim(user_agent="http",timeout=None)
        valid_response = geoloc.geocode(addr,addressdetails=True)
            

    #TODO:
    #Take address as input and return the desired format.
    return exist_response, valid_response

def headrows(conn,table_name,rows = 20):

    sql = '''SELECT * FROM "''' + table_name +'''" LIMIT '''+ str(rows) +''';'''
    df_head = pd.read_sql_query(sql,conn)
    return st.table(df_head)


def NameCheck(name):
    
    
    user,pwd,database,port,host = open_sign_in_file()
    conn = connect2GCP(database,user,pwd,host,port)
    cur = conn.cursor()
    
    name = name.split()[0]
    sql = "Select Name from Organization where lower(Name) like lower('%"+name+"%')"


    cur = conn.cursor()
    cur.execute(sql)
    
    exist_response = cur.fetchall()
    conn.close()
    
    return exist_response

@st.cache
def date_format(datein):
    """
    Function thst uses the trained RNN weights to predict dates in YYYY-MM-DD
    format given a date input in human readable format

    Parameters
    ----------
    datein : Input date in human readable format (e.g., Tue 10 April 2013)

    Returns
    -------
    dateout : Output date in YYYY-MM-DD format

    """

    n_s = 64 # number of units for the post-attention LSTM's hidden state "s"
    Tx = 30 # max length of the input text
    m = 1 # no. of inputs in each item of the input list
    
    # load the human vocabulary dictionary
    pfile = open('./date_model/human_vocab.pickle','rb')
    human_vocab = pickle.load(pfile)
    pfile.close()
    
    # load the inverted machine vocabulary dictionary
    pfile = open('./date_model/inv_machine_vocab.pickle','rb')
    inv_machine_vocab = pickle.load(pfile)
    pfile.close()
    
    # load the machine vocabulary dictionary
    #pfile = open('machine_vocab.pickle','rb')
    #machine_vocab = pickle.load(pfile)
    #pfile.close()

    
    model  = load_model('./date_model/trained_date_model.h5')
        
    #initialize hidden state weights
    s0 = numpy.zeros((m, n_s))
    c0 = numpy.zeros((m, n_s))
    
    #convert date text to integers
    datein_int= string_to_int(datein, Tx, human_vocab)
    
    #format date_int to model input structure
    datein_int = numpy.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), datein_int)))
    datein_int = datein_int[numpy.newaxis,:,:]
    
    # Predict the output
    pred = model.predict([datein_int, s0, c0])
    pred = numpy.argmax(pred, axis = -1)
    dateout = [inv_machine_vocab[int(i)] for i in pred]
    yr = ''.join(dateout[0:4])
    mn = ''.join(dateout[5:7])
    dy = ''.join(dateout[8:10])
    dateout = mn+'-'+dy+'-'+yr

    return dateout
    
def string_to_int(string, length, vocab):
    """
    Converts all strings in the vocabulary into a list of integers representing the positions of the
    input string's characters in the "vocab"
    
    Arguments:
    string -- input string, e.g. 'Wed 15 Dec 2017'
    length -- the number of time steps you'd like, determines if the output will be padded or cut
    vocab -- vocabulary, dictionary used to index every character of your "string"
    
    Returns:
    rep -- list of integers (or '<unk>') (size = length) representing the position of the string's character in the vocabulary
    """
    
    #make lower to standardize
    string = string.lower()
    string = string.replace(',','')
    
    if len(string) > length:
        string = string[:length]
    
    rep = list(map(lambda x: vocab.get(x, '<unk>'), string))
    
    if len(string) < length:
        rep += [vocab['<pad>']] * (length - len(string))
    
    return rep
    
def cleaned(name):
    name = re.sub('\W+',' ', unidecode.unidecode(name).lower())
    return name
    
#@st.cache
def read_preface():
    url = 'https://raw.githubusercontent.com/Yangxiao2498/fl_streamlit/main/Preface.md'
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

main()
