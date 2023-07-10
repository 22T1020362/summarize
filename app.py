import streamlit as st
import time
from transformers import pipeline
import langchain
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

from langchain import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
import streamlit.components.v1 as components

import pdf2image
from io import StringIO
from PIL import Image
import pytesseract
from pytesseract import Output, TesseractError
from functions import convert_pdf_to_txt_pages, convert_pdf_to_txt_file, save_pages, displayPDF, images_to_txt

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
# from langchain.chains import ConversationalRetrievalChain
from langchain.chains.summarize import load_summarize_chain
import tempfile

from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from hugchat import hugchat
from streamlit_chat import message

from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
import speech_recognition as sr
from os import path
from pydub import AudioSegment

st.title("Summarize text from File...")
st.write("")

#Session State
    # Generate empty lists for generated and past.
    ## generated stores AI generated responses
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
## past stores User's questions
if 'past' not in st.session_state:
    st.session_state['past'] = []
if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = 0

if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = []


global APIKEY
APIKEY = st.text_input('API_KEY', type = 'password')
if APIKEY:
    
    # #Summary
    
    
    # global llm
    # llm = ChatOpenAI(temperature=0.9)
    
    global prompt
    prompt = ChatPromptTemplate.from_template(
        "Summarize the following text {product}?"
    )

    # global chain
    # chain = LLMChain(llm=llm, prompt=prompt)
    
    @st.cache_resource
    def llm():
        model = OpenAI(temperature=0.0, openai_api_key= APIKEY)
        
        return model

    llm = llm()

    @st.cache_resource
    def chain():
        global memory
        memory = ConversationBufferMemory()
    
        chain = LLMChain(
            llm=llm, prompt=prompt, memory=memory
        )
        
        return chain

    global llm_chain
    llm_chain = chain()

    @st.cache_resource
    def conversation():
        global memory
        memory = ConversationBufferMemory()
    
        llm_conversation = ConversationChain(
            llm=llm, verbose=True, memory=memory
        )
        return llm_conversation


    llm_conversation = conversation()

    
    #Nguyen The Quang
    #Summarization with file
    global product
    product = ""
    
    st.write("Write a paragraph / verse / ...")
    plot = st.text_input(label="Text to summarize", placeholder="Write something to summaeize...")
    product = plot
    st.write("")
    st.write("Or upload a File...")
    uploaded_file = st.file_uploader("Choose a file")
    
    if uploaded_file is not None:
        st.session_state["uploaded_file"] = uploaded_file
        #convert pdf to txt
        path = uploaded_file.read()
        file_extension = uploaded_file.name.split(".")[-1]
        file_name = uploaded_file.name.split(".")[0]
        stringio = ""
        
        if file_extension == "pdf":
            stringio, nbPages = convert_pdf_to_txt_file(uploaded_file)
            totalPages = "Pages: "+str(nbPages)+" in total"
    
        elif file_extension == "txt":
            stringio = ""
    
        elif file_extension == "csv":
            stringio = pd.read_csv(path)
    
        #audio, mp3 file
        elif file_extension == "mp3" or file_extension == "wav":
            if file_extension == "mp3":
                ex = uploaded_file.name
                file_export = file_name + "wav"
                sound = AudioSegment.from_mp3(ex)
                sound.export(file_export, format="wav")
                r = sr.Recognizer()
                with sr.AudioFile(file_export) as source:
                    audio = r.record(source)  # read the entire audio file                  
                    stringio = r.recognize_google(audio)
            else:
                AUDIO_FILE = uploaded_file.name
                r = sr.Recognizer()
                with sr.AudioFile(AUDIO_FILE) as source:
                    audio = r.record(source)  # read the entire audio file                  
                    stringio = r.recognize_google(audio)
                
        #img,png,jpg
        else:
            pil_image = Image.open(uploaded_file)
            stringio = pytesseract.image_to_string(pil_image)
            
        global string_data
        string_data = ""
        
        if stringio == "":
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            string_data = stringio.read()
        
        else:
            string_data = stringio
        product = string_data

    def clear_msg():
        st.session_state['generated'] = []
        st.session_state['path'] = []
        memory = ConversationBufferMemory()
        llm_chain = chain()
        llm_conversation = conversation()
        st.session_state["file_uploader_key"] += 1
        st.experimental_rerun()
        
    st.sidebar.button("Clear", on_click=clear_msg)

    global x
    x = ""
    global check
    check = False
    global resans
    resans = ""
    if st.button("Summarize"):
        with st.spinner('Wait for it...'):
            time.sleep(1)
        if product != "":
            x = llm_chain.run(product)
            resans = st.text_area(label="Text after Summarize", value=x)
            check = True
            
        else:
            st.write("Error! Please write or upload a file")
    
   
    
    global inp
    inp = "Summarize" + product
    llm_conversation.memory.save_context({"input": inp}, 
                    {"output": x})
    
    with st.spinner('Wait for it...'):
        time.sleep(1)
    st.write("Asking about the context summarize above")
    
    # Layout of input/response containers
    input_container = st.container()
    colored_header(label='', description='', color_name='blue-30')
    response_container = st.container()
    
    
    with input_container:
        input_text = st.text_input("Question: ", "", key = "input", placeholder="Ask something...")
        query = input_text
    with response_container:  
            
        if st.button(label="Ask"):
            with st.spinner("generating..."):
                st.session_state.past.append(query)
                st.session_state.generated.append(llm_conversation.predict(input=query))
                
            llm_conversation.memory.save_context({"input": query}, 
                {"output": llm_conversation.predict(input=query)})


            # st.chat_message("user").markdown(query)
            # with st.chat_message("assistant"):
            #     st.markdown(llm_conversation.predict(input=query))

        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                message(st.session_state['generated'][i], key=str(i))

