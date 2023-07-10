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

from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from hugchat import hugchat
from streamlit_chat import message

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI

import speech_recognition as sr
from os import path
from pydub import AudioSegment

API_KEY = "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"
os.environ["OPENAI_API_KEY"] = API_KEY

#Summary

memory = ConversationBufferMemory()

llm = ChatOpenAI(temperature=0.9)
prompt = ChatPromptTemplate.from_template(
    "Summarize the following text {product}?"
)

chain = LLMChain(llm=llm, prompt=prompt)

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

#ChatBot function

#Session State
# Generate empty lists for generated and past.
## generated stores AI generated responses
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
## past stores User's questions
if 'past' not in st.session_state:
    st.session_state['past'] = []

#Nguyen The Quang
#Summarization with file
product = ""


st.title("Summarize text from File...")
st.write("")
st.write("Write a paragraph / verse / ...")
plot = st.text_input(label="Text to summarize", placeholder="Write something to summaeize...")
product = plot
st.write("")
st.write("Or upload a File...")
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:

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
        

    string_data = ""
    
    if stringio == "":
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        string_data = stringio.read()
    
    else:
        string_data = stringio
    product = string_data

x = ""
check = False
resans = ""
if st.button("Summarize"):
    with st.spinner('Wait for it...'):
        time.sleep(1)
    if product != "":
        x = chain.run(product)
        resans = st.text_area(label="Text after Summarize", value=chain.run(product))
        check = True
        
    else:
        print("Error! Please write or upload a file")
            
inp = ""

llm = OpenAI(temperature=0)
if resans:
    inp = "Summarize" + product
    memory.save_context({"input": inp}, 
                    {"output": x})
    conversation = ConversationChain(
        llm=llm, verbose=True, memory=memory
    )
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
                st.session_state.generated.append(conversation.predict(input=query))
                
            memory.save_context({"input": query}, 
                {"output": conversation.predict(input=query)})
            conversation = ConversationChain(
                llm=llm, verbose=True, memory=memory
            )

        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                message(st.session_state['generated'][i], key=str(i))
            
