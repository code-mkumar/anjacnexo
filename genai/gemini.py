import google.generativeai as genai
import streamlit as st
# 
genai.configure(api_key='AIzaSyBQyfBFD76jSEwqBuZswkbGpj88IX-s3jk')
model = genai.GenerativeModel('gemini-2.0-flash')

def get_gemini_response(combined_prompt):
   
    # print(data))
    response = model.generate_content(combined_prompt)
    # print(response)
    query=response.text
    return query
    
