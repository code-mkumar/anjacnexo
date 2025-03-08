import google.generativeai as genai
import streamlit as st
# 
genai.configure(api_key='AIzaSyBebu9JXr_EkOuruCAuFXWs-v5dbDKmNa0')
model = genai.GenerativeModel('gemini-2.0-flash')

def get_gemini_response(combined_prompt):
   
    # print(data))
    response = model.generate_content(combined_prompt)
    # print(response)
    query=response.text
    return query
    