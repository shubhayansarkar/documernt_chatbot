import streamlit as st
from dchatbot import chatbot

st.title("...Document Guide... :D")
docs= st.text_input("Enter a Document URL:", "", key = "0")
if docs:
    retrieval_chain = chatbot(docs)
    i= 0
    while True:
        i += 1
        query = st.text_input("You:", "", key = f"{i}")
        if query == 'ok bye':
            break
        else:
            
            response = retrieval_chain.invoke(query)
            # print(response['result'])
            st.text_area("Response:", response['result'], height=150, key = f"{i}{i}")