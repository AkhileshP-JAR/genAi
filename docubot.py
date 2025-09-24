
import streamlit as st
import time

#importing the backend
import helper as helper
# st.title("First Gen AI Document bot")

st.set_page_config(layout="wide")

OLLAMA=st.sidebar.text_input("Ollama Server","http://localhost:11434")

MODEL=st.sidebar.text_input("Model","gemma3:1b")

temp=st.sidebar.slider("Temperature",0.0,1.0,0.2,0.1)
prompt_style=st.sidebar.radio("Prompt Style",["v1(1)"])

left,right=st.columns(2,gap="large")


with left:
    st.title("PDF-> ASK-> ANSWER")
    #up is the pdf file 
    up=st.file_uploader("PDF upload",type=["pdf"])
    if up:
        extracted_text=helper.read_pdf_text(up)
        st.success(f"Pdf uploaded succesfully, Length:{len(extracted_text)}")

        with st.expander("PDF text"):
             st.write(extracted_text)


#right side cooum of ui
with right :
    st.subheader("ASk QnA")
    q=st.text_input("Question", placeholder=" Please ask quesion on uploaded doc")
    button=st.button("Ask Pdf", type="primary",use_container_width=True)

    if not up:
        st.warning("Please upload a pdf first")
    elif not q.strip():
        st.warning("Please ask Question")
    else:
        prompt=helper.make_prompt(prompt_style,q,extracted_text)

        with st.spinner(f"Calling model:{MODEL}"):
            t0=time.time()
            ans=helper.call_ollama(OLLAMA,MODEL, prompt, temp)
            st.write(ans)
            elapsed=time.time()-t0
            st.write(f"** Time Taken:{elapsed:.2f}")

            with st.expander("Prompt"):
                st.write(prompt)
