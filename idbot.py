import streamlit as st
import time
import json
from PIL import Image
import requests

#importing the backend
import helper_id 
# st.title("First Gen AI Document bot")

st.set_page_config(layout="wide",page_title="DOBID EXTRACTOR" ,page_icon="🍌")
st.title("Student ID-> NAME &DOB")

OLLAMA=st.sidebar.text_input("Ollama Server","http://localhost:11434").rstrip("/")
MODEL=st.sidebar.text_input("Model","qwen2.5vl:3b")
temp=st.sidebar.slider("Temperature",0.0,1.0,0.2,0.1)
prompt_style=st.sidebar.radio("Prompt Style",["v1(1)"])

left,right=st.columns(2,gap="large")


with left:
    st.title("Image-> ANSWER") 
    #up is the pdf file 
    up=st.file_uploader("PDF upload",type=["jpeg","jpg","png"])
    if up:
        img=Image.open(up)
        #st.success(f"Pdf uploaded succesfully, Length:{len(extracted_text)}")
        st.image(img, caption="Uploaded ID",use_container_width=True)

        # with st.expander("PDF text"):
        #      st.write(extracted_text)

#right side cooum of ui
with right :
    st.subheader("Image Extractor")
    # q=st.text_input("Question", placeholder=" Please ask quesion on uploaded doc")
    # button=st.button("Ask Pdf", type="primary",use_container_width=True)
    run=st.button("Extract Name and DOB", type="primary",use_container_width=True)
    if run:
        # st.warning("Please upload a Image first")
        if not up:
            st.warning("Please upload image first")
        else:
            prompt = helper_id.make_prompt(prompt_style)  # helper_id 
            b64 = helper_id.image_to_base64(img)
            with st.spinner(f"Calling model: {MODEL} via Ollama "):
                try:
                    t0 = time.time()
                    resp = helper_id.call_ollama(OLLAMA, MODEL, prompt, b64, temp)
                    elapsed = time.time() - t0
                    st.write(resp)
                except requests.RequestException as e:
                    st.error(str(e))
                    resp, elapsed = "", 0

        if resp:
            st.markdown("***Results***")
            
            # Assuming the extracted details are in a structured format
            try:
                extracted_data = {}

                # Parse the response and extract relevant info (Name, DOB, etc.)
                details = resp.split("\n")  # Adjust based on response structure
                
                for detail in details:
                    if "Name" in detail:
                        extracted_data["Name"] = detail.split(":")[1].strip()
                    elif "DOB" in detail:
                        extracted_data["Date_of_Birth"] = detail.split(":")[1].strip()
                    # Add other fields as necessary (e.g., Address, Gender)
                
                # Display extracted details in JSON format
                st.json(extracted_data)  # This will display the data in a formatted JSON block

            except Exception as e:
                st.error(f"Error parsing details: {str(e)}")

            # Display time taken
            st.write(f"⏲️ **Time Taken**: {elapsed:.2f} seconds")
            st.caption(f"Model: `{MODEL}` | Prompt: `{prompt_style}` | Temperature: {temp}")

#     if run:
#     # st.warning("Please upload a Image first")
#         if not up:
#             st.warning("Please upload image first")
#         else:
#             prompt = helper_id.make_prompt(prompt_style)  # helper_id 
#             b64 = helper_id.image_to_base64(img)
#             with st.spinner(f"Calling model: {MODEL} via Ollama "):
#                 try:
#                     t0 = time.time()
#                     resp = helper_id.call_ollama(OLLAMA, MODEL, prompt, b64, temp)
#                     elapsed = time.time() - t0
#                     st.write(resp)
#                 except requests.RequestException as e:
#                     st.error(str(e))
#                     resp, elapsed = "", 0

#         if resp:
#             st.markdown("***Results***")
            
#             # Show the extracted details in a point-wise format
#             st.markdown("**Extracted Details:**")
            
#             try:
#                 details = resp.split("\n")  # Adjust based on response structure
                
#                 # Point-wise display of extracted data
#                 point_count = 1
#                 for detail in details:
#                     # Display details if they contain specific keywords
#                     if "Name" in detail:
#                         st.write(f"{point_count}. **Name**: {detail.split(':')[1].strip()}")
#                         point_count += 1
#                     elif "DOB" in detail:
#                         st.write(f"{point_count}. **Date of Birth**: {detail.split(':')[1].strip()}")
#                         point_count += 1
#                     # Add more checks for other details (e.g., Address, Gender, etc.)
#                     else:
#                         st.write(f"{point_count}. **Other Info**: {detail}")
#                         point_count += 1
                        
#             except Exception as e:
#                 st.error(f"Error parsing details: {str(e)}")
            
#             # Display time taken
#             st.write(f"⏲️ **Time Taken**: {elapsed:.2f} seconds")
#             st.caption(f"Model: `{MODEL}` | Prompt: `{prompt_style}` | Temperature: {temp}")

            