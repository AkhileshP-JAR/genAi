import streamlit as st
import time
from PIL import Image

#importing the backend
import helper_id as helper
# st.title("First Gen AI Document bot")

st.set_page_config(page_title="ID Bot", layout="wide")
st.title("Student ID -> ID and Name")

OLLAMA=st.sidebar.text_input("Ollama Server","http://localhost:11434")

MODEL=st.sidebar.text_input("Vision Model","qwen2.5vl:3b")

temp=st.sidebar.slider("Temperature",0.0,1.0,0.2,0.1)
prompt_style=st.sidebar.radio("Prompt Style",["v1(1)"])

left,right=st.columns(2,gap="large")


with left:
    st.title("IMG-> Extract")
    #up is the pdf file 
    up=st.file_uploader("IMG upload",type=["jpg","png","jpeg"])
    if up:
        # extracted_text=helper.read_pdf_text(up)
        img = Image.open(up)
        st.image(img, caption='Uploaded Image.', use_container_width=True)
        # st.success(f"Pdf uploaded succesfully, Length:{len(extracted_text)}")

        with st.expander("PDF text"):
             st.write(img)


#right side cooum of ui
with right :
    st.subheader("Extraction")
    run = st.button("EXTRACT", type="primary", 
    use_container_width=True)
    if run:
        if not run:
            st.warning("Please upload a pdf first")
        else:
            prompt=helper.make_prompt(prompt_style)

            b64 = helper_id.img_to_base64(img)
            with st.spinner(f"Calling model:{MODEL}"):
                t0=time.time()
                ans=helper_id.call_ollama_img(OLLAMA,MODEL, prompt, b64, temp,b64)
                st.write(ans)
                elapsed=time.time()-t0
                st.write(f"** Time Taken:{elapsed:.2f}")

                with st.expander("Prompt"):
                    st.write(prompt)
            if resp:
                st.markdown("### Result")
                st.write(f'{resp}')
                st.write(f"** Time Taken:{elapsed:.2f} seconds")
                st.caption(f"Model: {MODEL} | Prompt Style: {prompt_style} | Temperature: {temp}")
    # if not up:
    #     st.warning("Please upload a pdf first")
    # elif not q.strip():
    #     st.warning("Please ask Question")
    # else:
    #     prompt=helper_id.make_prompt(prompt_style,q,extracted_text)

    #     with st.spinner(f"Calling model:{MODEL}"):
    #         t0=time.time()
    #         ans=helper_id.call_ollama(OLLAMA,MODEL, prompt, temp)
    #         st.write(ans)
    #         elapsed=time.time()-t0
    #         st.write(f"** Time Taken:{elapsed:.2f}")

    #         with st.expander("Prompt"):
    #             st.write(prompt)


