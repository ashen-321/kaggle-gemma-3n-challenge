import streamlit as st
import streamlit.components.v1 as components
import sys
import os
import io
import re
import datetime
import json
import random
from PIL import Image
import base64
import time
import hmac
import numpy as np
import logging
from streamlit_pdf_viewer import pdf_viewer

from configs.blog_writer import *
from configs.utility import *
from configs.utils import *
from configs.video_captioning import *
from configs.anthropic_tools import *
from configs.extract_urls import *
from configs.finance_analyzer_01 import *
from configs.nova import *
from configs.search_2_google import *


st.set_page_config(page_title="Gemma-3N Submission", page_icon="🩺", layout="wide")
st.title("Personal assistant")

file_path = "/home/alfred/demos/mmrag/data/"
video_file_name = "uploaded_video.mp4"
temp_audio_file = "audio_input.wav"
o1_sts_role_arn = "arn:aws:iam::905418197933:role/ovg_developer"
o1_region = "us-east-1"
llama33_70b_region = "us-east-2"
voice_prompt = ''
tokens = 0

aoss_host = read_key_value(".aoss_config.txt", "AOSS_host_name")
aoss_index = read_key_value(".aoss_config.txt", "AOSS_index_name")


# Logging handler
# os.environ["STREAMLIT_TRACING_ENABLED"] = "False"
class StreamlitLogHandler(logging.Handler):
    # Initializes a custom log handler with a Streamlit container for displaying logs
    def __init__(self, container):
        super().__init__()
        # Store the Streamlit container for log output
        self.container = container
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')  # Regex to remove ANSI codes
        self.log_area = self.container.empty()  # Prepare an empty conatiner for log output

    def emit(self, record):
        msg = self.format(record)
        clean_msg = self.ansi_escape.sub('', msg)  # Strip ANSI codes
        self.log_area.markdown(clean_msg)

    def clear_logs(self):
        self.log_area.empty()  # Clear previous logs


# Set up logging to capture all info level logs from the root logger
def setup_logging():
    root_logger = logging.getLogger()  # Get the root logger
    log_container = st.container()  # Create a container within which we display logs
    handler = StreamlitLogHandler(log_container)
    handler.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    return handler


def convert_to_markdown(data):
    # Extract the text from the tuple (first element)
    if isinstance(data, tuple):
        text = data[0]  # Get the actual text content
    else:
        text = data

    lines = text.split('\n')
    markdown_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            markdown_lines.append('')
        elif line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.')):
            markdown_lines.append(f"## {line}")
        elif line.startswith('*'):
            markdown_lines.append(f"- {line[1:].strip()}")
        elif 'AgentCore Memory' in line:
            markdown_lines.append(f"# {line}")
        else:
            markdown_lines.append(line)

    return '\n'.join(markdown_lines)


## Encapsulate another web page
def vto_encap_web():
    iframe_src = "https://agent.cavatar.info:7861"
    components.iframe(iframe_src)


# Avoid Overriding of current TracerProvider is not allowed messages with a decorator??

# Display Non_Englis charaters
def print_text():
    return st.session_state.user_input.encode('utf-8').decode('utf-8')
    # print(st.session_state.user_input)


# Password protection
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the passward is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


# Check password
if not check_password():
    st.stop()  # Do not continue if check_password is not True.

# @st.cache_data
# @st.cache_resource(TTL=300)
with st.sidebar:
    # ----- RAG  ------
    st.header(':green[Choose a topic] :eyes:')
    rag_search = rag_update = rag_retrieval = video_caption = image_caption = audio_transcibe = talk_2_pdf = pdf_exist = file_url_exist = blog_writer = image_argmentation = stock_recommend = vto = False
    rag_on = st.select_slider(
        '',
        value='Basic',
        options=['Basic', 'Multimodal', 'Search', 'Files'])  # , 'Insert', 'Retrieval', 'Stock'])
    if 'Search' in rag_on:
        doc_num = st.slider('Choose max number of documents', 1, 4, 2)
        # embedding_model_id = st.selectbox('Choose Embedding Model',('amazon.titan-embed-g1-text-02', 'amazon.titan-embed-image-v1'))
        embedding_model_id = 'amazon.titan-embed-g1-text-02'
        rag_search = True
    elif 'Multimodal' in rag_on:
        image = None
        bytes_data = []
        upload_files = st.file_uploader("Upload your image/video here.", accept_multiple_files=True,
                                        type=["jpg", "png", "webp", "mp4", "mov", "mp3", "wav"])
        image_url = st.text_input("Or Input Image/Video/Audio URL", key="image_url", type="default")
        if upload_files is not None and len(upload_files) > 0:
            # Check if the uploaded file is an image
            _, upload_file_extension = os.path.splitext(upload_files[0].name)
            if upload_file_extension in [".mp3", ".wav"]:
                audio_bytes = upload_files[0].read()
                with open(temp_audio_file, 'wb') as audio_file:
                    audio_file.write(audio_bytes)
                st.audio(audio_bytes, format="audio/wav")
                audio_transcibe = True
            elif upload_file_extension in [".jpg", ".png", ".webp"]:
                for file in upload_files:
                    file_bytes = file.read()
                    bytes_data.append(file_bytes)
                    image = (io.BytesIO(file_bytes))
                    st.image(image)

                image_caption = True
            elif upload_file_extension in [".mp4", ".mov"]:
                # Check if the uploaded file is a video
                video_bytes = upload_files[0].getvalue()
                with open(video_file_name, 'wb') as f:
                    f.write(video_bytes)
                st.video(video_bytes)
                video_caption = True
        elif len(image_url) > 4:
            response = requests.get(image_url, stream=True)
            if response.status_code == 200 and 'audio' in response.headers.get('Content-Type'):
                with open(temp_audio_file, 'wb') as audio_file:
                    for chunk in response.iter_content(chunk_size=1024):
                        audio_file.write(chunk)
                st.audio(response.content, format="audio/wav")
                audio_transcibe = True
            elif response.status_code == 200 and 'image' in response.headers.get('Content-Type'):
                image_data = response.content
                bytes_data.append(image_data)
                # Convert the image data to a BytesIO object
                image = io.BytesIO(image_data)
                st.image(image)
                image_caption = True
            elif response.status_code == 200 and 'video' in response.headers.get('Content-Type'):
                video_bytes = response.content
                with open(video_file_name, 'wb') as f:
                    f.write(video_bytes)
                st.video(video_bytes)
                video_caption = True
        else:
            image_argmentation = True
    elif 'Files' in rag_on:
        upload_docs = st.file_uploader("Upload your pdf/doc/txt files.", accept_multiple_files=True,
                                       type=["pdf", "csv", "json", "txt", "xml"])
        file_url = st.text_input("Or an input URL", key="file_urls", type="default")
        fnames = []
        # embedding_model_id = st.selectbox('Choose Embedding Model',('amazon.titan-embed-g1-text-02', 'amazon.titan-embed-image-v1'))
        embedding_model_id = 'amazon.titan-embed-g1-text-02'
        if upload_docs is not None and len(upload_docs) > 0:
            try:
                # pdf
                empty_directory(file_path)
                upload_doc_names = [file.name for file in upload_docs]
                for upload_doc in upload_docs:
                    bytes_data = upload_doc.read()
                    fnames.append(upload_doc.name)
                    full_filename = file_path + upload_doc.name
                    if os.path.isfile(full_filename):
                        pdf_exist = True
                    else:
                        with open(full_filename, 'wb') as f:
                            f.write(bytes_data)
                talk_2_pdf = True
                if is_pdf(full_filename):
                    pdf_viewer(input=bytes_data, width=1200)
                elif 'json' in upload_doc.name and isinstance(bytes_data, bytes):
                    string_data = bytes_data.decode('utf-8')
                    json_data = json.loads(string_data)
                    st.json(json_data)
                else:
                    st.write(bytes_data[:1000] + "......".encode())
            except:
                pass
        elif file_url is not None and len(file_url) > 4:
            file_url = file_url.rstrip()
            try:
                response = requests.get(file_url)
                st.write(f"Extracted: {len(response.content)} bytes")
            except:
                pass
            #    pdf_viewer(input=pdf_bytes, width=1200)
            file_url_exist = True

    # ----- Choose models  ------
    st.divider()
    st.title(':orange[Model Config] :pencil2:')
    if 'Search' in rag_on:
        option = st.selectbox('Choose Model', (
            'gemma-3N-finetune',
            'us.amazon.nova-lite-v1:0',
            'us.amazon.nova-micro-v1:0',
            'us.amazon.nova-pro-v1:0',
            'us.amazon.nova-premier-v1:0',
            'us.deepseek.r1-v1:0',
            'anthropic.claude-3-5-haiku-20241022-v1:0',
            'us.anthropic.claude-3-7-sonnet-20250219-v1:0',
            'us.anthropic.claude-sonnet-4-20250514-v1:0',
            'us.anthropic.claude-opus-4-20250514-v1:0',
            'Qwen/Qwen3-30B-A3B',  # qwen3-30b vllm
        ))
    elif 'Multimodal' in rag_on:
        option = st.selectbox('Choose Model', (
            'gemma-3N-finetune',
            'us.amazon.nova-lite-v1:0',
            'us.amazon.nova-pro-v1:0',
            'us.amazon.nova-premier-v1:0',
            'us.anthropic.claude-3-7-sonnet-20250219-v1:0',
            'us.anthropic.claude-sonnet-4-20250514-v1:0',
            'us.anthropic.claude-opus-4-20250514-v1:0',
            'anthropic.claude-3-haiku-20240307-v1:0',
            'flux1.dev.outfit',
        ))
    elif 'Files' in rag_on:
        option = st.selectbox('Choose Model', (
            'us.amazon.nova-lite-v1:0',
            'us.amazon.nova-micro-v1:0',
            'us.amazon.nova-pro-v1:0',
            'us.amazon.nova-premier-v1:0',
            'us.deepseek.r1-v1:0',
            'anthropic.claude-3-5-haiku-20241022-v1:0',
            'us.anthropic.claude-3-7-sonnet-20250219-v1:0',
            'us.anthropic.claude-sonnet-4-20250514-v1:0',
            'us.anthropic.claude-opus-4-20250514-v1:0',
            'anthropic.claude-3-5-sonnet-20241022-v2:0',
            'Qwen/Qwen3-30B-A3B'  # qwen3-30b vllm
        ))
    elif 'Blog' in rag_on:
        option = st.selectbox('Choose Model', ('anthropic.claude-3-5-haiku-20241022-v1:0',
                                               'us.anthropic.claude-sonnet-4-20250514-v1:0',
                                               'us.anthropic.claude-opus-4-20250514-v1:0',
                                               'meta.llama3-1-70b-instruct-v1:0',
                                               'mistral.mistral-large-2407-v1:0',
                                               ))
    else:
        option = st.selectbox('Choose Model', (
            'gemma-3N-finetune',
            'us.amazon.nova-lite-v1:0',
            'us.amazon.nova-micro-v1:0',
            'us.amazon.nova-pro-v1:0',
            'us.amazon.nova-premier-v1:0',
            'us.deepseek.r1-v1:0',
            'us.anthropic.claude-3-7-sonnet-20250219-v1:0',
            'us.anthropic.claude-sonnet-4-20250514-v1:0',
            'us.anthropic.claude-opus-4-20250514-v1:0',
            'anthropic.claude-3-5-haiku-20241022-v1:0',
            'meta.llama3-3-70b-instruct-v1:0',
            'Qwen/Qwen3-30B-A3B',  # qwen3-30b vllm
        ))

    # if 'Basic' in rag_on or 'Files' in rag_on or 'Multimodal' in rag_on:
    if 'Multimodal' not in rag_on or 'Blog' in rag_on or 'Stock' not in rag_on:
        st.write("------- Default parameters ----------")
        temperature = st.number_input("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
        max_token = st.number_input("Maximum Output Token", min_value=0, value=2048, step=64)
        top_p = st.number_input("Top_p: The cumulative probability cutoff for token selection", min_value=0.1,
                                value=0.85)
        top_k = st.number_input("Top_k: Sample from the k most likely next tokens at each step", min_value=1, value=20)
        # candidate_count = st.number_input("Number of generated responses to return", min_value=1, value=1)
        stop_sequences = st.text_input("The set of character sequences (up to 5) that will stop output generation",
                                       value="\n\n\nHuman")

    # --- Audio query -----#
    st.divider()
    st.header(':green[Enable voice input]')  # :microphone:')
    record_audio_bytes = st.audio_input("Toggle mic to start/stop recording")
    if record_audio_bytes:
        # st.audio(record_audio_bytes, format="audio/wav")#, start_time=0, *, sample_rate=None)
        with open(temp_audio_file, 'wb') as audio_file:
            audio_file.write(record_audio_bytes.getvalue())
        if os.path.exists(temp_audio_file):
            voice_prompt = get_asr(temp_audio_file).encode('utf-8').decode('unicode_escape')
            voice_prompt = "" if voice_prompt.lower() in ['please stop audio.', 'stop audio.'] else voice_prompt
    if 'Basic' not in rag_on:
        st.caption("Press space bar and hit ↩️ for activation")

    # ---- Clear chat history ----
    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages.clear()
        st.session_state["messages"] = [{"role": "assistant", "content": "I am your assistant. How can I help today?"}]
        record_audio = None
        voice_prompt = ""
        # del st.session_state[record_audio]

###
# Streamlist Body
###
start_time = time.time()
non_streaming = True
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "I am your assistant. How can I help today?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

#
if rag_search:
    if prompt := st.chat_input(placeholder=voice_prompt, on_submit=None, key="user_input"):
        prompt = voice_prompt if prompt == ' ' else prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        ## Tavily only        
        today = datetime.datetime.today().strftime("%D")
        prompt = f"{prompt}. The date today is {today}"
        if doc_num > 3:
            docs = tavily_search(prompt, num_results=doc_num)
            documents = docs['documents']
            urls = docs['urls'][0:doc_num]
        else:
            documents, urls = search_2_google(prompt, int(doc_num / 2) + 1)
        # documents, urls = all_search(prompt=prompt, num_results=int(doc_num/2)+1)

        if 'claude' in option:
            msg = retrieval_faiss(prompt, documents, option, embedding_model_id, 4000, 400, max_token, temperature,
                                  top_p, top_k, doc_num)
            # msg = retrieval_faiss(prompt, documents, option, embedding_model_id, 6000, 600, max_token, temperature, top_p, top_k, doc_num)
        elif 'deepseek' in option.lower():
            # msg = st.write_stream(retrieval_faiss_dsr1(prompt, documents, option, embedding_model_id, 6000, 600, max_token, temperature, top_p, top_k, doc_num))
            msg = retrieval_faiss_dsr1(prompt, documents, option, embedding_model_id, 2000, 200, max_token, temperature,
                                       top_p, top_k, doc_num)
            # msg_footer = f"Powered by deepseek"
        else:
            msg = retrieval_o1(prompt, documents, option, embedding_model_id, 4000, 400, max_token, temperature, top_p,
                               top_k, doc_num, role_arn=o1_sts_role_arn, region=o1_region)

        msg_footer = f"{msg}\n\n ✧***Sources:***\n\n" + '\n\n\r'.join(urls[0:doc_num])
        msg_footer += f"\n\n ✒︎***Content created by using:*** {option}, Latency: {(time.time() - start_time) * 1000:.2f} ms, Tokens In: {estimate_tokens(prompt, method='max')}, Out: {estimate_tokens(msg, method='max')}"
        st.session_state.messages.append({"role": "assistant", "content": msg_footer})
        st.chat_message("ai", avatar='👁️‍🗨️').write(msg_footer)
        if msg is not None and len(msg) > 2:
            st.audio(get_polly_tts(msg))

elif video_caption:
    if prompt := st.chat_input(placeholder=voice_prompt, on_submit=None, key="user_input"):
        prompt = voice_prompt if prompt == ' ' else prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        if 'claude' in option:
            b64frames, audio_file = process_video(video_file_name, seconds_per_frame=2)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                answer1 = executor.submit(videoCaptioning_claude_nn, option, prompt, b64frames, max_token, temperature,
                                          top_p, top_k)
                answer2 = executor.submit(get_asr, audio_file)
                captions, tokens = answer1.result()
                audio_transcribe = answer2.result()

            prompt2 = xml_prompt(captions, audio_transcribe, prompt)
            msg = bedrock_textGen(option, prompt2, max_token, temperature, top_p, top_k, stop_sequences)
        else:
            _, audio_file = process_video(video_file_name, seconds_per_frame=2)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                answer1 = executor.submit(o1_video, option, prompt, video_file_name, max_token, temperature, top_p,
                                          top_k, role_arn=o1_sts_role_arn, region=o1_region)
                answer2 = executor.submit(get_asr, audio_file)
                captions, tokens = answer1.result()
                audio_transcribe = answer2.result()

            prompt2 = xml_prompt(captions, audio_transcribe, prompt)
            msg = nova_textGen(option, prompt2, max_token, temperature, top_p, top_k, role_arn=o1_sts_role_arn,
                               region_name=o1_region)

        msg += "\n\n 🔊***Audio transcribe:*** " + audio_transcribe + "\n\n ✒︎***Content created by using:*** " + option + f", Latency: {(time.time() - start_time) * 1000:.2f} ms" + f", Tokens In: {tokens}+{estimate_tokens(prompt, method='max')}, Out: {estimate_tokens(msg, method='max')}"
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("ai", avatar='🎥').write(msg)
        # Ouptut TTS
        if msg is not None and len(msg) > 2:
            st.audio(get_polly_tts(msg))

###
# Audio
###
elif audio_transcibe:
    if prompt := st.chat_input(placeholder=voice_prompt, on_submit=None, key="user_input"):
        prompt = voice_prompt if prompt == ' ' else prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        asr_text = get_asr(temp_audio_file)
        prompt2 = f"{prompt}. Your answer should be strictly based on the context in {asr_text}."
        if 'nova' in option.lower():
            msg = nova_textGen(option, prompt2, max_token, temperature, top_p, top_k, role_arn=o1_sts_role_arn,
                               region=o1_region)
        else:
            msg = str(bedrock_textGen(option, prompt2, max_token, temperature, top_p, top_k, stop_sequences))

        msg_footer = f"{msg}\n\n ✒︎***Content created by using:*** {option}, Latency: {(time.time() - start_time) * 1000:.2f} ms, Tokens In: {estimate_tokens(prompt, method='max')}, Out: {estimate_tokens(msg, method='max')}"
        st.session_state.messages.append({"role": "assistant", "content": msg_footer})
        st.chat_message("ai", avatar='🔊').write(msg_footer)
        # Ouptut TTS
        if msg is not None and len(msg) > 2:
            st.audio(get_polly_tts(msg))
###
# Image
###
elif image_caption or image_argmentation:
    if prompt := st.chat_input(placeholder=voice_prompt, on_submit=None, key="user_input"):
        prompt = voice_prompt if prompt == ' ' else prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        if 'flux1' in option.lower():
            action = 'image generation'
        else:
            action = classify_query2(prompt, 'anthropic.claude-3-haiku-20240307-v1:0')

        if 'upscale' in action.lower():
            try:
                new_image = upscale_image_bytes(bytes_data, prompt)
                st.image(new_image, output_format="png", use_container_width='auto')
                msg_footer = "\n\n ✒︎***Content created by using:*** Aura V2 " + f", Latency: {(time.time() - start_time) * 1000:.2f} ms"
            except:
                msg_footer = "Server timeout. Please check image format and size and retry. " + f", Latency: {(time.time() - start_time) * 1000:.2f} ms"
                pass
        elif 'background' in action.lower():  # and hasattr(locals(), 'bytes_data'):
            try:
                option = 'amazon.titan-image-generator-v2:0'
                base64_str = bedrock_image_processing(option, prompt, action, iheight=1024, iwidth=1024,
                                                      src_image=bytes_data, color_string=None, image_quality='premium',
                                                      image_n=1, cfg=7.5, seed=random.randint(100, 500000))
                new_image = Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
                st.image(new_image, output_format="png", use_container_width='auto')
                msg_footer = "\n\n ✒︎***Content created by using:*** " + option + f", Latency: {(time.time() - start_time) * 1000:.2f} ms"
            except:
                msg_footer = "Image background removal failed. Make sure the image does not contain sensitive info." + f", Latency: {(time.time() - start_time) * 1000:.2f} ms"
                pass
        # elif 'segmentation' in action.lower():
        #    try:
        #       #new_image = pred_image_bytes(bytes_data, prompt)
        #        new_image, new_mask = seg_image(bytes_data, prompt)
        #        st.image(new_image, output_format="png", use_container_width='auto')
        #        st.image(new_mask, output_format="png", use_container_width='auto')
        #        msg = "\n\n ✒︎***Content created by using:*** EVF-SAM2 " + f", Latency: {(time.time() - start_time) * 1000:.2f} ms" 
        #    except:
        #        msg = "SAM2 server timeout. Please check image format and size and retry. " + f", Latency: {(time.time() - start_time) * 1000:.2f} ms" 
        #        pass
        elif 'to_be_implemented_conditioning' in action.lower():
            try:
                option = 'amazon.titan-image-generator-v2:0'
                base64_str = bedrock_image_processing(option, prompt, action, iheight=1024, iwidth=1024,
                                                      src_image=bytes_data, color_string=None, image_quality='premium',
                                                      image_n=1, cfg=7.5, seed=random.randint(100, 500000))
                new_image = Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
                st.image(new_image, output_format="png", use_container_width='auto')
                msg_footer = "\n\n ✒︎***Image conditioned by using:*** " + option + f", Latency: {(time.time() - start_time) * 1000:.2f} ms"
            except:
                msg_footer = "Image conditioning failed. Make sure the image does not contain sensitive info." + f" Latency: {(time.time() - start_time) * 1000:.2f} ms"
                pass
        elif 'image generation' in action.lower():
            msg = ''
            if 'titan' in prompt.lower():
                option = 'amazon.titan-image-generator-v2:0'
                base64_str = bedrock_imageGen(option, prompt, iheight=1024, iwidth=1024, src_image=None,
                                              image_quality='premium', image_n=1, cfg=random.uniform(3.2, 9.0),
                                              seed=random.randint(0, 500000))
                new_image = Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
                st.image(new_image, output_format="png", use_container_width='auto')
            elif 'sd3' in prompt.lower() or 'stable diffusion' in prompt.lower() or 'ultra' in prompt.lower():
                option = 'stability.stable-image-ultra-v1:0'  # 'SD3 Medium'
                # url = "http://infs.cavatar.info:8083/generate?prompt="
                # new_image = gen_photo_bytes(prompt, url)
                base64_str = bedrock_imageGen(option, prompt, iheight=1024, iwidth=1024, src_image=None,
                                              image_quality='premium', image_n=1, cfg=random.uniform(3.2, 9.0),
                                              seed=random.randint(0, 500000))
                new_image = Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
                st.image(new_image, output_format="png", use_container_width='auto')
            elif 'flux' in option.lower():
                # old_prompt = f"rephrase the following sentences to remove 'using flux xxx model to generate an imag' only and keep others as is': {prompt}"
                # option = 'flux.1.dev' #'stability.stable-diffusion-xl-v1:0' # Or 'amazon.titan-image-generator-v1'
                # new_prompt = str(bedrock_textGen(option, prompt, max_token, temperature, top_p, top_k, stop_sequences))
                url = "http://video.cavatar.info:8080/generate?prompt="
                new_image = gen_photo_bytes(prompt, url)
                st.image(new_image, output_format="png", use_container_width='auto')
            elif 'alpha' in prompt.lower():
                url = "http://video.cavatar.info:8085/generate?prompt="
                old_prompt = f"rephrase this for text to image generate and drop 'using xxx model': {prompt}"
                option = 'flux.1.alpha'  # 'stability.stable-diffusion-xl-v1:0' # Or 'amazon.titan-image-generator-v1'
                new_prompt = str(
                    bedrock_textGen(option, old_prompt, max_token, temperature, top_p, top_k, stop_sequences))
                new_image = gen_photo_bytes(new_prompt, url)
                st.image(new_image, output_format="png", use_container_width='auto')
            else:
                option = 'amazon.nova-canvas-v1:0'
                neg_prompt = "Bad anatomy, Bad proportions, Deformed, Disconnected limbs, Disfigured, Worst quality, Normal quality, Low quality, Low res, Blurry, Jpeg artifacts, Grainy."
                image_n = top_k if top_k < 5 else 1
                if image:
                    new_images = i2i_canvas(prompt, image, option, neg_prompt=neg_prompt, num_image=image_n,
                                            region_name=o1_region)
                else:
                    new_images = t2i_canvas(prompt, option, neg_prompt=neg_prompt, num_image=image_n,
                                            region_name=o1_region)
                if len(new_images) < 1:
                    msg = "Make sure your prompt meets guardrail requirements."
                for new_image in new_images:
                    st.image(new_image, output_format="png", use_container_width='auto')

            msg_footer = f"{msg}\n\n ✒︎***Content created by using:*** {option}, Latency: {(time.time() - start_time) * 1000:.2f} ms"
        elif 'video generation' in action.lower():
            if 't2v' in prompt.lower():
                option = 'T2V-Turbo_v2'
                url = "http://video.cavatar.info:8083/video_generate?prompt="
                response = requests.post(url + prompt)
                if response.status_code == 200:
                    mp4_bytes = response.content
                    st.video(mp4_bytes)
                    msg_footer = "\n\n ✒︎***Content created by using:*** " + option + f", Latency: {(time.time() - start_time) * 1000:.2f} ms"
                else:
                    msg_footer = f"Error generating video from {url}"
            elif 'luma' in prompt.lower():
                option = 'luma.ray-v2:0'
                file_name = t2v_luma(video_prompt=prompt, model_id=option, role_arn=o1_sts_role_arn, v_length=6,
                                     region="us-west-2")
                with open(file_name, 'rb') as file:
                    mp4_bytes = file.read()
                    st.video(mp4_bytes)
                    msg_footer = "\n\n ✒︎***Content created by using:*** " + option + f", Latency: {(time.time() - start_time) * 1000:.2f} ms"
            else:
                option = 'amazon.nova-reel-v1:0'
                file_name = t2v_reel(video_prompt=prompt, model_id=option, role_arn=o1_sts_role_arn, v_length=6,
                                     region=o1_region)
                with open(file_name, 'rb') as file:
                    mp4_bytes = file.read()
                    st.video(mp4_bytes)
                    msg_footer = "\n\n ✒︎***Content created by using:*** " + option + f", Latency: {(time.time() - start_time) * 1000:.2f} ms"
        elif 'music generation' in action.lower():
            option2 = 'MusicGen-Large'
            url = "http://video.cavatar.info:8084/music_generate?prompt="
            music_prompt = prompt + '&length=' + str(top_k)
            response = requests.post(url + music_prompt)
            if response.status_code == 200:
                audio_tensor_restored = np.load(io.BytesIO(response.content))
                st.audio(audio_tensor_restored, format="audio/wav", sample_rate=32000)
                msg_footer = "\n\n ✒︎***Content created by using:*** " + option2 + f", Latency: {(time.time() - start_time) * 1000:.2f} ms"
            else:
                msg_footer = f"Error generating music from {url} Latency: {(time.time() - start_time) * 1000:.2f} ms"
        elif 'virtual try-on' in action.lower():
            msg = "Click to launch virtual try-on [demo](http://video.cavatar.info:7861)"
            msg_footer = f"{msg}\n\n ✒︎***Content created by using:*** [IDM-VTON](https://github.com/yisol/IDM-VTON)"
        else:
            if "claude" in option.lower() and image is not None:
                # msg = anthropic_imageCaption(option, prompt, image, max_token, temperature, top_p, top_k)
                msg = bedrock_get_img_description(option, prompt, image, max_token, temperature, top_p, top_k,
                                                  stop_sequences)
            elif 'gemma-3n' in option.lower() and image is not None:
                msg = convert_to_markdown(local_gemma3n_image(option, prompt, image, max_token, ))

                msg_footer = f"{msg}\n\n ✒︎***Content created by using:*** {option}, Latency: {(time.time() - start_time) * 1000:.2f} ms"
            elif 'nova' in option.lower() and len(bytes_data) > 0:
                msg = nova_image(option, prompt, bytes_data, max_token, temperature, top_p, top_k,
                                 role_arn=o1_sts_role_arn, region=o1_region)
                width, height = Image.open(image).size
                tokens = int((height * width) / 750)
            # elif image is not None:
            #    msg = o1_image(option, prompt, image, max_token, temperature, top_p, top_k, role_arn=o1_sts_role_arn, region=o1_region)
            #    width, height = Image.open(image).size
            #    tokens = int((height * width)/750)
            else:
                msg = str(bedrock_textGen(option, prompt, max_token, temperature, top_p, top_k, stop_sequences))
            # width, height = Image.open(image).size
            # tokens = int((height * width)/750)
            msg_footer = f"{msg}\n\n ✒︎***Content created by using:*** {option}, Latency: {(time.time() - start_time) * 1000:.2f} ms"
        st.session_state.messages.append({"role": "assistant", "content": msg_footer})
        st.chat_message("ai", avatar='🖼️').write(msg_footer)
        # if msg is not None and len(msg)> 2:
        #    st.audio(get_polly_tts(msg))
###
# Pdf parser        
###      
elif talk_2_pdf:
    if prompt := st.chat_input(placeholder=voice_prompt, on_submit=None, key="user_input"):
        prompt = voice_prompt if prompt == ' ' else prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # file_path, fnames
        if not pdf_exist:
            xml_texts = ''
            for fname in fnames:
                if is_pdf(os.path.join(file_path, fname)):
                    texts, tables = parser_pdf(file_path, fname)
                    xml_text = parse_pdf_to_xml(file_path + fname)
                    xml_texts += xml_text
                    # tables += table
                else:
                    with open(os.path.join(file_path, fname), 'r') as file:
                        file_content = file.read()
                    xml_texts += file_content

        prompt2 = f"{prompt}. Your answer should be strictly based on the context in {xml_texts}."
        if 'claude' in option:
            msg = bedrock_textGen(option, prompt2, max_token, temperature, top_p, top_k, stop_sequences)
        elif 'qwen' in option.lower():
            msg = local_openai_textGen(option, prompt2, max_token, temperature, top_p, top_k)
        else:
            msg = nova_textGen(option, prompt2, max_token, temperature, top_p, top_k, role_arn=o1_sts_role_arn,
                               region_name=o1_region)
        msg_footer = f"{msg}\n\n ✒︎***Content created by using:*** {option}, Latency: {(time.time() - start_time) * 1000:.2f} ms, Tokens In: {estimate_tokens(prompt2, method='max')}, Out: {estimate_tokens(msg, method='max')}"
        st.session_state.messages.append({"role": "assistant", "content": msg_footer})
        st.chat_message("ai", avatar='🗂️').write(msg_footer)
        # Ouptut TTS
        if msg is not None and len(msg) > 2:
            st.audio(get_polly_tts(msg))
###
# File from urls
###
elif file_url_exist:
    if prompt := st.chat_input(placeholder=voice_prompt, on_submit=None, key="user_input"):
        prompt = voice_prompt if prompt == ' ' else prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        if 'nova' in option.lower():
            msg = extract_urls_o1([file_url], prompt, option, embedding_model_id, max_token, temperature, top_p, top_k,
                                  role_arn=o1_sts_role_arn, region=o1_region)
        else:
            msg = extract_urls([file_url], prompt, option, embedding_model_id)
        msg_footer = f"{msg}\n\n ✒︎***Content created by using:*** {option}, Latency: {(time.time() - start_time) * 1000:.2f} ms, Tokens In: {estimate_tokens(prompt, method='max')}, Out: {estimate_tokens(msg, method='max')}"
        st.session_state.messages.append({"role": "assistant", "content": msg_footer})
        st.chat_message("ai", avatar='🗃️').write(msg_footer)
        # Ouptut TTS
        st.audio(get_polly_tts(msg))
####
# RAG injection
####
elif rag_update:
    # Update AOSS
    if upload_docs:
        empty_directory(file_path)
        upload_doc_names = [file.name for file in upload_docs]
        for upload_doc in upload_docs:
            bytes_data = upload_doc.read()
            with open(file_path + upload_doc.name, 'wb') as f:
                f.write(bytes_data)
        stats, status, kb_id = bedrock_kb_injection(file_path)
        msg = f'Total {stats}  to Amazon Bedrock Knowledge Base: {kb_id}  with status: {status}.' + f", Latency: {(time.time() - start_time) * 1000:.2f} ms"
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("ai", avatar='🖊️').write(msg)

elif rag_retrieval:
    if prompt := st.chat_input(placeholder=voice_prompt, on_submit=None, key="user_input"):
        prompt = voice_prompt if prompt == ' ' else prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Do retrieval
        msg = bedrock_kb_retrieval(prompt, option)
        # msg = bedrock_kb_retrieval_advanced(prompt, option, max_token, temperature, top_p, top_k, stop_sequences)
        # msg = bedrock_kb_retrieval_decomposition(prompt, option, max_token, temperature, top_p, top_k, stop_sequences)
        msg += "\n\n ✒︎***Content created by using:*** " + option + f", Latency: {(time.time() - start_time) * 1000:.2f} ms" + f", Tokens In: {estimate_tokens(prompt, method='max')}, Out: {estimate_tokens(msg, method='max')}"
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("ai", avatar='📈').write(msg)

elif (record_audio_bytes and len(voice_prompt) > 3):
    # if prompt := st.chat_input(placeholder=voice_prompt, on_submit=None, key="user_input"):
    #    prompt=voice_prompt if prompt==' ' else prompt
    # prompt=voice_prompt.encode('utf-8').decode('unicode_escape')
    prompt = voice_prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    # action = classify_query(prompt, 'image generation, image upscaling, news, others', 'anthropic.claude-3-haiku-20240307-v1:0')

    if 'nova' in option.lower():
        msg = st.write_stream(
            nova_textGen_streaming(option, prompt, max_token, temperature, top_p, top_k, role_arn=o1_sts_role_arn,
                                   region_name=o1_region))
    elif 'llama3-3' in option.lower():
        msg = bedrock_textGen_cris(option, prompt, max_token, temperature, top_p, top_k, region_name=llama33_70b_region)
        msg_footer = f"{msg}\n\n ✒︎***Content created by using:*** {option}, Latency: {(time.time() - start_time) * 1000:.2f} ms"
        st.write(msg_footer)
    elif 'qwen' in option.lower():
        st.write_stream(local_openai_textGen_streaming(option, prompt, max_token, temperature, top_p, top_k))
        # msg_footer =f"{msg}\n\n ✒︎***Content created by using:*** {option}, Latency: {(time.time() - start_time) * 1000:.2f} ms, {reasoning_usage}"
        # st.write(msg_footer)
        non_streaming = False
    elif 'gemma-3n' in option.lower():
        option = "alfredcs/gemma-3N-finetune"
        # st.write_stream(local_openai_textGen_streaming(option, prompt, max_token, temperature, top_p, top_k))
        st.write_stream(local_gemma3n_textGen(option, prompt, max_token, temperature, top_p, top_k))
        msg = ''
        non_streaming = True
    elif 'deepseek' in option.lower():
        st.write_stream(deepseek_streaming(option, prompt, max_token, temperature, top_p))
        non_streaming = False
    elif 'claude-3-7' in option.lower() or 'claude-4' in option.lower():
        st.write_stream(bedrock_textGen_thinking_stream(option, prompt, max_token))
        # msg = bedrock_textGen_thinking(option, prompt, max_token)
        non_streaming = False
    else:
        msg = str(bedrock_textGen(option, prompt, max_token, temperature, top_p, top_k, stop_sequences))
        msg_footer = f"{msg}\n\n ✒︎***Content created by using:*** {option}, Latency: {(time.time() - start_time) * 1000:.2f} ms"
        st.write(msg_footer)
        # msg=str(bedrock_textGen_streaming(option, prompt, max_token, temperature, top_p, top_k, stop_sequences))
        # msg =  st.write_stream(bedrock_textGen_streaming(option, prompt, max_token, temperature, top_p, top_k))
    if isinstance(msg, set):
        msg = str(sorted(list(msg)))
    msg_footer = f"{msg}\n\n ✒︎***Content created by using:*** {option}, Latency: {(time.time() - start_time) * 1000:.2f} ms"
    # st.session_state.messages.append({"role": "assistant", "content": msg_footer})
    # st.chat_message("ai", avatar='🎙️').write(msg_footer)
    st.write(msg_footer)
    # Ouptut TTS
    if msg is not None and len(msg) > 2:
        st.audio(get_polly_tts(msg))
elif blog_writer:
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        blog_crew = blogCrew(prompt, option)
        msg = blog_crew.run().raw
        msg_footer = f"{msg}\n\n ✒︎***Content created by using:*** {option}, latency: {(time.time() - start_time) * 1000:.2f} ms"
        st.session_state.messages.append({"role": "assistant", "content": msg_footer})
        st.chat_message("ai", avatar='📝').write(msg_footer)
        # if msg is not None and len(msg)> 2 and non_streaming:
        #    st.audio(get_polly_tts(msg[:1024]))
###
# The rest
###
else:
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        # try:
        # action = classify_query2(prompt, 'anthropic.claude-3-haiku-20240307-v1:0')
        if 'nova' in option.lower():
            msg = nova_textGen(option, prompt, max_token, temperature, top_p, top_k, role_arn=o1_sts_role_arn,
                               region_name=o1_region)
        elif 'deepseek' in option.lower():
            st.write_stream(deepseek_streaming(option, prompt, max_token, temperature, top_p))
            # reasoning_usage = ''
            msg = ''
            non_streaming = False
        elif 'qwen' in option.lower():
            # st.write_stream(local_openai_textGen_streaming(option, prompt, max_token, temperature, top_p, top_k))
            st.write_stream(local_openai_textGen(option, prompt, max_token, temperature, top_p, top_k))
            msg = ''
            non_streaming = True
        elif 'gemma-3n' in option.lower():
            option = "alfredcs/gemma-3N-finetune"
            # st.write_stream(local_openai_textGen_streaming(option, prompt, max_token, temperature, top_p, top_k))
            st.write_stream(local_gemma3n_textGen(option, prompt, max_token, temperature, top_p, top_k))
            msg = ''
            non_streaming = True
        elif 'llama3-3' in option.lower():
            msg = bedrock_textGen_cris(option, prompt, max_token, temperature, top_p, top_k,
                                       region_name=llama33_70b_region)
        elif 'claude' in option.lower():
            msg = bedrock_textGen_thinking(option, prompt, max_token)
        else:
            msg = str(bedrock_textGen(option, prompt, max_token, temperature, top_p, top_k, stop_sequences))
        # except:
        #    msg = "Server error. Check the model access permision"
        #    pass
        if 'o1' in option:
            msg_footer = f"{msg}\n\n ✒︎***Content created by using:*** {option}, Latency: {(time.time() - start_time) * 1000:.2f} ms, Tokens In: {estimate_tokens(prompt, method='max')}, Out: {estimate_tokens(msg, method='max')}, Reasoning tokens: {reasoning_token}"
        elif 'qwen' in option.lower() and not non_streaming:
            msg_footer = f"{msg}\n\n ✒︎***Content created by using:*** {option}, Latency: {(time.time() - start_time) * 1000:.2f} ms"
        else:
            msg_footer = f"{msg}\n\n ✒︎***Content created by using:*** {option}, Latency: {(time.time() - start_time) * 1000:.2f} ms, Tokens In: {estimate_tokens(prompt, method='max')}, Out: {estimate_tokens(msg, method='max')}"
        st.session_state.messages.append({"role": "assistant", "content": msg_footer})
        if len(msg_footer) > 2:
            st.chat_message("ai", avatar='🤵').write(msg_footer)
        # Ouptut TTS
        # if msg is not None and len(msg)> 2 and non_streaming:
        #    st.audio(get_polly_tts(msg))
