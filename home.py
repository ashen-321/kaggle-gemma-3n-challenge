import streamlit as st
import streamlit.components.v1 as components
import sys
import os
import io
import asyncio
import re
import json
from PIL import Image
import time
import logging
from streamlit_pdf_viewer import pdf_viewer
from configs.utility import *
from configs.utils import *
from configs.video_captioning import get_asr

file_path = os.path.dirname(__file__)
input_file_path = os.path.join(file_path, "input-files")
voice_prompt = ''
tokens = 0
audio_extensions = [".mp3", ".wav"]
image_extensions = [".jpg", ".jpeg", ".png", ".webp"]

os.environ["OPENAI_API_KEY"] = "EMPTY"
os.environ["OPENAI_BASE_URL"] = "http://video.cavatar.info:8087/v1"

# --------------------------------------------------------------------------------------------
# Webpage Setup ------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------

# Variables
aoss_host = read_key_value(".aoss_config.txt", "AOSS_host_name")
aoss_index = read_key_value(".aoss_config.txt", "AOSS_index_name")
input_image_file = "input_image"
input_audio_file = "input_audio"
query_audio_file = "query_audio.wav"
last_uploaded_files = None
model_id = 'gemma-3N-finetune'

st.set_page_config(page_title="Gemma-3N", page_icon="🩺", layout="wide")
st.title("Personal assistant")


# Logger
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


# Encapsulate another web page
def vto_encap_web():
    iframe_src = "https://agent.cavatar.info:7861"
    components.iframe(iframe_src)


# Display Non_English charaters
def print_text():
    return st.session_state.user_input.encode('utf-8').decode('utf-8')


# --------------------------------------------------------------------------------------------
# Sidebar ------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------


with st.sidebar:
    st.header(':green[Settings]')

    # File upload box
    upload_file = st.file_uploader("Upload your image/audio here:", accept_multiple_files=True,
                                   type=["jpg", "jpeg", "png", "webp", "mp3", "wav"])
    file_url = st.text_input("Or input a valid file URL:", key="file_url", type="default")

    # Only update input file directory if something has changed
    if upload_file != last_uploaded_files:
        # File saving
        audio_file_indexes = []
        image_file_indexes = []

        # Clear input file directory
        empty_directory(input_file_path)

        # Save file type indexes
        if upload_file is not None:
            for i in range(len(upload_file)):
                _, upload_file_extension = os.path.splitext(upload_file[i].name)

                if upload_file_extension in audio_extensions:
                    audio_file_indexes.append(i)

                elif upload_file_extension in image_extensions:
                    image_file_indexes.append(i)

        # Read file indexes and save accordingly
        # Audio upload
        for i in range(len(audio_file_indexes)):
            index = audio_file_indexes[i]
            audio_bytes = upload_file[index].read()
            st.audio(audio_bytes, format="audio/wav")

            input_file = os.path.join(input_file_path, input_audio_file + f"_{i}" + upload_file_extension)
            with open(input_file, 'wb') as audio_file:
                audio_file.write(audio_bytes)

        # Image upload
        for i in range(len(image_file_indexes)):
            index = image_file_indexes[i]
            image_bytes = upload_file[index].read()
            st.image(io.BytesIO(image_bytes))

            input_file = os.path.join(input_file_path, input_image_file + f"_{i}" + upload_file_extension)
            with open(input_file, 'wb') as image_file:
                image_file.write(image_bytes)

        last_uploaded_files = upload_file

        # Configuration sliders
        temperature = st.number_input("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
        max_tokens = st.number_input("Maximum Output Tokens", min_value=0, value=4096, step=64)
        top_p = st.number_input("Top_p: The cumulative probability cutoff for token selection", min_value=0.1, value=0.85)

        # --- Audio query -----#
        st.divider()
        st.header(':green[Enable voice input]')
        record_audio_bytes = st.audio_input("Toggle mic to start/stop recording")
        if record_audio_bytes:
            with open(query_audio_file, 'wb') as audio_file:
                audio_file.write(record_audio_bytes.getvalue())
            if os.path.exists(query_audio_file):
                voice_prompt = get_asr(query_audio_file).encode('utf-8').decode('unicode_escape')
                voice_prompt = "" if voice_prompt.lower() in ['please stop audio.', 'stop audio.'] else voice_prompt

        # ---- Clear chat history ----
        st.divider()
        if st.button("Clear Chat History"):
            st.session_state.messages.clear()
            st.session_state["messages"] = [
                {"role": "assistant", "content": "I am your assistant. How can I help today?"}]
            record_audio = None
            voice_prompt = ""

# --------------------------------------------------------------------------------------------
# GUI ----------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------


start_time = time.time()

# Message tracking
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "I am your assistant. How can I help today?"}]

for msg in st.session_state.messages[1:]:
    st.chat_message(msg["role"]).write(msg["content"])

# OpenAI Client
if "openai_client" not in st.session_state:
    st.session_state["openai_client"] = OpenAI()


if prompt := st.chat_input():
    # Override
    if len(voice_prompt):
        prompt = voice_prompt

    # Add prompt to messages
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Generate response
    response = st.session_state["openai_client"].chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system",
             "content": "You are a helpful assistant. Please answer the user question accurately and truthfully. Also please make sure to think carefully before answering"},
            *st.session_state.messages[1:],
        ],
        stream=False,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p
    )
    usages = f'Completion_tokens: {response.usage.completion_tokens}, Prompt_tokens: {response.usage.prompt_tokens}, Total_tokens:{response.usage.total_tokens}'

    # Display text and save to messages
    response_formatted = f"{response}\n\n ✒︎***Content created by using:*** {model_id}, Latency: {(time.time() - start_time) * 1000:.2f} ms, {usages}"
    st.session_state.messages.append({"role": "assistant", "content": response_formatted})
    st.chat_message("ai", avatar='🤵').write(response_formatted)
