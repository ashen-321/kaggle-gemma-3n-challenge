import base64
import requests
from io import BytesIO
import cv2
import boto3
import os
import ffmpeg
import json
import time
import sys
import io

#from langchain.vectorstores import Chroma
import random
from PIL import Image
import xml.etree.ElementTree as ET
import concurrent.futures

region = 'us-west-2'
module_paths = ["./", "./configs"]
for module_path in module_paths:
    sys.path.append(os.path.abspath(module_path))
    
from utils import  classify_query

def get_bedrock_client(region):
    bedrock_client = boto3.client("bedrock-runtime", region_name=region)
    return bedrock_client

def get_embedding(image_base64=None, text_description=None):
    input_data = {}

    if image_base64 is not None:
        input_data["inputImage"] = image_base64
    if text_description is not None:
        input_data["inputText"] = text_description

    if not input_data:
        raise ValueError("At least one of image_base64 or text_description must be provided")

    body = json.dumps(input_data)

    response = boto3.client(service_name="bedrock-runtime").invoke_model(
        body=body,
        modelId="amazon.titan-embed-image-v1",
        accept="application/json",
        contentType="application/json"
    )

    response_body = json.loads(response.get("body").read())
    return response_body.get("embedding")

def convert_image_to_base64(BytesIO_image):
    # Convert the image to RGB (optional, depends on your requirements)
    rgb_image = BytesIO_image.convert('RGB')
    # Prepare the buffer
    buffered = BytesIO()
    # Save the image to the buffer
    rgb_image.save(buffered, format="JPEG")
    # Get the byte data
    img_data = buffered.getvalue()
    # Encode to base64
    base64_encoded = base64.b64encode(img_data)
    return base64_encoded.decode('utf-8')
    
def bedrock_get_img_description(model_id, prompt, image, max_token:int=2048, temperature:float=0.01, top_p:float=0.90, top_k:int=40, stop_sequences="\n\nHuman"):
    stop_sequence = [stop_sequences]
    #encoded_string = base64.b64encode(image)
    if isinstance(image, io.BytesIO):
        image = Image.open(image)

    # Resize to image resolution by half to make sure input to Claude 3 is < 5M
    width, height = image.size
    if width > 2048 or height > 2048:
        new_size = (int(width/2), int(height/2))
        image = image.resize(new_size, Image.Resampling.LANCZOS) # or Image.ANTIALIAS for Pillow < 7.0.0
        
    #base64_string = encoded_string.decode('utf-8')
    payload = {
        "modelId": model_id,
        "contentType": "application/json",
        "accept": "application/json",
        "body": {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_token,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            #"stop_sequences": stop_sequence,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": 'image/png', #get_image_type(image),
                                "data": convert_image_to_base64(image)
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        }
    }
    
    # Convert the payload to bytes
    body_bytes = json.dumps(payload['body']).encode('utf-8')
    
    # Invoke the model
    response = get_bedrock_client(region).invoke_model(
        body=body_bytes,
        contentType=payload['contentType'],
        accept=payload['accept'],
        modelId=payload['modelId']
    )
    
    # Process the response
    response_body = response['body'].read().decode('utf-8')
    data = json.loads(response_body)
    return data['content'][0]['text']

def openai_image_getDescription(option, prompt, image, max_output_tokens, temperature, top_p):
    openai_api_key = os.getenv("openai_api_token")
    headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {openai_api_key}"
    }
    if isinstance(image, io.BytesIO):
        image = Image.open(image)
    payload = {
      "model": option,
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": prompt 
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{convert_image_to_base64(image)}"
              }
            }
          ]
        }
      ],
      "max_tokens": max_output_tokens,
      "temperature": temperature,
      "top_p": top_p,
    }
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        return response.json()['choices'][0]['message']['content'].replace("\n", "",2)
    except:
        return f"API to {model_name} has been denied!"

def videoCaptioning_claude_nn(option, prompt, b64frames, max_token, temperature, top_p, top_k):
    ## Take random 20 images due to Claude 3 limit
    #samples = 20 if len(base64Frames) > 20 else  len(base64Frames)
    #base64Frames_20 = random.sample(base64Frames, samples)

    captions = ''
    tokens = 0
    for i in range(0, len(b64frames), 20):
        samples = 20 if len(b64frames[i: i+20]) >= 20 else  len(b64frames[i: i+20])
        base64Frames_20 = random.sample(b64frames[i: i+20], samples)
    
        # Get resolution to compute tokens
        image_bytes = base64.b64decode(base64Frames_20[0])
        # Create a BytesIO object from the decoded image bytes
        image_buffer = BytesIO(image_bytes)
        # Open the image using PIL
        width, height = Image.open(image_buffer).size
        video_tokens = int((height * width)/750)* samples
        
        payload = {
            "modelId": option,
            "contentType": "application/json",
            "accept": "application/json",
            "body": {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_token,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                #"stop_sequences": stop_sequence,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                        *map(lambda x: {
                            "type": "image",
                            "source": {
                                    "type": "base64",
                                    "media_type": 'image/png', #get_image_type(image),
                                     "data": f'{x}'}}, base64Frames_20),
                        {
                            "type": "text",
                            "text": prompt
                        }
                        ]
                    }
                ]
            }
        }
        # Convert the payload to bytes
        body_bytes = json.dumps(payload['body']).encode('utf-8')
        
        # Invoke the model
        response = get_bedrock_client(region).invoke_model(
            body=body_bytes,
            contentType=payload['contentType'],
            accept=payload['accept'],
            modelId=payload['modelId']
        )
        
        # Process the response
        response_body = response['body'].read().decode('utf-8')
        data = json.loads(response_body)
        captions += data['content'][0]['text']
        tokens += video_tokens
    return captions, tokens
    
def getBase64Frames(video_file_name):
    video = cv2.VideoCapture(video_file_name)
    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    return base64Frames

def process_video(video_path, seconds_per_frame=2):
    base64Frames = []
    base_video_path, _ = os.path.splitext(video_path)

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps * seconds_per_frame)
    curr_frame=0

    # Loop through the video and extract frames at specified sampling rate
    while curr_frame < total_frames - 1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip
    video.release()

    # Extract audio from video
    '''
    audio_path = f"{base_video_path}.mp3"
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, bitrate="32k")
    clip.audio.close()
    clip.close()

    print(f"Extracted {len(base64Frames)} frames")
    print(f"Extracted audio to {audio_path}")
    '''
    video2 = ffmpeg.input(video_path)
    audio_file = f"{base_video_path}.mp3"
    if os.path.exists(audio_file):
        os.remove(audio_file)
    # Extract the audio stream
    audio = video2.audio
    # Save the audio as an MP3 file
    try:
        ffmpeg.output(audio, audio_file).run()
    except ffmpeg.Error as e:
        with open(audio_file, 'wb') as f:
             pass
    
    return base64Frames, audio_file

def get_asr(audio_filename):
    file_size = os.path.getsize(audio_filename)
    if file_size == 0:
        return 'No audio.'
    # Set the API endpoint
    #url = 'http://infs.cavatar.info:8081/asr?task=transcribe&encode=true&output=txt'
    url = 'http://video.cavatar.info:8082/generate'
    # Define headers
    headers = {
        'Accept': 'application/json',
        #'Content-Type': 'multipart/form-data'
    }

    # Define the file to be uploaded
    files = {
        'audio_file': (audio_filename, open(audio_filename, 'rb'), 'audio/mpeg')
    }

    # Make the POST request
    response = requests.post(url, headers=headers, files=files)
    #output = response.text.rstrip() # if using infs.cavatar.info:8081
    if response.status_code==200:
        output =json.dumps(response.json(), indent=3).replace('"', '')
        return output
    else:
        return ""


'''
def bedrock_textGen(model_id, prompt, max_tokens, temperature, top_p, top_k, stop_sequences):
    stop_sequence = [stop_sequences]
    if  "anthropic.claude-v2" in model_id.lower():
        inference_modifier = {
            "max_tokens_to_sample": max_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "stop_sequences": stop_sequence,
        }
    
        textgen_llm = Bedrock(
            model_id=model_id,
            client=boto3_bedrock,
            model_kwargs=inference_modifier,
        )     
        return textgen_llm(prompt)
    elif "anthropic.claude-3" in model_id.lower():
        payload = {
            "modelId": model_id,
            "contentType": "application/json",
            "accept": "application/json",
            "body": {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                #"stop_sequences": stop_sequence,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            }
                        ]
                    }
                ]
            }
        }
        
        # Convert the payload to bytes
        body_bytes = json.dumps(payload['body']).encode('utf-8')
        # Invoke the model
        response = get_bedrock_client(region).invoke_model(
            body=body_bytes,
            contentType=payload['contentType'],
            accept=payload['accept'],
            modelId=payload['modelId']
        )
        
        # Process the response
        response_body = response['body'].read().decode('utf-8')
        data = json.loads(response_body)
        return data['content'][0]['text']
    
    else:
        return f"Incorrect Bedrock model ID {model_id.lower()} selected!"
'''
### Claude 3 promt with xml style
def xml_prompt(caption:str, transcribe:str, query:str):
    # Create the root element
    root = ET.Element('prompt')
    
    # Add the task element
    task = ET.SubElement(root, 'task')
    if "question-and-answer" in classify_query(query, 'question-and-answer, video-caption, others', 'anthropic.claude-3-haiku-20240307-v1:0'):
        task.text = 'Analyze the provided {aggregated_caption} and {audio_transcribe} in detail and provide truthful and accurate answer to {input_query} with comprehensive, vivid detail that truthfully captures the figures, actions, and overall sentiment of the scene. Your answer should be consistently, correctly and meticulously crafted to convey the nuances and complexities of the scene, leaving no aspect unexamined.'
    else:
        task.text = 'Analyze the provided {aggregated_caption} and {audio_transcribe} in detail and craft a comprehensive, vivid description that truthfully captures the figures, actions, and overall sentiment of the scene. Strive to paint a rich, immersive picture through your words, drawing the reader into the moment and allowing them to fully experience the depicted events and emotions. Your description should directly address the {input_query} and your answer should be consistently, correctly and meticulously crafted to convey the nuances and complexities of the scene, leaving no aspect unexamined.'
    
    # Add the context element
    context = ET.SubElement(root, 'input_query')
    context.text = query
    context = ET.SubElement(root, 'aggregated_caption')
    context.text = caption
    context = ET.SubElement(root, 'audio_transcribe')
    context.text = transcribe
    
    # Add the output_instructions element
    output_instructions = ET.SubElement(root, 'output_instructions')
    output_instructions.text = "To capture all scene truefully and consistently with detail for a searchable video understanding and indexing."
    
    # Add the persona element
    persona = ET.SubElement(root, 'persona')
    persona.text = 'Write the caption from the perspective of the viewer. Use a conversational and engaging tone to draw the reader into the narrative. Use the word video instead of image.'
    
    # Add the style element
    style = ET.SubElement(root, 'style')
    style.text = 'The caption should have all details including names and numbers as well as a vivid descriptions of the action and its effects. Focus on the context, emotional and narrative aspects of the scene.'
    
    # Convert the XML tree to a string
    xml_string = ET.tostring(root, encoding='utf-8', xml_declaration=True).decode('utf-8')

    return xml_string
    

if __name__ == "__main__":
    start_time = time.time()
    b64frames, audio_file = process_video('/home/alfred/mm/notebooks/data/chiefs-bills-2021.mp4', seconds_per_frame=2)
    ### Process video captioing by loop all keyframes,  20 frames per batch due to Claude 3 limit.
    option = 'anthropic.claude-3-haiku-20240307-v1:0'
    prompt = 'Carefully analyze the sequenced images in detail then craft a detail, comprehensive and vivid description that truthfully captures the figures, actions, and overall sentiment of the scene. Strive to paint a rich, immersive picture through the action, drawing the reader into the moment and allowing them to fully experience the depicted events and emotions. Your description should be correctly and meticulously crafted to convey the nuances and complexities of the scene, leaving no aspect unexamined.'
    with concurrent.futures.ThreadPoolExecutor() as executor:
        answer1 = executor.submit(videoCaptioning_claude_nn, option, prompt, b64frames, 2048, 0., 0.9, 40)
        answer2 =  executor.submit(get_asr, audio_file)
        captions, tokens = answer1.result()
        audio_transcribe = answer2.result()

    prompt2 = xml_prompt(captions, audio_transcribe, prompt)
    msg = bedrock_textGen(option, prompt2, 2048, 0.1, 0.9, 50, "\n\n\n")
    print(f"Caption: {msg}, \n\nAudio: {audio_transcribe}, \n\nTokens: {tokens} \n\n Latency: {(time.time() - start_time) * 1000:.2f} ms")
