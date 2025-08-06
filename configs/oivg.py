import json
import boto3
import botocore
from PIL import Image
import os
import base64
import imghdr
import io
from random import randint
#from datetime import datetime
import datetime
from IPython.display import display
from dateutil.tz import tzlocal
import time
import requests

from langchain import hub
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_aws.embeddings.bedrock import BedrockEmbeddings
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer



retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

def assume_role(account_id, role_name, session_name):
    """
    Assume an IAM role in another account

    Parameters:
    account_id (str): The AWS account ID where the role exists
    role_name (str): The name of the role to assume
    session_name (str): An identifier for the assumed role session

    Returns:
    boto3.Session: A boto3 session with the assumed role credentials
    """
    # Create an STS client
    sts_client = boto3.client('sts')

    # Construct the role ARN
    role_arn = f'arn:aws:iam::{account_id}:role/{role_name}'

    try:
        # Assume the role
        assumed_role_object = sts_client.assume_role(
            RoleArn=role_arn,
            RoleSessionName=session_name
        )

        # Extract the temporary credentials
        credentials = assumed_role_object['Credentials']

        # Create a new session with the temporary credentials
        session = boto3.Session(
            aws_access_key_id=credentials['AccessKeyId'],
            aws_secret_access_key=credentials['SecretAccessKey'],
            aws_session_token=credentials['SessionToken']
        )

        return session

    except Exception as e:
        print(f"Error assuming role: {str(e)}")
        raise

def assumed_role_session(role_arn: str, base_session: botocore.session.Session = None):
    base_session = base_session or boto3.session.Session()._session
    fetcher = botocore.credentials.AssumeRoleCredentialFetcher(
        client_creator = base_session.create_client,
        source_credentials = base_session.get_credentials(),
        role_arn = role_arn,
        extra_args = {
        #    'RoleSessionName': None # set this if you want something non-default
        }
    )
    creds = botocore.credentials.DeferredRefreshableCredentials(
        method = 'assume-role',
        refresh_using = fetcher.fetch_credentials,
        time_fetcher = lambda: datetime.datetime.now(tzlocal())
    )
    botocore_session = botocore.session.Session()
    botocore_session._credentials = creds
    return boto3.Session(botocore_session = botocore_session)

def check_urls(urls: list, kind: str):
    pdf_urls = []
    for url in urls:
        try:
            response = requests.head(url)
            if kind in response.url.lower():
                pdf_urls.append(url)
        except requests.exceptions.RequestException:
            pass
    return pdf_urls

###
# Text-to-image
###
def t2i_olympus(prompt:str, neg_prompt:str, cfgScale:float=7.0, num_image: int=1, width:int=1280, height:int=768, quality: str='premium'  ):
    # Create a new BedrockRuntime client.
    bedrock_runtime = assumed_role_session("arn:aws:iam::905418197933:role/ovg_developer").client(
        "bedrock-runtime",
        region_name="us-east-1",  # You must use us-east-1 during the beta period.
        endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com",
    )

    # Configure the inference parameters.
    inference_params = {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": prompt,
            "negativeText": neg_prompt  # A list of items that should not appear in the image.
        },
        "imageGenerationConfig": {
            "numberOfImages": num_image,  # Number of images to generate, up to 5.
            "width": width,  # See README for supported resolutions.
            "height": height,  # See README for supported resolutions.
            "cfgScale": cfgScale,  # How closely the prompt will be followed.
            "quality": quality,  # "standard" or "premium"
            "seed": randint(
                0, 2147483646
            ),  # Using a random seed value guarantees we get different results each time this code is executed.
        },
    }
    
    # Display the random seed.
    print(f"Generating with seed: {inference_params['imageGenerationConfig']['seed']}")
    
    start_time = datetime.datetime.now()
    
    # Invoke the model.
    response = bedrock_runtime.invoke_model(
        modelId="amazon.olympus-image-generator-v1:0",
        body=json.dumps(inference_params),
    )
    # Convert the JSON-formatted response to a dictionary.
    response_body = json.loads(response["body"].read())
    return_images = []
    try:
        images = response_body["images"]
        for i in range(len(images)):
            image_data = images[i]
            image_bytes = base64.b64decode(image_data)
            #image = Image.open(io.BytesIO(image_bytes))
            return_images.append(Image.open(io.BytesIO(image_bytes)))
    except:
        pass
    return return_images

def download_video_for_invocation_arn(invocation_arn: str, role_arn:str, bucket_name: str="ovg-videos", destination_folder:str="./output"):
    """
    This function downloads the video file for the given invocation ARN.
    """
    invocation_id = invocation_arn.split("/")[-1]

    # Create the local file path
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"{timestamp}_{invocation_id}.mp4"
    import os

    output_folder = os.path.abspath(destination_folder)
    local_file_path = os.path.join(output_folder, file_name)

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Create an STS client
    sts_client = boto3.client('sts')

    # Assume the specified role
    assumed_role = sts_client.assume_role(
        RoleArn=role_arn,
        RoleSessionName='CallBedrockOVG'
    )
    # Extract temporary credentials
    credentials = assumed_role['Credentials']
    
    # Create an S3 client
    s3 = boto3.client("s3",
                        aws_access_key_id=credentials['AccessKeyId'],
                        aws_secret_access_key=credentials['SecretAccessKey'],
                        aws_session_token=credentials['SessionToken'],
                     )

    # List objects in the specified folder
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=invocation_id)
    
    # Find the first MP4 file and download it.
    for obj in response.get("Contents", []):
        object_key = obj["Key"]
        if object_key.endswith(".mp4"):
            print(f"""Downloading "{object_key}"...""")
            s3.download_file(bucket_name, object_key, local_file_path)
            print(f"Downloaded to {local_file_path}")
            return local_file_path

    # If we reach this point, no MP4 file was found.
    print(f"Problem: No MP4 file was found in S3 at {bucket_name}/{invocation_id}")

def t2v_ovg(video_prompt:str, role_arn:str, v_length:int, region: str='us-east-1', s3_destination_bucket:str="ovg-videos"):
    """
    Assume an IAM role and create an S3 bucket

    :param role_arn: ARN of the IAM role to assume
    :param bucket_name: Name of the S3 bucket to create
    :param region: AWS region to create the bucket in
    """
    # Configure the inference parameters.
    model_input = {
        "taskType": "TEXT_VIDEO",  # This is the only task type supported in Beta. Other tasks types will be supported at launch
        "textToVideoParams": {"text": video_prompt},
        "videoGenerationConfig": {
            "durationSeconds": v_length,  # 6 is the only supported value currently.
            "fps": 24,  # 24 is the only supported value currently.
            "dimension": "1280x720",  # "1280x720" is the only supported value currently.
            "seed": randint(
                -2147483648, 2147483648
            ),  # A random seed guarantees we'll get a different result each time this code runs.
        },
    }
    
    # Create an STS client
    sts_client = boto3.client('sts')

    # Assume the specified role
    assumed_role = sts_client.assume_role(
        RoleArn=role_arn,
        RoleSessionName='CallBedrockOVG'
    )

    # Extract temporary credentials
    credentials = assumed_role['Credentials']

    # Call Bedrock
    bedrock_runtime = boto3.client(
        "bedrock-runtime",
        region_name=region,  # You must use us-east-1 during the beta period.
        endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com",
        aws_access_key_id=credentials['AccessKeyId'],
        aws_secret_access_key=credentials['SecretAccessKey'],
        aws_session_token=credentials['SessionToken'],
    )
    invocation_jobs = bedrock_runtime.start_async_invoke_model(
        modelId="amazon.olympus-video-generator-v1:0",
        modelInput=model_input,
        outputDataConfig={"s3OutputDataConfig": {"s3Uri": f"s3://{s3_destination_bucket}"}},
    )
    

    # Check the status of the job until it's complete.
    invocation_arn = invocation_jobs["invocationArn"]
    file_path = ''
    while True:
        invocation_job = bedrock_runtime.get_async_invoke_model(
            invocationArn=invocation_arn
        )
    
        status = invocation_job["status"]
        if status == "InProgress":
            time.sleep(1)
        elif status == "Completed":
            print("\nJob complete!")
            # Save the video to disk.
            s3_bucket = (
                invocation_job["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"]
                .split("//")[1]
                .split("/")[0]
            )
            file_path = download_video_for_invocation_arn(invocation_arn, role_arn, s3_destination_bucket, "./output")
            break
        else:
            print("\nJob failed!")
            print("\nResponse:")
            print(json.dumps(invocation_job, indent=2, default=str))
    return file_path

##
# Text
##
def olympus_textGen(model_id, prompt, max_tokens, temperature, top_p, top_k, role_arn, region='us-east-1'):
    # Create an STS client
    sts_client = boto3.client('sts')

    # Assume the specified role
    assumed_role = sts_client.assume_role(
        RoleArn=role_arn,
        RoleSessionName='CallBedrockOVG'
    )

    # Extract temporary credentials
    credentials = assumed_role['Credentials']

    # Call Bedrock
    bedrock_runtime = boto3.client(
        "bedrock-runtime",
        region_name=region,  # You must use us-east-1 during the beta period.
        endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com",
        aws_access_key_id=credentials['AccessKeyId'],
        aws_secret_access_key=credentials['SecretAccessKey'],
        aws_session_token=credentials['SessionToken'],
    )
    
    request_body = {
        "schemaVersion": "messages-v1",
        "system": [
            {"text": "You are a smart AI assistant whi can answer user questions by providing clear, factual, and dependable information."}
        ],
        "messages": [
            {
                "role": "user",
                "content": [{"text": prompt}]
            },
        ],
        "inferenceConfig": {
            "max_new_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
        }
    }
    
    # Invoke the model and extract the response body.
    response = bedrock_runtime.invoke_model(
        modelId=model_id, #"amazon.Olympus-micro-v1:0",
        body=json.dumps(request_body)
    )
    model_response = json.loads(response["body"].read())
    return model_response["output"]["message"]["content"][0]["text"]

## Streaming
def olympus_textGen_streaming(model_id, prompt, max_tokens, temperature, top_p, top_k, role_arn, region='us-east-1'):
    # Create an STS client
    sts_client = boto3.client('sts')

    # Assume the specified role
    assumed_role = sts_client.assume_role(
        RoleArn=role_arn,
        RoleSessionName='CallBedrockOVG'
    )

    # Extract temporary credentials
    credentials = assumed_role['Credentials']

    # Call Bedrock
    bedrock_runtime = boto3.client(
        "bedrock-runtime",
        region_name=region,  # You must use us-east-1 during the beta period.
        endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com",
        aws_access_key_id=credentials['AccessKeyId'],
        aws_secret_access_key=credentials['SecretAccessKey'],
        aws_session_token=credentials['SessionToken'],
    )
    
    request_body = {
        "schemaVersion": "messages-v1",
        "system": [
            {"text": "You are a smart AI assistant whi can answer user questions by providing clear, factual, and dependable information."}
        ],
        "messages": [
            {
                "role": "user",
                "content": [{"text": prompt}]
            },
        ],
        "inferenceConfig": {
            "max_new_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
        }
    }
    start_time = datetime.datetime.now()
    # Invoke the model and extract the response body.
    response = bedrock_runtime.invoke_model_with_response_stream(
        modelId=model_id, #"amazon.Olympus-micro-v1:0",
        body=json.dumps(request_body)
    )
    
    request_id = response.get("ResponseMetadata").get("RequestId")
    chunk_count = 0
    stream_str = ''
    time_to_first_token = None
    # Process the response stream
    stream = response.get("body")
    if stream:
        for event in stream:
            chunk = event.get("chunk")
            if chunk:
                # Print the response chunk
                chunk_json = json.loads(chunk.get("bytes").decode())
                # Pretty print JSON
                # print(json.dumps(chunk_json, indent=2, ensure_ascii=False))
                content_block_delta = chunk_json.get("contentBlockDelta")
                if content_block_delta:
                    if time_to_first_token is None:
                        time_to_first_token = datetime.datetime.now() - start_time
                        print(f"Time to first token: {time_to_first_token}")
    
                    chunk_count += 1
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")
                    # print(f"{current_time} - ", end="")
                    #stream_str = content_block_delta.get("delta").get("text")
                    yield  content_block_delta.get("delta").get("text")
        #print(f"Total chunks: {chunk_count}")
        #return stream_str
    else:
        yield "No response stream received."

    yield f"\n\n ✒︎***Content created by using:*** {model_id}, latency: {str((datetime.datetime.now() - start_time)).replace('0:00:', '')} sec" # * 1000:.2f} ms"
    #model_response = json.loads(response["body"].read())
    #return model_response["output"]["message"]["content"][0]["text"]

###
# O1 image 
###
def o1_image(option, prompt, image_binary_data, max_token, temperature, top_p, top_k, role_arn, region="us-east-1"):
    # Create an STS client
    sts_client = boto3.client('sts')

    # Assume the specified role
    assumed_role = sts_client.assume_role(
        RoleArn=role_arn,
        RoleSessionName='CallBedrockOVG'
    )

    # Extract temporary credentials
    credentials = assumed_role['Credentials']

    # Call Bedrock
    bedrock_runtime = boto3.client(
        "bedrock-runtime",
        region_name=region,  # You must use us-east-1 during the beta period.
        endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com",
        aws_access_key_id=credentials['AccessKeyId'],
        aws_secret_access_key=credentials['SecretAccessKey'],
        aws_session_token=credentials['SessionToken'],
    )
    
    #Define your system prompt(s).
    system_list = [
        {
            "type": "system",
            "content": [{"text": "You are an expert artist. When the user provides you with an image, provide 3 potential art titles"}],
        }
    ]
    
    # Define a "user" message including both the image and a text prompt.
    content = [
        {
            "text": prompt
        }
    ]

    for image_bin in image_binary_data:
        base_64_encoded_data = base64.b64encode(image_bin)
        image_base64_string = base_64_encoded_data.decode("utf-8")
        image_type = imghdr.what(io.BytesIO(image_bin))

        content.append(
            {
                "image": {
                    "format": image_type, #"png",
                    "source": {"bytes": image_base64_string},
                }
            }
        )
    
    message_list = [
        {
            "role": "user",
            "content": content
        }
    ]
    
    # Configure the inference parameters.
    inf_params = {"max_new_tokens": max_token, "top_p": top_p, "top_k": top_k, "temperature": temperature}
    
    native_request = {
        "schemaVersion": "messages-v1",
        "messages": message_list,
        "system": system_list,
        "inferenceConfig": inf_params,
    }
    
    # Invoke the model and extract the response body.
    response = bedrock_runtime.invoke_model(modelId=option, body=json.dumps(native_request))
    model_response = json.loads(response["body"].read())

    
    # Return the text content for easy readability.
    return model_response["output"]["message"]["content"][0]["text"]

def o1_video(option, prompt, video_file_name, max_token, temperature, top_p, top_k, role_arn, region="us-east-1"):
    # Create an STS client
    sts_client = boto3.client('sts')

    # Assume the specified role
    assumed_role = sts_client.assume_role(
        RoleArn=role_arn,
        RoleSessionName='CallBedrockOVG'
    )

    # Extract temporary credentials
    credentials = assumed_role['Credentials']

    # Call Bedrock
    bedrock_runtime = boto3.client(
        "bedrock-runtime",
        region_name=region,  # You must use us-east-1 during the beta period.
        endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com",
        aws_access_key_id=credentials['AccessKeyId'],
        aws_secret_access_key=credentials['SecretAccessKey'],
        aws_session_token=credentials['SessionToken'],
    )
    
    #Define your system prompt(s).
    system_list = [
        {
            "type": "system",
            "content": [{"text": "You are an expert video interpreter. Please analyze the provided video  and generate a comprehensive description, including main subjects and their actions, Key events and their temporal sequence, Setting and environmental context and Notable objects and their interactions. Make sure to inlcude Any text or graphical overlays, Audio elements including speech, music, and sound effects, camera movements and transitions, emotional tone and atmosphere, relevant temporal markers or timestamps and any significant changes in scene composition."}],
        }
    ]
    
    # Define a "user" message including both the image and a text prompt. 
    # Value at 'body' failed to satisfy constraint: Member must have length less than or equal to 25000000
    with open(video_file_name, "rb") as video_file:
        video_binary_data = video_file.read()
        base_64_encoded_data = base64.b64encode(video_binary_data) #.getvalue())
        video_base64_string = base_64_encoded_data.decode("utf-8")
        
    message_list = [
        {
            "role": "user",
            "content": [
                {
                    "video": {
                        "format": "mp4",
                        "source": {"bytes": video_base64_string},
                    }
                },
                {
                    "text": prompt
                }
            ],
        }
    ]
    
    # Configure the inference parameters.
    inf_params = {"max_new_tokens": max_token, "top_p": top_p, "top_k": top_k, "temperature": temperature}
    
    native_request = {
        "schemaVersion": "messages-v1",
        "messages": message_list,
        "system": system_list,
        "inferenceConfig": inf_params,
    }
    
    # Invoke the model and extract the response body.
    response = bedrock_runtime.invoke_model(modelId=option, body=json.dumps(native_request))
    model_response = json.loads(response["body"].read())

    
    # Return the text content for easy readability.
    return model_response["output"]["message"]["content"][0]["text"], len(video_binary_data)
    
###
# Search
###
def retrieval_o1(query, documents, model_id, embedding_model_id, chunk_size, over_lap, max_tokens, temperature, top_p, top_k, doc_num, role_arn, region):
    doc2text = "\n\n".join([doc.page_content for doc in documents])
    
    prompt2 = f"{query}. Your answer should be strictly based on the context in {doc2text}."
    return olympus_textGen(model_id, prompt2, max_tokens, temperature, top_p, top_k, role_arn, region)
    
def retrieval_faiss_o1(query, documents, model_id, embedding_model_id, chunk_size, over_lap, max_tokens, temperature, top_p, top_k, doc_num, role_arn, region):
    
    # Create an STS client
    sts_client = boto3.client('sts')

    # Assume the specified role
    assumed_role = sts_client.assume_role(
        RoleArn=role_arn,
        RoleSessionName='CallBedrockOVG'
    )

    # Extract temporary credentials
    credentials = assumed_role['Credentials']

    # Call Bedrock
    bedrock_runtime = boto3.client(
        "bedrock-runtime",
        region_name=region,  # You must use us-east-1 during the beta period.
        endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com",
        aws_access_key_id=credentials['AccessKeyId'],
        aws_secret_access_key=credentials['SecretAccessKey'],
        aws_session_token=credentials['SessionToken'],
    )
    
    request_body = {
        "schemaVersion": "messages-v1",
        "system": [
            {"text": "You are a smart AI assistant whi can answer user questions by providing clear, factual, and dependable information."}
        ],
        "messages": [
            {
                "role": "user",
                "content": [{"text": query}]
            },
        ],
        "inferenceConfig": {
            "max_new_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
        }
    }
    
    # Invoke the model and extract the response body.
    chat = bedrock_runtime.invoke_model(
        modelId=model_id, #"amazon.Olympus-micro-v1:0",
        body=json.dumps(request_body)
    )
    
    # Process doc
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=over_lap, length_function=len, is_separator_regex=False,)
    docs = text_splitter.split_documents(documents)
    
    # Prepare embedding function
    embedding_bedrock = BedrockEmbeddings(client=boto3.client('bedrock-runtime'), model_id=embedding_model_id)
    
    # Try to get vectordb with FAISS
    db = FAISS.from_documents(docs, embedding_bedrock)
    retriever = db.as_retriever(search_kwargs={"k": doc_num})


    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    messages = [
        ("system", """Your are a helpful assistant to provide comprehensive and truthful answers to questions, \n
                    drawing upon all relevant information contained within the specified in {context}. \n 
                    You add value by analyzing the situation and offering insights to enrich your answer. \n
                    Simply say I don't know if you can not find any evidence to match the question. \n
                    """),
        #MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
    
    prompt_template = ChatPromptTemplate.from_messages(messages)

    # Reranker
    compression_retriever = ContextualCompressionRetriever(
        base_compressor= FlashrankRerank(), base_retriever=retriever
    )

    rag_chain = (
        #{"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
        #| RunnableParallel(answer=hub.pull("rlm/rag-prompt") | chat |format_docs, question=itemgetter("question") ) 
        RunnableParallel(context=compression_retriever | format_docs, question=RunnablePassthrough() )
        | prompt_template
        | chat
        | StrOutputParser()
    )

    results = rag_chain.invoke(query)
    return results

### 
#Extract_url using Plympus
###
def extract_urls_o1(urls: list, query: str, model_id: str, embedding_model_id: str,  max_tokens: int, temperature: float, top_p: float, top_k: int, role_arn: str, region: str):
    '''
    # Create an STS client
    sts_client = boto3.client('sts')

    # Assume the specified role
    assumed_role = sts_client.assume_role(
        RoleArn=role_arn,
        RoleSessionName='CallBedrockOVG'
    )

    # Extract temporary credentials
    credentials = assumed_role['Credentials']

    # Call Bedrock
    bedrock_runtime = boto3.client(
        "bedrock-runtime",
        region_name=region,  # You must use us-east-1 during the beta period.
        endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com",
        aws_access_key_id=credentials['AccessKeyId'],
        aws_secret_access_key=credentials['SecretAccessKey'],
        aws_session_token=credentials['SessionToken'],
    )
    
    request_body = {
        "schemaVersion": "messages-v1",
        "system": [
            {"text": "You are a smart AI assistant whi can answer user questions by providing clear, factual, and dependable information."}
        ],
        "messages": [
            {
                "role": "user",
                "content": [{"text": query}]
            },
        ],
        "inferenceConfig": {
            "max_new_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
        }
    }
    
    # Invoke the model and extract the response body.
    llm_c3 = bedrock_runtime.invoke_model(
        modelId=model_id, #"amazon.Olympus-micro-v1:0",
        body=json.dumps(request_body)
    )
    '''
    
    loaders = []
    # Prepare embedding function
    br_embedding = BedrockEmbeddings(client=boto3.client('bedrock-runtime'), model_id=embedding_model_id)

    # Load
    xml_loader = WebBaseLoader(urls[0])
    xml_loader.requests_per_second = 1
    loaders.append(xml_loader)
    if check_urls(urls, '.html') or check_urls(urls, '.htm'):
        html_loader = AsyncHtmlLoader(urls)
        loaders.append(html_loader)
    if check_urls(urls, '.pdf'):
        pdf_loader = PyMuPDFLoader(urls[0])
        loaders.append(pdf_loader)
    loader_all = MergedDataLoader(loaders=loaders)
    docs_all = loader_all.load()

    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs_all)
    text_str = ''
    for doc_transformed in docs_transformed:
        text_str += doc_transformed.page_content

    ''' # Olympus has NOT been tested with LangChain
    docs_all_transformed = html2text.transform_documents(docs_all)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=8000,
        chunk_overlap=800,
        length_function=len,
    )
    
    splits = text_splitter.split_documents(docs_all_transformed)
    # Create a FAISS vector store and embed the document chunks
    vectorstore = FAISS.from_documents(splits, br_embedding)

    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 1})
    results = retriever.invoke(query, filter={"source": "news"})

    # Create chain
    combine_docs_chain = create_stuff_documents_chain(
           llm_c3, retrieval_qa_chat_prompt
       )
    # Retrivals
    retrieval_chain = create_retrieval_chain(
       vectorstore.as_retriever(), combine_docs_chain
    )
    res = retrieval_chain.invoke({"input":query})
    '''
    prompt2 = f"{query}. Your answer should be strictly based on the context in {text_str}."
    return olympus_textGen(model_id, prompt2, max_tokens, temperature, top_p, top_k, role_arn, region)


###
# Main
###
if __name__ == "__main__":
    prompt = "A high res, 4k image of fall foliage with tree on different layers ofcolors from red, orange, yellow to all season green with snow covered mountain peaks in the backgroung and running creak at the front vivid color and photoreastic"
    neg_prompt = "human, single color tree"
    images = t2i_olympus(prompt, neg_prompt, num_image=3)
    print(f"Image size: {len(images)}")
    video_prompt = "Closeup of a scence of fall foliage in Sierra with snow covered mountain peaks and running stream, frothy gentle wind blowing tree leaves. Camera zoom in."
    file_name = t2v_ovg(video_prompt=video_prompt, role_arn="arn:aws:iam::905418197933:role/ovg_developer")
    print(f"Video file: {file_name}")