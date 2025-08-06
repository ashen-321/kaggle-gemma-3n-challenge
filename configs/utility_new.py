import json
import boto3
import random
import time
import os
import requests
import shutil

from typing import Optional
from botocore.config import Config
from langchain.llms.bedrock import Bedrock

import faiss
from bs4 import BeautifulSoup 
from urllib.parse import unquote
from langchain_community.vectorstores import FAISS
from langchain.memory import VectorStoreRetrieverMemory
from langchain_community.docstore import InMemoryDocstore
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.chat_models import BedrockChat
from langchain_core.documents.base import Document
from readabilipy import simple_json_from_html_string # Required to parse HTML to pure text

config_filename = '.aoss_config.txt'
suffix = random.randrange(200, 900)
boto3_session = boto3.session.Session()
region_name = boto3_session.region_name
iam_client = boto3_session.client('iam')
account_number = boto3.client('sts').get_caller_identity().get('Account')
identity = boto3.client('sts').get_caller_identity()['Arn']

# S3
sts_client = boto3.client('sts')
bedrock_agent_client = boto3_session.client('bedrock-agent', region_name=region_name)
bedrock_client = boto3.client("bedrock-runtime", region_name=region_name)
bedrock_agent_runtime_client = boto3.client("bedrock-agent-runtime", region_name=region_name)
s3_client = boto3.client('s3')
account_id = sts_client.get_caller_identity()["Account"]
s3_suffix = f"{region_name}-{account_id}"
bucket_name = f'bedrock-kb-{s3_suffix}' # replace it with your bucket name.


encryption_policy_name = f"bedrock-sample-rag-sp-{suffix}"
network_policy_name = f"bedrock-sample-rag-np-{suffix}"
access_policy_name = f'bedrock-sample-rag-ap-{suffix}'
bedrock_execution_role_name = f'AmazonBedrockExecutionRoleForKnowledgeBase_{suffix}'
fm_policy_name = f'AmazonBedrockFoundationModelPolicyForKnowledgeBase_{suffix}'
s3_policy_name = f'AmazonBedrockS3PolicyForKnowledgeBase_{suffix}'
oss_policy_name = f'AmazonBedrockOSSPolicyForKnowledgeBase_{suffix}'


def get_bedrock_client(
    assumed_role: Optional[str] = None,
    region: Optional[str] = None,
    runtime: Optional[bool] = True,
):
    """Create a boto3 client for Amazon Bedrock, with optional configuration overrides

    Parameters
    ----------
    assumed_role :
        Optional ARN of an AWS IAM role to assume for calling the Bedrock service. If not
        specified, the current active credentials will be used.
    region :
        Optional name of the AWS Region in which the service should be called (e.g. "us-east-1").
        If not specified, AWS_REGION or AWS_DEFAULT_REGION environment variable will be used.
    runtime :
        Optional choice of getting different client to perform operations with the Amazon Bedrock service.
    """
    if region is None:
        target_region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION"))
    else:
        target_region = region

    #print(f"Create new client\n  Using region: {target_region}")
    session_kwargs = {"region_name": target_region}
    client_kwargs = {**session_kwargs}

    profile_name = os.environ.get("AWS_PROFILE")
    if profile_name:
        #print(f"  Using profile: {profile_name}")
        session_kwargs["profile_name"] = profile_name

    retry_config = Config(
        region_name=target_region,
        retries={
            "max_attempts": 10,
            "mode": "standard",
        },
    )
    session = boto3.Session(**session_kwargs)

    if assumed_role:
        #print(f"  Using role: {assumed_role}", end='')
        sts = session.client("sts")
        response = sts.assume_role(
            RoleArn=str(assumed_role),
            RoleSessionName="langchain-llm-1"
        )
        #print(" ... successful!")
        client_kwargs["aws_access_key_id"] = response["Credentials"]["AccessKeyId"]
        client_kwargs["aws_secret_access_key"] = response["Credentials"]["SecretAccessKey"]
        client_kwargs["aws_session_token"] = response["Credentials"]["SessionToken"]

    if runtime:
        service_name='bedrock-runtime'
    else:
        service_name='bedrock'

    bedrock_client = session.client(
        service_name=service_name,
        config=retry_config,
        **client_kwargs
    )

    #print("boto3 Bedrock client successfully created!")
    #print(bedrock_client._endpoint)
    return bedrock_client
    
def create_bedrock_execution_role(bucket_name):
    foundation_model_policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "bedrock:InvokeModel",
                ],
                "Resource": [
                    f"arn:aws:bedrock:{region_name}::foundation-model/amazon.titan-embed-text-v1"
                ]
            }
        ]
    }

    s3_policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:ListBucket"
                ],
                "Resource": [
                    f"arn:aws:s3:::{bucket_name}",
                    f"arn:aws:s3:::{bucket_name}/*"
                ],
                "Condition": {
                    "StringEquals": {
                        "aws:ResourceAccount": f"{account_number}"
                    }
                }
            }
        ]
    }

    assume_role_policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "bedrock.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }
    # create policies based on the policy documents
    fm_policy = iam_client.create_policy(
        PolicyName=fm_policy_name,
        PolicyDocument=json.dumps(foundation_model_policy_document),
        Description='Policy for accessing foundation model',
    )

    s3_policy = iam_client.create_policy(
        PolicyName=s3_policy_name,
        PolicyDocument=json.dumps(s3_policy_document),
        Description='Policy for reading documents from s3')

    # create bedrock execution role
    bedrock_kb_execution_role = iam_client.create_role(
        RoleName=bedrock_execution_role_name,
        AssumeRolePolicyDocument=json.dumps(assume_role_policy_document),
        Description='Amazon Bedrock Knowledge Base Execution Role for accessing OSS and S3',
        MaxSessionDuration=3600
    )

    # fetch arn of the policies and role created above
    bedrock_kb_execution_role_arn = bedrock_kb_execution_role['Role']['Arn']
    s3_policy_arn = s3_policy["Policy"]["Arn"]
    fm_policy_arn = fm_policy["Policy"]["Arn"]

    # attach policies to Amazon Bedrock execution role
    iam_client.attach_role_policy(
        RoleName=bedrock_kb_execution_role["Role"]["RoleName"],
        PolicyArn=fm_policy_arn
    )
    iam_client.attach_role_policy(
        RoleName=bedrock_kb_execution_role["Role"]["RoleName"],
        PolicyArn=s3_policy_arn
    )
    return bedrock_kb_execution_role


def create_oss_policy_attach_bedrock_execution_role(collection_id, bedrock_kb_execution_role):
    # define oss policy document
    oss_policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "aoss:APIAccessAll"
                ],
                "Resource": [
                    f"arn:aws:aoss:{region_name}:{account_number}:collection/{collection_id}"
                ]
            }
        ]
    }
    oss_policy = iam_client.create_policy(
        PolicyName=oss_policy_name,
        PolicyDocument=json.dumps(oss_policy_document),
        Description='Policy for accessing opensearch serverless',
    )
    oss_policy_arn = oss_policy["Policy"]["Arn"]
    print("Opensearch serverless arn: ", oss_policy_arn)

    iam_client.attach_role_policy(
        RoleName=bedrock_kb_execution_role["Role"]["RoleName"],
        PolicyArn=oss_policy_arn
    )
    return None


def create_policies_in_oss(vector_store_name, aoss_client, bedrock_kb_execution_role_arn):
    encryption_policy = aoss_client.create_security_policy(
        name=encryption_policy_name,
        policy=json.dumps(
            {
                'Rules': [{'Resource': ['collection/' + vector_store_name],
                           'ResourceType': 'collection'}],
                'AWSOwnedKey': True
            }),
        type='encryption'
    )

    network_policy = aoss_client.create_security_policy(
        name=network_policy_name,
        policy=json.dumps(
            [
                {'Rules': [{'Resource': ['collection/' + vector_store_name],
                            'ResourceType': 'collection'}],
                 'AllowFromPublic': True}
            ]),
        type='network'
    )
    access_policy = aoss_client.create_access_policy(
        name=access_policy_name,
        policy=json.dumps(
            [
                {
                    'Rules': [
                        {
                            'Resource': ['collection/' + vector_store_name],
                            'Permission': [
                                'aoss:CreateCollectionItems',
                                'aoss:DeleteCollectionItems',
                                'aoss:UpdateCollectionItems',
                                'aoss:DescribeCollectionItems'],
                            'ResourceType': 'collection'
                        },
                        {
                            'Resource': ['index/' + vector_store_name + '/*'],
                            'Permission': [
                                'aoss:CreateIndex',
                                'aoss:DeleteIndex',
                                'aoss:UpdateIndex',
                                'aoss:DescribeIndex',
                                'aoss:ReadDocument',
                                'aoss:WriteDocument'],
                            'ResourceType': 'index'
                        }],
                    'Principal': [identity, bedrock_kb_execution_role_arn],
                    'Description': 'Easy data policy'}
            ]),
        type='data'
    )
    return encryption_policy, network_policy, access_policy


def delete_iam_role_and_policies():
    fm_policy_arn = f"arn:aws:iam::{account_number}:policy/{fm_policy_name}"
    s3_policy_arn = f"arn:aws:iam::{account_number}:policy/{s3_policy_name}"
    oss_policy_arn = f"arn:aws:iam::{account_number}:policy/{oss_policy_name}"
    iam_client.detach_role_policy(
        RoleName=bedrock_execution_role_name,
        PolicyArn=s3_policy_arn
    )
    iam_client.detach_role_policy(
        RoleName=bedrock_execution_role_name,
        PolicyArn=fm_policy_arn
    )
    iam_client.detach_role_policy(
        RoleName=bedrock_execution_role_name,
        PolicyArn=oss_policy_arn
    )
    iam_client.delete_role(RoleName=bedrock_execution_role_name)
    iam_client.delete_policy(PolicyArn=s3_policy_arn)
    iam_client.delete_policy(PolicyArn=fm_policy_arn)
    iam_client.delete_policy(PolicyArn=oss_policy_arn)
    return 0


def interactive_sleep(seconds: int):
    dots = ''
    for i in range(seconds):
        dots += '.'
        print(dots, end='\r')
        time.sleep(1)
    print('Done!')
    
# Get AOSS key and values
def read_key_value(file_path, key1):
    with open(file_path, 'r') as file:
        for line in file:
            key_value_pairs = line.strip().split(':')
            if key_value_pairs[0] == key1:
                return key_value_pairs[1].lstrip()
    return None

#Upload to S3
def uploadDirectory(path,bucket_name):
    for root,dirs,files in os.walk(path):
        for file in files:
            s3_client.upload_file(os.path.join(root,file),bucket_name,file)

def empty_directory(directory_path):
    if os.path.exists(directory_path):
        for item in os.scandir(directory_path):
            if item.is_file():
                os.remove(item.path)
            elif item.is_dir():
                #os.rmdir(item.path)
                shutil.rmtree(item.path)
        return True
    else:
        return False

def empty_versioned_s3_bucket(bucket_name):
    s3r = boto3.resource('s3')
    bucket = s3r.Bucket(bucket_name)
    bucket.object_versions.delete()
    return True

def bedrock_kb_injection(path):
    kb_id = read_key_value(config_filename, 'KB_id')
    ds_id = read_key_value(config_filename, 'DS_id')
    region_name = read_key_value(config_filename, 'Region')
    bucket_name = read_key_value(config_filename, 'S3_bucket_name')
    # Bedrock KB syncs with designated S3 bucket so be careful
    empty_versioned_s3_bucket(bucket_name)
    uploadDirectory(path,bucket_name)
    #ds = bedrock_agent_client.get_data_source(knowledgeBaseId = kb_id, dataSourceId = ds_id)
    start_job_response = bedrock_agent_client.start_ingestion_job(knowledgeBaseId = kb_id, dataSourceId = ds_id)
    job = start_job_response["ingestionJob"]
    while(job['status']!='COMPLETE' ):
      get_job_response = bedrock_agent_client.get_ingestion_job(
          knowledgeBaseId = kb_id,
            dataSourceId = ds_id,
            ingestionJobId = job["ingestionJobId"]
      )
      job = get_job_response["ingestionJob"]
    interactive_sleep(40)
    return job['statistics'], job['status'], kb_id

def bedrock_kb_retrieval(query: str, model_id: str) -> str:
    kb_id = read_key_value(config_filename, 'KB_id')
    model_arn = f'arn:aws:bedrock:{region_name}::foundation-model/{model_id}'
    response = bedrock_agent_runtime_client.retrieve_and_generate(
        input={
            'text': query
        },
        retrieveAndGenerateConfiguration={
            'type': 'KNOWLEDGE_BASE',
            'knowledgeBaseConfiguration': {
                'knowledgeBaseId': kb_id,
                'modelArn': model_arn
            }
        },
    )

    generated_text = response['output']['text']
    return generated_text


def bedrock_textGen(model_id, prompt, max_tokens, temperature, top_p, top_k, stop_sequences):
    stop_sequence = [stop_sequences]

    if  "anthropic.claude-v2" in model_id.lower() or "anthropic.claude-instant" in model_id.lower():
        inference_modifier = {
            "max_tokens_to_sample": max_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "stop_sequences": stop_sequence,
        }
    
        textgen_llm = Bedrock(
            model_id=model_id,
            client=bedrock_client,
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
                "stop_sequences": stop_sequence,
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
        response = bedrock_client.invoke_model(
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

###
#-- perplexity ---
###

#----------- Parse out web content -----------
def scrape_and_parse(url: str) -> Document:
    """Scrape a webpage and parse it into a Document object"""
    req = requests.get(url)
    article = simple_json_from_html_string(req.text, use_readability=True)
    # The following line seems to work with the package versions on my local machine, but not on Google Colab
    # return Document(page_content=article['plain_text'][0]['text'], metadata={'source': url, 'page_title': article['title']})
    return Document(page_content='\n\n'.join([a['text'] for a in article['plain_text']]), metadata={'source': url, 'page_title': article['title']})
    
#--- Configure Bedrock -----
def config_bedrock(embedding_model_id, model_id, max_tokens, temperature, top_p, top_k):
    embedding_bedrock = BedrockEmbeddings(client=bedrock_client, model_id=embedding_model_id)
    model_kwargs =  { 
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        #"stop_sequences": ["\n\nHuman"],
    }
    chat = BedrockChat(
        model_id=model_id, client=bedrock_client, model_kwargs=model_kwargs
    )
    #llm = Bedrock(
    #    model_id=model_id, client=bedrock_client, model_kwargs=model_kwargs
    #)

    return chat, embedding_bedrock

# -- search news --- 
class newsSearcher:
    def __init__(self):
        self.google_url = "https://www.google.com/search?q="
        self.bing_url = "https://www.bing.com/search?q="
        #self.bing_url = "https://www.bing.com/search?q={query.replace(' ', '+')}"

    def search(self, query, count: int=10):
        google_urls = self.search_goog(query, count)
        bing_urls = self.search_bing(query, count)
        combined_urls = google_urls + bing_urls
        urls = list(set(combined_urls))  # Remove duplicates
        return [scrape_and_parse(f) for f in urls], urls # Scrape and parse all the url

    def search_goog(self, query, count):
        #response = requests.get(f"https://www.google.com/search?q={query}") # Make the request
        params = {
            "q": query,
            "num": count  # Number of results to retrieve
        }
        response = requests.get(self.google_url, params=params)
        soup = BeautifulSoup(response.text, "html.parser") # Parse the HTML
        links = soup.find_all("a") # Find all the links in the HTML
        urls = []
        for l in [link for link in links if link["href"].startswith("/url?q=")]:
            # get the url
            url = l["href"]
            # remove the "/url?q=" part
            url = url.replace("/url?q=", "")
            # remove the part after the "&sa=..."
            url = unquote(url.split("&sa=")[0])
            # special case for google scholar
            if url.startswith("https://scholar.google.com/scholar_url?url=http"):
                url = url.replace("https://scholar.google.com/scholar_url?url=", "").split("&")[0]
            elif 'google.com/' in url: # skip google links
                continue
            elif 'youtube.com/' in url:
                continue
            elif 'search?q=' in url:
                continue
            if url.endswith('.pdf'): # skip pdf links
                continue
            if '#' in url: # remove anchors (e.g. wikipedia.com/bob#history and wikipedia.com/bob#genetics are the same page)
                url = url.split('#')[0]
            # print the url
            urls.append(url)
        return urls
        
    def search_google(self, query, count):
        params = {
            "q": query,
            "num": count  # Number of results to retrieve
        }
        response = requests.get(self.google_url, params=params)
        soup = BeautifulSoup(response.text, "html.parser")
        urls = [link.get("href") for link in soup.select(".yuRUbf a")]
        return urls

    def search_bing(self, query, count):
        params = {
            "q": query,
            "count": count # Number of results to retrieve
        }
        response = requests.get(self.bing_url, params=params)
        soup = BeautifulSoup(response.text, "html.parser")
        urls = [link.get("href") for link in soup.select(".b_algo h2 a")]
        return urls[:count]


def bedrock_textGen_perplexity_memory(option, prompt, max_token, temperature, top_p, top_k, stop_sequences, embd_model_id):
    if "titan-embed-text" in embd_model_id:
        embedding_size = 1536 # Dimensions of the amazon.titan-embed-text-v1
    elif "titan-embed-image" in embd_model_id:
        embedding_size = 1024 #amazon.titan-embed-image-v1
    else: 
        embedding_size = 4096

    # Configure Bedrock LLM and Chat
    chat, embd = config_bedrock(embd_model_id, option, max_tokens=max_token, temperature=temperature, top_p=top_p, top_k=top_k)

    # Get FAISS host string
    index = faiss.IndexFlatL2(embedding_size)
    #embedding_fn = OpenAIEmbeddings().embed_query
    #embedding_fn = embd.embed_query
    vectorstore = FAISS(embd, index, InMemoryDocstore({}), {})
    #print(aoss_host)

    searcher = newsSearcher()
    documents, urls = searcher.search(prompt)
    #print(documents)
    # Splitter
    text_splitter = CharacterTextSplitter(separator=' ', chunk_size=8000, chunk_overlap=800)
    texts = text_splitter.split_documents(documents)
    docsearcher = FAISS.from_documents(texts, embd)
    retriever = docsearcher.as_retriever(search_kwargs={"k": 5})

    # Query using conversational chain
    if 'naive' in stop_sequences.lower():
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | hub.pull("rlm/rag-prompt")
            | chat
            | StrOutputParser()
        )
        results = rag_chain.invoke(prompt)
    else:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        messages = [
            ("system", """Your are a helpful assistant to provide omprehensive and truthful answers to questions, \n
                        drawing upon all relevant information contained within the specified in {context}. \n 
                        You add value by analyzing the situation and offering insights to enrich your answer. \n
                        Simply say I don't know if you can not find any evidence to match the question. \n
                        Extract the corresponding sources and add the clickable, relevant and unaltered URLs in hyperlink format to the end of your answer."""),
            #MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
        prompt_template = ChatPromptTemplate.from_messages(messages)
        
        #chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | chat | StrOutputParser())
        chain = ({"context": retriever, "question": RunnablePassthrough(chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("chat_history"))} | prompt_template | chat | StrOutputParser())
        results = chain.invoke(prompt)
    return results, urls