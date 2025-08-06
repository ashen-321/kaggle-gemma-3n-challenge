import arxiv
from langchain_core.documents.base import Document
import requests
import os
import boto3
import datetime
import base64
from io import BytesIO
from botocore.config import Config
import json
import re
import string
import signal
from functools import wraps
import multiprocessing
from bs4 import BeautifulSoup
import requests # Required to make HTTP requests
from bs4 import BeautifulSoup  # Required to parse HTML
from urllib.parse import unquote # Required to unquote URLs
from xml.etree import ElementTree
from botocore.exceptions import ClientError


from operator import itemgetter
from langchain import hub
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.chat_models import BedrockChat
from langchain_aws import ChatBedrockConverse
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.messages import SystemMessage, HumanMessage
#from langchain.utilities import GoogleSearchAPIWrapper
from langchain.retrievers.web_research import WebResearchRetriever
from langchain_community.utilities import GoogleSearchAPIWrapper
from readabilipy import simple_json_from_html_string # Required to parse HTML to pure text
from langchain.agents import AgentExecutor, create_react_agent, initialize_agent, AgentType,load_tools
from langchain_community.utilities.serpapi import SerpAPIWrapper
from serpapi import GoogleSearch #, BingSearch
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_deepseek import ChatDeepSeek


from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
# To address  RuntimeError: maximum recursion depth exceeded

#from langchain_community.llms import HuggingFaceTextGenInference
from langchain_community.llms.huggingface_text_gen_inference import (
    HuggingFaceTextGenInference,
)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# For TGI textGen
from langchain.chains import LLMChain
#from langchain_community.llms import TextGen
from langchain_core.prompts import PromptTemplate

# Pdf
from unstructured.partition.pdf import partition_pdf
import fitz
from PIL import Image
import io
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import pdfplumber
import xml.etree.ElementTree as ET
from xml.dom import minidom
import magic

# OpenAi
from openai import OpenAI

import sys   
sys.setrecursionlimit(10000)

#---  Set timeout -----
class TimeoutError(Exception):
    pass

def set_timeout(seconds=10, error_message='Function call timed out'):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def _handle_timeout(signum, frame):
                raise TimeoutError(error_message)

            old_handler = signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            except TimeoutError as exc:
                raise exc
            finally:
                signal.signal(signal.SIGALRM, old_handler)
                signal.alarm(0)
            return result
        return wrapper
    return decorator


## SEarch engines ----

#----------- Parse out web content -----------
def scrape_and_parse(url: str) -> Document:
    """Scrape a webpage and parse it into a Document object"""
    req = requests.get(url)
    article = simple_json_from_html_string(req.text, use_readability=True)
    # The following line seems to work with the package versions on my local machine, but not on Google Colab
    # return Document(page_content=article['plain_text'][0]['text'], metadata={'source': url, 'page_title': article['title']})
    return Document(page_content='\n\n'.join([a['text'] for a in article['plain_text']]), metadata={'source': url, 'page_title': article['title']})

def extract_keywords(input_string):
    # Remove punctuation and convert to lowercase
    input_string = input_string.translate(str.maketrans('', '', string.punctuation)).lower()
    # Split the string into words
    words = input_string.split()
 
    # Define a regular expression pattern to match web searchable keywords
    pattern = r'^[a-zA-Z0-9]+$'
    # Filter out non-keyword words
    keywords = [word for word in words if re.match(pattern, word)]
    # Join the keywords with '+'
    output_string = '+'.join(keywords)
    return re.sub(r'[.-:/"\']', ' ', output_string)
    
# Search google and bing with a query and return urls
def google_search(query: str, num_results: int=5):
    gsearch = GoogleSearchAPIWrapper(google_api_key=os.getenv("google_api_key"), google_cse_id=os.getenv("google_cse_id"))
    params = {
        "num_results": num_results,  # Number of results to return
        #"exclude": "youtube.com"  # Exclude results from YouTube
    }
    #key_words = re.sub(r'[^a-zA-Z0-9\s]', ' ', extract_keywords(query))
    google_results = gsearch.results(extract_keywords(query), **params)
    #google_results = gsearch.results(query, num_results=num_results)
    documents = []
    urls = []
    for item in google_results:
        try:
            # Send a GET request to the URL
            if ('link' not in item) or ('youtube' in item['link'].lower()):
                continue
            response = requests.get(item['link'])
            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            # Extract the text content from the HTML
            content = soup.get_text()
            if "404 Not Found" not in content or "400 Client Error" not in content:
                # Create a LangChain document
                doc = Document(page_content=content, metadata={'title': item['title'],'source': item['link']})
                documents.append(doc)
                urls. append(item['link'])
    
        except requests.exceptions.RequestException as e:
            print(f"Error parsing URL: {e}")
            pass

    return documents, urls

## Tavily Search 
def tavily_search(query: str, num_results: int=3):
    def _set_if_undefined(var: str):
        if not os.environ.get(var):
            os.environ[var] = os.getenv(var.replace("KEY", "token").lower())

    _set_if_undefined("OPENAI_API_KEY")
    _set_if_undefined("LANGCHAIN_API_KEY")
    _set_if_undefined("TAVILY_API_KEY")
    
    documents = []
    web_search_tool = TavilySearchResults(k=num_results)
    # Web search
    docs = web_search_tool.invoke({"query": query})
    try:
        if "404 Not Found" not in docs or '400 Client Error' not in docs:
            #print(f"Web_search docs: {docs}")
            web_results = "\n".join([d["content"] for d in docs])
            web_urls = ", ".join([d["url"] for d in docs])
            web_results = Document(page_content=web_results, metadata={'source': web_urls})
            #web_results = Document(page_content=web_results)
            if documents is not None:
                documents.append(web_results)
            else:
                documents = [web_results]
        return {"documents": documents, "urls":  [d["url"] for d in docs]}
    except:
        return {"documents": [], "urls": []}
    
'''
class newsSearcher:
    def __init__(self):
        self.google_url = "https://www.google.com/search?q="
        self.bing_url = "https://www.bing.com/search?q="
        #self.bing_url = "https://www.bing.com/search?q={query.replace(' ', '+')}"

    def search(self, query, count: int=3):
        with multiprocessing.Pool(processes=2) as pool:
            google_results = pool.apply_async(self.search_google, args=(query,count))
            bing_results = pool.apply_async(self.search_bing, args=(query,count))
    
            # Get results from both processes
            google_urls = google_results.get()
            bing_urls = bing_results.get()
        #google_urls = self.search_goog(query, count)
        #bing_urls = self.search_bing(query, count)
        combined_urls = google_urls + bing_urls
        urls = list(set(combined_urls))  # Remove duplicates
        return [scrape_and_parse(f) for f in google_urls], urls # Scrape and parse all the url

    def search_goog(self, query, count):
        #response = requests.get(f"https://www.google.com/search?q={query}") # Make the request
        params = {
            "q": query,
            "num": count  # Number of results to retrieve
        }
        response = requests.get(self.google_url, params=params)
        soup = BeautifulSoup(response.text, "html.parser") # Parse the HTML
        #soup = BeautifulSoup(response.text, "html.parser").get_text() # Parse the HTML
        links = soup.find_all("a", recursive=True) # Find all the links in the HTML
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
'''
# --- is a file pdf ---
def is_pdf(file_path):
    """
    Checks if a file is a PDF.

    Args:
        file_path (str): The path to the file.

    Returns:
        bool: True if the file is a PDF, False otherwise.
    """
    try:
        with open(file_path, 'rb') as file:
            file_type = magic.from_buffer(file.read(1024), mime=True)
            return file_type == 'application/pdf'
    except FileNotFoundError:
        return False
    except Exception:
        return False
        

# ---- Search arxiv -----
#--- Configure Bedrock -----
def config_bedrock(embedding_model_id, model_id, max_tokens, temperature, top_p, top_k):   
    bedrock_client = boto3.client('bedrock-runtime')
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

def config_bedrock_converse(embedding_model_id, model_id, max_tokens, temperature, top_p):
    my_config = Config(read_timeout=70, #secs, default 60
                       connect_timeout=100, #secs, default 60
                       retries = {
                        'max_attempts': 10,
                        'mode': 'adaptive' #or standard or lagacy
                    }
                )
    bedrock_client = boto3.client('bedrock-runtime', config=my_config)
    embedding_bedrock = BedrockEmbeddings(client=bedrock_client, model_id=embedding_model_id)
    chat = ChatBedrockConverse(
        model=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )
    return chat, embedding_bedrock
    
    
# Convert to Document
def convert_to_document2(entry):
    """Convert an entry to a LangChain Document object."""
    # Adjust attributes according to the actual Document class definition
    document = Document(
        page_content=entry.summary,
        metadata={
            'title': entry.title,
            'authors': entry.authors,
            'id': entry.entry_id,
            'link': entry.links,
            'categories':entry.categories,
            'published': entry.published
        }
    )
    return document


def parse_response(xml_data):
    """Parse the XML response from arXiv and extract relevant data."""
    namespace = {'arxiv': 'http://www.w3.org/2005/Atom'}
    root = ElementTree.fromstring(xml_data)
    entries_data = []
    
    for entry in root.findall('arxiv:entry', namespace):
        entry_data = {
            'title': entry.find('arxiv:title', namespace).text,
            'summary': entry.find('arxiv:summary', namespace).text,
            'authors': [author.find('arxiv:name', namespace).text for author in entry.findall('arxiv:author', namespace)],
            'id': entry.find('arxiv:id', namespace).text,
            'link': entry.find('arxiv:link[@rel="alternate"]', namespace).attrib.get('href'),
            'published': entry.find('arxiv:published', namespace).text
        }
        entries_data.append(entry_data)
    
    return entries_data

def download_pdf(entries, dest_filepath):
    for entry in entries:
        paper_id = entry['id'].split("/")[-1]
        file_name = f"{dest_filepath}/{paper_id}.pdf"
        paper = next(arxiv.Client().results(arxiv.Search(id_list=[paper_id])))
        paper.download_pdf(filename=file_name)  # Downloads the paper
    return True

def download_pdf2(entries, dest_filepath):
    for entry in entries:
        paper_id = entry['id'].split("/")[-1]
        file_name = f"{dest_filepath}/{paper_id}.pdf"
        paper = next(arxiv.Client().results(arxiv.Search(id_list=[paper_id])))
        paper.download_pdf(filename=file_name)  # Downloads the paper
    return True

def convert_to_document(entry):
    """Convert an entry to a LangChain Document object."""
    # Adjust attributes according to the actual Document class definition
    document = Document(
        page_content=entry['summary'],
        metadata={
            'title': entry['title'],
            'authors': entry['authors'],
            'id': entry['id'],
            'link': entry['link'],
            'published': entry['published']
        }
    )
    return document

def search_and_convert(query, max_results=10, filepath='pdfs'):
    """Search arXiv, parse the results, and convert them to LangChain Document objects."""
    params = {"search_query": query, "start": 0, "max_results": max_results}
    base_url = "http://export.arxiv.org/api/query?"
    os.makedirs(filepath, exist_ok=True)
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        entries = parse_response(response.text)
        download_pdf(entries,filepath) 
        return [convert_to_document(entry) for entry in entries]
    else:
        print(f"Error fetching results from arXiv: {response.status_code}")
        return []

# Construct the default API client
def search_arxiv(query:str, max_results=10, filepath: str='pdfs'):
    client = arxiv.Client()
    docs = []
    # Search for articles matching the keyword "question and answer"
    search = arxiv.Search(
      query = query,
      max_results = max_results,
      sort_by = arxiv.SortCriterion.SubmittedDate
    )
    results = client.results(search)
    #download_pdf2(results, filepath)
    for result in results:
        docs.append(convert_to_document2(result))
    return docs

# ----  Search with google and  FAISS -----
#@set_timeout(seconds=25, error_message='Function took too long to execute. Please try again.')
def retrieval_faiss(query, documents, model_id, embedding_model_id:str, chunk_size:int=6000, over_lap:int=600, max_tokens: int=2048, temperature: int=0.01, top_p: float=0.90, top_k: int=25, doc_num: int=3):
    #text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=over_lap)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=over_lap, length_function=len, is_separator_regex=False,)
    docs = text_splitter.split_documents(documents)
    
    # Prepare embedding function
    if 'meta' in model_id.lower() or 'claude' in model_id.lower():
        chat, embedding = config_bedrock_converse(embedding_model_id, model_id, max_tokens, temperature, top_p)
    else:
        chat, embedding = config_bedrock(embedding_model_id, model_id, max_tokens, temperature, top_p, top_k)
    
    # Try to get vectordb with FAISS
    db = FAISS.from_documents(docs, embedding)
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

def retrieval_faiss_dsr1(query, documents, model_id, embedding_model_id:str, chunk_size:int=6000, over_lap:int=600, max_tokens: int=2048, temperature: int=0.01, top_p: float=0.90, top_k: int=25, doc_num: int=3):

    DEEPSEEK_API_KEY = "EMPTY"
    LOCAL_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    LOCAL_API_URL = "http://agent.cavatar.info:8080/v1"
    today = datetime.datetime.today().strftime("%D")
    
    #text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=over_lap)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=over_lap, length_function=len, is_separator_regex=False,)
    docs = text_splitter.split_documents(documents)
    
    # Prepare embedding function
    _, embedding = config_bedrock(embedding_model_id, model_id, max_tokens, temperature, top_p, top_k)
    chat = ChatDeepSeek(
        model=LOCAL_MODEL, 
        api_key=DEEPSEEK_API_KEY, 
        base_url=LOCAL_API_URL,
        max_tokens=4096,
        temperature=0.01,
        timeout=None,
        max_retries=2,
    )
    
    # Try to get vectordb with FAISS
    db = FAISS.from_documents(docs, embedding)
    retriever = db.as_retriever(search_kwargs={"k": doc_num})


    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    messages = [
        ("system", """Your are a helpful assistant who can provide comprehensive and truthful answers to questions, \n
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
    #for text in rag_chain.stream(query):
    #    yield text
    

def retrieval_chroma(query, model_id, embedding_model_id:str, chunk_size:int=6000, over_lap:int=600, max_tokens: int=2048, temperature: int=0.01, top_p: float=0.90, top_k: int=25, doc_num: int=3):

    # LLM   and  embedding function
    chat, embedding = config_bedrock(embedding_model_id, model_id, max_tokens, temperature, top_p, top_k)
    # Define on-memeory vector store
    vectorstore = Chroma(embedding_function=embedding)    
    # Search
    search = GoogleSearchAPIWrapper(google_api_key=os.getenv("google_api_key"), google_cse_id=os.getenv("google_cse_id"))
    # Initialize
    web_research_retriever = WebResearchRetriever.from_llm(
        vectorstore=vectorstore, llm=chat, search=search
    )

    # Form retrievl chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    messages = [
        ("system", """Your are a helpful assistant to provide comprehensive and truthful answers to questions, \n
                    drawing upon all relevant information contained within the specified in {context}. \n 
                    You add value by analyzing the situation and offering insights to enrich your answer. \n
                    Simply say I don't know if you can not find any evidence to match the question. \n
                    Display the source urls with clicable hyperlinks at the end of your answer.
                    """),
        #MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
    prompt_template = ChatPromptTemplate.from_messages(messages)
    
    # Reranker
    compression_retriever = ContextualCompressionRetriever(
        base_compressor= FlashrankRerank(), base_retriever=web_research_retriever
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
    

def tgi_textGen(option, prompt, max_token, temperature, top_p, top_k):
    try:
        llm = HuggingFaceTextGenInference(
            inference_server_url=option,
            max_new_tokens=max_token,
            top_k=top_k,
            top_p=top_p,
            typical_p=top_p,
            truncate=None,
            #callbacks=callbacks,
            streaming=True,
            watermark=False,
            temperature=temperature,
            repetition_penalty=1.03,
        )
    except Excepption as err:
        print(f"An error occurred in tgi_textGen: {err}")

    c_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional programmer who can write Python codes based on the user input."),
        ("user", "{input}")
    ])
    output_parser = StrOutputParser()
    chain = c_prompt | llm | output_parser
    output = chain.invoke({"input": prompt})
    return output.strip().replace('Assistant:', '')

def tgi_textGen2(option, question, max_token, temperature, top_p, top_k):
    try:
        llm = HuggingFaceTextGenInference(
            inference_server_url=option,
            max_new_tokens=max_token,
            top_k=top_k,
            top_p=top_p,
            typical_p=top_p,
            truncate=None,
            callbacks=[StreamingStdOutCallbackHandler()],
            streaming=False,
            watermark=False,
            temperature=temperature,
            repetition_penalty=1.13,
        )
    except Excepption as err:
        print(f"An error occurred in tgi_textGen: {err}")
        
    #template = """Question: {question}
    #Answer: Let's think first and answer the question with your best effort with comprehensive and accurate info."""
    
    template = """
                Assistant:You are a world class assistant. Please think first and answer the {question} with comprehensive and accurate info.
                Question:{question}
                Answer:
               """
    
    prompt = PromptTemplate.from_template(template)

    
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    
    return llm_chain.run(question)

def tgi_client(option, question, max_token, temperature, top_p, top_k):
 
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    
    payload = {
        "inputs": question,
        "parameters": {
            "adapter_id": None,
            "best_of": 1,
            "decoder_input_details": False,
            "details": False,
            "do_sample": False,
            "frequency_penalty": 0.1,
            "grammar": None,
            "max_new_tokens": max_token,
            "repetition_penalty": 1.03,
            "return_full_text": False,
            "seed": None,
            "stop": [
                "<|eot_id|>"
            ],
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "truncate": None,
            "watermark": False
        }
    }
    
    response = requests.post(option, headers=headers, json=payload)
    
    if response.status_code == 200:
        response = json.dumps(response.json(), indent=2)
        return (response)
    else:
        print(f"Request failed with status code: {response.status_code}")
        return(response.text)
    '''
    try:
        llm = HuggingFaceTextGenInference(
            inference_server_url=option,
            max_new_tokens=max_token,
            top_k=top_k,
            top_p=top_p,
            typical_p=top_p,
            truncate=None,
            callbacks=[StreamingStdOutCallbackHandler()],
            streaming=False,
            watermark=False,
            temperature=temperature,
            repetition_penalty=1.13,
        )
    except Excepption as err:
        print(f"An error occurred in tgi_textGen: {err}")
        
    template = """
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        {system_prompt}
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        {user_prompt}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """

    # Added prompt template
    sys_template_str = "You are a world class assistant who can provide answers with comprehensive and accurate info"
    human_template_str = "Please think first and answer the user question: {question} ?"

    prompt = PromptTemplate.from_template(template.format(system_prompt = sys_template_str,
                                                          user_prompt = human_template_str))

    #llm_chain = LLMChain(prompt=prompt, system_prompt=system_promt, llm=llm)
    session = prompt | llm | StrOutputParser
    return session.invoke({"question":question})
    '''

# ------ Classification ----
def classify_query(query, classes: str, modelId: str):
    """
    Classify a query into 'Tech', 'Health', or 'General' using an LLM.

    :param query: The query string to classify.
    :param openai_api_key: Your OpenAI API key.
    :return: A string classification: 'Tech', 'Health', or 'General'.
    """
    bedrock_client = boto3.client('bedrock-runtime')
    
    # Constructing the prompt for the LLM
    prompt = f"Carefully analyze the {query} and select the most likely category from {classes}. Please answer directly with the category name from the {classes} only."
    payload = {
            "modelId": modelId,
            "contentType": "application/json",
            "accept": "application/json",
            "body": {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "temperature": 0.01,
                "top_k": 250,
                "top_p": 0.95,
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
    try:
        # Convert the payload to bytes
        body_bytes = json.dumps(payload['body']).encode('utf-8')
        # Invoke the model
        response = bedrock_client.invoke_model(
            body=body_bytes,
            contentType=payload['contentType'],
            accept=payload['accept'],
            modelId=payload['modelId']
        )


        #response = bedrock_client.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
        response_body = json.loads(response.get('body').read())
        classification = ''.join([item['text'] for item in response_body['content'] if item.get('type') == 'text'])
        # Assuming the most likely category is returned directly as the output text
        #classification = response.choices[0].text.strip()
        return classification
    except Exception as e:
        print(f"Error classifying query: {e}")
        return "Error"

def classify_query2(query: str, modelId: str):
    question_category_prompt = '''You are a seasoned text classifier and your task is to classify precisely the incoming questions to match the action type. Depending on your answer, question will be routed to the right action based on your type choice, so your task is crucial. There are 8 possible question types: 
    - Image Generation - Generate a new image out of the input text explicitly call out image genreration or image creation.
    - Video Generation - Generate a new video clip out of the input text explicitly call out video genreration or video creation.
    - Music Generation - Generate a new music clip out of the input text explicitly call out music genreration or msuic creation.
    - Background - Remove the background from the input image. 
    - Upscale - Enhance or upscale the image for higher resolution.
    - Segmentation - To segment out the objects specified in text and creating masks.
    - Conditioning - Condition the input image and genderate a similiar one guided by the input text.
    - Virtual Try-on - Explicitly call out virtual fashion demo to showcase on-line fashion styling by selecting a model and a garment.
    - Others - Image understanding and image caption creation.
Return in the output only one word to represent the top matched type from (IAMGE GENERATION, VIDEO GENERATION, MUSIC GENERATION, BACKGROUND, UPSCALE, SEGMENTATION, CONDITIONING or OTHERS).
'''

    chat, _ = config_bedrock(embedding_model_id="", model_id=modelId, max_tokens=128, temperature=0.01, top_p=0.99, top_k=20)
    messages = [
        SystemMessage(content=question_category_prompt), 
        HumanMessage(content=query)
    ]
    response = chat.invoke(messages)
    return response.content

# ---- Image gen --------
def image_to_base64(img) -> str:
    """Converts a PIL Image or local image file path to a base64 string"""
    if isinstance(img, str):
        if os.path.isfile(img):
            print(f"Reading image from file: {img}")
            with open(img, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        else:
            raise FileNotFoundError(f"File {img} does not exist")
    elif isinstance(img, Image.Image):
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    else:
        raise ValueError(f"Expected str (filename) or PIL Image. Got {type(img)}")

def bedrock_imageGen(model_id:str, prompt:str, iheight:int, iwidth:int, src_image, image_quality:str, image_n:int, cfg:float, seed:int):
    negative_prompts = [
                "poorly rendered",
                "poor background details",
                "poorly drawn objects",
                "poorly focused objects",
                "disfigured object features",
                "cartoon",
                "animation"
            ]
    titan_negative_prompts = ','.join(negative_prompts)
    #try:
    if 'amazon.titan-image-generator' in model_id:
        if cfg > 10.0:
           cfg = 10.0
        if src_image:
            src_img_b64 = image_to_base64(src_image)
            body = json.dumps(
                {
                    "taskType": "IMAGE_VARIATION",
                    "imageVariationParams": {
                        "text":prompt,   # Required
                        "negativeText": titan_negative_prompts,  # Optional
                        "images": [src_img_b64]
                    },
                    "imageGenerationConfig": {
                        "numberOfImages": image_n,   # Range: 1 to 5 
                        "quality": image_quality,  # Options: standard or premium
                        "height": iheight,         # Supported height list in the docs 
                        "width": iwidth,         # Supported width list in the docs
                        "cfgScale": cfg,       # Range: 1.0 (exclusive) to 10.0
                        "seed": seed             # Range: 0 to 214783647
                    }
                }
            )
        else:
            body = json.dumps(
                {
                    "taskType": "TEXT_IMAGE",
                    "textToImageParams": {
                        "text":prompt,   # Required
                        "negativeText": titan_negative_prompts  # Optional
                    },
                    "imageGenerationConfig": {
                        "numberOfImages": image_n,   # Range: 1 to 5 
                        "quality": image_quality,  # Options: standard or premium
                        "height": iheight,         # Supported height list in the docs 
                        "width": iwidth,         # Supported width list in the docs
                        "cfgScale": cfg,       # Range: 1.0 (exclusive) to 10.0
                        "seed": seed             # Range: 0 to 214783647
                    }
                }
            )
    elif 'stability:sd3-large' in model_id:
        style_preset = "photographic"  # (e.g. photographic, digital-art, cinematic, ...)
        clip_guidance_preset = "FAST_GREEN" # (e.g. FAST_BLUE FAST_GREEN NONE SIMPLE SLOW SLOWER SLOWEST)
        sampler = "K_DPMPP_2S_ANCESTRAL" # (e.g. DDIM, DDPM, K_DPMPP_SDE, K_DPMPP_2M, K_DPMPP_2S_ANCESTRAL, K_DPM_2, K_DPM_2_ANCESTRAL, K_EULER, K_EULER_ANCESTRAL, K_HEUN, K_LMS)
        if src_image:
            src_img_b64 = image_to_base64(src_image)
            body = json.dumps({
                "text_prompts": (
                        [{"text": prompt, "weight": 1.0}]
                        + [{"text": negprompt, "weight": -1.0} for negprompt in negative_prompts]
                    ),
                "cfg_scale": cfg,
                "seed": seed,
                "steps": 60,
                "style_preset": style_preset,
                "clip_guidance_preset": clip_guidance_preset,
                "sampler": sampler,
                "width": iwidth,
                "init_image": src_img_b64,
            })
        else:
            body = json.dumps({
                "text_prompts": (
                        [{"text": prompt, "weight": 1.0}]
                        + [{"text": negprompt, "weight": -1.0} for negprompt in negative_prompts]
                    ),
                "cfg_scale": cfg,
                "seed": seed,
                "steps": 60,
                "style_preset": style_preset,
                "clip_guidance_preset": clip_guidance_preset,
                "sampler": sampler,
                "width": iwidth,
            })
    elif 'ultra' in model_id:
        body = json.dumps({
                "prompt":prompt,
                "mode": "text-to-image",
                "aspect_ratio": "1:1",
                "output_format": "jpeg",
            })
            
    bedrock_client = boto3.client("bedrock-runtime",  region_name="us-west-2")
    response = bedrock_client.invoke_model(
        body=body, 
        modelId=model_id,
        accept="application/json", 
        contentType="application/json"
    )
    response_body = json.loads(response["body"].read())
    if "amazon.titan-image-generator" in model_id or "ultra" in model_id or 'sd3-large' in model_id:
        base64_image_data = response_body["images"][0]
    elif "stability.stable-diffusion" in  model_id:
        base64_image_data = response_body["artifacts"][0].get("base64")

    return base64_image_data

    #except ClientError:
    #    logger.error("Couldn't invoke Titan Image Generator Model")
    #    raise

# --- Titan color guided image generation and image background removal ---
def bedrock_image_processing(model_id:str, prompt:str, action_type:str, iheight:int, iwidth:int, src_image, color_string, image_quality:str, image_n:int, cfg:float, seed:int):
    negative_prompts = [
                "poorly rendered",
                "poor background details",
                "poorly drawn objects",
                "poorly focused objects",
                "disfigured object features",
                "cartoon",
                "animation"
            ]
    titan_negative_prompts = ','.join(negative_prompts)
    
    if 'titan-image-generator-v2' not in model_id:
        return "Please choose the Titan Image Generator v2 model."
        
    if cfg > 10.0:
       cfg = 10.0
    if src_image and 'generation' in action_type.lower():
        input_image = base64.b64encode(src_image).decode("utf-8")
        body = json.dumps(
            {
                "taskType": "COLOR_GUIDED_GENERATION",
                "colorGuidedGenerationParams": {
                    "text":prompt,   # Required
                    "negativeText": titan_negative_prompts,  # Optional
                    "referenceImage": input_image,
                    "colors": [color_string] # list of color hex codes
                },
                "imageGenerationConfig": {
                    "numberOfImages": image_n,   # Range: 1 to 5 
                    "quality": image_quality,  # Options: standard or premium
                    "height": iheight,         # Supported height list in the docs 
                    "width": iwidth,         # Supported width list in the docs
                    "cfgScale": cfg,       # Range: 1.0 (exclusive) to 10.0
                    "seed": seed             # Range: 0 to 214783647
                }
            }
        )
    elif src_image and 'conditioning' in action_type.lower():
        #resize if needed
        image = Image.open(BytesIO(src_image))
        width, height = image.size
        if width > 1408 or height > 1408:
            #image = image.resize((1408,1408), resample=Image.BICUBIC)
            image = image.resize((1408,1408), resample=Image.Resampling.LANCZOS)
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            resized_base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        else:
            resized_base64_image = base64.b64encode(src_image).decode("utf-8")
        
        body = json.dumps(
            {
                "taskType": "TEXT_IMAGE",
                "textToImageParams": {
                    "text": prompt,
                    "negativeText": negative_prompts,
                    "conditionImage": resized_base64_image,
                    "controlMode": "CANNY_EDGE", # or  SEGMENTATION
                    "controlStrength":0.75,
                },
                "imageGenerationConfig": {
                    "numberOfImages": image_n,   # Range: 1 to 5 
                    #"quality": image_quality,  # Options: standard or premium
                    "height": iheight,         # Supported height list in the docs 
                    "width": iwidth,         # Supported width list in the docs
                    "cfgScale": cfg,       # Range: 1.0 (exclusive) to 10.0
                    "seed": seed             # Range: 0 to 214783647
                }
            }
        )
    elif src_image and 'background' in action_type.lower():
        #resize if needed
        image = Image.open(BytesIO(src_image))
        width, height = image.size
        if width > 1408 or height > 1408:
            #image = image.resize((1408,1408), resample=Image.BICUBIC)
            image = image.resize((1408,1408), resample=Image.Resampling.LANCZOS)
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            resized_base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        else:
            resized_base64_image = base64.b64encode(src_image).decode("utf-8")
        
        body = json.dumps(
            {
                "taskType": "BACKGROUND_REMOVAL",
                "backgroundRemovalParams": {
                    "image": resized_base64_image,
                }
            }
        )

    bedrock_client = boto3.client("bedrock-runtime",  region_name="us-west-2")
    response = bedrock_client.invoke_model(
        body=body, 
        modelId=model_id,
        accept="application/json", 
        contentType="application/json"
    )
    response_body = json.loads(response["body"].read())
    if "amazon.titan-image-generator" in  model_id :
        base64_image_data = response_body["images"][0]
    elif "stability.stable-diffusion" in  model_id :
        base64_image_data = response_body["artifacts"][0].get("base64")

    return base64_image_data


#------------ Retrivel from serpapi search -----
def serp_search(query, model_id, embedding_model_id:str, max_tokens: int=2048, temperature: int=0.01, top_p: float=0.90, top_k: int=40,  doc_num: int=3):
    
    # Get the API token and prompt to use - you can modify this!
    prompt = hub.pull("hwchase17/react")
    os.environ['SERPAPI_API_KEY'] = os.getenv('serp_api_token')
    
    # Choose the LLM to use
    llm, embedding = config_bedrock(embedding_model_id, model_id, max_tokens, temperature, top_p, top_k)
    
    # Set up tools
    #tool1 = [TavilySearchResults(max_results=3, api_wrapper=tavily_search)]
    tool2 = load_tools(["serpapi"], llm=llm)
    
    # Construct the ReAct agent
    agent = create_react_agent(llm, tool2, prompt)
    
    # Create an agent executor by passing in the agent and tools
    agent_executor = AgentExecutor(agent=agent, tools=tool2, verbose=True, handle_parsing_errors=True)
    results = agent_executor.invoke({"input": query})

    # Get urls
    params = {"engine": "google", "q": query, "api_key": os.getenv("serp_api_token"), "num_results": doc_num}
    goog_search = GoogleSearch(params)
    data = goog_search.get_dict()
    urls = []
    for i in range(doc_num+1):
        url = data['organic_results'][i]['link']
        if 'youtube.com' not in url:
            print(url)
            urls.append(url)

    return results['output'], urls

# --- Estimate tokens ---
def estimate_tokens(text, method="max"):
    """
    Estimates the number of tokens in the given text.

    Parameters:
    text (str): The input text.
    method (str): The method to use for estimation. Can be "average", "words", "chars", "max", or "min".
        - "average": The average of the word-based and character-based estimates.
        - "words": The word count divided by 0.75.
        - "chars": The character count divided by 4.
        - "max": The maximum of the word-based and character-based estimates.
        - "min": The minimum of the word-based and character-based estimates.

    Returns:
    int: The estimated number of tokens.
    """
    if text is None:
        word_count = char_count = 0
    else:
        if isinstance(text, set):
            text = str(sorted(list(text)))
        word_count = len(text.split())
        char_count = len(text)

    tokens_word_est = word_count / 0.75
    tokens_char_est = char_count / 4

    if method == "average":
        return int((tokens_word_est + tokens_char_est) / 2)
    elif method == "words":
        return int(tokens_word_est)
    elif method == "chars":
        return int(tokens_char_est)
    elif method == "max":
        return int(max(tokens_word_est, tokens_char_est))
    elif method == "min":
        return int(min(tokens_word_est, tokens_char_est))
    else:
        raise ValueError("Invalid method. Use 'average', 'words', 'chars', 'max', or 'min'.")


#----- Talk to pdf  ------
def extract_text_and_tables(pdf_path):
    # Partition the PDF
    elements = partition_pdf(
                    filename=pdf_path,
                    extract_images_in_pdf=False,
                    infer_table_structure=True,
                    chunking_strategy="by_title", #https://docs.unstructured.io/api-reference/api-services/chunking#by-page-chunking-strategy
                    max_characters=200000,
                    new_after_n_chars=200000,
                    strategy="fast",
                    combine_text_under_n_chars=200000,
                )

    # Initialize lists to store text and tables
    texts = []
    tables = []

    # Iterate through elements
    for element in elements:
        #if isinstance(element, Table):
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(element)
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))

    return texts, tables
    
def image_parser(pdf_filename):
    # Open the PDF file
    pdf_file = fitz.open(pdf_filename)
    image_files = []
    text_image_pairs = defaultdict(list)
    # Iterate through each page
    for page_index in range(len(pdf_file)):
        page = pdf_file[page_index]
        image_list = page.get_images()
        text = page.get_text()
        
        # If there are images on the page
        if image_list:
            print(f"[+] Found {len(image_list)} images on page {page_index}")
    
            for image_index, img in enumerate(image_list, start=1):
                # Get the image xref
                xref = img[0]
    
                # Extract the image bytes
                base_image = pdf_file.extract_image(xref)
                image_bytes = base_image["image"]
    
                # Save the image to disk
                with open(f"page_{page_index+1}_image_{image_index}.png", "wb") as f:
                    f.write(image_bytes)
                    image_files.append(f"page_{page_index+1}_image_{image_index}.png")

                # Convert image bytes to PIL Image
                image = Image.open(io.BytesIO(image_bytes))
                
                # Extract text from image using OCR
                image_text = bedrock_get_img_description(image_2_text_model, image_caption_promt, image, max_tokens, temperature, top_p, top_k, stop_sequences)

                # Form teh image/text correlation
                text_image_pairs[page_index].append({
                    "text": text,
                    "image": image,
                    "image_text": image_text
                })
                
    # Close the PDF file
    pdf_file.close()
    return image_files, text_image_pairs


def parser_pdf(fpath: str, fname:str):
    # Combine the file path and file name
    full_path = os.path.join(fpath, fname)

    # Check if the file exists
    if not os.path.isfile(full_path):
        return "File not found", ""
        

    with ThreadPoolExecutor() as executor:
        # Get text, tables
        answer1 = executor.submit(extract_text_and_tables, full_path)
        # Get images
        #answer2 = executor.submit(image_parser, fpath+fname)
        texts, tables = answer1.result()
        #img_files, text_image_pairs = answer2.result()
  
    return texts, tables #, img_files, text_image_pairs 

def parse_pdf_to_xml(pdf_path):
    # Create the root element
    root = ET.Element("document")

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # Create a page element
            page_elem = ET.SubElement(root, "page", number=str(page_num + 1))

            # Extract text and preserve some structure
            words = page.extract_words(keep_blank_chars=True, use_text_flow=True)

            current_line = ET.SubElement(page_elem, "line")
            current_top = None

            for word in words:
                if current_top is None or abs(word['top'] - current_top) > 2:
                    # New line detected
                    current_line = ET.SubElement(page_elem, "line")
                    current_top = word['top']

                # Add word to the current line
                word_elem = ET.SubElement(current_line, "word")
                word_elem.text = word['text']
                #word_elem.set("x", str(word['x0']))
                #word_elem.set("y", str(word['top']))
                #word_elem.set("width", str(word['width']))
                #word_elem.set("height", str(word['height']))

    # Convert the XML tree to a string
    xml_string = ET.tostring(root, encoding='unicode')

    return xml_string

# --- OpenAi -----
def openai_textGen(model_name, prompt, max_output_tokens, temperature, top_p):
    openai_client = OpenAI(api_key=os.getenv('openai_api_token'))
    response = openai_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
            {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return response.choices[0].message.content

def o1_textGen(model_name, prompt, top_k):
    ''' Best Prompting Practices:
    1. Keep prompts concise and clear: The models thrive on short, direct instructions without the need for elaborate explanations.
    3. Avoid chain-of-thought prompts: These models handle reasoning internally, so there’s no need to ask them to “think step by step” or “explain your reasoning.”
    4. Utilize delimiters for clarity: Incorporate delimiters like triple quotation marks, XML tags, or section headers to clearly define different parts of the input.
    5. Limit extra context in retrieval-augmented generation (RAG): When adding context or documents, include only the most pertinent information to avoid complicating the model’s response.
    '''
    openai_client = OpenAI(api_key=os.getenv('openai_api_token'))
    response_with_limit = openai_client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_completion_tokens=top_k
    )
    return(response_with_limit.choices[0].message.content, response_with_limit.usage)


# --- Photo gen --- 
def gen_photo_bytes(prompt: str, url:str):
    response = requests.get(url+prompt)
    if response.status_code == 200:
        image_bytes = response.content
        return image_bytes
    else:
        print(f"Error fetching image from {url}")
        return None

# --- Image upscaleing  --- 
def upscale_image_bytes(image_bytes, prompt):
    files = {"file": image_bytes, "prompt": prompt}    
    #url = "http://infs.cavatar.info:8085//upscale/?image_bytes="
    url = "http://agent.cavatar.info:8085/upscale"
    response = requests.post(url, files=files)
    if response.status_code == 200:
        image_bytes_out = response.content
        return image_bytes_out
    else:
        print(f"Error fetching image from {url}")
        return None

# --- Image segmentation  --- 
def pred_image_bytes(image_bytes, prompt):
    files = {"file": image_bytes, "prompt": prompt}    
    #url = "http://infs.cavatar.info:8085//upscale/?image_bytes="
    url = "http://video.cavatar.info:8081/pred_image"
    response = requests.post(url, files=files)
    if response.status_code == 200:
        image_bytes_out = response.content
        return image_bytes_out
    else:
        print(f"Error calling SAM2 from {url}")
        return None

# -- llama 3.x pretty print
def llama_pretty_print(text):
    # Find the start of the assistant's response
    start = text.find("<|start_header_id|>assistant<|end_header_id|>\n\n")
    if start == -1:
        return "Assistant's response not found."

    # Extract the content between the start and end tags
    content = text[start + len("<|start_header_id|>assistant<|end_header_id|>\n\n"):]
    end = content.rfind("<|eot_id|>")
    if end != -1:
        content = content[:end]

    # Remove any remaining tags
    content = re.sub(r'<\|.*?\|>', '', content)

    # Strip leading/trailing whitespace and return
    return content.strip()