from langchain_community.document_loaders import WebBaseLoader, AsyncHtmlLoader, PyPDFLoader, PyMuPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader, PyPDFLoader, PyMuPDFLoader, UnstructuredPDFLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_aws import BedrockEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from PIL import Image
import pdfkit
import io
import time
import os
import requests

#import { WebPDFLoader } from "langchain/document_loaders/web/pdf";
import nest_asyncio
from bedrock import * 

nest_asyncio.apply()
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
#os.environ['USER_AGENT'] = 'langchain_agent'
#os.environ["LANGCHAIN_API_KEY"] = os.getenv("langsmith_api_token")

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

def url_to_pdf(url):
    try:
        # Convert the URL page to PDF
        pdf_bytes = pdfkit.from_url(url, False)
        return pdf_bytes
    except Exception as e:
        print(f"Error converting URL to PDF: {e}")
        return None
        
def url_to_image(url, width=1920, height=1080):
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode (no GUI)
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument(f"--window-size={width},{height}")

    # Path to your ChromeDriver
    service = Service('./chromedriver')  # Update this path
    # Initialize the Chrome driver
    driver = webdriver.Chrome(service=service, options=chrome_options)
    image = None
    try:
        # Navigate to the URL
        driver.get(url)

        # Wait for the page to load (you may need to adjust the wait time)
        time.sleep(2)

        # Capture the screenshot
        screenshot = driver.get_screenshot_as_png()

        # Convert the screenshot to a PIL Image
        image = Image.open(io.BytesIO(screenshot))  

    finally:
        # Close the browser
        driver.quit()

    return image
        
def extract_urls(urls: list, query: str, model_id: str, embedding_model_id: str):
    loaders = []
    llm_c3 = get_llm(model_id=model_id)
    br_embedding = get_embedding(model_id=embedding_model_id)\

    # Load
    xml_loader = WebBaseLoader(urls)
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
    from langchain_community.document_transformers import Html2TextTransformer

    html2text = Html2TextTransformer()
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
    return res['answer']

if __name__ == "__main__":
    start_time = time.time()
    url_strings = "https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/pdf/, https://lilianweng.github.io/posts/2023-06-23-agent/"
    urls = url_strings.split(",")
    embedding_model_id = "amazon.titan-embed-text-v2:0"
    model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    query = "Explain the differences between using unstructured and pymupdf to load pdf files"
    answer = extract_urls(urls, query, model_id, embedding_model_id)
    print(f"Answer: {answer}, Latency: {(time.time() - start_time) * 1000:.2f} ms")