import anthropic
import os
import boto3
import io
from io import BytesIO
import base64
import json
from PIL import Image
from langchain_anthropic import ChatAnthropic
#from langchain_aws.embeddings.bedrock import BedrockEmbeddings
from langchain_community.embeddings import BedrockEmbeddings
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


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


def anthropic_textGen(option, prompt, max_token, temperature, top_p, top_k, stop_sequences):
    #vlm = ChatAnthropic(model=option)   
    client = anthropic.Anthropic(api_key=os.environ.get("anthropic_api_token"))
    base64Frames_20 = []
    
    message = client.messages.create(
        model=option,
        max_tokens=max_token,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        stop_sequences=[stop_sequences],
        system="Your are a domain expert. Please answer the query accurately and precisely. ",
        messages=[
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
    )
    text_strings = [block.text for block in message.content]
    return(text_strings[0])

def anthropic_imageCaption(option, prompt, image, max_token, temperature, top_p, top_k):
    client = anthropic.Anthropic(api_key=os.environ.get("anthropic_api_token"))
    if isinstance(image, io.BytesIO):
        image = Image.open(image)
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=max_token,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": 'image/jpeg',
                            "data": convert_image_to_base64(image),
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ],
            }
        ],
    )
    #message_dict = json.loads(message)
    return message.content[0].text
    
def retrieval_faiss_anthropic(query, documents, model_id, embedding_model_id:str, max_tokens: int=2048, temperature: int=0.01, top_p: float=0.90, top_k: int=25, doc_num: int=3):
    #text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=over_lap)
    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=over_lap, length_function=len, is_separator_regex=False,)
    #docs = text_splitter.split_documents(documents)
    # No need to split doc since C3.5 allows 200K 
    # Prepare embedding function
    #chat, embedding = config_bedrock(embedding_model_id, model_id, max_tokens, temperature, top_p, top_k)
    bedrock_client = boto3.client('bedrock-runtime')
    embedding_bedrock = BedrockEmbeddings(client=bedrock_client, model_id=embedding_model_id)
    os.environ['ANTHROPIC_API_KEY'] = os.environ.get("anthropic_api_token")
    chat =  ChatAnthropic(model=model_id,temperature=temperature, max_tokens=max_tokens, top_p=top_p, top_k=top_k)
    # Try to get vectordb with FAISS
    db = FAISS.from_documents(documents, embedding_bedrock)
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