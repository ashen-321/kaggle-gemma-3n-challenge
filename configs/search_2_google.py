import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from googleapiclient.discovery import build
import html2text
from langchain.schema import Document
from concurrent.futures import ThreadPoolExecutor
import os

def google_search(query, num_results=5):
    """
    Perform a Google search and return top URLs using requests and BeautifulSoup

    Parameters:
        query (str): Search query
        num_results (int): Number of results to return (default: 5)

    Returns:
        list: List of URLs
    """
    try:
        # Format the query and create the URL
        search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"

        # Headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # Send the request
        response = requests.get(search_url, headers=headers)
        response.raise_for_status()

        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all search result divs
        search_results = soup.find_all('div', class_='yuRUbf')

        # Extract URLs
        urls = []
        for result in search_results[:num_results]:
            link = result.find('a')
            if link:
                url = link.get('href')
                if url and url.startswith('http'):
                    urls.append(url)

        return urls

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return []

def search_google(query, num_results=5):
    # You need to get these from Google Cloud Console
    GOOGLE_API_KEY = os.getenv('google_api_key')
    SEARCH_ENGINE_ID = os.getenv('google_cse_id')

    # Create a service object for the Custom Search API
    service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)

    try:
        # Execute the search
        result = service.cse().list(
            q=query,
            cx=SEARCH_ENGINE_ID,
            num=num_results
        ).execute()

        # Extract URLs from the search results
        urls = []
        if 'items' in result:
            for item in result['items']:
                urls.append(item['link'])

        return urls

    except Exception as e:
        print(f"An error occurred during the search: {e}")
        return []

def fetch_and_parse_webpage(url):
    try:
        # Send HTTP request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Convert HTML to plain text
        h = html2text.HTML2Text()
        h.ignore_links = True
        text = h.handle(str(soup))

        # Clean up the text
        text = ' '.join(text.split())

        return text

    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None


def create_documents_from_urls(urls):
    documents = []

    for url in urls:
        content = fetch_and_parse_webpage(url)
        if content:
            # Create a Document object
            doc = Document(
                page_content=content,
                metadata={
                    "source": url,
                    "type": "webpage"
                }
            )
            documents.append(doc)

    return documents

def search_2_google(search_query:str, num_results:int=3):
    with ThreadPoolExecutor() as executor:
        answer1 =  executor.submit(google_search, search_query, num_results)
        answer2 =  executor.submit(search_google, search_query, num_results)
        url1 = answer1.result()
        url2 = answer2.result()
        #d_answer1 =  executor.submit(create_documents_from_urls, url1)
        #d_answer2 =  executor.submit(create_documents_from_urls, url2)
    urls= list(set(url1 + url2))
    # Convert content into Lancgahin document
    documents = create_documents_from_urls(urls)
    return documents, urls

    
def main():
    # Example usage
    search_query = input("Enter your search query:")

    print("\nSearching...")
    results_1 = google_search(search_query)
    results_2 = search_google(search_query)
    results= list(set(results_1 + results_2))

    if results:
        print("\nTop 5 results:")
        for i, url in enumerate(results, 1):
            print(f"{i}. {url}")
    else:
        print("No results found or an error occurred.")

    # Convert content into Lancgahin document
    documents = create_documents_from_urls(results)

    # Print results
    print(f"\nCreated {len(documents)} documents:")
    for doc in documents:
        print(f"\nSource: {doc.metadata['source']}")
        print(f"Content preview: {doc.page_content[:200]}...")

if __name__ == "__main__":
    main()