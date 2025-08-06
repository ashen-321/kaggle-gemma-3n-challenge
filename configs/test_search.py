import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote
import multiprocessing

def search_engines(query, num_results=10):
    def google_search(query):
        
        url = f"https://www.google.com/search?q={quote(query)}"
        print(url)
        #url = f"https://www.google.com/search?q={query}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        links = soup.select(".yuRUbf a")
        return [link["href"] for link in links]

    def bing_search(query):
        url = f"https://www.bing.com/search?q={quote(query)}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        links = soup.select(".b_algo h2 a")
        return [link["href"] for link in links]
 
    with ThreadPoolExecutor() as executor:
        google_future = executor.submit(google_search, query)
        bing_future = executor.submit(bing_search, query)

        google_urls = google_future.result()
        bing_urls = bing_future.result()
    '''
    with multiprocessing.Pool(processes=2) as pool:
        google_results = pool.apply_async(google_search, args=(query))
        bing_results = pool.apply_async(bing_search, args=(query))

        # Get results from both processes
        google_urls = google_results.get()
        bing_urls = bing_results.get()
    '''
    combined_results = google_urls + bing_urls
    combined_results = list(set(combined_results))  # Remove duplicates
    #combined_results.sort(key=lambda x: combined_results.index(x))  # Preserve order

    return combined_results[:num_results]


if __name__ == "__main__":
    #create a keyword search string from the following input and generate search string with format 'keyword1+keyword2+...': MOE mixture of experts mistral
    search_keywords = "moe+Mixture+experts+model+router".lower()  # Replace with your actual search terms
    top_urls = search_engines(search_keywords, 10)
    print("Top URLs:", top_urls)