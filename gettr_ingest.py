import os
import requests
from langchain.document_loaders import SeleniumURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import xml.etree.ElementTree as ET
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from langchain.document_loaders import PyPDFLoader

from xml.etree.ElementTree import ElementTree

# Setup Chrome Driver, may need to change based on system
service = Service("/Users/vtbloise/Downloads/chromedriver_mac64/chromedriver")
options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
driver = webdriver.Chrome(service=service, options=options)

def extract_urls_from_sitemap(sitemap_url):
    #read from local file
    tree = ElementTree()
    root = tree.parse("sitemap.xml")
    
    #response = requests.get(sitemap_url)
    #if response.status_code != 200:
    #    print(f"Failed to fetch sitemap: {response.status_code}")
    #    return []
    #parser = ET.XMLParser(encoding='utf-8-sig')
    #sitemap_content = response.content
    #root = ET.fromstring(sitemap_content, parser=parser)

    # Extract the URLs from the sitemap
    urls = [
        elem.text
        for elem in root.iter("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
    ]

    return urls

def load_pdf_text(pdfs):
    loader = PyPDFLoader(pdfs)
    data = loader.load_and_split()

    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    #texts = text_splitter.split_documents(data)

    return data

def load_html_text(sitemap_urls):
    loader = SeleniumURLLoader(urls=sitemap_urls)
    data = loader.load()
    #print("data: ", data)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    texts = text_splitter.split_documents(data)
    pdfs = "https://arxiv.org/pdf/2108.05876.pdf" #
    texts.append(load_pdf_text(pdfs)) 
    pdfs = "https://workshop-proceedings.icwsm.org/pdf/2022_62.pdf"
    texts.append(load_pdf_text(pdfs))
    pdfs = "Gettr-ing_Deep_Insights_from_the_Social_Network_Ge.pdf"
    texts.append(load_pdf_text(pdfs))

    #print("texts: ", texts)

    return texts

def load_html_text_tester(sitemap_urls):
    loader = SeleniumURLLoader(urls=sitemap_urls)
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    texts = text_splitter.split_documents(data)

    return texts


def embed_text_OpenAI(texts, save_loc):
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    docsearch = FAISS.from_documents(texts, embeddings)

    docsearch.save_local(save_loc)

def embed_text_Bedrock(texts, save_loc):
    embeddings = BedrockEmbeddings(
        credentials_profile_name="default", region_name="us-east-1"
    )
    docsearch = FAISS.from_documents(texts, embeddings)

    docsearch.save_local(save_loc)

def embed_text_Bedrock_with_timeout_avoid_logic(texts, save_loc):
    embeddings = BedrockEmbeddings(
        credentials_profile_name="default", region_name="us-east-1"
    )

    # Split texts to batchs of 100, then merge each batch
    final_db = None
    for i in range(0, len(texts), 1):
        # An array of i to i+50
        print(f"Getting docs from {i} to {i+1} of {len(texts)}")
        sample_texts = texts[i : i + 1]
        try:
            temp_db = FAISS.from_documents(sample_texts, embeddings)
        except Exception as e:
            print("Failed o well")
            continue

        if i == 0:
            final_db = temp_db

        if i > 0:
            final_db.merge_from(temp_db)

    final_db.save_local(save_loc)


def main() -> None:
    """
    Purpose:
        Ingest data into a a local db
    Args:
        N/A
    Returns:
        N/A
    """
    # Site maps for the AWS Well-Architected Framework
    sitemap_url_list = [
        "https://gettr.com/sitemap.xml",
    ]

    # Get all links from the sitemaps
    full_sitemap_list = []
    for sitemap in sitemap_url_list:
        full_sitemap_list.extend(extract_urls_from_sitemap(sitemap))

    midpoint = int(len(full_sitemap_list)/8)
    # no need to split url list
    print(midpoint)
    #half_sitemap_list = full_sitemap_list[:midpoint]
    half_sitemap_list = full_sitemap_list
    print(half_sitemap_list)
    # get the raw html text
    texts = load_html_text(half_sitemap_list)
    print("TEXTS\n")
    print(texts)
    # Save embeddings to local_index
    embed_text_Bedrock_with_timeout_avoid_logic(texts, "local_index_gettr")


if __name__ == "__main__":
    main()
