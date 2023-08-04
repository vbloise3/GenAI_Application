# GenAIApplication
Generative AI application that implements Retrieval Augmented Generatioin using Titan text and Titan embeddings

# Create a demo that uses RAG with your customer's data with a vector database created with the Titan embeddings model

# Instructions
To load data into your pickle file, run ingest.py after you change the location of the sitemap.xml url in the code. 
The sitemap.xml url should match your customer's sitemap so you can demo RAG using you customer's data.

Then edit the bedrock_tools.py code. In bedrock_tools.py add a class for your customer embeddings that you ingested with ingest.py. 

Finally, edit the bedrock_tools_st.py code to include your customer class in the appropriate places.

You'll see in ingest.py that you need to install the Chrome Driver for Selenium to work. You'll need to install streamlit as well.

Also, run pip install -r requirements.txt using the requirements.txt file.

To run the app: streamlit run bedrock_tools_st.py 
