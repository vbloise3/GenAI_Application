import json

import boto3
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from transformers import Tool
import os
import pinecone
import ai21

ai21.api_key = os.environ.get('AI21_API_KEY')

bedrock = boto3.client(
    service_name="bedrock",
    region_name="us-east-1",
    endpoint_url="https://bedrock.us-east-1.amazonaws.com",
)

# get api key from app.pinecone.io
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY') 
# find your environment next to the api key in pinecone console
PINECONE_ENV = os.environ.get('PINECONE_ENVIRONMENT') 

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

def call_ai21(prompt):
    response = ai21.Completion.execute(
    model='j1-large',
    prompt=prompt,
    temperature=0.65,
    minTokens=4,
    maxTokens=32,
    numResults=1
)
    return response

def call_bedrock(prompt):
    prompt_config = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 4096,
            "stopSequences": [],
            "temperature": 0.5,
            "topP": 0.2,
        },
    }

    body = json.dumps(prompt_config)

    modelId = "amazon.titan-tg1-large" #"ai21.j2-ultra" 
    accept = "application/json"
    contentType = "application/json"

    response = bedrock.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("results")[0].get("outputText")
    return results


class AWSWellArchTool(Tool):
    name = "well_architected_tool"
    description = "Use this tool for any AWS related question to help customers understand best practices on building on AWS. It will use the relevant context from the AWS Well-Architected Framework to answer the customer's query. The input is the customer's question. The tool returns an answer for the customer using the relevant context."
    inputs = ["text"]
    outputs = ["text"]

    def __call__(self, query, translation_language):
        # Find docs
        embeddings = BedrockEmbeddings()
        vectorstore = FAISS.load_local("local_index", embeddings)
        docs = vectorstore.similarity_search(query)
                
        context = ""

        doc_sources_string = ""
        for doc in docs:
            doc_sources_string += doc.metadata["source"] + "\n"
            context += doc.page_content

        prompt = f"""Use the following pieces of context to answer the question at the end. Give a very detailed, long answer.

        {context}

        Question: {query}
        Answer:"""
        # print prompt for log
        print("prompt:\n")
        print(prompt)
        print("\nend of prompt\n")

        generated_text = call_bedrock(prompt)
        #print(generated_text)
        #translate
        if translation_language:
            print("translating\n")
            translation_prompt = generated_text 
            try:
                translated_text = call_ai21(translation_prompt)
                resp_json = {"ans": str(translated_text['prompt']['text']), "docs": doc_sources_string}
            except:
                resp_json = "none"
        else:
            print("not translating\n")
            resp_json = {"ans": str(generated_text), "docs": doc_sources_string}
        return resp_json

class MyelomaFoundationTool(Tool):
    name = "myeloma_foundation_tool"
    description = "Use this tool for any Myeloma Foundation related question to help customers understand best practices on building on AWS. It will use the relevant context from the AWS Well-Architected Framework to answer the customer's query. The input is the customer's question. The tool returns an answer for the customer using the relevant context."
    inputs = ["text"]
    outputs = ["text"]

    def __call__(self, query, translation_language):
        # Find docs
        embeddings = BedrockEmbeddings()
        vectorstore = FAISS.load_local("local_index_myeloma_foundation", embeddings)
        docs = vectorstore.similarity_search(query)
        
        context = ""

        doc_sources_string = ""
        for doc in docs:
            doc_sources_string += doc.metadata["source"] + "\n"
            context += doc.page_content

        prompt = f"""Use the following pieces of context to answer the question at the end. Give a very detailed, long answer.

        {context}

        Question: {query}
        Answer:"""

        # print prompt for log
        print("prompt:\n")
        print(prompt)
        print("\nend of prompt\n")

        generated_text = call_bedrock(prompt)
        #print(generated_text)
        #translate
        if translation_language:
            print("translating\n")
            translation_prompt = generated_text 
            try:
                translated_text = call_ai21(translation_prompt)
                resp_json = {"ans": str(translated_text['prompt']['text']), "docs": doc_sources_string}
            except:
                resp_json = "none"
        else:
            print("not translating\n")
            resp_json = {"ans": str(generated_text), "docs": doc_sources_string}
        return resp_json
class WhoWePlayForTool(Tool):
    name = "who_we_play_for_tool"
    description = "Use this tool for any Who We Play For related question to help customers understand best practices on building on AWS. It will use the relevant context from the AWS Well-Architected Framework to answer the customer's query. The input is the customer's question. The tool returns an answer for the customer using the relevant context."
    inputs = ["text"]
    outputs = ["text"]

    def __call__(self, query, translation_language):
        # Find docs
        embeddings = BedrockEmbeddings()
        vectorstore = FAISS.load_local("local_index_who_we_play_for", embeddings)
        docs = vectorstore.similarity_search(query)
        
        context = ""

        doc_sources_string = ""
        for doc in docs:
            doc_sources_string += doc.metadata["source"] + "\n"
            context += doc.page_content

        prompt = f"""Use the following pieces of context to answer the question at the end. Give a very detailed, long answer.

        {context}

        Question: {query}
        Answer:"""

        # print prompt for log
        print("prompt:\n")
        print(prompt)
        print("\nend of prompt\n")

        generated_text = call_bedrock(prompt)
        #print(generated_text)
        #translate
        if translation_language:
            print("translating\n")
            translation_prompt = generated_text 
            try:
                translated_text = call_ai21(translation_prompt)
                resp_json = {"ans": str(translated_text['prompt']['text']), "docs": doc_sources_string}
            except:
                resp_json = "none"
        else:
            print("not translating\n")
            resp_json = {"ans": str(generated_text), "docs": doc_sources_string} 
        return resp_json

class BroadviewFCUTool(Tool):
    name = "broadview_tool"
    description = "Use this tool for any Broadview FCU related question to help customers understand best practices on building on AWS. It will use the relevant context from the AWS Well-Architected Framework to answer the customer's query. The input is the customer's question. The tool returns an answer for the customer using the relevant context."
    inputs = ["text"]
    outputs = ["text"]

    def __call__(self, query, translation_language):
        # Find docs
        embeddings = BedrockEmbeddings()
        vectorstore = FAISS.load_local("local_index_broadview_fcu", embeddings)
        docs = vectorstore.similarity_search(query)
        
        context = ""

        doc_sources_string = ""
        for doc in docs:
            doc_sources_string += doc.metadata["source"] + "\n"
            context += doc.page_content

        prompt = f"""Use the following pieces of context to answer the question at the end. Give a very detailed, long answer.

        {context}

        Question: {query}
        Answer:"""

        # print prompt for log
        print("prompt:\n")
        print(prompt)
        print("\nend of prompt\n")

        generated_text = call_bedrock(prompt)
        #print(generated_text)
        #translate
        if translation_language:
            print("translating\n")
            translation_prompt = generated_text 
            try:
                translated_text = call_ai21(translation_prompt)
                resp_json = {"ans": str(translated_text['prompt']['text']), "docs": doc_sources_string}
            except:
                resp_json = "none"
        else:
            print("not translating\n")
            resp_json = {"ans": str(generated_text), "docs": doc_sources_string}
        return resp_json
    
class TEDTool(Tool):
    name = "ted_tool"
    description = "Use this tool for any TED related question to help customers understand best practices on building on AWS. It will use the relevant context from the AWS Well-Architected Framework to answer the customer's query. The input is the customer's question. The tool returns an answer for the customer using the relevant context."
    inputs = ["text"]
    outputs = ["text"]

    def __call__(self, query, translation_language):
        # Find docs
        embeddings = BedrockEmbeddings()
        vectorstore = FAISS.load_local("local_index_ted", embeddings)
        docs = vectorstore.similarity_search(query)
        
        context = ""

        doc_sources_string = ""
        for doc in docs:
            doc_sources_string += doc.metadata["source"] + "\n"
            context += doc.page_content

        prompt = f"""Use the following pieces of context to answer the question at the end. Give a very detailed, long answer.

        {context}

        Question: {query}
        Answer:"""

        # print prompt for log
        print("prompt:\n")
        print(prompt)
        print("\nend of prompt\n")
        generated_text = call_bedrock(prompt)
        # print(generated_text)
        #translate
        if translation_language:
            print("translating\n")
            translation_prompt = generated_text 
            try:
                translated_text = call_ai21(translation_prompt)
                resp_json = {"ans": str(translated_text['prompt']['text']), "docs": doc_sources_string}
            except:
                resp_json = "none"
        else:
            print("not translating\n")
            resp_json = {"ans": str(generated_text), "docs": doc_sources_string}
        return resp_json

class GettrTool(Tool):
    name = "gettr_tool"
    description = "Use this tool for any Gettr related question to help customers understand best practices on building on AWS. It will use the relevant context from the AWS Well-Architected Framework to answer the customer's query. The input is the customer's question. The tool returns an answer for the customer using the relevant context."
    inputs = ["text"]
    outputs = ["text"]

    def __call__(self, query, translation_language):
        # Find docs
        embeddings = BedrockEmbeddings()
        vectorstore = FAISS.load_local("local_index_gettr", embeddings)
        docs = vectorstore.similarity_search(query)
        
        context = ""

        doc_sources_string = ""
        for doc in docs:
            doc_sources_string += doc.metadata["source"] + "\n"
            context += doc.page_content

        prompt = f"""Use the following pieces of context to answer the question at the end. Give a very detailed, long answer.

        {context}

        Question: {query}
        Answer:"""

        # print prompt for log
        print("prompt:\n")
        print(prompt)
        print("\nend of prompt\n")

        generated_text = call_bedrock(prompt)
        #print(generated_text)
        #translate
        if translation_language:
            print("translating\n")
            translation_prompt = generated_text 
            try:
                translated_text = call_ai21(translation_prompt)
                resp_json = {"ans": str(translated_text['prompt']['text']), "docs": doc_sources_string}
            except:
                resp_json = "none"
        else:
            print("not translating\n")
            resp_json = {"ans": str(generated_text), "docs": doc_sources_string}
        return resp_json
    
class CK_12Tool(Tool):
    name = "ck_12_tool"
    description = "Use this tool for any CK-12 related question to help customers understand best practices on building on AWS. It will use the relevant context from the AWS Well-Architected Framework to answer the customer's query. The input is the customer's question. The tool returns an answer for the customer using the relevant context."
    inputs = ["text"]
    outputs = ["text"]

    def __call__(self, query, translation_language):
        # Find docs
        embeddings = BedrockEmbeddings()
        vectorstore = FAISS.load_local("local_index_ck_12", embeddings)
        docs = vectorstore.similarity_search(query)
        
        context = ""

        doc_sources_string = ""
        for doc in docs:
            doc_sources_string += doc.metadata["source"] + "\n"
            context += doc.page_content

        prompt = f"""Use the following pieces of context to answer the question at the end. Give a very detailed, long answer.

        {context}

        Question: {query}
        Answer:"""

        # print prompt for log
        print("prompt:\n")
        print(prompt)
        print("\nend of prompt\n")

        generated_text = call_bedrock(prompt)
        #print(generated_text)
        #translate
        if translation_language:
            print("translating\n")
            translation_prompt = generated_text 
            try:
                translated_text = call_ai21(translation_prompt)
                resp_json = {"ans": str(translated_text['prompt']['text']), "docs": doc_sources_string}
            except:
                resp_json = "none"
        else:
            print("not translating\n")
            resp_json = {"ans": str(generated_text), "docs": doc_sources_string}
        return resp_json
    
class CFATool(Tool):
    name = "cfa_tool"
    description = "Use this tool for any CFA related question to help customers understand best practices on building on AWS. It will use the relevant context from the AWS Well-Architected Framework to answer the customer's query. The input is the customer's question. The tool returns an answer for the customer using the relevant context."
    inputs = ["text"]
    outputs = ["text"]

    def __call__(self, query, translation_language):
        # Find docs
        embeddings = BedrockEmbeddings()
        vectorstore = FAISS.load_local("local_index_cfa_institute", embeddings)
        docs = vectorstore.similarity_search(query)
        
        context = ""

        doc_sources_string = ""
        for doc in docs:
            doc_sources_string += doc.metadata["source"] + "\n"
            context += doc.page_content

        prompt = f"""Use the following pieces of context to answer the question at the end. Give a very detailed, long answer.

        {context}

        Question: {query}
        Answer:"""

        # print prompt for log
        print("prompt:\n")
        print(prompt)
        print("\nend of prompt\n")

        generated_text = call_bedrock(prompt)
        #print(generated_text)
        #translate
        if translation_language:
            print("translating\n")
            translation_prompt = generated_text 
            try:
                translated_text = call_ai21(translation_prompt)
                resp_json = {"ans": str(translated_text['prompt']['text']), "docs": doc_sources_string}
            except:
                resp_json = "none"
        else:
            print("not translating\n")
            resp_json = {"ans": str(generated_text), "docs": doc_sources_string}
        return resp_json
    
class JehovasWitnessTool(Tool):
    name = "jw_tool"
    description = "Use this tool for any Jehova's Witness related question to help customers understand best practices on building on AWS. It will use the relevant context from the AWS Well-Architected Framework to answer the customer's query. The input is the customer's question. The tool returns an answer for the customer using the relevant context."
    inputs = ["text"]
    outputs = ["text"]

    def __call__(self, query, translation_language):
        # Find docs
        embeddings = BedrockEmbeddings()
        vectorstore = FAISS.load_local("local_index_jehovah_witness", embeddings)
        docs = vectorstore.similarity_search(query)
        
        context = ""

        doc_sources_string = ""
        for doc in docs:
            doc_sources_string += doc.metadata["source"] + "\n"
            context += doc.page_content

        prompt = f"""Use the following pieces of context to answer the question at the end. Give a very detailed, long answer.

        {context}

        Question: {query}
        Answer:"""

        # print prompt for log
        #print("prompt:\n")
        #print(prompt)
        #print("\nend of prompt\n")

        generated_text = call_bedrock(prompt)
        #print(generated_text)
        #translate
        if translation_language:
            print("translating\n")
            translation_prompt = generated_text 
            try:
                translated_text = call_ai21(translation_prompt)
                resp_json = {"ans": str(translated_text['prompt']['text']), "docs": doc_sources_string}
            except:
                resp_json = "none"
        else:
            print("not translating\n")
            resp_json = {"ans": str(generated_text), "docs": doc_sources_string}
        return resp_json

class CodeGenerationTool(Tool):
    name = "code_generation_tool"
    description = "Use this tool only when you need to generate code based on a customers's request. The input is the customer's question. The tool returns code that the customer can use."

    inputs = ["text"]
    outputs = ["text"]

    def __call__(self, prompt, translation_language):
        generated_text = call_bedrock(prompt)
        return generated_text



#### Testing Well Architected Tool
# query = "How can I design secure VPCs?"
# well_arch_tool = AWSWellArchTool()
# output = well_arch_tool(query)
# print(output)


#### Testing Code Generation Tool
# query = "Write a function in Python to upload a file to Amazon S3"
# code_gen_tool = CodeGenerationTool()
# output = code_gen_tool(query)
# print(output)
