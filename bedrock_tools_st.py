import json

import boto3
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from transformers import Tool
import streamlit as st
from bedrock_tools import AWSWellArchTool, CodeGenerationTool, CustomerExampleTool

well_arch_tool = AWSWellArchTool()
code_gen_tool = CodeGenerationTool()
customer_example_tool = CustomerExampleTool()


bedrock = boto3.client(
    service_name="bedrock",
    region_name="us-east-1",
    endpoint_url="https://bedrock.us-east-1.amazonaws.com",
)


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

    modelId = "amazon.titan-tg1-large"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("results")[0].get("outputText")
    return results





#### Testing Well Architected Tool
#query = "How can I design secure VPCs?"
#well_arch_tool = AWSWellArchTool()
#output = well_arch_tool(query)
#print(output)


#### Testing Code Generation Tool
# query = "Write a function in Python to upload a file to Amazon S3"
# code_gen_tool = CodeGenerationTool()
# output = code_gen_tool(query)
# print(output)

def app() -> None:
    """
    Purpose:
        Controls the app flow
    Args:
        N/A
    Returns:
        N/A
    """

    # Choose tool
    current_tool = st.selectbox(
        "Choose Tool:", ["AWS Well Architected Tool", "Code Generation Tool", "Customer Q&A"]
    )

    query = st.text_input("Query:")

    if st.button("Submit Query"):
        with st.spinner("Generating..."):
            if current_tool == "AWS Well Architected Tool":
                answer = well_arch_tool(query)
            elif current_tool == "Customer Q&A":
                answer = customer_example_tool(query)
            elif current_tool == "Code Generation Tool":
                print("codegen")
                answer = code_gen_tool(query)

            if type(answer) == dict:
                st.markdown(answer["ans"])
                docs = answer["docs"].split("\n")

                with st.expander("Resources"):
                    for doc in docs:
                        st.write(doc)
            else:
                st.markdown(answer)


def main() -> None:
    """
    Purpose:
        Controls the flow of the streamlit app
    Args:
        N/A
    Returns:
        N/A
    """

    # Start the streamlit app
    st.title("Bedrock Q&A")

    app()


if __name__ == "__main__":
    main()
