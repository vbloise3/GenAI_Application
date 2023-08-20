import json
import sys
import boto3
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from transformers import Tool
import streamlit as st
from bedrock_tools import AWSWellArchTool, CodeGenerationTool, GettrTool, CK_12Tool, CFATool, JehovasWitnessTool, BroadviewFCUTool, TEDTool, MyelomaFoundationTool, WhoWePlayForTool

well_arch_tool = AWSWellArchTool()
code_gen_tool = CodeGenerationTool()
gettr_tool = GettrTool()
ck_12_tool = CK_12Tool()
cfa_tool = CFATool()
jw_tool = JehovasWitnessTool()
broadview_tool = BroadviewFCUTool()
ted_tool = TEDTool()
myeloma_foundation_tool = MyelomaFoundationTool()
who_we_play_for_tool = WhoWePlayForTool()


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

    modelId = "amazon.titan-tg1-large" #"ai21.j2-ultra" 
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

def app(multilingual) -> None:
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
        "Choose Tool:", ["AWS Well Architected Tool", "Code Generation Tool", "Broadview FCU Q&A", "Myeloma Foundation Q&A", "TED Q&A", "Gettr Q&A", "CK-12 Q&A", "Jehova's Witness Q&A", "CFA Institute Q&A", "Who We Play For Q&A"]
    )

    query = st.text_input("Query:")

    #translation_language = st.selectbox(
    #    "Choose Language:", ["English", "French", "Spanish", "German", "Dutch", "Italian", "Arabic", "Bengali", "Esperanto"]
    #)
    translation_language = multilingual

    if st.button("Submit Query"):
        with st.spinner("Generating..."):
            if current_tool == "AWS Well Architected Tool":
                answer = well_arch_tool(query, translation_language)
            elif current_tool == "Broadview FCU Q&A":
                answer = broadview_tool(query, translation_language)
            elif current_tool == "TED Q&A":
                answer = ted_tool(query, translation_language)
            elif current_tool == "Gettr Q&A":
                answer = gettr_tool(query, translation_language)
            elif current_tool == "CK-12 Q&A":
                answer = ck_12_tool(query, translation_language)
            elif current_tool == "Jehova's Witness Q&A":
                answer = jw_tool(query, translation_language)
            elif current_tool == "CFA Institute Q&A":
                answer = cfa_tool(query, translation_language)
            elif current_tool == "Myeloma Foundation Q&A":
                answer = myeloma_foundation_tool(query, translation_language)
            elif current_tool == "Who We Play For Q&A":
                answer = who_we_play_for_tool(query, translation_language)
            elif current_tool == "Code Generation Tool":
                print("codegen")
                answer = code_gen_tool(query, translation_language)

            if type(answer) == dict:
                st.markdown(answer["ans"])
                docs = answer["docs"].split("\n")

                with st.expander("Resources"):
                    for doc in docs:
                        st.write(doc)
            else:
                st.markdown(answer)


def main(multilingual) -> None:
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

    app(multilingual)


if __name__ == "__main__":
    multilingual = bool(int(sys.argv[1]))
    print('multilingual: ' + str(multilingual))
    main(multilingual)
