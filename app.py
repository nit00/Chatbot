from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import AwaEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import requests
from flask import Flask,request,jsonify

app = Flask(__name__)


#LangchainSummarization

# repo_id="google/flan-t5-xxl"
# llm = HuggingFaceHub(
#     repo_id=repo_id, model_kwargs={"temperature": 0.5}
# )


@app.route('/query', methods=['POST'])
def chatbot_process():
    loader = TextLoader("test.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    embeddings =GPT4AllEmbeddings()
    os.environ["GOOGLE_API_KEY"]="AIzaSyC1v_n6qGnxEbQ1RLLkDjooFaWximsp0B4"
    db = FAISS.from_documents(docs,embeddings)
    data=request.get_json()
    user_query=data['query']
    docs = db.similarity_search(user_query)
    text=(docs[0].page_content)
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    template = """Answer the given question by using the below given context:
    Question: {query},
    Context: {text}

    And give response in a summarised manner in 50 words.
    """

    prompt = PromptTemplate(template=template, input_variables=["query","text"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response=llm_chain.run({"query":user_query,"text":text})
    print(response)
    return str(response)


if __name__ == '__main__':
    app.run(debug=True)