# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import AwaEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_community.document_loaders.csv_loader import CSVLoader
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
    #CSV Loader
    loader = CSVLoader(file_path="SampleData2.csv", encoding="utf-8",csv_args={
                'delimiter': ','})
    data = loader.load()
    # loader = TextLoader("test.txt")
    # documents = loader.load()
    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # docs = text_splitter.split_documents(data)
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    embeddings =GPT4AllEmbeddings()
    os.environ["GOOGLE_API_KEY"]="AIzaSyC1v_n6qGnxEbQ1RLLkDjooFaWximsp0B4"
    #Saving index local
    # db = FAISS.from_documents(data,embeddings)
    # db.save_local("faiss_index")
    new_db = FAISS.load_local("faiss_index", embeddings)
    data=request.get_json()
    user_query=data['query']
    docs = new_db.similarity_search(user_query)
    text2=""
    for doc in docs:
        print(doc.page_content)
        text2+=doc.page_content
    # text=(docs[0].page_content)
    # print(text)
    text=text2
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    template = """Answer the given question by using the below given context:
    Question: {query},
    Context: {text}

    Return the response after summarising the answer in a chatbot style if "status" field is set to "Y" or"YES"
    Else return "Not Valid Answer Present" as response.
    """

    prompt = PromptTemplate(template=template, input_variables=["query","text"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response=llm_chain.run({"query":user_query,"text":text})
    print(response)
    return jsonify({"llm_response":response,
                    "semantic_search_response":text})


if __name__ == '__main__':
    app.run(debug=True)