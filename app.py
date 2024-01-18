from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import AwaEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
import os
import requests

loader = TextLoader("test.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
embeddings =GPT4AllEmbeddings()

db = FAISS.from_documents(docs,embeddings)

query = "Where was Robinson born?"
docs = db.similarity_search(query)
text=(docs[0].page_content)
#LangchainSummarization

repo_id="google/flan-t5-xxl"
llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.5}
)

template = """Answer the given question by using the below given context:
Question: {query},
Context: {text}

Answer to the query:"""

prompt = PromptTemplate(template=template, input_variables=["query","text"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

print(llm_chain.run({"query":query,"text":text}))

