# https://velog.io/@euisuk-chung/RAG-From-Scratch-Overview
# 이 프로젝트는 위 게시글을 통해 작성되었습니다.

from dotenv import load_dotenv 

import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import tiktoken

load_dotenv()

# 1. 토큰 개수 계산
def num_tokens_from_string(string: str, encoding_name: str) -> int:
  """Returns the number of tokens in a text string."""
  encoding = tiktoken.get_encoding(encoding_name)
  num_tokens = len(encoding.encode(string))
  return num_tokens

# 2. 텍스트 임베딩 모델
# embd = OpenAIEmbeddings()
# query_result = embd.embed_query(question)
# document_result = embd.embed_query(document)
# len(query_result)

# 3. 코사인 유사도 계산
import numpy as np

def cosine_similarity(vec1, vec2):
  # 두 벡터 사이의 각도를 기반으로 유사성 측정
  # 1에 가까울 수록 유사
  dot_product = np.dot(vec, vec2)
  norm_vec1 = np.linalg.norm(vec1)
  norm_vec2 = np.linalg.norm(vec2)
  return dot_product / (norm_vec1 * norm_vec2)

# similarity = cosine_similarity(query_result, document_result)
# print("Consine Similarity: ", similarity)

### INDEXING ###

# Load Documents
# 4. 문서 로더
loader = WebBaseLoader(
  web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent",), 
  bs_kwargs=dict(
    parse_only=bs4.SoupStrainer(
      class_=("post-content", "post-title", "post-header")
    )
  ),
)

blog_docs = loader.load()

# Split
# 5. 텍스트 분할기
## chunk_size: 각 덩어리의 최대 크기
## chunk_overlap: 덩어리 간의 겹치는 부분의 크기
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
splits = text_splitter.split_documents(blog_docs)

# Embed
# Vectorstore를 사용해 대규모 데이터셋에서 효율적인 유사성 검색 수행
# 6. 벡터 스토어
vectorstore = Chroma.from_documents(documents=splits,
                                    embedding=OpenAIEmbeddings())

# 검색기 생성
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
# ----- 이 코드들을 `langchain` 라이브러리로 대체할 수 있음 -----
# docs = retriever.get_relevant_documents("What is Task Decomposition?")

# # 프롬프트 템플릿 정의
# template = """Answer the question based only on the following context:
# {context}

# Question:
# {question}
# """

# from langchain.prompts import ChatPromptTemplate
# # 템플릿을 ChatPromptTemplate 객체로 변환
# prompt = ChatPromptTemplate.from_template(template)

# # LLM 생성
# llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# # 체인(chain) 생성: 프롬프트와 LLM 연결
# chain = prompt | llm

# chain.invoke({"context": docs, "question": "What is Task Decomposition?"})
# ----- 이 코드들을 `langchain` 라이브러리로 대체할 수 있음 -----

### RETRIEVAL and GENERATION ###
# Prompt
prompt = hub.pull("rlm/rag-prompt")

# LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# Post-processing
def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs)

# RAG 체인 생성
rag_chain = (
  {"context": retriever | format_docs, "question": RunnablePassthrough()}
  | prompt # 호출한 template 사용
  | llm
  | StrOutputParser()
)

# Question
rag_chain.invoke("What is Task Decomposition?")

# Answer
# Task Decomposition is a technique used to break down complex tasks into smaller, more manageable steps.
# It involves methods like Chain of Thought (CoT) and Tree of Thoughts, which guide models to think step by step and explore multiple reasoning possibilities.
# This approach enhances model performance by simplifying and structuring tasks systematically.