# https://velog.io/@euisuk-chung/RAG-From-Scratch-5-9
# 이 프로젝트는 위 게시글을 통해 작성되었습니다.

# Multi Query
# 사용자의 질문을 여러 형태로 변환하여 문서 검색 성능을 개선
# 질문을 여러 가지 형태로 변환하여 다양한 관점에서 검색을 수행
# > 검색 범위 확대에 중점

from dotenv import load_dotenv 

import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.load import dumps, loads
from operator import itemgetter

load_dotenv()

loader = WebBaseLoader(
  web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent",), 
  bs_kwargs=dict(
    parse_only=bs4.SoupStrainer(
      class_=("post-content", "post-title", "post-header")
    )
  ),
)

blog_docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
splits = text_splitter.split_documents(blog_docs)

vectorstore = Chroma.from_documents(documents=splits,
                                    embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

# 1. 다중 쿼리 생성을 위한 프롬프트 정의
template = """
You are an AI language model assistant. 
Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database. 
    
By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search.

Provide these alternative questions separated by newlines. 
    
Original question: {question}
"""

prompt_perspectives = ChatPromptTemplate.from_template(template)

# 주어진 질문에 대해 여러 관점의 쿼리를 생성
# 2. 다중 쿼리를 이용한 검색 및 문서 통합
generate_queries = (
  prompt_perspectives
  | ChatOpenAI(model_name="gpt-4o-mini", temperature=0) # ChatOpenAI 모델을 통해 프롬프트를 처리
  | StrOutputParser() # StrOutputParser로 모델의 출력을 파싱
  | (lambda x: x.split("\n")) # 결과를 개행 문자로 분할하여 여러 쿼리로 만듬
)

def get_unique_union(documents: list[list]):
  """" Unique union of retrieved docs """
  flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
  unique_docs = list(set(flattened_docs))
  return [loads(doc) for doc in unique_docs]

# retriever.map()을 사용해 각 생성된 쿼리에 대해 문서 검색
# get_unique_union()을 사용해 중복된 문서를 제거
retrieval_chain = generate_queries | retriever.map() | get_unique_union

template = """
Answer the question based only on the following context:
{context}

Question:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# 3. RAG 체인 생성
final_rag_chain = (
  # retrieval_chain이 컨텍스트를 제공
  # itemgetter("question")가 입력에서 질문을 추출
  # itemgetter는 딕셔너리나 시퀀스에서 특정 키나 인덱스 값을 추출하는 callable 객체 생성(여기서는 "question" 키의 값 추출)
  {"context": retrieval_chain, "question": itemgetter("question")}
  # 두 요소가 prompt tempalte에 삽입
  | prompt
  | llm
  | StrOutputParser()
)

question = "What is task decomposition for LLM agents?"
final_rag_chain.invoke({"question": question})