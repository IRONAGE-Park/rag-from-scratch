# https://velog.io/@euisuk-chung/RAG-From-Scratch-5-9
# 이 프로젝트는 위 게시글을 통해 작성되었습니다.

# HyDE
# 가상 문서(Hypothetical Document)를 생성하여 문서 검색을 수행하는 방법
# 질문의 임베딩이 부족할 수 있는 경우에도 가상의 문서가 더 유사한 실제 문서와 잘 일치할 수 있도록 함
# > 질문이 짧거나 구조가 명확하지 않은 경우에 유용하며, 도메인에 맞게 가상 문서 생성 프롬프트를 조정할 수 있음

from dotenv import load_dotenv 

import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

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

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# 1. HyDE를 위한 문서 생성 프롬프트 정의 및 생성
template = """
Please write a scientific paper passage to answer the question
Question: {question}
Passage:
"""

prompt_hyde = ChatPromptTemplate.from_template(template)

generate_docs_for_retrieval = (
  prompt_hyde | llm | StrOutputParser()
)

# 예시 질문
question = "What is task decomposition for LLM agents?"

# 검색 체인
# 2. 생성된 가상 문서를 사용한 문서 검색
retrieval_chain = generate_docs_for_retrieval | retriever
retireved_docs = retrieval_chain.invoke({"question":question})
retireved_docs

# 3. 검색된 문서를 바탕으로 최종 답변 생성
# RAG 프롬프트
template = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

final_rag_chain = (
  prompt
  | llm
  | StrOutputParser()
)

# 최종 RAG 체인 실행
final_rag_chain.invoke({"context":retireved_docs,"question":question})