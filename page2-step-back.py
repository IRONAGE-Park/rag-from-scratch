# https://velog.io/@euisuk-chung/RAG-From-Scratch-5-9
# 이 프로젝트는 위 게시글을 통해 작성되었습니다.

# Step Back
# 질문을 더 추상적인 질문으로 변환하여 고차원적 개념을 중심으로 검색 수행
# 원래 질문이 너무 구체적일 때, 더 일반적인 질문을 생성하여 더 넓은 범위의 정보를 검색
# > 문서의 구조가 개념적 내용과 구체적 내용으로 나뉠 때, 개념적인 지식을 바탕으로 검색해야 하는 도메인에서 유용

from dotenv import load_dotenv 

import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda 
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
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

# fewshot
examples = [
  {
    "input": "Could the members of The Police perform lawful arrests?",
    "output": "What can the members of The Police do?",
  },
  {
    "input": "Jan Sindel’s was born in what country?",
    "output": "What is Jan Sindel’s personal history?",
  },
]

example_prompt = ChatPromptTemplate.from_messages(
  [
    ("human", "{input}"),
    ("ai", "{output}"),
  ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
  example_prompt=example_prompt,
  examples=examples,
)

prompt = ChatPromptTemplate.from_messages(
  [
    (
      "system",
      """
      You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. 
      Here are a few examples:
      """,
    ),
    few_shot_prompt,
    ("user", "{question}"),
  ]
)

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

generate_queries_step_back = prompt | llm | StrOutputParser()

question = "What is task decomposition for LLM agents?"
response_prompt_template = """
You are an expert of world knowledge. I am going to ask you a question. 
Your response should be comprehensive and not contradicted with the following context if they are relevant. 
Otherwise, ignore them if they are not relevant.

# {normal_context}
# {step_back_context}

# Original Question: {question}
# Answer:
"""

response_prompt = ChatPromptTemplate.from_template(response_prompt_template)

chain = (
  {
    # 원래 질문에 대한 검색
    "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
    # Step Back 질문에 대한 검색
    "step_back_context": generate_queries_step_back | retriever,
    "question": lambda x: x["question"],
  }
  | response_prompt
  | llm
  | StrOutputParser()
)

chain.invoke({"question": question})