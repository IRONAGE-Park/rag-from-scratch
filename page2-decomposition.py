# https://velog.io/@euisuk-chung/RAG-From-Scratch-5-9
# 이 프로젝트는 위 게시글을 통해 작성되었습니다.

# Decomposition
# 질문을 하위 문제로 분해하여 각 문제를 순차적으로 해결하는 접근
# Chain-of-Thought
# Least-to-Most Prompting: 하위 문제를 순차적으로 제공하여 사용자의 이해도를 높임
# Interleaving Retrieval with hain-of-Thought Reasoning: 정보 검색과 COT 추론을 상호보완적으로 결합
# > 점진적으로 문제를 해결

from dotenv import load_dotenv 

import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
# from operator import itemgetter

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

# 예시 질문
question = "What are the main components of an LLM-powered autonomous agent system?"

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# 1. Decomposition 용 프롬프트 정의 / LLM을 이용한 하위 질문 생성
template = """
You are a helpful assistant that generates multiple sub-questions related to an input question. \n

The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n

Generate multiple search queries related to: {question} \n

Output (3 queries):
"""

prompt_decomposition = ChatPromptTemplate.from_template(template)

generate_queries = (
  prompt_decomposition
  | llm 
  | StrOutputParser()
  | (lambda x: x.split("\n"))
)

# questions = generate_queries.invoke({"question": question})

# ----- 이 코드들을 `langchain` 라이브러리로 대체할 수 있음 -----
# # 2. 하위 질문별로 답변 생성 및 연속적인 처리
# template = """
# Here is the question you need to answer:

# \n --- \n {question} \n --- \n

# Here is any available background question + answer pairs:

# \n --- \n {q_a_pairs} \n --- \n

# Here is additional context relevant to the question: 

# \n --- \n {context} \n --- \n

# Use the above context and any background question + answer pairs to answer the question: \n {question}
# """

# prompt_decomposition = ChatPromptTemplate.from_template(template)

# def format_qa_pair(question, answer):
#   """
#   Format question and answer pairs for inclusion in the prompt
#   """
#   formatted_string = ""
#   formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
#   return formatted_string.strip()


# # Initialize an empty string to accumulate question-answer pairs
# q_a_pairs = ""

# for q in questions:
#   rag_chain = (
#     {
#         "context": itemgetter("question") | retriever, 
#         "question": itemgetter("question"),
#         "q_a_pairs": itemgetter("q_a_pairs")
#     } 
#     | prompt_decomposition
#     | llm
#     | StrOutputParser()
#   )
  
#   # n + 1번째 질문에 대한 답변 시 n번째 답변을 참고하도록 함
#   answer = rag_chain.invoke({"question": q, "q_a_pairs": q_a_pairs})
#   q_a_pair = format_qa_pair(q, answer)
#   q_a_pairs = q_a_pairs + "\n---\n" + q_a_pair

# # 3. 최종 답변 작성

# # After processing sub-questions and accumulating q_a_pairs
# final_prompt_template = """
# You are a knowledgeable assistant.

# Here is the original question:

# {original_question}

# Here are the relevant question and answer pairs that may help you:

# {q_a_pairs}

# Using the information above, please provide a detailed and comprehensive answer to the original question.
# """

# final_prompt = ChatPromptTemplate.from_template(final_prompt_template)

# # Create the chain
# final_chain = (
#     final_prompt
#     | llm
#     | StrOutputParser()
# )

# # Invoke the chain to get the final answer
# final_answer = final_chain.invoke({"original_question": question, "q_a_pairs": q_a_pairs})
# print("Final Answer:\n", final_answer)
# ----- 이 코드들을 `langchain` 라이브러리로 대체할 수 있음 -----

# RAG 프롬프트
prompt_rag = hub.pull("rlm/rag-prompt")

def retrieve_and_rag(question, prompt_rag, sub_question_generator_chain):
  """하위 질문에 대한 RAG 수행"""
  sub_questions = sub_question_generator_chain.invoke({"question":question})
  rag_results = []

  for sub_question in sub_questions:
    retrieved_docs = retriever.get_relevant_documents(sub_question)
    answer = (prompt_rag | llm | StrOutputParser()).invoke({"context": retrieved_docs,
                                                            "question": sub_question})
    rag_results.append(answer)

  return rag_results, sub_questions

answers, questions = retrieve_and_rag(question, prompt_rag, generate_queries)

def format_qa_pairs(questions, answers):
  """질문과 답변을 포맷팅"""
  formatted_string = ""
  for i, (question, answer) in enumerate(zip(questions, answers), start=1):
    formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
  return formatted_string.strip()

context = format_qa_pairs(questions, answers)

# 최종 RAG 프롬프트
template = """Here is a set of Q+A pairs:

{context}

Use these to synthesize an answer to the question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

final_rag_chain = (
    prompt
    | llm
    | StrOutputParser()
)

final_rag_chain.invoke({"context":context,"question":question})