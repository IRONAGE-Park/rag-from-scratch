# https://velog.io/@euisuk-chung/RAG-From-Scratch-5-9
# 이 프로젝트는 위 게시글을 통해 작성되었습니다.

# RAG Fusion
# RRF(Reciprocal Rank Fusion)을 사용하여 검색된 문서를 재정렬하는 방식
# > 결과의 품질 개선에 중점

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

# 1. RAG Fusion 용 프롬프트 정의 및 다중 쿼리 생성
template = """
You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n

Output (4 queries):
"""

prompt_rag_fusion = ChatPromptTemplate.from_template(template)

generate_queries = (
  prompt_rag_fusion
  | ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
  | StrOutputParser()
  | (lambda x: x.split("\n"))
)

# RRF function
# 2. RRF 함수 정의 및 검색 수행
def reciprocal_rank_fusion(results: list[list], k=60):
  """ Reciprocal_rank_fusion taht takes multiple lists of ranked documents
      and an optional parameter k used in the RRF formula """
  
  # Initialize a dictionary to hold fused scores for each unique document
  fused_scores = {}

  # Iterate through each list of ranked documents
  for docs in results:
    # Iterate through each document in the list, with it's rank (position in the list)
    for rank, doc in enumerate(docs):
      # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
      doc_str = dumps(doc)
      # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
      if doc_str not in fused_scores:
          fused_scores[doc_str] = 0
      # Retrieve the current score of the document, if any
      previous_score = fused_scores[doc_str]
      # Update the score of the document using the RRF formula: 1 / (rank + k)
      fused_scores[doc_str] = previous_score + 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results

retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion

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
  {"context": retrieval_chain_rag_fusion, "question": itemgetter("question")}
  | prompt
  | llm
  | StrOutputParser()
)

question = "What is task decomposition for LLM agents?"
final_rag_chain.invoke({"question": question})