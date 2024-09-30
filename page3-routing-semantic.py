# https://velog.io/@euisuk-chung/RAG-From-Scratch-10-11
# 이 프로젝트는 위 게시글을 통해 작성되었습니다.

# 의미적 라우팅
# 질문과 여러 프롬프트 간의 의미적 유사성을 기반으로 적합한 프롬프트를 선택하는 방식
# 임베딩한 후 유사도를 계산하여 가장 유사한 프롬프트 선택
# - 미리 정의된 데이터베이스가 아니라, 질문의 의미를 파악해 가장 적합한 프롬프트를 선택하는 방식
# > 질문이 단순한 정보 조회가 아니라, 의미적으로 유사한 여러 가능성을 고려해야 할 때 적합한 방법

from dotenv import load_dotenv 

from langchain.utils.math import cosine_similarity # 질문과 프롬프트 간의 유사도를 계산하고, 가장 유사한 프롬프트를 선택
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# 두 가지 프롬프트 정의(물리학, 수학)
physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise and easy to understand manner. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{query}"""

math_template = """You are a very good mathematician. You are great at answering math questions. \
You are so good because you are able to break down hard problems into their component parts, \
answer the component parts, and then put them together to answer the broader question.

Here is a question:
{query}"""

# 프롬프트 임베딩
embeddings = OpenAIEmbeddings() # 질문과 프롬프트를 임베딩
prompt_templates = [physics_template, math_template]
prompt_embeddings = embeddings.embed_documents(prompt_templates)

# 질문을 적합한 프롬프트로 라우팅
def prompt_router(input):
  query_embedding = embeddings.embed_query(input["query"])
  similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
  most_similar = prompt_templates[similarity.argmax()]
  print("Using MATH" if most_similar == math_template else "Using PHYSICS")
  return PromptTemplate.from_template(most_similar)

chain = (
        {"query": RunnablePassthrough()}
        | RunnableLambda(prompt_router)
        | ChatOpenAI()
        | StrOutputParser()
)

print(chain.invoke("What's a black hole"))