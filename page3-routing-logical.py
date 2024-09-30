# https://velog.io/@euisuk-chung/RAG-From-Scratch-10-11
# 이 프로젝트는 위 게시글을 통해 작성되었습니다.

# Routing
# 라우팅은 질문을 적절한 데이터 소스로 전달하는 프로세스
# - RAG에서 라우팅은 특정 질문을 처리하기 위해 적절한 데이터베이스나 프롬프트에 연결하는 역할

# 논리적 라우팅
# 시스템이 다양한 데이터 소스 중 어떤 소스를 사용할지 미리 설정된 규칙에 따라 결정하는 방식
# - 주로 구조화된 출력을 사용하여 라우팅을 수행. 즉, 시스템은 질문을 미리 정의된 규칙에 따라 분류하고 그에 맞는 데이터 소스를 선택
# > 질문이 명확히 구분 가능한 주제나 데이터베이스와 관련이 있을 때 매우 적합한 방법

from dotenv import load_dotenv 

from typing import Literal
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

load_dotenv()

# Data model
# 사용자 질문을 적절한 데이터 소스로 라우팅하기 위해 미리 정으디ㅚㄴ `datasource` 옵션을 포함하는 클래스
class RouteQuery(BaseModel):
  """Route a user query to the most relevant datasource."""
  datasource: Literal["python_docs", "js_docs", "golang_docs"] = Field(
    # ...은 필드를 필수 필드로 지정하는 것을 의미
    ..., description="Given a user question choose which datasource would be most relevant for answering their question",
  )

# LLM with function call 
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# 구조화된 출력 형식을 사용하도록 설정된 LLM
structured_llm = llm.with_structured_output(RouteQuery)

# Prompt 정의
system = """
You are an expert at routing a user question to the appropriate data source.

Based on the programming language the question is referring to, route it to the relevant data source.
"""

prompt = ChatPromptTemplate.from_messages(
  [("system", system), ("human", "{question}")])

# Define router 
router = prompt | structured_llm

# 질문 예시
question = """
Why doesn't the following code work:

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
prompt.invoke("french")

"""

# LLM이 질문을 분석하여 적절한 데이터 소스를 반환
# result = router.invoke({"question": question})

# 선택된 데이터 소스에 따라 추가적인 처리 수행
def choose_route(result):
  if "python_docs" in result.datasource.lower():
    return "chain for python_docs"
  elif "js_docs" in result.datasource.lower():
    return "chain for js_docs"
  else:
    return "chain for golang_docs"

# 최종적으로 선택된 경로에 따라 라우팅
full_chain = router | RunnableLambda(choose_route)

full_chain.invoke({"question": question})  # 'chain for python_docs'