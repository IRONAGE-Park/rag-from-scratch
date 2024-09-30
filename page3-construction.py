# https://velog.io/@euisuk-chung/RAG-From-Scratch-10-11
# 이 프로젝트는 위 게시글을 통해 작성되었습니다.

# 쿼리 구조화
# 자연어로 된 질문을 특정 데이터베이스나 도메인에 맞는 구조화된 쿼리로 변환하는 과정(벡터 스토어에서 메타 데이터 필터를 사용하여 질의를 처리하는 방법에 중점)
# > 질문이 단순한 정보 조회가 아니라, 의미적으로 유사한 여러 가능성을 고려해야 할 때 적합한 방법

import datetime
from typing import Literal, Optional, Tuple
from dotenv import load_dotenv 

from langchain_community.document_loaders import YoutubeLoader
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

load_dotenv()

# 1. 자연어 질문 -> 구조화된 쿼리
docs = YoutubeLoader.from_youtube_url(
  "https://www.youtube.com/watch?v=pbAd8O1Lvm4",
  add_video_info=True
).load()

# 2. 메타데이터 필터 사용
# 3. LLM과 함수 호출
class TutorialSearch(BaseModel):
  """Search over a database of tutorial videos about a software library."""

  content_search: str = Field(
    ...,
    description="Similarity search query applied to video transcripts.",
  )
  title_search: str = Field(
    ...,
    description=(
      "Alternate version of the content search query to apply to video titles. "
      "Should be succinct and only include key words that could be in a video "
      "title."
    ),
  )
  min_view_count: Optional[int] = Field(
    None,
    description="Minimum view count filter, inclusive. Only use if explicitly specified.",
  )
  max_view_count: Optional[int] = Field(
    None,
    description="Maximum view count filter, exclusive. Only use if explicitly specified.",
  )
  earliest_publish_date: Optional[datetime.date] = Field(
    None,
    description="Earliest publish date filter, inclusive. Only use if explicitly specified.",
  )
  latest_publish_date: Optional[datetime.date] = Field(
    None,
    description="Latest publish date filter, exclusive. Only use if explicitly specified.",
  )
  min_length_sec: Optional[int] = Field(
    None,
    description="Minimum video length in seconds, inclusive. Only use if explicitly specified.",
  )
  max_length_sec: Optional[int] = Field(
    None,
    description="Maximum video length in seconds, exclusive. Only use if explicitly specified.",
  )

  def pretty_print(self) -> None:
    for field in self.__fields__:
      if getattr(self, field) is not None and getattr(self, field) != getattr(
              self.__fields__[field], "default", None
      ):
        print(f"{field}: {getattr(self, field)}")


# Create the Prompt Template:
system = """You are an expert at converting user questions into database queries.
You have access to a database of tutorial videos about a software library for building LLM-powered applications.
Given a question, return a database query optimized to retrieve the most relevant results."""

# Initialize the Language Model (LLM)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 4. User Question 처리
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}")])

structured_llm = llm.with_structured_output(TutorialSearch)

# Prompt and LLM into a Query Analyzer
query_analyzer = prompt | structured_llm

# 2023년에 게시된 chat langchain 비디오
# query_analyzer.invoke({"question": "videos on chat langchain published in 2023"}).pretty_print()
# 2024년 이전에 게시된 chat langchain 비디오
# query_analyzer.invoke({"question": "videos that are focused on the topic of chat langchain that are published before 2024"}).pretty_print()
# 5분 이하의 멀티모달 모델 관련 비디오
query_analyzer.invoke({"question": "how to use multi-modal models in an agent, only videos under 5 minutes"}).pretty_print()