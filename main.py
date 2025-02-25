import os

import pandas as pd
import streamlit as st
from custom_chatbot import ExcelPDFChatbot

# page title
st.set_page_config(page_title="🦜🕸️ 엑셀 및 PDF 문서를 모두 활용하는 챗봇")
st.title("🦜🕸️ 엑셀 데이터와 PDF 문서를 모두 활용하는 챗봇")

# file_path = "data/키오스크(무인정보단말기) 이용실태 조사.pdf"
file_path = "data/kiosk_survey.pdf"
# file_description = "키오스크(무인정보단말기)"
file_description = "kiosk_survey"
data_path = "data/InkjetDB_preprocessing.csv"
data_description = "잉크젯 데이터베이스"
df = pd.read_csv(data_path)


@st.cache_resource
def init_chatbot():
    chatbot = ExcelPDFChatbot(
        df,
        data_description,
        file_path,
        file_description,
    )
    return chatbot


# Streamlit app은 app code를 계속 처음부터 재실행하는 방식으로 페이지를 갱신합니다.
# Chatbot을 state에 포함시키지 않으면 매 질문마다 chatbot을 다시 초기화 합니다.
if "chatbot" not in st.session_state:
    with st.spinner("챗봇 초기화 중입니다, 약 3분 정도 소요됩니다."):
        chatbot = init_chatbot()
        st.session_state.chatbot = chatbot
    st.write("챗봇 초기화를 완료했습니다.")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.write("챗봇을 이용해보세요!")
st.markdown(
    """
- 예시 질문 (시장 조사 문서 활용): 올해 키오스크 시장의 전망을 알려줘
- 예시 질문 (잉크젯 데이터 활용): 잉크젯 데이터의 각 컬럼의 평균값을 알려줘
- 예시 질문 (잉크젯 그래프): 잉크젯 데이터의 분포를 그래프로 그려줘
- 예시 질문 (데이터 무관): 오늘 저녁 뭐 먹을까?
"""
)

for conversation in st.session_state.messages:
    with st.chat_message(conversation["role"]):
        st.write(conversation["content"])

# React to user input
if prompt := st.chat_input("질문을 입력하면 챗봇이 답변을 제공합니다."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

if prompt is not None:
    response = st.session_state.chatbot.invoke(prompt)
    generation = response["generation"]
    with st.chat_message("assistant"):
        st.markdown(generation)
        if "data" in response.keys() and response["data"] == "plot.png":
            st.image("plot.png")
            # Remove the plot image after displaying it
            os.remove("plot.png")

    st.session_state.messages.append({"role": "assistant", "content": generation})

# faiss 업데이트 : pip install --upgrade faiss-cpu
# web 실행 : streamlit run main.py