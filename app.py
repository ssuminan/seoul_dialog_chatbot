import streamlit as st
import json
import os
import openai
from sentence_transformers import SentenceTransformer, util
import torch
from streamlit_chat import message

# Data load
@st.cache_data
def load_data():
    conv_data = json.load(open('./data/대화데이터.json', 'r'))['data']
    intent_data = json.load(open('./data/의도데이터.json', 'r'))['data']
    return conv_data, intent_data

conv_data, intent_data = load_data()

# API load
@st.cache_data
def generate(prompt, temperature):
    API_KEY = "xxxxxx"
    openai.api_key = API_KEY
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt,
        temperature=temperature
    )
    return completion.choices[0].message["content"].strip()

# convert conversation documents to text embeddings
@st.cache_resource
def embedding(conv_data):
    embedder = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    corpus = [conv['대화전문'] for conv in conv_data]
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    return embedder, corpus, corpus_embeddings

embedder, corpus, corpus_embeddings = embedding(conv_data)

def cal_cos(query):
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results_3 = torch.topk(cos_scores, k=3)
    top_results_5 = torch.topk(cos_scores, k=5)

    return top_results_3, top_results_5

# convert query to text embedding and do semantic search
def search(top_results_3):
    # query_embedding = embedder.encode(query, convert_to_tensor=True)
    #
    # cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    # top_results = torch.topk(cos_scores, k=3)

    documents = '```\n' + '\n```\n\n```\n'.join([corpus[idx] for idx in top_results_3[1]][::-1]) + '\n```'

    return documents

# extract and summarize related imformation from documents
def knowledge(conv, docs):
    prompt = [
        {"role": "user", "content": f"현재 대화:\n```\n{conv}```\n\n관련 대화:\n{docs}\n\n현재 대화와 관련 있는 정보를 구체적으로 정리해줘."},
    ]
    knowledge_text = generate(prompt, 0.7)

    return knowledge_text

# generate user intent given user question and conversation
def intent(query, conv, top_results_5):
    # query_embedding = embedder.encode(query, convert_to_tensor=True)
    #
    # cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    # top_results = torch.topk(cos_scores, k=5)

    few_shot = '\n\n###\n\n'.join([
                                      f"대화:\n```\n{intent_data[idx]['대화전문']}\n```\n\n고객질문: {intent_data[idx]['고객질문'].strip('고객: ')}\n\n고객의도: {intent_data[idx]['고객의도']}"
                                      for idx in top_results_5[1]][::-1])
    few_shot += f"\n\n###\n\n대화:\n```\n{conv}\n\n고객질문:{query.strip('고객: ')}\n\n고객의도:"

    prompt = [
        {"role": "user", "content": few_shot},
    ]
    intent_text = generate(prompt, 0.7)

    return intent_text

# generate answer to user question
def dialog(query, conv):
    top_results_3, top_results_5 = cal_cos(query)
    docs_text = search(top_results_3)
    knowledge_text = knowledge(conv, docs_text)
    intent_text = intent(query, conv, top_results_5)

    # print(docs_text)

    prompt = [
        {"role": "user",
         "content": f"관련대화:\n{docs_text}\n\n관련정보:\n```\n{knowledge_text}\n```\n\n주어진 정보를 바탕으로 현재 고객의 질문에 답변해줘\n\n고객의도:\n```\n{intent_text}\n```\n\n현재대화:\n```\n{conv}\n상담사:"},
    ]
    dialog_text = generate(prompt, 0.7).split('\n')[0].strip('```').strip()

    return dialog_text


st.title('다산콜 AI ChatBot')

if 'conv' not in st.session_state:
    st.session_state['conv'] = ''

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []


with st.form('form', clear_on_submit=True):
    user_input = st.text_input('질문 : ', '')
    submitted = st.form_submit_button('전송')

# User write a question and then click button
if submitted and user_input:
    processed_query = '고객: ' + user_input
    st.session_state['conv'] += processed_query
    answer = dialog(user_input, st.session_state['conv'])
    processed_answer = '상담사: ' + answer
    st.session_state['conv'] += f"\n{processed_answer}\n"

    st.session_state.past.append(user_input)
    st.session_state.generated.append(answer)

# Show conversation
for i in range(len(st.session_state['past'])):
    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    if len(st.session_state['generated']) > i:
        message(st.session_state['generated'][i], key=str(i) + '_bot')