# Seoul Dialog ChatBot
ChatGPT를 활용한 서울시 생성 AI ChatBot Demo 입니다.

## Description
- `data/.` : JSON 형식의 AI-Hub 민원(콜센터) 질의-응답 데이터셋입니다. 대화데이터셋과 대화의도데이터셋으로 구성되어 있습니다.
- `app.py` : ChatBot Web Demo 입니다.
- `prompt_engineering.ipynb` : ChatBot Demo 입니다.

## Requirements
```python
pip install pytorch
pip install streamlit
pip install openai
pip install sentence_transformers
pip install streamlit_chat
```

## Usage

```python
streamlit run app.py
```

## Example
<img width="1582" alt="스크린샷 2023-05-18 오후 6 37 36" src="https://github.com/ssuminan/seoul_dialog_chatbot/assets/109983468/bfcad4eb-62b9-4fde-aba0-bcccb9ee0f7b">

