# Seoul Dialog ChatBot
ChatGPT를 활용한 서울시 생성 AI ChatBot Demo 입니다.

## Description
- `data/.` : JSON 형식의 AI-Hub 민원(콜센터) 질의-응답 데이터셋입니다. 대화데이터셋과 대화의도데이터셋으로 구성되어 있습니다.
- `app.py` : ChatBot Web Demo 입니다.
- `prompt_engineering.ipynb` : ChatBot Demo 입니다.

## Requirements
```python
conda install pytorch::pytorch torchvision torchaudio -c pytorch
conda install streamlit
conda install openai
conda install sentence_transformers
conda install streamlit_chat
```

## Usage

```python
streamlit run app.py
```

## Example
<img width="1582" alt="스크린샷 2023-05-17 오후 7 56 54" src="https://github.com/ssuminan/seoul_dialog_chatbot/assets/109983468/9fd7538c-af44-4435-9933-3c4b9edf2ee4">
