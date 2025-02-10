from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import requests
from bs4 import BeautifulSoup
import re

def scrape_article(url):
    # Send a GET request to the URL
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code != 200:
        print("Failed to retrieve the webpage")
        return None
    
    # Parse the HTML content of the page
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the article text (this depends on the structure of the website)
    # For example, many websites use <article> or <div class="article"> for articles
    article = soup.find('article')  # Modify this line as needed based on the page structure
    
    # If an article is found, return its text content
    if article:
        return article.get_text()
    else:
        print("Article not found")
        return None

app = FastAPI()

class model_input(BaseModel):
    text: str

fake_news_model = pickle.load(open('fake_news_model.sav', 'rb'))

@app.post("/predict")
def predict(input: model_input):
    if input.text is None:
        raise HTTPException(status_code=400, detail="text is a required field")
    print(input.text)

    article_text = scrape_article(input.text)
    print(article_text)
    print("tokenizing")
    tokenizer = Tokenizer(num_words=10000)  # Adjust num_words as needed
    tokenizer.fit_on_texts([article_text])  # Fit the tokenizer on the input text
    new_seq = tokenizer.texts_to_sequences([article_text])
    new_pad = pad_sequences(new_seq, maxlen=200)
    print("predicting")
    prediction = fake_news_model.predict(new_pad)
    prediction = float(prediction[0][0])
    
    print(f'Prediction: {prediction}')  # Outputs a probability between 0 and 1
    if prediction > 0.5:
        prediction_percent = (prediction - 0.5) * 200
    else:
        prediction_percent = (0.5 - prediction) * 200
    return {
        "label": "fake news" if prediction > 0.5 else "real news",
        "prediction": prediction_percent,
    }


@app.get("/")
async def read_root():
    return FileResponse("templates/index.html")
