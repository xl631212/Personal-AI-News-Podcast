import openai
import streamlit as st
import os
import pprint
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.llms.openai import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
import requests
from bs4 import BeautifulSoup
# from TTS.api import TTS
from gnews import GNews
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from datetime import datetime
import edge_tts
import subprocess

current_date = datetime.now().date()

system_message = '''
                You are a very talented news editor, skilled at consolidating 
                fragmented information and introductions into a cohesive news script, without missing any details.
                Compile the news article based on the information in 【】.  
                '''


os.environ["SERPER_API_KEY"] = st.secrets["SERPER_API_KEY"]
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
openai.api_key = os.environ["OPENAI_API_KEY"]
st.set_page_config(layout="wide")

def get_completion_from_messages(messages,
                                 model="gpt-3.5-turbo-16k",
                                 temperature=0.5, max_tokens=7000):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message["content"]

def fetch_gnews_links(query, language='en', country='US', period='1d', start_date=None, end_date=None, max_results=5, exclude_websites=None):
    """
    Fetch news links from Google News based on the provided query.

    Parameters:
    - query (str): The search query for fetching news.
    - ... (other params): Additional parameters for customizing the news fetch.

    Returns:
    - List[str]: List of URLs based on the search query.
    """

    # Ensure that the exclude_websites parameter is a list
    content = {'title':[], 'summary':[], 'url':[]}

    # Initialize GNews
    google_news = GNews(language=language, country=country, period=period, start_date=start_date, end_date=end_date, max_results=max_results, exclude_websites=exclude_websites)
    
    # Fetch news based on the query
    news_items = google_news.get_news(query)
    print(news_items)
    # Extract URLs
    urls = [item['url'] for item in news_items]
    content['title'] = [item['title'] for item in news_items]

    for url in urls:
      content['url'].append(url)
      content['summary'].append(summarize_website_content(url))

    return content

def summarize_website_content(url, temperature=0, model_name="gpt-3.5-turbo-16k", chain_type="stuff"):
    """
    Summarize the content of a given website URL.

    Parameters:
    - url (str): The website URL to fetch and summarize.
    - temperature (float, optional): Temperature parameter for ChatOpenAI model. Default is 0.
    - model_name (str, optional): The model name for ChatOpenAI. Default is "gpt-3.5-turbo-16k".
    - chain_type (str, optional): The type of summarization chain to use. Default is "stuff".

    Returns:
    - The summarized content.
    """
    
    # Load the content from the given URL
    loader = WebBaseLoader(url)
    docs = loader.load()

    # Initialize the ChatOpenAI model
    llm = ChatOpenAI(temperature=temperature, model_name=model_name)
    
    # Load the summarization chain
    chain = load_summarize_chain(llm, chain_type=chain_type)

    # Run the chain on the loaded documents
    summarized_content = chain.run(docs)

    return summarized_content

def get_all_text_from_url(url):
    # Fetch the content using requests
    response = requests.get(url)
    response.raise_for_status()  # Raise an error if the request failed

    # Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract all text
    return ' '.join(soup.stripped_strings)  # `stripped_strings` generates strings by stripping extra whitespaces

def contains_keywords(s):
    keywords = ["AI", "GPT", "LLM"]
    return any(keyword in s for keyword in keywords)


def heacker_news_content():
    hn = HackerNews()
    content = {'title':[], 'summary':[], 'url':[]}
    for news in hn.top_stories()[:25]:
        if contains_keywords(hn.item(news).title):
            if 'url' in dir(hn.item(news)):
                content['title'].append(hn.item(news).title)
                content['url'].append(hn.item(news).url)
                content['summary'].append(summarize_website_content(hn.item(news).url))
    return content


def generate_tts_audio(text, output_file="output.wav"):
    """
    Generate audio from text using the first available TTS model and save to a file.

    Parameters:
    - text (str): The text to convert to speech.
    - output_file (str): The file path where the audio should be saved. Default is "output.wav".

    Returns:
    - None
    """

    # Determine the computation device
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Get the first available TTS model
    model_name = TTS().list_models()[0]
    
    # Initialize the TTS with the model
    tts = TTS(model_name).to("cpu")
    
    # Convert text to speech and save to file
    tts.tts_to_file(text=text, speaker=tts.speakers[0], language=tts.languages[0], file_path=output_file)



st.title("Stay Ahead: Real-time News and Podcasts with LLM")
st.sidebar.image("https://www.groundzeroweb.com/wp-content/uploads/2017/05/Funny-Cat-Memes-11.jpg", width=210)
st.subheader(current_date)

with st.sidebar.expander("About the App"):
    st.markdown("""
        Dive deep into the forefront of Large Language Model (LLM) developments with **Stay Ahead**. Our platform delivers:

        - Real-time collection of the latest in LLM research, blogs, discussions, and news.
        - Content sourced from esteemed platforms including:
        - [Openai Blog](#)
        - [HackerNews](#)
        - [BAIR Blog](https://bair.berkeley.edu/blog/)
        - [MIT AI News](https://news.mit.edu/topic/artificial-intelligence2)
        - [Google News](#)
        
        Personalize your experience:
        - [WIP] Receive a curated markdown daily report tailored to your interests.
        - Opt for our broadcast mode to stay updated via audio, making it easier than ever to keep pace with LLM advancements on-the-go.

        Discover, learn, and stay ahead with **This App**.

        """)
    
option = st.sidebar.selectbox(
    'Select language',
    ('English', 'Chinese', 'Spanish'))


if option == 'English':
    if st.button('Generate LLM newsletter'):
        with st.sidebar.status("Generating newsletter..."):
            st.write("Searching for Openai Blog...")
            openai_blog = summarize_website_content("https://openai.com/blog")
            st.write("Searching for BAIR Blog...")
            bair_blog = summarize_website_content("https://bair.berkeley.edu/blog/")
            st.write("Searching for MIT Blog...")
            mit_blog = summarize_website_content('https://news.mit.edu/topic/artificial-intelligence2')
            st.write("Searching Google News...")
            google_news = fetch_gnews_links(query='AI LLM')
            st.write("Writing Newsletter...")
            query = 'news from google news' + str(google_news['summary']) + 'news from openai blog: ' + openai_blog + 'news from bair blog' + bair_blog + 'news from mit blog' + str(mit_blog)
                 
            query = query.replace('<|endoftext|>', '')
            user_message = query
            messages =  [
                        {'role':'system',
                            'content': system_message},
                        {'role':'user',
                            'content': f"【{query}】"},]
            response = get_completion_from_messages(messages)
            st.write("Generating boardcast...")
            TEXT = response
            VOICE = "en-GB-SoniaNeural"
            updated_text = response
            # 构建 edge-tts 命令
            command = f'edge-tts --text "{updated_text}" --write-media hello.mp3'

            # 使用 subprocess 运行命令
            subprocess.run(command, shell=True)

        st.subheader('News Broadcast', divider='red')
        audio_file = open('hello.mp3', 'rb')
        audio_bytes = audio_file.read()


        st.audio(audio_bytes, format='wav')
        with st.sidebar.expander("See the full news script"):
            st.markdown(response)

        st.subheader('Google news', divider='rainbow')
        for i in range(len(google_news['title'])):
            st.markdown(f"### {google_news['title'][i]}\n")
            st.markdown(google_news['summary'][i])
            st.markdown(f"[more on]({google_news['url'][i]})\n")
        
        st.subheader('Openai blog', divider='green')
        st.markdown(openai_blog)
        st.markdown(f"[more on](https://openai.com/blog)\n")

        st.subheader('BAIR blog', divider='blue')
        st.markdown(bair_blog)
        st.markdown(f"[more on](https://bair.berkeley.edu/blog/)\n")

        st.subheader('MIT blog', divider='gray')
        st.markdown(mit_blog)
        st.markdown(f"[more on](https://news.mit.edu/topic/artificial-intelligence2)\n")

else:
    st.subheader('Work in Progress', divider='rainbow')


