import openai
import streamlit as st
import os
import pprint
import requests
from bs4 import BeautifulSoup
from gnews import GNews
from datetime import datetime
import edge_tts
import subprocess
import arxiv
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.llms.openai import OpenAI
from youtubesearchpython import *
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.llms.openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain

system_message = '''
                You are a very talented news editor, skilled at consolidating 
                fragmented information and introductions into a cohesive news script, without missing any details.
                Compile the news article based on the information in 【】.  
                '''

system_message_2 = '''
                You are a linguist, skilled in summarizing textual content and presenting it in 3 bullet points using markdown. 
                no more than 150 words in total.
                '''


os.environ["SERPER_API_KEY"] = st.secrets["SERPER_API_KEY"]
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
openai.api_key = os.environ["OPENAI_API_KEY"]


def fetch_videos_from_channel(channel_id):
    playlist = Playlist(playlist_from_channel_id(channel_id))
    while playlist.hasMoreVideos:
        playlist.getNextVideos()
    return playlist.videos


def get_transcript(video_id):
    raw_data = YouTubeTranscriptApi.get_transcript(video_id)
    texts = [item['text'] for item in raw_data]
    return ' '.join(texts)


def split_text_into_documents(long_string, max_docs=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len,
    )
    texts = text_splitter.split_text(long_string)
    docs = [Document(page_content=t) for t in texts[:max_docs]]

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=0
    )
    split_docs = text_splitter.split_documents(docs)
    return split_docs


def summarize_documents(split_docs):
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(split_docs)
    return summary


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


def input_page(st, **state):
    st.markdown("<h1 style='text-align: center; color: black;'>LLM <span style='color: pink;'>Personal Podcast</span></h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: black;'>Stay Ahead: Real-time News and Podcasts with LLM </h2>", unsafe_allow_html=True)
    st.markdown("""
    <h4 style='text-align: center; color: black;'>
        Select <span style='color: red;'>⭕️</span> either of the modes at the bottom and double-click 👆 the button below. 
        <br>
        Wait for approximately <span style='color: blue;'>3 mins</span> to generate your personalized LLM podcast
    </h4>
    """, 
    unsafe_allow_html=True)



    # Custom CSS to modify the button appearance
    st.markdown("""
    <style>
        .stButton>button {
            width: 40%;
            height: 70px;
            color: white;
            background-color: pink;
            border: none;
            border-radius: 10px;
            margin: auto;
            font-weight: bold; 
            font-size: 500px; 
            display: flex;            /* Use flexbox */
            justify-content: center;  /* Center children horizontally */
            align-items: center;  
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
        .stRadio > div[role="radiogroup"] {
            justify-content: center;
        }
        .stRadio > div[role="radiogroup"] > label{
            font-size: 1000px !important;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
        .stSelectbox {
            width: 10% !important;
            margin-left: calc(50% - 20%) !important;
        }
    </style>
    """, unsafe_allow_html=True)

    button_placeholder = st.empty()

    choice = st.radio(
        "",
        ["Auto-generation (recommended for first-time use)", "Advanced setting"],
        key="visibility",
        disabled=False,
        horizontal=True,
    )

    language = st.selectbox(
        "language",
        ("English", "Spanish", "Chinese"),
        label_visibility="visible",
        disabled=False,
    )

    with button_placeholder:
        # 添加按钮样式
        st.markdown("""
        <style>
            .stButton > button {
                width: 20%;
                height: 70px;
                color: white;
                background-color: pink;
                border: none;
                border-radius: 10px;
                margin: auto;
                font-weight: bold; 
                font-size: 70px; 
                justify-content: center;
                align-items: center;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # 创建按钮
        if st.button("👆 Double-Click Generation"):
            st.session_state.page = "two"
            st.session_state.choice = choice
            st.session_state.language = language


def compute_page(st, **state):
    st.markdown("<h1 style='text-align: center; color: black;'>LLM <span style='color: pink;'>Personal Podcast</span></h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: black;'>Stay Ahead: Real-time News and Podcasts with LLM </h2>", unsafe_allow_html=True)
    st.markdown("""
    <style>
        /* This styles the main content excluding h1 and h2 */
        #root .block-container {
            width: 75%;
            margin: auto;
        }
    </style>
    """, unsafe_allow_html=True)
    radio_placeholder = st.empty()
    progress_placeholder = st.empty()
    progress_text = "Searching for Openai Blog..."
    my_bar = progress_placeholder.progress(0, text=progress_text)
    openai_blog = summarize_website_content("https://openai.com/blog")

    my_bar.progress(10, text="Searching for BAIR Blog...")
    bair_blog = summarize_website_content("https://bair.berkeley.edu/blog/")

    my_bar.progress(20, text="Searching for MIT Blog...")
    mit_blog = summarize_website_content('https://news.mit.edu/topic/artificial-intelligence2')
    
    my_bar.progress(30, text="Searching for a16z Blog...")
    a16z_blog = summarize_website_content('https://a16z.simplecast.com/')
    
    my_bar.progress(40, text='Searching for lexi friman boardcast...')
    lexi_boardcast = summarize_website_content('https://www.youtube.com/c/lexfridman')

    my_bar.progress(50, text="Searching for arxiv ...")
    search = arxiv.Search(
        query = "AI, LLM",
        max_results = 3,
        sort_by = arxiv.SortCriterion.SubmittedDate
    )
    ariv_essay = ''
    for result in search.results():
        ariv_essay += result.summary
    
    my_bar.progress(60, text="Searching Google News...")
    google_news = fetch_gnews_links(query='AI LLM')

    my_bar.progress(80, text="Writing Newsletter...")
    query = 'news from google news' + str(google_news['summary']) + 'news from bair blog' + bair_blog + 'news from mit blog' + str(mit_blog) \
             + 'news from a15z blog' + a16z_blog + 'news from lexi broadcast' + lexi_boardcast + 'news from openai blog: ' + openai_blog + 'new arxiv essay' \
             + ariv_essay
    
    query = query.replace('<|endoftext|>', '')
    messages =  [
                    {'role':'system',
                    'content': system_message},
                    {'role':'user',
                    'content': f"【{query}】"},]
    response = get_completion_from_messages(messages)

    my_bar.progress(90, text="Generating Podcast...")
    my_bar.empty()

    updated_text = response
    # 构建 edge-tts 命令
    command = f'edge-tts --text "{updated_text}" --write-media hello.mp3'
    # 使用 subprocess 运行命令
    subprocess.run(command, shell=True)

    my_bar.progress(90, text="Generating Summary...")
    my_bar.empty()

    query = response
    messages =  [
                    {'role':'system',
                    'content': system_message_2},
                    {'role':'user',
                    'content': f"【{query}】"},]
    summary = get_completion_from_messages(messages)


    with radio_placeholder:
        audio_file = open('hello.mp3', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='wav')

    st.subheader('Summary and Commentary', divider='rainbow')
    st.markdown(summary)

    st.subheader('Technology News', divider='rainbow')
    for i in range(len(google_news['title'])):
        st.markdown(f"### {google_news['title'][i]}\n")
        st.markdown(google_news['summary'][i])
        st.markdown(f"[more on]({google_news['url'][i]})\n")

    st.subheader('Podcast and Speeches', divider='orange')
    st.markdown(lexi_boardcast)
    st.markdown(f"[more on](https://www.youtube.com/c/lexfridman)\n")
    st.markdown(a16z_blog)
    st.markdown(f"[more on](https://a16z.simplecast.com/)\n")
    
    st.subheader('Technology Blogs', divider='green')
    st.markdown(openai_blog)
    st.markdown(f"[more on](https://openai.com/blog)\n")
    st.markdown(bair_blog)
    st.markdown(f"[more on](https://bair.berkeley.edu/blog/)\n")
    st.markdown(mit_blog)
    st.markdown(f"[more on](https://news.mit.edu/topic/artificial-intelligence2)\n")

    st.subheader('Arxiv Essay', divider='green')
    for result in search.results():
        st.markdown(f"### {result.title}\n")
        st.markdown(result.summary)
        st.markdown(f"[more on]({result.entry_id})\n")

def page_one():
    input_page(st)

def page_two():
    compute_page(st)


def main():
    # 初始化session状态
    if "page" not in st.session_state:
        st.session_state.page = "one"

    if "choice" not in st.session_state:
        st.session_state.choice = ""
    
    if "language" not in st.session_state:
        st.session_state.language = ""


    # 根据session状态来渲染页面
    if st.session_state.page == "one":
        page_one()
    elif st.session_state.page == "two":
        page_two()

if __name__ == "__main__":
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
    main()
