import os
import pprint
import requests
from bs4 import BeautifulSoup
from gnews import GNews
from datetime import datetime
import edge_tts
import arxiv
import subprocess
import base64
import openai
import streamlit as st
from langchain.utilities import GoogleSerperAPIWrapper
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



os.environ["SERPER_API_KEY"] = st.secrets["SERPER_API_KEY"]
os.environ["OPENAI_API_KEY"]= st.secrets["OPENAI_API_KEY"]
openai.api_key = os.environ["OPENAI_API_KEY"]


system_message = '''
                You are a very talented news editor, skilled at consolidating 
                fragmented information and introductions into a cohesive news script, without missing any details.
                Compile the news article based on the information in 【】.  
                '''

system_message_2 = '''
                You are a linguist, skilled in summarizing textual content and presenting it in 3 bullet points using markdown. 
                '''

system_message_3 = '''
                你是个语言学家，擅长把英文翻译成中文。要注意表达的流畅和使用中文的表达习惯。不要返回多余的信息，只把文字翻译成中文。
                '''

def is_link_accessible(url):
    """Check if a link is accessible."""
    try:
        response = requests.get(url, timeout=10)  # setting a timeout to avoid waiting indefinitely
        # Check if the status code is 4xx or 5xx
        if 400 <= response.status_code < 600:
            return False
        return True
    except requests.RequestException:
        return False
    
def get_latest_aws_ml_blog():
    url = 'https://aws.amazon.com/blogs/machine-learning/'
    
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Failed to retrieve webpage. Status code: {response.status_code}")
        return None, None
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    articles = soup.find_all('div', class_='lb-col lb-mid-18 lb-tiny-24')
    
    if not articles:
        print("No articles found.")
        return None, None
    
    title = articles[0].find('h2').text
    link = articles[0].find('a')['href']
    
    return title, link

def fetch_videos_from_channel(channel_id):
    playlist = Playlist(playlist_from_channel_id(channel_id))
    while playlist.hasMoreVideos:
        playlist.getNextVideos()
    return playlist.videos

def get_h1_text(url):
    """Fetches the text content of the first h1 element from the given URL."""
    
    # Get the HTML content of the URL
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the first h1 element and get its text
    h1_element = soup.find('h1', class_='entry-title')
    if h1_element:
        return h1_element.text.strip()  # Remove any extra whitespaces
    else:
        return None
    
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


def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls style="width: 100%;">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )

def get_h1_from_url(url):
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 根据class查找<h1>标签
        h1_tag = soup.find("h1", class_="f-display-2")
        if h1_tag:
            return h1_tag.text
        else:
            print("Couldn't find the <h1> tag with the specified class on the page.")
            return None
    else:
        print(f"Failed to fetch the webpage. Status code: {response.status_code}")
        return None
    

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

def fetch_gnews_links(query, language='en', country='US', period='1d', start_date=None, end_date=None, max_results=8, exclude_websites=None):
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
    if is_link_accessible(url):
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
    
    else:
        return 'No result'


def get_transcript_link(url):
    """Fetches the first 'Transcript' link from the given URL."""
    
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    transcript_link_element = soup.find('a', string="Transcript")

    if transcript_link_element:
        return transcript_link_element['href']
    else:
        return None

def get_youtube_link(url):
    """Fetches the first 'Transcript' link from the given URL."""
    
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    transcript_link_element = soup.find('a', string="Video")

    if transcript_link_element:
        return transcript_link_element['href']
    else:
        return None

def get_latest_openai_blog_url():
    base_url = "https://openai.com"
    blog_url = f"{base_url}/blog"

    response = requests.get(blog_url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 查找具有特定类名的<a>标签
        target_link = soup.find("a", class_="ui-link group relative cursor-pointer") 
        if target_link:
            # Combining base URL with the relative path
            post_url = base_url + target_link['href']
            return post_url
        else:
            print("Couldn't find the target post URL.")
            return None
    else:
        print(f"Failed to fetch the webpage. Status code: {response.status_code}")
        return None

def extract_blog_link_info(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.3'
    }

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        return None, None

    soup = BeautifulSoup(response.content, 'html.parser')

    # 由于网站可能有多个这样的链接，我们只选择第一个匹配的项
    link_element = soup.find('a', class_='f-post-link')

    if link_element:
        text_content = link_element.h3.text.strip()
        href_link = link_element['href']
        return text_content, href_link
    else:
        return None, None


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
    st.markdown("""
    <h1 style='text-align: center; color: black;'>
        LLM <span style='color: #FF4B4B; font-size: 0.85em;'>Personal Podcast</span>
    </h1>
    """, 
    unsafe_allow_html=True
    )
    st.markdown("<h2 style='text-align: center; color: black;'>Stay Ahead: Real-time News and Podcasts with LLM </h2>", unsafe_allow_html=True)
    st.markdown("""
    <h4 style='text-align: center; color: black;'>
        Select <span style='color: red;'>⭕️</span> either of the modes at the bottom and double-click 👆 the button below. 
        <br>
        Wait for approximately <span style='color: blue; font-size: 1.2em;'>3 mins</span> to generate your personalized LLM podcast
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
            background-color: #FF4B4B;
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
    st.markdown("""
        <style>
            .stSelectbox {
                width: 30% !important;
                margin: 0 auto !important;
            }
        </style>
        """, unsafe_allow_html=True)
    
    language_placeholder = st.empty()
    with language_placeholder:
        language = st.selectbox(
            "Language",  # Removing the label here as we manually placed it in the left column
            ("English",  "中文"))
    

    if choice == 'Advanced setting':
        language_placeholder.empty()
        # CSS to adjust column content alignment and container width
        st.markdown("""
        <style>
            .centered-container {
                width: 40% !important;
                margin: 0 auto !important;
            }

            .col1-content {
                text-align: right !important;
            }
            .col2-content {
                text-align: left !important;
            }
        </style>
        """, unsafe_allow_html=True)

        # Using a container to center content
        with st.container():
            st.markdown("<div class='centered-container'>", unsafe_allow_html=True)
            
            col1, col2 = st.columns([3,4])

            with col1:
                st.markdown("<div class='col1-content'>", unsafe_allow_html=True)
                language = st.selectbox(
                    "Language",
                    ("English", "中文"),
                    key='ahaha'
                )
                audio_length = st.selectbox(
                    'Audio Length',
                    ['1 min', '3 min', '5 min'],
                    key='opt2'
                )
                st.session_state.audio_length = audio_length
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='col2-content'>", unsafe_allow_html=True)
                options_2 = st.selectbox(
                    'In a tone of',
                    ['News', 'Enthusiastic', 'Humor'],
                    key='opt3'
                )
                time_period = st.selectbox(
                    'In a period of',
                    ['Today', 'Yesterday'],
                    key='opt4'
                )
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)




    with button_placeholder:
        # 添加按钮样式
        st.markdown("""
        <style>
            .stButton button {
                font-size:50px;
                width: 10%;
                box-sizing: 5%;
                height: 30em;
                color: white;
                background-color: #FF4B4B;
                border: none;
                border-radius: 10px;
                margin: auto;
                font-weight: bold; 
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
    st.markdown("<h1 style='text-align: center; color: black;'>LLM <span style='color: #FF4B4B;'>Personal Podcast</span></h1>", unsafe_allow_html=True)
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
    openai_blog_url = get_latest_openai_blog_url()
    if openai_blog_url:
        
        openai_title = get_h1_from_url(openai_blog_url)
        openai_blog = summarize_website_content(openai_blog_url)

    my_bar.progress(10, text="Searching for Microsoft Blog...")
    url = "https://blogs.microsoft.com/"
    M_title, link = extract_blog_link_info(url)
    bair_blog = summarize_website_content(link)

    my_bar.progress(20, text="Searching for Amazon Blog...")
    A_title, A_link = get_latest_aws_ml_blog()
    mit_blog = summarize_website_content(A_link)

    my_bar.progress(30, text="Searching for Apple Blog...")
    Apple_blog = summarize_website_content('https://machinelearning.apple.com/')
    
    my_bar.progress(40, text='Searching for lexi friman boardcast...')
    url = "https://lexfridman.com/podcast/"
    link = get_transcript_link(url)
    L_title = get_h1_text(link)
    youtube_link = get_youtube_link(url)
    lexi_boardcast = summarize_website_content(youtube_link)

    my_bar.progress(50, text="Searching for arxiv ...")
    search = arxiv.Search(
        query = "AI, LLM, machine learning, NLP",
        max_results = 3,
        sort_by = arxiv.SortCriterion.SubmittedDate
    )
    ariv_essay = ''
    for result in search.results():
        ariv_essay += result.summary
    
    my_bar.progress(60, text="Searching Google News...")
    google_news = fetch_gnews_links(query='AI, LLM, Machine learning')

    my_bar.progress(80, text="Writing Newsletter...")
    print(google_news['summary'], bair_blog, mit_blog, openai_blog, ariv_essay)
    query = 'news from google news' + str(google_news['summary'])  + bair_blog  + str(mit_blog) \
              + openai_blog + 'new arxiv essay' + ariv_essay
    
    query = query.replace('<|endoftext|>', '')
    messages =  [
                    {'role':'system',
                    'content': system_message},
                    {'role':'user',
                    'content': f"【{query}】"},]
    response = get_completion_from_messages(messages)

    my_bar.progress(90, text="Generating Podcast...")
    if st.session_state.language == 'English':
        updated_text = response
        # 构建 edge-tts 命令
        command = f'edge-tts --text "{updated_text}" --write-media hello.mp3'
        # 使用 subprocess 运行命令
        subprocess.run(command, shell=True)

        my_bar.progress(90, text="Generating Summary...")

        query = response
        messages =  [
                        {'role':'system',
                        'content': system_message_2 + "keep it within {} minutes.".format(st.session_state.audio_length)},
                        {'role':'user',
                        'content': f"【{query}】"},]
        summary = get_completion_from_messages(messages)
    
    else:
        before = response
        summary = before.replace('<|endoftext|>', '')
        messages =  [
                        {'role':'system',
                        'content': system_message_3},
                        {'role':'user',
                        'content': f"【{summary}】"},]
        after = get_completion_from_messages(messages)
        # 构建 edge-tts 命令
        command = f'edge-tts --voice zh-CN-XiaoyiNeural --text "{after}" --write-media hello.mp3'
        # 使用 subprocess 运行命令
        subprocess.run(command, shell=True)


    my_bar.progress(100, text="Almost there...")

    with radio_placeholder:
        #audio_file = open('hello.mp3', 'rb')
        #audio_bytes = audio_file.read()
        #st.audio(audio_bytes, format='wav')
        autoplay_audio("hello.mp3")

    my_bar.empty()
    if st.session_state.language == 'English':
        st.subheader('Summary and Commentary', divider='rainbow')
        st.markdown(summary)

        st.subheader('Technology News', divider='red')
        for i in range(len(google_news['title'])):
            if google_news['summary'][i] != 'No result':
                st.markdown(f'<a href="{google_news["url"][i]}" style="color: #2859C0; text-decoration: none; \
                font-size: 20px;font-weight: bold;"> {google_news["title"][i]} </a>\
                    <span style="margin-left: 10px; background-color: white; padding: 0px 7px; border: 1px solid rgb(251, 88, 88); border-radius: 20px; font-size: 7px; color: rgb(251, 88, 88)">Google News</span>', unsafe_allow_html=True)
                st.markdown(google_news['summary'][i])

        st.subheader('Podcast and Speeches', divider='orange')

        st.markdown(f'<a href="https://lexfridman.com/podcast/" style="color:  #2859C0; text-decoration: none; \
            font-size: 20px;font-weight: bold;">{L_title}</a>\
                    <span style="margin-left: 10px; background-color: white; padding: 0px 7px; border: 1px solid rgb(251, 88, 88); border-radius: 20px; font-size: 7px; color: rgb(251, 88, 88)">Lexi Fridman</span>', unsafe_allow_html=True)
        st.markdown(lexi_boardcast)
        
        st.subheader('Technology Blogs', divider='green')
        st.markdown(f'<a href= {openai_blog_url} style="color:  #2859C0; text-decoration: none; \
            font-size: 20px;font-weight: bold;"> {openai_title}</a>\
                <span style="margin-left: 10px; background-color: white; padding: 0px 7px; border: 1px solid rgb(251, 88, 88); border-radius: 20px; font-size: 7px; color: rgb(251, 88, 88)">Openai</span>', unsafe_allow_html=True)
        st.markdown(openai_blog)  

        st.markdown(f'<a href={link} style="color:  #2859C0; text-decoration: none; \
            font-size: 20px;font-weight: bold;"> {M_title}</a>\
                <span style="margin-left: 10px; background-color: white; padding: 0px 7px; border: 1px solid rgb(251, 88, 88); border-radius: 20px; font-size: 7px; color: rgb(251, 88, 88)">Microsoft</span>', unsafe_allow_html=True)
        st.markdown(bair_blog)
        
        st.markdown(f'<a href="{A_link}" style="color:  #2859C0; text-decoration: none; \
            font-size: 20px;font-weight: bold;"> {A_title}</a>\
                    <span style="margin-left: 10px; background-color: white; padding: 0px 7px; border: 1px solid rgb(251, 88, 88); border-radius: 20px; font-size: 7px; color: rgb(251, 88, 88)">Amazon</span>', unsafe_allow_html=True)
        st.markdown(mit_blog)

        st.markdown(
            f'<a href="https://machinelearning.apple.com/" style="color:  #2859C0; text-decoration: none; font-size: 20px; font-weight: bold;">Recent research</a>\
            <span style="margin-left: 10px; background-color: white; padding: 0px 7px; border: 1px solid rgb(251, 88, 88); border-radius: 20px; font-size: 7px; color: rgb(251, 88, 88)">Apple</span>', 
            unsafe_allow_html=True
        )
        st.markdown(Apple_blog)


        st.subheader('Cutting-edge Papers', divider='green')
        for result in search.results():
            st.markdown(f'<a href="{result.entry_id}" style="color:  #2859C0; text-decoration: none; \
            font-size: 20px;font-weight: bold;"> {result.title} </a>\
             <span style="margin-left: 10px; background-color: white; padding: 0px 7px; border: 1px solid rgb(251, 88, 88); border-radius: 20px; font-size: 7px; color: rgb(251, 88, 88)">{result.primary_category}</span>\
                ', unsafe_allow_html=True)
            st.markdown(result.summary)
            

    elif st.session_state.language == '中文':
        st.subheader('摘要与评论', divider='rainbow')
        summary = summary.replace('<|endoftext|>', '')
        messages =  [
                        {'role':'system',
                        'content': system_message_3},
                        {'role':'user',
                        'content': f"【{summary}】"},]
        summary = get_completion_from_messages(messages)
        st.markdown(summary)

        st.subheader('科技新闻', divider='rainbow')
        for i in range(len(google_news['title'])):
            title = google_news['title'][i]
            messages =  [
                        {'role':'system',
                        'content': system_message_3},
                        {'role':'user',
                        'content': f"【{title}】"},]
            
            title = get_completion_from_messages(messages)
            news_summary = google_news['summary'][i]
            messages =  [
                        {'role':'system',
                        'content': system_message_3},
                        {'role':'user',
                        'content': f"【{news_summary}】"},]
            news_summary = get_completion_from_messages(messages)
 
            st.markdown(f'<a href="{google_news["url"][i]}" style="color:  #2859C0; text-decoration: none;">#### {title}</a>', unsafe_allow_html=True)
            st.markdown(news_summary)


        st.subheader('播客与演讲', divider='orange')
        lexi_boardcast = lexi_boardcast.replace('<|endoftext|>', '')
        messages =  [
                        {'role':'system',
                        'content': system_message_3},
                        {'role':'user',
                        'content': f"【{lexi_boardcast}】"},]
        lexi_boardcast = get_completion_from_messages(messages)
        st.markdown(lexi_boardcast)
        st.markdown(f"[more on](https://www.youtube.com/@lexfridman/videos)\n")
        
        st.subheader('科技博客', divider='green')
        openai_blog = openai_blog.replace('<|endoftext|>', '')
        messages =  [
                        {'role':'system',
                        'content': system_message_3},
                        {'role':'user',
                        'content': f"【{openai_blog}】"},]
        openai_blog = get_completion_from_messages(messages)
        st.markdown(openai_blog)
        st.markdown(f"[more on](https://openai.com/)\n")

        bair_blog = bair_blog.replace('<|endoftext|>', '')
        messages =  [
                        {'role':'system',
                        'content': system_message_3},
                        {'role':'user',
                        'content': f"【{bair_blog}】"},]
        bair_blog = get_completion_from_messages(messages)
        st.markdown(bair_blog)
        st.markdown(f"[more on](https://bair.berkeley.edu/blog/)\n")

        mit_blog = mit_blog.replace('<|endoftext|>', '')
        messages =  [
                        {'role':'system',
                        'content': system_message_3},
                        {'role':'user',
                        'content': f"【{mit_blog}】"},]
        mit_blog = get_completion_from_messages(messages)
        st.markdown(mit_blog)
        st.markdown(f"[more on](https://news.mit.edu/topic/artificial-intelligence2)\n")

        st.subheader('尖端论文', divider='green')
        for result in search.results():
            title = result.title
            result_summary = result.summary
            messages =  [
                        {'role':'system',
                        'content': system_message_3},
                        {'role':'user',
                        'content': f"【{title}】"},]
            result_title = get_completion_from_messages(messages)

            messages =  [
                        {'role':'system',
                        'content': system_message_3},
                        {'role':'user',
                        'content': f"【{result_summary}】"},]
            result_summary = get_completion_from_messages(messages)

            st.markdown(f'<a href="{result.entry_id}" style="color:  #2859C0; text-decoration: none;">#### {result_title}</a>', unsafe_allow_html=True)
            st.markdown(result_summary)


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

    if "audio_length" not in st.session_state:
        st.session_state.audio_length = '5'
    


    # 根据session状态来渲染页面
    if st.session_state.page == "one":
        page_one()
    elif st.session_state.page == "two":
        page_two()

if __name__ == "__main__":
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
    main()

