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
                You are a very talented editor, skilled at consolidating 
                fragmented information and introductions into a cohesive script, without missing any details.
                Compile the news article based on the information in 【】.  
                '''

system_message_2 = '''
                You are a linguist, skilled in summarizing textual content and presenting it in 3 bullet points using markdown. 
                '''

system_message_3 = '''
                你是个语言学家，擅长把英文翻译成中文。要注意表达的流畅和使用中文的表达习惯。不要返回多余的信息，只把文字翻译成中文。
                '''

def find_next_link_text(url, target_link, target_text):
    """
    Find the first link and text after the given target link and text on the specified URL.
    
    Parameters:
        url (str): The URL of the webpage to scrape.
        target_link (str): The specific link to be found.
        target_text (str): The specific link text to be found.
        
    Returns:
        tuple: A tuple containing the next link and its text. Returns (None, None) if not found.
    """
    
    # Send a GET request
    response = requests.get(url)
    response.raise_for_status()  # This will raise an exception if there's an error
    
    # Parse the content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find all the <ul> elements
    ul_elems = soup.find_all('ul')
    
    # Initialize a list to store all links and their texts
    all_links = []
    
    # Extract links and texts from all <ul> elements
    for ul_elem in ul_elems:
        links = [(link.get('href'), link.text) for link in ul_elem.find_all('a')]
        all_links.extend(links)
    
    # Extract the first link and text after the specified link-text pair
    found = False
    for link, text in all_links:
        if found:
            return link, text
        if link == target_link and text == target_text:
            found = True
            
    return None, None
  
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

def extract_data_from_url(url, class_name):
    """
    从指定的URL中提取特定类名的<a>标签的href属性和文本内容。

    参数:
    - url (str): 要提取数据的网页URL。
    - class_name (str): 要查找的<a>标签的类名。

    """

    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        target_a = soup.find('a', class_=class_name)

        if target_a:
            data_mrf_link = target_a.get('href')
            text = target_a.get_text().strip()
            return (data_mrf_link, text)
        else:
            raise ValueError("找不到目标元素。")
    else:
        raise ConnectionError("请求失败。")
    
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
            <audio controls autoplay style="width: 100%;">
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
    llm = ChatOpenAI(temperature=1, model_name="gpt-3.5-turbo-16k")
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(split_docs)
    return summary


def get_completion_from_messages(messages,
                                 model="gpt-3.5-turbo-16k",
                                 temperature=1.5, max_tokens=7000):
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



def summarize_website_content(url, temperature=1, model_name="gpt-3.5-turbo-16k", chain_type="stuff"):
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
    if True:
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


def input_page(st, **state):
    # Include Font Awesome CSS
    st.markdown(
        """
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
        """,
        unsafe_allow_html=True,
    )

    # Style and position the GitHub and Twitter icons at the bottom left corner
    st.markdown(
        """
        <style>
            .social-icons {
                gap: 10px;  # Space between icons
            }
            .social-icons a i {
                color: #6c6c6c;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    # Add the GitHub and Twitter icons with hyperlinks
    github_url = "https://github.com/xl631212/llm_newsletter/tree/main"  # replace with your GitHub repo URL
    twitter_url = "https://twitter.com/li_xuying"  # replace with your Twitter profile URL

    st.markdown("""
    <h1 style='text-align: center; color: black;'>
        Your Personal <span style='color: #FF4B4B; font-size: 1.25em;'>AI News</span> Podcast
    </h1>
    <div class="social-icons" style='text-align: center; color: black;'>
            <a href="https://github.com/xl631212/llm_newsletter/tree/main" target="_blank"><i class="fab fa-github fa-2x"></i></a>
            <a href="https://twitter.com/li_xuying" target="_blank"><i class="fab fa-twitter fa-2x"></i></a>
        </div>
    """, 
    unsafe_allow_html=True
    )
    st.markdown("<h3 style='text-align: center; color: black;'>Empower Your Day with Real-Time Insights: Leveraging AI for Instant News <br> and Podcast Updates.</h3>", unsafe_allow_html=True)
    st.markdown("""
        <h4 style='text-align: center; color: #6C6C6C;'>
            Choose your preferred options🔘 at the bottom, then double-click👆 the button below to initiate. 
                <br>
                Sit back and relax while we craft your personalized LLM podcast within <span style='color: #2859C0; font-size: 1.15em;'>3 mins</span>.
        </h4>
        <br><br>
        """, 
        unsafe_allow_html=True)
    

    button_placeholder = st.empty()
    st.markdown("<br>", unsafe_allow_html=True)

    with st.container():
        col3a, col4a, col5a= st.columns([4,7,4])
        with col3a:
            pass
        with col4a:
            col1a, col2a, col8a = st.columns([3,1,3])
            with col1a:
                st.write("**Options🔘:**")
        with col5a:
            pass

    st.markdown("""
        <style>
            .stButton > button {
                font-size: 100px;
                width: 35%;  /* 设置一个固定的宽度 */
                height: 50px; /* 设置一个固定的高度 */
                color: white;
                background-color: #FF4B4B;
                border: none;
                border-radius: 15px;
                margin: auto;
                font-weight: bold; 
                display: flex;
                justify-content: center;
                align-items: center;
            }

            .stButton > button:hover {
                background-color: #EFEFEF; /* 为按钮添加简单的悬停效果 */
                color: #9A9A9A;
            }

            .stButton > button div p {
                font-size: 24px;  /* 改变按钮文本的字号 */
                margin: 0;  /* 移除段落的默认边距 */
            }
                
            .stButton > button div p:hover {
                font-size: 20px;
            }
        </style>
        """, unsafe_allow_html=True)
    

    with st.container():
        col3, col4, col5= st.columns([4,7,4])
        with col3:
            pass
        with col4:

            col1, col2, col8 = st.columns([4,2,4])
            with col1:
                language = st.selectbox(
                    "Language",
                    ("English", "中文"),
                    key='ahaha'
                )
                audio_length_adjust = st.select_slider('Audio length', options=['short', 'medium', 'long'],value=('medium'))
                if audio_length_adjust == 'short':
                    audio_length = 200
                elif audio_length_adjust == 'medium':
                    audio_length = 350
                else:
                    audio_length = 500
                st.session_state.audio_length = audio_length

            
            with col8:
                options_2 = st.selectbox(
                    'In a tone of',
                    ['Informal', 'Professional', 'Humorous'],
                    key='opt3'
                )
                day = st.select_slider('Information volume', options=['small', 'medium', 'large'],value=('medium'))
                if day == 'small':
                    st.session_state.day = 2
                    st.session_state.arxiv = 2
                elif day == 'medium':
                    st.session_state.day = 4
                    st.session_state.arxiv = 3
                else:
                    st.session_state.day = 6
                    st.session_state.arxiv = 4

        with col5:
            pass


    with button_placeholder:     
        # 创建按钮
        if st.button("👆 Double-Click Generation"):
            st.session_state.page = "two"
            st.session_state.language = language
            if options_2 == 'Informal':
                st.session_state.tone = """read news and present them in a casual and conversational tone. 
                You should use everyday language, contractions, and slang to engage the audience and make the news more relatable. """
            elif options_2 == 'Humorous':
                st.session_state.tone = """read news and present in a comical and amusing tone. 
                You should be able to recognize and exaggerate humorous elements of each article along with jokes and deliver them in a way 
                that will make the audience laugh."""


    st.markdown("""
        <style>
            .footer {
                position: fixed;
                bottom: 0;
                left: 10px;
                width: auto;
                background-color: transparent;
                text-align: right;
                padding-right: 10px;
                padding-bottom: 10px;
            }
        </style>
        <div class="footer">Made with ❤️ by Xuying Li</div>
    """, unsafe_allow_html=True)
        
        
      
def compute_page(st, **state):
    # Include Font Awesome CSS
    st.markdown(
        """
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
        """,
        unsafe_allow_html=True,
    )

    # Style and position the GitHub and Twitter icons at the bottom left corner
    st.markdown(
        """
        <style>
            .social-icons {
                gap: 10px;  # Space between icons
            }
            .social-icons a i {
                color: #6c6c6c;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    # Add the GitHub and Twitter icons with hyperlinks
    github_url = "https://github.com/xl631212/llm_newsletter/tree/main"  # replace with your GitHub repo URL
    twitter_url = "https://twitter.com/li_xuying"  # replace with your Twitter profile URL

    st.markdown("""
    <h1 style='text-align: center; color: black;'>
        Your Personal <span style='color: #FF4B4B; font-size: 1.25em;'>AI News</span> Podcast
    </h1>
    <div class="social-icons" style='text-align: center; color: black;'>
            <a href="https://github.com/xl631212/llm_newsletter/tree/main" target="_blank"><i class="fab fa-github fa-2x"></i></a>
            <a href="https://twitter.com/li_xuying" target="_blank"><i class="fab fa-twitter fa-2x"></i></a>
        </div>
    """, 
    unsafe_allow_html=True
    )
   
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
    M_title, Microsoft_link = extract_blog_link_info(url)
    bair_blog = summarize_website_content(Microsoft_link)


    my_bar.progress(20, text="Searching for Amazon Blog...")
    A_title, A_link = get_latest_aws_ml_blog()
    mit_blog = summarize_website_content(A_link)

    my_bar.progress(30, text="Searching for Apple Blog...")
    url = 'https://machinelearning.apple.com/'
    
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # 根据提供的HTML片段，定位到文章的标题和链接
    article = soup.select_one('h3.post-title a')
    apple_link = 'https://machinelearning.apple.com'+ article['href']
    
    Apple_blog_title = article.text
    Apple_blog = summarize_website_content(apple_link)

    my_bar.progress(35, text='Searching for machine learning street talk...')
    channel_id = "UCMLtBahI5DMrt0NPvDSoIRQ"
    playlist = Playlist(playlist_from_channel_id(channel_id))

    while playlist.hasMoreVideos:
        playlist.getNextVideos()

    machine_title = playlist.videos[0]['title']
    machine_link = playlist.videos[0]['link']
    machine_learning_boardcast = summarize_website_content(machine_link)

    my_bar.progress(40, text='Searching for lex friman boardcast...')
    url = "https://lexfridman.com/podcast/"
    link = get_transcript_link(url)
    L_title = get_h1_text(link)
    youtube_link = get_youtube_link(url)
    lexi_boardcast = summarize_website_content(youtube_link)
    

    my_bar.progress(50, text="Searching for arxiv ...")
    search = arxiv.Search(
        query = "AI, LLM, machine learning, NLP",
        max_results = st.session_state.arxiv,
        sort_by = arxiv.SortCriterion.SubmittedDate
    )
    ariv_essay = ''
    for result in search.results():
        ariv_essay += result.summary
    
    my_bar.progress(60, text="Searching Google News...")
    google_news = fetch_gnews_links(query='AI, LLM, Machine learning', max_results = st.session_state.day)

    my_bar.progress(70, text="Searching Techcrunch...")
    url = 'https://techcrunch.com/category/artificial-intelligence/'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    articles = soup.select('.post-block__title a')

    data_mrf_link, h_title = articles[0]['href'],articles[0].text
    h_content = summarize_website_content(data_mrf_link)

    my_bar.progress(75, text="Nvidia Podcast...")
    url = "https://blogs.nvidia.com/ai-podcast/"
    target_link = "https://blogs.nvidia.com/ai-podcast/"
    target_text = "AI Podcast"
    next_link, Nvidia_title = find_next_link_text(url, target_link, target_text)
    n_content = summarize_website_content(next_link)


    my_bar.progress(80, text="Writing Newsletter...")

    query = n_content + str(google_news['summary'])  + str(mit_blog)  \
              + openai_blog
    
    query = query.replace('<|endoftext|>', '')
    messages =  [
                    {'role':'system',
                    'content': system_message + "keep it equal to {} words.".format(st.session_state.audio_length) + st.session_state.tone},
                    {'role':'user',
                    'content': f"【{query}】"},]
    response = get_completion_from_messages(messages)
    print(response)
    my_bar.progress(90, text="Generating Podcast...")
    if st.session_state.language == 'English':
        updated = response.replace('-', '').replace('--', '').replace('"', '').replace('“', '')
        command = f'edge-tts --text "{updated}" --write-media hello.mp3'
        subprocess.run(command, shell=True)
        my_bar.progress(90, text="Generating Summary...")

        query = response
        messages =  [
                        {'role':'system',
                        'content': system_message_2},
                        {'role':'user',
                        'content': f"【{query}】"},]
        summary = get_completion_from_messages(messages)
    
    else:
        before = response
        before = before.replace('<|endoftext|>', '')
        messages =  [
                        {'role':'system',
                        'content': system_message_3},
                        {'role':'user',
                        'content': f"【{before}】"},]
        after = get_completion_from_messages(messages)
        # 构建 edge-tts 命令
        command = f'edge-tts --voice zh-CN-XiaoyiNeural --text "{after}" --write-media hello2.mp3'
        # 使用 subprocess 运行命令
        subprocess.run(command, shell=True)


    my_bar.progress(100, text="Almost there...")

    with radio_placeholder:
        #audio_file = open('hello.mp3', 'rb')
        #audio_bytes = audio_file.read()
        #st.audio(audio_bytes, format='wav')
        if st.session_state.language == 'English':
          autoplay_audio("hello.mp3")
        else:
          autoplay_audio("hello2.mp3")
        

    my_bar.empty()
    if st.session_state.language == 'English':
        st.subheader('Summary and Commentary', divider='rainbow')
        st.markdown(summary)

        st.subheader('Technology News', divider='red')
        for i in range(len(google_news['title'])):
            if len(google_news['summary'][i]) > 100:
                st.markdown(f'<a href="{google_news["url"][i]}" style="color: #2859C0; text-decoration: none; \
                font-size: 20px;font-weight: bold;"> {google_news["title"][i]} </a>\
                    <span style="margin-left: 10px; background-color: white; padding: 0px 7px; border: 1px solid rgb(251, 88, 88); border-radius: 20px; font-size: 7px; color: rgb(251, 88, 88)">Google News</span>', unsafe_allow_html=True)
                st.markdown(google_news['summary'][i])
        
        st.markdown(f'<a href="{data_mrf_link}" style="color:  #2859C0; text-decoration: none; \
            font-size: 20px;font-weight: bold;">{h_title}</a>\
                    <span style="margin-left: 10px; background-color: white; padding: 0px 7px; border: 1px solid rgb(251, 88, 88); border-radius: 20px; font-size: 7px; color: rgb(251, 88, 88)">Techcrunch</span>', unsafe_allow_html=True)
        st.markdown(h_content)

        st.subheader('Podcast and Speeches', divider='orange')

        st.markdown(f'<a href="{youtube_link}" style="color:  #2859C0; text-decoration: none; \
            font-size: 20px;font-weight: bold;">{L_title}</a>\
                    <span style="margin-left: 10px; background-color: white; padding: 0px 7px; border: 1px solid rgb(251, 88, 88); border-radius: 20px; font-size: 7px; color: rgb(251, 88, 88)">Lex Fridman</span>', unsafe_allow_html=True)
        st.markdown(lexi_boardcast)

        st.markdown(f'<a href="{next_link}" style="color:  #2859C0; text-decoration: none; \
            font-size: 20px;font-weight: bold;">{Nvidia_title}</a>\
                    <span style="margin-left: 10px; background-color: white; padding: 0px 7px; border: 1px solid rgb(251, 88, 88); border-radius: 20px; font-size: 7px; color: rgb(251, 88, 88)">Nvidia</span>', unsafe_allow_html=True)
        st.markdown(n_content)

        st.markdown(f'<a href="{machine_link}" style="color:  #2859C0; text-decoration: none; \
            font-size: 20px;font-weight: bold;">{machine_title}</a>\
                    <span style="margin-left: 10px; background-color: white; padding: 0px 7px; border: 1px solid rgb(251, 88, 88); border-radius: 20px; font-size: 7px; color: rgb(251, 88, 88)">Machine Learning Street Talk</span>', unsafe_allow_html=True)
        st.markdown(machine_learning_boardcast)
      
        st.subheader('Technology Blogs', divider='green')
        st.markdown(f'<a href= {openai_blog_url} style="color:  #2859C0; text-decoration: none; \
            font-size: 20px;font-weight: bold;"> {openai_title}</a>\
                <span style="margin-left: 10px; background-color: white; padding: 0px 7px; border: 1px solid rgb(251, 88, 88); border-radius: 20px; font-size: 7px; color: rgb(251, 88, 88)">Openai</span>', unsafe_allow_html=True)
        st.markdown(openai_blog)  

        st.markdown(f'<a href={Microsoft_link} style="color:  #2859C0; text-decoration: none; \
            font-size: 20px;font-weight: bold;"> {M_title}</a>\
                <span style="margin-left: 10px; background-color: white; padding: 0px 7px; border: 1px solid rgb(251, 88, 88); border-radius: 20px; font-size: 7px; color: rgb(251, 88, 88)">Microsoft</span>', unsafe_allow_html=True)
        st.markdown(bair_blog)
        
        st.markdown(f'<a href="https://aws.amazon.com/blogs/machine-learning/" style="color:  #2859C0; text-decoration: none; \
            font-size: 20px;font-weight: bold;"> {A_title}</a>\
                    <span style="margin-left: 10px; background-color: white; padding: 0px 7px; border: 1px solid rgb(251, 88, 88); border-radius: 20px; font-size: 7px; color: rgb(251, 88, 88)">Amazon</span>', unsafe_allow_html=True)
        st.markdown(mit_blog)

        st.markdown(
            f'<a href={apple_link} style="color:  #2859C0; text-decoration: none; font-size: 20px; font-weight: bold;">{Apple_blog_title}</a>\
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
        summary = after.replace('<|endoftext|>', '')
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
 
            st.markdown(f'<a href="{google_news["url"][i]}" style="color: #2859C0; text-decoration: none; \
                font-size: 20px;font-weight: bold;"> {title} </a>\
                    <span style="margin-left: 10px; background-color: white; padding: 0px 7px; border: 1px solid rgb(251, 88, 88); border-radius: 20px; font-size: 7px; color: rgb(251, 88, 88)">Google News</span>', unsafe_allow_html=True)
            st.markdown(news_summary)
        news_summary = h_title
        messages =  [
                        {'role':'system',
                        'content': system_message_3},
                        {'role':'user',
                        'content': f"【{news_summary}】"},]
        h_title = get_completion_from_messages(messages)
        st.markdown(f'<a href="{data_mrf_link}" style="color:  #2859C0; text-decoration: none; \
            font-size: 20px;font-weight: bold;">{h_title}</a>\
                    <span style="margin-left: 10px; background-color: white; padding: 0px 7px; border: 1px solid rgb(251, 88, 88); border-radius: 20px; font-size: 7px; color: rgb(251, 88, 88)">Techcrunch</span>', unsafe_allow_html=True)
        news_summary = h_content
        messages =  [
                        {'role':'system',
                        'content': system_message_3},
                        {'role':'user',
                        'content': f"【{news_summary}】"},]
        h_content = get_completion_from_messages(messages)
        st.markdown(h_content)

        st.subheader('播客与博客', divider='orange')
        news_summary = L_title
        messages =  [
                        {'role':'system',
                        'content': system_message_3},
                        {'role':'user',
                        'content': f"【{news_summary}】"},]
        L_title = get_completion_from_messages(messages)
        st.markdown(f'<a href="{youtube_link}" style="color:  #2859C0; text-decoration: none; \
            font-size: 20px;font-weight: bold;">{L_title}</a>\
                    <span style="margin-left: 10px; background-color: white; padding: 0px 7px; border: 1px solid rgb(251, 88, 88); border-radius: 20px; font-size: 7px; color: rgb(251, 88, 88)">Lex Fridman</span>', unsafe_allow_html=True)
        news_summary = lexi_boardcast
        messages =  [
                        {'role':'system',
                        'content': system_message_3},
                        {'role':'user',
                        'content': f"【{news_summary}】"},]
        lexi_boardcast = get_completion_from_messages(messages)
        st.markdown(lexi_boardcast)

        news_summary = Nvidia_title
        messages =  [
                        {'role':'system',
                        'content': system_message_3},
                        {'role':'user',
                        'content': f"【{news_summary}】"},]
        Nvidia_title = get_completion_from_messages(messages)
        st.markdown(f'<a href="{next_link}" style="color:  #2859C0; text-decoration: none; \
            font-size: 20px;font-weight: bold;">{Nvidia_title}</a>\
                    <span style="margin-left: 10px; background-color: white; padding: 0px 7px; border: 1px solid rgb(251, 88, 88); border-radius: 20px; font-size: 7px; color: rgb(251, 88, 88)">Nvidia</span>', unsafe_allow_html=True)
        news_summary = n_content
        messages =  [
                        {'role':'system',
                        'content': system_message_3},
                        {'role':'user',
                        'content': f"【{news_summary}】"},]
        n_content = get_completion_from_messages(messages)
        st.markdown(n_content)

        news_summary = machine_title
        messages =  [
                        {'role':'system',
                        'content': system_message_3},
                        {'role':'user',
                        'content': f"【{news_summary}】"},]
        machine_title = get_completion_from_messages(messages)
        st.markdown(f'<a href="{machine_link}" style="color:  #2859C0; text-decoration: none; \
            font-size: 20px;font-weight: bold;">{machine_title}</a>\
                    <span style="margin-left: 10px; background-color: white; padding: 0px 7px; border: 1px solid rgb(251, 88, 88); border-radius: 20px; font-size: 7px; color: rgb(251, 88, 88)">Machine Learning Street Talk</span>', unsafe_allow_html=True)
        
        news_summary = machine_learning_boardcast
        messages =  [
                        {'role':'system',
                        'content': system_message_3},
                        {'role':'user',
                        'content': f"【{news_summary}】"},]
        machine_learning_boardcast = get_completion_from_messages(messages)
        st.markdown(machine_learning_boardcast)

        st.subheader('科技博客', divider='green')
        openai_blog = openai_blog.replace('<|endoftext|>', '')
        messages =  [
                        {'role':'system',
                        'content': system_message_3},
                        {'role':'user',
                        'content': f"{openai_blog}"},]
        openai_blog = get_completion_from_messages(messages)


        messages =  [
                        {'role':'system',
                        'content': system_message_3},
                        {'role':'user',
                        'content': f"【{openai_title}】"},]
        openai_title = get_completion_from_messages(messages)

        st.markdown(f'<a href= {openai_blog_url} style="color:  #2859C0; text-decoration: none; \
            font-size: 20px;font-weight: bold;"> {openai_title}</a>\
                <span style="margin-left: 10px; background-color: white; padding: 0px 7px; border: 1px solid rgb(251, 88, 88); border-radius: 20px; font-size: 7px; color: rgb(251, 88, 88)">Openai</span>', unsafe_allow_html=True)
        st.markdown(openai_blog)

        bair_blog = bair_blog.replace('<|endoftext|>', '')
        messages =  [
                        {'role':'system',
                        'content': system_message_3},
                        {'role':'user',
                        'content': f"【{bair_blog}】"},]
        bair_blog = get_completion_from_messages(messages)

        messages =  [
                        {'role':'system',
                        'content': system_message_3},
                        {'role':'user',
                        'content': f"{M_title}"},]
        M_title = get_completion_from_messages(messages)
        st.markdown(f'<a href={link} style="color:  #2859C0; text-decoration: none; \
            font-size: 20px;font-weight: bold;"> {M_title}</a>\
                <span style="margin-left: 10px; background-color: white; padding: 0px 7px; border: 1px solid rgb(251, 88, 88); border-radius: 20px; font-size: 7px; color: rgb(251, 88, 88)">Microsoft</span>', unsafe_allow_html=True)
        st.markdown(bair_blog)

        mit_blog = mit_blog.replace('<|endoftext|>', '')
        messages =  [
                        {'role':'system',
                        'content': system_message_3},
                        {'role':'user',
                        'content': f"【{mit_blog}】"},]
        mit_blog = get_completion_from_messages(messages)

        messages =  [
                        {'role':'system',
                        'content': system_message_3},
                        {'role':'user',
                        'content': f"{A_title}"},]
        A_title = get_completion_from_messages(messages)
        st.markdown(f'<a href="{A_link}" style="color:  #2859C0; text-decoration: none; \
            font-size: 20px;font-weight: bold;"> {A_title}</a>\
                    <span style="margin-left: 10px; background-color: white; padding: 0px 7px; border: 1px solid rgb(251, 88, 88); border-radius: 20px; font-size: 7px; color: rgb(251, 88, 88)">Amazon</span>', unsafe_allow_html=True)
        st.markdown(mit_blog)
        

        st.subheader('尖端论文', divider='green')
        for result in search.results():
            title = result.title
            result_summary = result.summary
            messages =  [
                        {'role':'system',
                        'content': system_message_3},
                        {'role':'user',
                        'content': f"{title}"},]
            result_title = get_completion_from_messages(messages)

            messages =  [
                        {'role':'system',
                        'content': system_message_3},
                        {'role':'user',
                        'content': f"{result_summary}"},]
            result_summary = get_completion_from_messages(messages)

            st.markdown(f'<a href="{result.entry_id}" style="color:  #2859C0; text-decoration: none; \
            font-size: 20px;font-weight: bold;"> {result_title} </a>\
             <span style="margin-left: 10px; background-color: white; padding: 0px 7px; border: 1px solid rgb(251, 88, 88); border-radius: 20px; font-size: 7px; color: rgb(251, 88, 88)">{result.primary_category}</span>\
                ', unsafe_allow_html=True)
            st.markdown(result_summary)
    st.markdown("""
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 10px;
            width: auto;
            background-color: transparent;
            text-align: left;
            padding-left: 10px;
            padding-top: 10px;
        }
    </style>
    <div class="footer">Made with ❤️ by Xuying Li</div>
""", unsafe_allow_html=True)

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
        st.session_state.language = "English"

    if "audio_length" not in st.session_state:
        st.session_state.audio_length = '5'

    if "day" not in st.session_state:
        st.session_state.day = 0
        st.session_state.arxiv = 0
    
    if "tone" not in st.session_state:
        st.session_state.tone = ''


    # 根据session状态来渲染页面
    if st.session_state.page == "one":
        page_one()
    elif st.session_state.page == "two":
        page_two()

if __name__ == "__main__":
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
    main()



