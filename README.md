# Stay Ahead: Real-time News and Podcasts with LLM

Stay Ahead leverages Large Language Models (LLM) and web scraping techniques to curate a real-time newsletter based on the latest updates, research, blogs, and discussions in the field of AI.

## App Preview

<p float="left">
  <img src="截屏2023-09-19 23.10.09.png" width="100%" />
</p>

## Features:

1. **Real-time Content Aggregation**: Gathers the freshest research, blogs, discussions, and news related to AI.
2. **Wide Source Integration**: Summarizes content from esteemed platforms including:
   - Openai Blog
   - HackerNews
   - BAIR Blog
   - MIT AI News
   - Google News
3. **Newsletter and Podcast Generation**: Not only presents written content but also converts it into an immersive audio experience.
4. **Interactive User Interface**: (If applicable) Provides a user-friendly web interface for customizing content preferences.

## How It Works:

1. **Data Collection**: Uses web scraping tools to fetch information from a plethora of online AI sources.
2. **Content Summarization**: Leverages OpenAI's Chat Model to condense and highlight crucial points from the aggregated content.
3. **Audio Broadcast**: Produces an audio podcast, enabling users to listen on-the-go.
4. **User-friendly Display**: Showcases the refined content in an easy-to-read format, ensuring a seamless user experience.

## Installation and Setup

1. Make sure you have Python installed.
2. Install the required packages: `pip install -r requirements.txt` (You will need to create a `requirements.txt` file containing the required libraries)
3. Set up environment variables:
    ```
    export SERPER_API_KEY=<Your SERPER API Key>
    export OPENAI_API_KEY=<Your OpenAI API Key>
    ```
4. Run the app using Streamlit: `streamlit run app.py` (or whatever you name the main script).

## Contribute

Contributions are always welcome! Please ensure any pull requests are made against the `develop` branch.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
