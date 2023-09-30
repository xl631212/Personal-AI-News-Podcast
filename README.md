# Your Personal AI News Podcast

Empower Your Day with Real-Time Insights: Leveraging AI for Instant News
and Podcast Updates.

## App Preview

<p float="left">
  <img src="截屏2023-09-30 10.58.48.png" width="100%" />
</p>

## App Link：https://llmnewsletter-cup7ts82sxzkzyszvquh4l.streamlit.app/
## Features:

1. **Real-time Content Aggregation**: Gathers the freshest research, blogs, discussions, and news related to AI.
2. **Wide Source Integration**: Summarizes content from esteemed platforms including:
   - Openai Blog
   - HackerNews
   - Nvidia Blog
   - Techcrunch News
   - Google News
3. **Newsletter and Podcast Generation**: Not only presents written content but also converts it into an immersive audio experience.
[TBD]4. **Interactive User Interface**: (If applicable) Provides a user-friendly web interface for customizing content preferences.

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

## Future Plans

As we continue to improve Stay Ahead, we have several exciting features and enhancements lined up:

- [✅] **Add More Information Sources**: Integrate even more data sources to provide a comprehensive view of the latest developments in the field of AI.
   
- [ ] **Advanced Settings**: Give users more control over their podcast experience with:
   - [✅] **2.1 Voice Duration**: Adjust the speed and length of the audio to fit your schedule.
   - [✅] **2.2 Linguistic Style**: Choose between formal, casual, or other voice styles to match your preference.
   - [✅] **2.3 Language Selection**: Multiple language options to cater to our diverse user base.
   - [ ] **2.4 Source Filtering**: Select and prioritize the sources you're most interested in.
   
- [ ] **User Feedback Functionality**: Implement a direct feedback mechanism within the app to gather invaluable insights from users.

Stay tuned for these updates and more as we strive to make Stay Ahead your go-to AI news and podcast platform.


## Contribute

Contributions are always welcome! Please ensure any pull requests are made against the `develop` branch.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
