# Stay Ahead: Real-time News and Podcasts with LLM

Stay Ahead leverages Large Language Models (LLM) and web scraping techniques to curate a real-time newsletter based on the latest updates, research, blogs, and discussions in the field of AI.

![App Preview](https://www.groundzeroweb.com/wp-content/uploads/2017/05/Funny-Cat-Memes-11.jpg)

## Features

- Real-time collection of the latest in LLM research, blogs, discussions, and news.
- Summarizes and presents content from platforms including:
  - [Openai Blog](#)
  - [HackerNews](#)
  - [BAIR Blog](https://bair.berkeley.edu/blog/)
  - [MIT AI News](https://news.mit.edu/topic/artificial-intelligence2)
  - [Google News](#)
- Generates newsletters and supports text-to-speech functionalities for an audio-based experience.
  
## How It Works

1. The app uses web scraping utilities to collect data from various online sources.
2. Gathers and summarizes articles using OpenAI's Chat Model.
3. Generates a broadcast in the form of an audio file which users can listen to.
4. Displays the summarized content from each source for a user-friendly reading experience.

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
