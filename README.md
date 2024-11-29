# MRSL.Tech

MRSL.Tech is an AI-powered tool designed to help users interact with and extract meaningful metadata from images. This project uses Streamlit for building interactive web applications and integrates **Ollama** for natural language processing to assist with queries. Whether you're exploring EXIF data or asking general questions, MRSL.Tech provides an intelligent assistant to make your experience easy and interactive.

---

## Features

- **Image Metadata Extraction:** Easily extract and view EXIF data from images.
- **Natural Language Processing:** Use Ollama for answering user queries related to image metadata or general questions.
- **Interactive Chat Interface:** Seamlessly interact with MRSL via a chat interface built with Streamlit.
- **Snow Effect & Particles Animation:** A visually appealing experience with animations (snow effect and particle animation).

---

## Technologies Used

- **Streamlit**: For building the interactive web app and displaying real-time chat.
- **Ollama**: Used for NLP and chatbot interaction.
- **Pandas**: For handling image metadata and EXIF data manipulation.
- **LangChain**: To handle the chatbot pipeline, including prompt templates and output parsing.
- **HTML & CSS**: For styling the web interface, including animations and formatting.

---

## Installation

1. **Clone this repository:**

   ```bash
   git clone https://github.com/your-username/MRSL-Tech.git
   ```

2. **Install required dependencies:**

   Make sure you have Python 3.7+ installed. Then, install the necessary libraries:

   ```bash
   pip install -r requirements.txt
   ```

   Alternatively, you can manually install the dependencies:

   ```bash
   pip install streamlit ollama pandas langchain
   ```

3. **Run the application:**

   Start the Streamlit app locally:

   ```bash
   streamlit run app.py
   ```

   You should now be able to interact with MRSL.Tech on your browser.

---

## Features Walkthrough

### Sidebar

- **Logo and Header**: The sidebar features a small logo and header (`MRSL.Tech`) for branding.
- **Developer Information**: "Developed by Muhammad Rasoul" with a link to LinkedIn. You can also find options to contact via email or explore further.
  
### Main Interface

- **Chat Interface**: Users can interact with MRSL.Tech by typing their queries. MRSL answers based on the metadata of uploaded images or general questions related to the EXIF data.

- **Ollama Integration**: Ollama handles the NLP tasks, generating intelligent responses based on user input.

- **Animations**: Snowflakes fall on the page when interacting, and there's an option for particle animations that can be toggled.

---

## Usage

Once the app is running, follow these simple steps:

1. Upload an image or provide a URL of an image to extract metadata.
2. Ask MRSL questions related to the image or general queries.
3. Explore the answers that MRSL provides, powered by Ollama's natural language understanding.

---

## Contributing

Feel free to fork the repository and submit issues or pull requests. Contributions are welcome!

### Steps to contribute:

1. Fork the repository.
2. Clone it to your local machine.
3. Make necessary changes and improvements.
4. Push the changes to your fork.
5. Create a pull request describing your changes.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

- **Author**: [Muhammad Rasoul](http://www.linkedin.com/in/muhammad-rasoul-sahibzadah-b97a47218/)
- **Email**: rasoul.sahibbzadah@auaf.edu.af

--- 

### Notes

- This project was developed as a tool to interact with image metadata in a fun and useful way, using the power of Streamlit and AI.

