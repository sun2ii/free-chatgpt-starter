## Free ChatGPT Starter (No GPU Required)

A starter template for building and running your custom chatbot using OpenAI's [distilgpt2](https://huggingface.co/distilbert/distilgpt2) model.
This project provides a simple foundation for interacting with AI in Python.

### Features

- Easy-to-use Python chatbot script.
- Customizable for various use cases.

### Getting Started

Follow these steps to set up and run the chatbot locally. You do NOT need a GPU for smaller models.

### Prerequisites

Python: Ensure Python 3.7+ is installed on your system.

- Download Python from https://www.python.org/.

### Setup Instructions

**1) Clone the Repository**

```bash
git clone git@github.com:sun2ii/free-chatgpt-starter.git
cd free-chatgpt-starter
```

**2) Create a Virtual Environment (Optional but Recommended)**

```bash
python -m venv venv

source venv/bin/activate # On macOS/Linux
venv\Scripts\activate    # On Windows
```

**3) Install Dependencies**

```bash
pip install -r requirements.txt
```

**4) Run the Chatbot**

```bash
python chatbot.py
```

### How to Use

- Type a message into the terminal to chat with the bot.
- The chatbot will respond (sometimes unintelligently) using OpenAI's GPT model.

## Contributing

We welcome contributions to improve the chatbot! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- Inspired by the growing interest in conversational AI.
