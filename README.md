# Next Sentence Prediction using Generative AI

A web application that uses GPT-2 and HuggingFace Transformers to predict the next sentence based on user input. Built with Python and Streamlit for an interactive user experience.

## 🚀 Features

- **Real-time Text Generation**: Uses GPT-2 model for generating contextually relevant sentence completions
- **Interactive Web Interface**: Clean, modern UI built with Streamlit
- **Customizable Parameters**: Adjust temperature, max length, and number of predictions
- **Example Sentences**: Pre-built examples to test the model quickly
- **Export Functionality**: Download generated predictions as text files
- **Responsive Design**: Works on desktop and mobile devices

## 🛠️ Technologies Used

- **Python 3.8+**
- **Streamlit** - Web application framework
- **HuggingFace Transformers** - Pre-trained language models
- **PyTorch** - Deep learning framework
- **GPT-2** - Generative language model

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- At least 4GB RAM (for model loading)
- Internet connection (for initial model download)

## 🔧 Installation

1. **Clone the repository**
   \`\`\`bash
   git clone https://github.com/yourusername/next-sentence-prediction.git
   cd next-sentence-prediction
   \`\`\`

2. **Create a virtual environment** (recommended)
   \`\`\`bash
   python -m venv venv
   
   # On Windows
   venv\\Scripts\\activate
   
   # On macOS/Linux
   source venv/bin/activate
   \`\`\`

3. **Install required packages**
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

## 🚀 Usage

1. **Run the Streamlit application**
   \`\`\`bash
   streamlit run app.py
   \`\`\`

2. **Open your web browser** and navigate to `http://localhost:8501`

3. **Enter your text** in the input area or click on example sentences

4. **Adjust settings** in the sidebar (optional):
   - Temperature: Controls creativity (0.1 = focused, 2.0 = creative)
   - Max Length: Maximum tokens to generate
   - Number of Predictions: How many predictions to show

5. **Click "Generate Predictions"** to see AI-generated completions

6. **Export results** using the download button if needed

## 🧪 Testing

Run the test script to verify everything is working:

\`\`\`bash
python test_model.py
\`\`\`

This will test:
- Model loading functionality
- Input validation
- Prediction generation
- Performance timing

## 📁 Project Structure

\`\`\`
next-sentence-prediction/
│
├── app.py                 # Main Streamlit application
├── model_utils.py         # Model utilities and helper functions
├── test_model.py          # Testing script
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── .gitignore           # Git ignore file
\`\`\`

## 🎯 How It Works

1. **Model Loading**: The application loads a pre-trained GPT-2 model from HuggingFace
2. **Text Processing**: User input is tokenized and prepared for the model
3. **Generation**: The model generates multiple possible continuations using sampling techniques
4. **Post-processing**: Generated text is cleaned and formatted for display
5. **Display**: Results are shown in an interactive web interface

## ⚙️ Configuration Options

### Generation Parameters

- **Temperature** (0.1-2.0): Controls randomness in generation
  - Lower values (0.1-0.7): More focused and deterministic
  - Higher values (0.8-2.0): More creative and diverse

- **Max Length** (20-100): Maximum number of tokens to generate

- **Top-k Sampling** (50): Limits vocabulary to top k most likely tokens

- **Top-p Sampling** (0.95): Uses nucleus sampling for better quality

## 🔍 Example Usage

**Input**: "I went to the park to"

**Generated Predictions**:
1. I went to the park to meet my friends and play basketball for two hours.
2. I went to the park to walk my dog and enjoy the beautiful weather.
3. I went to the park to read a book under my favorite oak tree.

## 🚨 Troubleshooting

### Common Issues

1. **Model Loading Error**
   - Ensure you have stable internet connection
   - Check if you have enough RAM (4GB minimum)
   - Try restarting the application

2. **Slow Performance**
   - Reduce max_length parameter
   - Use fewer predictions
   - Consider using a smaller model variant

3. **Poor Quality Predictions**
   - Adjust temperature settings
   - Try different input phrasing
   - Ensure input text is clear and contextual

### Error Messages

- **"Failed to load model"**: Check internet connection and system resources
- **"Could not generate predictions"**: Try different input text or adjust parameters
- **"Input text too short/long"**: Follow the input length guidelines (3-200 characters)

## 🔮 Future Enhancements

- [ ] Support for multiple language models (GPT-3, BERT, etc.)
- [ ] Fine-tuning capabilities for domain-specific text
- [ ] Batch processing for multiple inputs
- [ ] API endpoint for programmatic access
- [ ] User feedback system for model improvement
- [ ] Advanced text preprocessing options
- [ ] Model comparison features

## 📊 Performance Metrics

- **Model Size**: ~500MB (GPT-2 base)
- **Loading Time**: 10-30 seconds (first run)
- **Generation Time**: 1-3 seconds per prediction
- **Memory Usage**: 2-4GB RAM

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for the GPT-2 model
- **HuggingFace** for the Transformers library
- **Streamlit** for the web framework
- **PyTorch** team for the deep learning framework

## 📞 Contact

- **Author**: [Your Name]
- **Email**: [your.email@example.com]
- **GitHub**: [https://github.com/yourusername](https://github.com/yourusername)
- **Project Link**: [https://github.com/yourusername/next-sentence-prediction](https://github.com/yourusername/next-sentence-prediction)

---

**Built with ❤️ for educational purposes**
