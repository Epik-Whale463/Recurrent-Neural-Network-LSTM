# Secret Garden Text Generator

Welcome to the **Secret Garden Text Generator**, a project that demonstrates the power of machine learning in generating human-like text using Recurrent Neural Networks (RNNs) with Long Short-Term Memory (LSTM) cells.

This project is inspired by *"The Secret Garden"* and allows users to generate new text based on patterns learned from the original book. 

---

## Features

- **Seed-Based Text Generation**: Provide a starting phrase, and the model generates a continuation based on it.
- **Temperature Control**: Adjust the randomness of the generated text for more creative or predictable results.
- **Word Length Customization**: Choose how many words to generate.
- **Educational Interface**: Learn the basics of RNNs and LSTMs through intuitive explanations on the web page.

---

## How It Works

1. **Training the Model**: 
   The model is trained on the text of *"The Secret Garden"*, learning patterns and relationships between words.

2. **Generating Text**: 
   - **Input**: Provide a seed phrase and parameters like word count and temperature.
   - **Prediction**: The model predicts the next word and continues generating text iteratively.
   - **Output**: A coherent or creative block of text based on the input seed.

3. **Temperature**: 
   - Low values (e.g., 0.3) result in predictable and coherent text.
   - High values (e.g., 1.5) create more surprising and varied outputs.

---

## Getting Started

### Prerequisites

To run this project, ensure you have the following installed:
- Python 3.8+
- Flask
- TensorFlow/Keras

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/secret-garden-text-generator.git
   cd secret-garden-text-generator
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

4. Open your browser and visit:
   ```
   http://localhost:5000
   ```

---

## Usage

1. Enter a **seed text** to start generating.
2. Set the number of words to generate.
3. Adjust the **temperature** for creativity or coherence.
4. Click "Generate Text" and view the output!

---

## Project Structure

```
Secret-Garden-Text-Generator/
├── templates/                # HTML templates for the web interface
├── static/                   # CSS and JS files
├── app.py                    # Flask application
├── model/                    # Pre-trained model and training scripts
├── README.md                 # Documentation
├── requirements.txt          # Python dependencies
└── LICENSE                   # Project license
```

---

## Screenshots

### Input Form:
![Input Form](screenshots/input-form.png)

### Generated Text Output:
![Generated Text](screenshots/generated-text.png)

---

## Technologies Used

- **Framework**: Flask
- **Machine Learning**: TensorFlow, Keras
- **Frontend**: HTML, CSS (minimal styling)

---

## Contributing

Contributions are welcome! If you have suggestions or want to improve the project:

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/your-feature
   ```
5. Open a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Inspired by *"The Secret Garden"* by Frances Hodgson Burnett.
- Powered by open-source libraries and frameworks.
- Special thanks to the machine learning community for their invaluable resources.

---

## Contact

For questions or suggestions, feel free to reach out:
- **Email**: your-email@example.com
- **GitHub**: [Your GitHub Profile](https://github.com/your-username)
