"""
Utility functions for model operations and text processing
"""

import re
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class NextSentencePredictor:
    """
    A class to handle next sentence prediction using GPT-2
    """
    
    def __init__(self, model_name="gpt2"):
        """
        Initialize the predictor with a specified model
        
        Args:
            model_name (str): Name of the GPT-2 model to use
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load the GPT-2 model and tokenizer"""
        try:
            print(f"Loading {self.model_name} model...")
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
            
            # Add padding token
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def clean_generated_text(self, text, input_text):
        """
        Clean and process generated text
        
        Args:
            text (str): Generated text from model
            input_text (str): Original input text
            
        Returns:
            str: Cleaned prediction text
        """
        # Remove input text from generated text
        if text.startswith(input_text):
            text = text[len(input_text):].strip()
        
        # Split by sentence endings and take first complete sentence
        sentences = re.split(r'[.!?]+', text)
        
        if sentences and sentences[0].strip():
            # Clean up the sentence
            clean_sentence = sentences[0].strip()
            
            # Remove any incomplete words at the end
            words = clean_sentence.split()
            if words:
                # Check if last word seems incomplete (very short or has special chars)
                if len(words[-1]) < 2 or not words[-1].isalpha():
                    words = words[:-1]
                
                return ' '.join(words)
        
        return text.strip()
    
    def generate_predictions(self, input_text, num_predictions=3, max_length=50, temperature=0.8, top_k=50, top_p=0.95):
        """
        Generate multiple predictions for input text
        
        Args:
            input_text (str): Input text to complete
            num_predictions (int): Number of predictions to generate
            max_length (int): Maximum length of generated text
            temperature (float): Sampling temperature
            top_k (int): Top-k sampling parameter
            top_p (float): Top-p sampling parameter
            
        Returns:
            list: List of generated predictions
        """
        if not self.model or not self.tokenizer:
            return []
        
        predictions = []
        
        try:
            # Encode input text
            input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
            
            for i in range(num_predictions * 2):  # Generate more to filter better ones
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids,
                        max_length=input_ids.shape[1] + max_length,
                        temperature=temperature,
                        do_sample=True,
                        top_k=top_k,
                        top_p=top_p,
                        pad_token_id=self.tokenizer.eos_token_id,
                        num_return_sequences=1,
                        repetition_penalty=1.1
                    )
                
                # Decode generated text
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Clean the prediction
                clean_prediction = self.clean_generated_text(generated_text, input_text)
                
                # Filter out very short or repetitive predictions
                if (clean_prediction and 
                    len(clean_prediction.split()) >= 3 and 
                    clean_prediction not in predictions and
                    len(clean_prediction) >= 10):
                    predictions.append(clean_prediction)
                
                # Stop if we have enough good predictions
                if len(predictions) >= num_predictions:
                    break
            
            return predictions[:num_predictions]
        
        except Exception as e:
            print(f"Error generating predictions: {str(e)}")
            return []

def validate_input(text):
    """
    Validate input text
    
    Args:
        text (str): Input text to validate
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not text or not text.strip():
        return False, "Input text cannot be empty"
    
    if len(text.strip()) < 3:
        return False, "Input text is too short (minimum 3 characters)"
    
    if len(text.strip()) > 200:
        return False, "Input text is too long (maximum 200 characters)"
    
    return True, ""

def get_example_sentences():
    """
    Get a list of example sentences for testing
    
    Returns:
        list: List of example sentences
    """
    return [
        "I went to the park to",
        "The weather today is",
        "After finishing my homework I",
        "My favorite book is about",
        "When I wake up in the morning I usually",
        "Last weekend I decided to",
        "If I could travel anywhere I would",
        "The most interesting thing about my city is",
        "Yesterday I learned that",
        "My biggest dream is to",
        "The best advice I ever received was",
        "When I feel stressed I like to"
    ]
