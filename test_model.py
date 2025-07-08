"""
Test script for the Next Sentence Prediction model
"""

from model_utils import NextSentencePredictor, validate_input, get_example_sentences
import time

def test_model_loading():
    """Test if the model loads correctly"""
    print("Testing model loading...")
    predictor = NextSentencePredictor()
    
    if predictor.model and predictor.tokenizer:
        print("✅ Model loaded successfully!")
        return True
    else:
        print("❌ Failed to load model")
        return False

def test_predictions():
    """Test prediction generation with example sentences"""
    print("\nTesting prediction generation...")
    
    predictor = NextSentencePredictor()
    
    if not predictor.model:
        print("❌ Model not loaded, skipping prediction tests")
        return False
    
    test_sentences = [
        "I went to the store to",
        "The weather today is",
        "My favorite hobby is"
    ]
    
    for sentence in test_sentences:
        print(f"\nInput: '{sentence}'")
        
        start_time = time.time()
        predictions = predictor.generate_predictions(sentence, num_predictions=3)
        end_time = time.time()
        
        print(f"Generation time: {end_time - start_time:.2f} seconds")
        
        if predictions:
            for i, pred in enumerate(predictions, 1):
                print(f"  {i}. {sentence} {pred}")
        else:
            print("  ❌ No predictions generated")
    
    return True

def test_input_validation():
    """Test input validation function"""
    print("\nTesting input validation...")
    
    test_cases = [
        ("", False),  # Empty string
        ("Hi", False),  # Too short
        ("This is a good sentence", True),  # Valid
        ("A" * 250, False),  # Too long
    ]
    
    for text, expected in test_cases:
        is_valid, error = validate_input(text)
        status = "✅" if is_valid == expected else "❌"
        print(f"{status} '{text[:20]}...' - Valid: {is_valid}")

def main():
    """Run all tests"""
    print("🧪 Running Next Sentence Prediction Tests\n")
    
    # Test model loading
    if not test_model_loading():
        return
    
    # Test input validation
    test_input_validation()
    
    # Test predictions
    test_predictions()
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    main()
