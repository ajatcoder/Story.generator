import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re
import time

# Configure page
st.set_page_config(
    page_title="Next Generation Sentence to Story Predictor",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .prediction-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .example-button {
        background: #e3f2fd;
        border: 1px solid #2196f3;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        cursor: pointer;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        border-top: 1px solid #eee;
        margin-top: 3rem;
    }

/* Add hover effects for portfolio button */
@keyframes pulse {
    0% { box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3); }
    50% { box-shadow: 0 6px 25px rgba(102, 126, 234, 0.5); }
    100% { box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3); }
}

.portfolio-btn:hover {
    animation: pulse 2s infinite;
}

/* Responsive design for footer */
@media (max-width: 768px) {
    .footer > div:first-child {
        flex-direction: column !important;
        text-align: center !important;
    }
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load and cache the GPT-2 model and tokenizer"""
    try:
        model_name = "gpt2"
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Add padding token
        tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def generate_stories(input_text, model, tokenizer, num_stories=3, max_length=150, temperature=0.8):
    """Generate multiple story continuations for the input text"""
    stories = []
    
    try:
        # Encode input text
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        
        # Generate multiple stories
        for i in range(num_stories * 2):  # Generate more to filter better ones
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                    num_return_sequences=1,
                    repetition_penalty=1.1
                )
            
            # Decode the generated text
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new part (remove input text)
            new_text = generated_text[len(input_text):].strip()
            
            # Clean up the text and create story paragraphs
            sentences = re.split(r'[.!?]+', new_text)
            if len(sentences) >= 2:  # Ensure we have at least 2 sentences for a story
                # Take first 3-4 sentences to form a story paragraph
                story_sentences = [s.strip() for s in sentences[:4] if s.strip()]
                if len(story_sentences) >= 2:
                    clean_story = '. '.join(story_sentences) + '.'
                    if len(clean_story) > 50 and clean_story not in stories:  # Ensure minimum length and uniqueness
                        stories.append(clean_story)
        
        return stories[:num_stories]
    
    except Exception as e:
        st.error(f"Error generating stories: {str(e)}")
        return []

@st.dialog("📚 Generated Stories")
def show_stories_dialog(stories, input_text):
    """Display generated stories in a popup dialog"""
    st.markdown("### Your AI-Generated Stories")
    
    for i, story in enumerate(stories, 1):
        with st.container():
            st.markdown(f"""
            <div style="
                background: #f8f9fa;
                padding: 1.5rem;
                border-radius: 10px;
                border-left: 4px solid #667eea;
                margin: 1rem 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            ">
                <h4 style="color: #667eea; margin-bottom: 1rem;">📖 Story Version {i}</h4>
                <p style="font-size: 1.1em; line-height: 1.8; text-align: justify; margin: 0;">
                    <strong style="color: #667eea;">{input_text}</strong> {story}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Export functionality in dialog
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Prepare text for download
        export_text = f"Story Prompt: {input_text}\n\n"
        export_text += "="*50 + "\n"
        export_text += "GENERATED STORIES\n"
        export_text += "="*50 + "\n\n"
        
        for i, story in enumerate(stories, 1):
            export_text += f"Story Version {i}:\n"
            export_text += f"{input_text} {story}\n\n"
            export_text += "-"*30 + "\n\n"
        
        st.download_button(
            label="📄 Download Stories",
            data=export_text,
            file_name="generated_stories.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col2:
        if st.button("✨ Generate New Stories", use_container_width=True):
            st.rerun()

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📚 Next Generation Sentence to Story Predictor</h1>
        <p style="margin-top: 1rem; font-size: 1.2em; opacity: 0.9;">Transform your ideas into captivating stories using AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading GPT-2 model... This may take a moment on first run."):
        model, tokenizer = load_model()
    
    if model is None or tokenizer is None:
        st.error("Failed to load the model. Please refresh the page and try again.")
        return
    
    st.success("✅ GPT-2 model loaded successfully!")
    
    # Sidebar for settings
    st.sidebar.header("⚙️ Story Generation Settings")
    temperature = st.sidebar.slider(
        "Creativity Level", 
        min_value=0.1, 
        max_value=2.0, 
        value=0.9, 
        step=0.1,
        help="Higher values make stories more creative and unpredictable"
    )
    
    max_length = st.sidebar.slider(
        "Story Length", 
        min_value=50, 
        max_value=200, 
        value=150, 
        step=25,
        help="Maximum number of words in each story"
    )
    
    num_stories = st.sidebar.selectbox(
        "Number of Stories", 
        options=[1, 2, 3, 4, 5], 
        index=2,
        help="How many different story variations to generate"
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("✍️ Enter Your Story Prompt")
        
        # Text input
        input_text = st.text_area(
            "Enter your story beginning or prompt:",
            placeholder="Once upon a time, in a magical forest...",
            height=120,
            help="Start your story with an interesting opening and let AI continue the narrative"
        )
        
        # Generate button
        col_btn1, col_btn2 = st.columns([1, 1])
        
        with col_btn1:
            generate_btn = st.button("📖 Generate Stories", type="primary", use_container_width=True)
        
        with col_btn2:
            clear_btn = st.button("🔄 Clear", use_container_width=True)
        
        if clear_btn:
            st.rerun()
    
    with col2:
        st.header("💡 Try These Story Starters")
        
        example_prompts = [
            "Once upon a time, in a small village",
            "The old lighthouse keeper discovered",
            "On a rainy Tuesday morning, Sarah found",
            "The mysterious package arrived exactly at midnight",
            "Deep in the Amazon rainforest, the expedition team",
            "The last person on Earth sat alone",
            "In the year 2150, robots and humans",
            "The antique music box began playing"
        ]
        
        for example in example_prompts:
            if st.button(example, key=f"example_{example}", use_container_width=True):
                st.session_state.selected_example = example
                st.rerun()
        
        # Handle example selection
        if hasattr(st.session_state, 'selected_example'):
            input_text = st.session_state.selected_example
            del st.session_state.selected_example
    
    # Generate stories and show in dialog
    if generate_btn and input_text.strip():
        with st.spinner("🤖 AI is crafting your stories... Please wait..."):
            # Add a small delay for better UX
            time.sleep(2)
            
            stories = generate_stories(
                input_text.strip(), 
                model, 
                tokenizer, 
                num_stories=num_stories,
                max_length=max_length,
                temperature=temperature
            )
        
        if stories:
            show_stories_dialog(stories, input_text.strip())
        else:
            st.error("⚠️ Could not generate stories. Please try a different prompt or adjust the settings.")
    
    elif generate_btn and not input_text.strip():
        st.warning("⚠️ Please enter a story prompt before generating stories.")
    
    # Initial state message
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 10px; margin: 2rem 0;">
        <h3>🚀 Ready to Create Amazing Stories!</h3>
        <p style="font-size: 1.1em; color: #666;">
            Enter a story beginning or prompt above, and our AI will generate multiple creative 
            story continuations using advanced natural language processing.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model information
    with st.expander("ℹ️ About the Story Generator"):
        st.markdown("""
        **How it works:**
        - **Input Processing**: Your story prompt is analyzed for context and style
        - **Creative Generation**: GPT-2 model generates multiple story continuations
        - **Smart Filtering**: Stories are refined for coherence and readability
        - **Multiple Variations**: Different creative directions for your story
        
        **Tips for better stories:**
        - Start with an engaging opening line
        - Include specific details (characters, settings, emotions)
        - Use descriptive language to set the scene
        - Try different creativity levels for varied results
        """)
    
    # Footer
    st.markdown("""
<div class="footer">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 2rem; margin-bottom: 1rem;">
        <div style="flex: 1; min-width: 300px;">
            <h4 style="color: #667eea; margin-bottom: 1rem;">🛠️ Tech Stack</h4>
            <div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">
                <span style="background: #e3f2fd; color: #1976d2; padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.85em;">Python</span>
                <span style="background: #f3e5f5; color: #7b1fa2; padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.85em;">Streamlit</span>
                <span style="background: #e8f5e8; color: #388e3c; padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.85em;">PyTorch</span>
                <span style="background: #fff3e0; color: #f57c00; padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.85em;">HuggingFace</span>
                <span style="background: #fce4ec; color: #c2185b; padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.85em;">GPT-2</span>
                <span style="background: #e1f5fe; color: #0277bd; padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.85em;">Transformers</span>
            </div>
        </div>
        <div style="text-align: center;">
            <a href="https://raghavmaheshwari-rv49.vercel.app/" target="_blank" style="text-decoration: none;">
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 1rem 2rem;
                    border-radius: 25px;
                    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
                    transition: all 0.3s ease;
                    cursor: pointer;
                    border: none;
                    font-weight: bold;
                    font-size: 1rem;
                    display: inline-flex;
                    align-items: center;
                    gap: 0.5rem;
                " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 20px rgba(102, 126, 234, 0.4)'"
                   onmouseout="this.style.transform='translateY(0px)'; this.style.boxShadow='0 4px 15px rgba(102, 126, 234, 0.3)'">
                    🚀 Visit My Portfolio
                </div>
            </a>
        </div>
    </div>
    <hr style="border: none; height: 1px; background: linear-gradient(90deg, transparent, #ddd, transparent); margin: 1.5rem 0;">
    <div style="text-align: center;">
        <p style="margin: 0.5rem 0;"><strong>Next Generation Sentence to Story Predictor</strong></p>
        <p style="margin: 0; color: #888; font-size: 0.9em;">Built with ❤️ for educational purposes | College Project</p>
    </div>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
