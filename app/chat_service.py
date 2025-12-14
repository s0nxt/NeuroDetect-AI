import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    try:
        # List all available models to debug
        print("Scanning for available Gemini models...")
        all_models = list(genai.list_models())
        supported_models = [m for m in all_models if 'generateContent' in m.supported_generation_methods]
        
        print(f"Found {len(supported_models)} supported models:")
        for m in supported_models:
            print(f" - {m.name}")

        # Strategy: Prefer Flash -> Pro -> Any
        # We look for specific keywords in the model name
        selected_model = None
        
        # 1. Try to find a Flash model (usually best for free tier)
        for m in supported_models:
            if 'flash' in m.name.lower() and 'exp' not in m.name.lower():
                selected_model = m.name
                break
        
        # 2. If no stable Flash, try any Flash
        if not selected_model:
            for m in supported_models:
                if 'flash' in m.name.lower():
                    selected_model = m.name
                    break

        # 3. If no Flash, try standard Pro (avoid experimental if possible)
        if not selected_model:
            for m in supported_models:
                if 'pro' in m.name.lower() and 'exp' not in m.name.lower():
                    selected_model = m.name
                    break

        # 4. Last resort: Pick the first available one
        if not selected_model and supported_models:
            selected_model = supported_models[0].name

        if selected_model:
            print(f"Selected Gemini model: {selected_model}")
            model = genai.GenerativeModel(selected_model)
        else:
            model = None
            print("CRITICAL: No supported Gemini models found for this API key.")
            
    except Exception as e:
        print(f"Error configuring Gemini model: {e}")
        model = None
else:
    model = None
    print("Warning: GOOGLE_API_KEY not found. Chat feature will be disabled.")

def get_ai_response(user_question, context):
    """
    Generates a response from the AI based on the user's question and medical context.
    """
    if not model:
        return "AI Chat is not configured. Please set the GOOGLE_API_KEY environment variable."

    try:
        # Construct the prompt
        prompt = f"""
        You are Dr. Neuro, an advanced AI medical assistant specializing in medical image analysis (Brain Tumors, Diabetic Retinopathy, Lung Cancer).
        
        Current Patient Context:
        - Diagnosis: {context.get('diagnosis', 'Unknown')}
        - Confidence: {context.get('confidence', 'N/A')}
        - Tumor Type Description: {context.get('description', 'N/A')}
        
        User Question: {user_question}
        
        Instructions:
        1. Answer the user's question clearly and compassionately.
        2. Use the provided context to give specific answers.
        3. You may answer general questions, but prioritize the medical context if relevant.
        4. Always include a disclaimer that you are an AI and this is not professional medical advice.
        5. Keep the response concise (under 150 words) unless asked for details.
        
        Answer:
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating AI response: {e}")
        return f"Error: {str(e)}. Please check your API key and connection."
