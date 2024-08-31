import transformers 
import pipeline
import gradio as gr

# Load the text generation model using Bloom
model_name = "bigscience/bloom-560m"  # Smaller version of Bloom for faster execution
text_generator = pipeline("text-generation", model=model_name)

# Define a function for generating text for specific SDG 4 questions
def sdg4_responses(question):
    responses = {
        "How can technology improve access to quality education?": text_generator(
            "Technology can improve access to quality education by", max_length=150)[0]["generated_text"],
        "The importance of early childhood education for lifelong learning.": text_generator(
            "Early childhood education is important for lifelong learning because", max_length=150)[0]["generated_text"]
    }
    return responses.get(question, "Please ask one of the predefined questions about SDG 4.")

# Create a Gradio interface for predefined questions
questions = ["How can technology improve access to quality education?", 
             "The importance of early childhood education for lifelong learning."]

demo = gr.Interface(
    fn=sdg4_responses,
    inputs=gr.Dropdown(questions, label="Select a question related to SDG 4"),
    outputs=gr.Textbox(label="Generated response"),
    title="SDG 4 Text Generation",
    description="Generate responses for specific questions related to SDG 4 (Quality Education) using the Bloom model.",
)

# Launch the Gradio app
demo.launch()
