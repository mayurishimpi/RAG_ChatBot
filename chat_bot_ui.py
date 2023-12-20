import os
from typing import Optional, Tuple
from threading import Lock

import gradio as gr

from query_document import qa_chain
class ChatWrapper:
   
    def __init__(self):
        self.lock = Lock()

    def __call__(
        self, inp: str, history: Optional[Tuple[str, str]]
    ):
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            history = history or []
            
            chain = qa_chain()

            output = chain({"question": inp})["answer"]
            history.append((inp, output))
        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return history, history


# Graphical user interface using Gradio: creates easy-to-use, customizable UI components for ML models
chat = ChatWrapper()
theme = gr.themes.Default(primary_hue="blue", secondary_hue="orange").set(
    body_background_fill = "*primary_100",
    button_primary_background_fill="*primary_3",
    button_primary_background_fill_hover="orange",

)

with gr.Blocks(theme=theme) as demo:

    # Creating a row with a Markdown title
    with gr.Row():
        gr.Markdown(
            "<h3><center>Learn more about Kristin Hannah's Best Selling novels!</center></h3>")


    # Initializing a Gradio Chatbot component
    chatbot = gr.Chatbot( )

    # Creating another row with a Textbox for user input and a Submit button
    with gr.Row():
        message = gr.Textbox(
            label="What's your question?",
            placeholder="Type a message.....",
            lines=1,
            scale = 4
        )
        submit = gr.Button(value="Send",size = "sm",  scale = 1)

    # Providing examples for user convenience
    gr.Examples(
        examples=[
            "Who is the author of these novels?",
            "Compare the themes of the two novels",
            "Why should I read this novel?",

        ],
        inputs=message,
    )


    state = gr.State()
    agent_state = gr.State()

    submit.click(chat, inputs=[ message, state], outputs=[chatbot, state])
    message.submit(chat, inputs=[message, state], outputs=[chatbot, state])


demo.launch(debug=True)
