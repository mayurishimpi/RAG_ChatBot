from query_document import qa_chain
from rich.console import Console
from rich.prompt import Prompt
if __name__ == "__main__":
    # Command-line application
    c = Console()
    
    chain = qa_chain()
    
    c.print("[bold]Chat about the novel 'The Nightingale'!")
    c.print("[blue]**********************************************")

    while True:
        default_question = "What the name of the author of the novel 'The Nightingale'?"
        question = Prompt.ask("Your Question: ", default=default_question)
      
        result = chain({"question": question})
        c.print("[green]Answer: [/green]" + result['answer'])


        c.print("[grey]**********************************************")
