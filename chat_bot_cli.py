import click
from query_document import qa_chain
CYAN = "\033[96m"
GREEN = "\033[92m"
BOLD = "\033[1m"
RESET = "\033[0m"

@click.command()
@click.option('--question', prompt='Your Question', default="who is the author of the novel?", help='The question you want to ask.')
def main(question):
    # Initialize the QA chain
    chain = qa_chain()
  

    history = []

    while True:
        # Retriving the answer based on the provided question
        result = chain({"question": question})

        click.echo(f"Answer: {result['answer']}")

        # Adding the question and answer to the history
        history.append((question, result['answer']))

        # Printing the history
        formatted_history =  f"{BOLD}Question History:{RESET}"

        click.echo(f"{formatted_history}")
        for q, a in history:
            formatted_question = f"{BOLD}{CYAN}{q}{RESET}"
            formatted_answer = f"{BOLD}{GREEN}{a}{RESET}"
            click.echo(f"{formatted_question} -> {formatted_answer}")


        # Prompting the user for a new question
        question = click.prompt('\nYour Question', default="Who is the author of the novel?")

if __name__ == '__main__':
    main()