from search import search_prompt

EXIT_COMMANDS = {"sair", "exit", "quit", "q"}


def main():
    print("Faça sua pergunta:")
    print("(digite 'sair' para encerrar)\n")

    while True:
        question = input("PERGUNTA: ").strip()

        if not question:
            print("RESPOSTA: Pergunta vazia. Tente novamente.\n")
            continue

        if question.lower() in EXIT_COMMANDS:
            print("Encerrando chat. Até mais!")
            break

        answer = search_prompt(question)
        print(f"RESPOSTA: {answer}\n")


if __name__ == "__main__":
    main()
