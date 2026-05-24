from tokenizer.tokenizer import Tokenizer
import os

# =========================================================
# TRAIN USER TOKENIZER
# =========================================================

def user_train(corpus_path, save_path="saved_user", num_merges=500):

    tokenizer = Tokenizer()

    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    print("Training tokenizer...")

    tokenizer.train(text, num_merges=num_merges)

    tokenizer.save(save_path)

    print("Training complete.")


# =========================================================
# CLI APP
# =========================================================

def run_cli():

    print("1. Use pretrained tokenizer")
    print("2. Train tokenizer")
    print("3. Load custom tokenizer")
    
    print("Press 'q' anytime to exit.\n")

    task = input("Enter task (1 , 2 or 3): ")

    # =====================================================
    # PRETRAINED MODE
    # =====================================================

    if task == "1":

        tokenizer = Tokenizer()
        tokenizer.load("saved")

        while True:

            text = input("\nEnter text: ")

            if text.lower() == "q":
                break

            tokens, token_info = tokenizer.encode(text)

            decoded_text = tokenizer.decode(tokens)

            print("\nToken IDs:")
            print(tokens)

            print("\nDecoded Text:")
            print(decoded_text)

            print("\nToken Breakdown:")

            for token in token_info:

                token_id = token["token_id"]
                token_text = token["token_text"]

                token_type = (
                    "Base Token"
                    if token_id < 256
                    else "Merged Token"
                )

                print(
                    f"{token_id:<6} | "
                    f"{repr(token_text):<12} | "
                    f"{token_type}"
                )

            print("\n" + "-" * 40)

    # =====================================================
    # CUSTOM TRAINING MODE
    # =====================================================

    elif task == "2":

        corpus_path = input(
            "\nEnter corpus file path: "
        ).strip()

        if not os.path.exists(corpus_path):
            print(f"File path '{corpus_path}' does not exist")
            return

        save_path = input(
            "Enter save folder name: "
        ).strip()

        os.makedirs(save_path, exist_ok=True)

        num_merges = input(
            "Enter number of merges (default 500): "
        ).strip()

        if num_merges == "":
            num_merges = 500
        else:
            num_merges = int(num_merges)

        # -----------------------------
        # TRAIN TOKENIZER
        # -----------------------------
        user_train(
            corpus_path=corpus_path,
            save_path=save_path,
            num_merges=num_merges
        )

        # -----------------------------
        # ASK TO USE MODEL
        # -----------------------------
        outputs = input(
            "\nWould you like to use the trained tokenizer? (y/n): "
        ).lower()

        if outputs == "y":

            tokenizer = Tokenizer()
            tokenizer.load(save_path)

            print("\nCustom tokenizer loaded.\n")

            while True:

                text = input("Enter text: ")

                if text.lower() == "q":
                    break

                tokens, token_info = tokenizer.encode(text)

                decoded_text = tokenizer.decode(tokens)

                print("\nToken IDs:")
                print(tokens)

                print("\nDecoded Text:")
                print(decoded_text)

                print("\nToken Breakdown:")
                print(token_info)

                print("\n" + "-" * 40)


    # =====================================================
    #   LOAD CUSTOM TOKENIZER
    # =====================================================

    elif task == "3":

        model_path = input(
            "\nEnter tokenizer folder path: "
        ).strip()

        if not os.path.exists(model_path):
            print("Model path does not exist")
            return

        tokenizer = Tokenizer()
        tokenizer.load(model_path)

        print("\nCustom tokenizer loaded.\n")

        while True:

            text = input("Enter text: ")

            if text.lower() == "q":
                break

            tokens, token_info = tokenizer.encode(text)

            decoded_text = tokenizer.decode(tokens)

            print("\nToken IDs:")
            print(tokens)

            print("\nDecoded Text:")
            print(decoded_text)

            print("\nToken Breakdown:")
            print(token_info)

            print("\n" + "-" * 40)
    
    
    # =====================================================
    # INVALID OPTION
    # =====================================================

    elif task == 'q':
        return


    else:
        print("\n Invalid option")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    run_cli()



