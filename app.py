import streamlit as st
import pandas as pd
import io
import zipfile
import json

from tokenizer.tokenizer import Tokenizer


# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "home"

if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None


# =========================================================
# LOAD PRETRAINED TOKENIZER
# =========================================================

tokenizer = Tokenizer()
tokenizer.load("saved")


# =========================================================
# HOME PAGE
# =========================================================

def home():

    st.title("Tokenizer Studio")
    st.caption("A Byte-Pair-Encoding tokenizer built from scratch")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Use Pretrained Model")
        st.caption(
            "Encode and decode using pretrained tokenizer"
        )

        if st.button(
            "Use Pretrained Model",
            use_container_width=True
        ):
            st.session_state.page = "pretrained"
            st.rerun()

    with col2:

        st.markdown("### Train Your Own Model")
        st.caption(
            "Train your own tokenizer on custom data"
        )

        if st.button(
            "Train Your Own Model",
            use_container_width=True
        ):
            st.session_state.page = "train"
            st.rerun()

    st.divider()


# =========================================================
# PRETRAINED TOKENIZER PAGE
# =========================================================

def pretrained_page():

    st.title("Pretrained Tokenizer")

    text = st.text_area("Enter text")

    if st.button("Encode"):

        tokens, token_info = tokenizer.encode(text)

        # -------------------------------------------------
        # TOKEN IDS
        # -------------------------------------------------

        st.subheader("Token IDs")
        st.write(tokens)

        st.divider()

        # -------------------------------------------------
        # TOKEN VISUALIZATION
        # -------------------------------------------------

        rows = []

        for token in token_info:

            token_id = token["token_id"]
            token_text = token["token_text"]

            token_type = (
                "Base Token"
                if token_id < 256
                else "Merged Token"
            )

            rows.append({
                "Token ID": token_id,
                "Token Text": repr(token_text),
                "Type": token_type
            })

        df = pd.DataFrame(rows)

        st.subheader("BPE Token Visualization")

        st.dataframe(
            df,
            use_container_width=True
        )

        st.divider()

        # -------------------------------------------------
        # DECODED TEXT
        # -------------------------------------------------

        decoded_text = tokenizer.decode(tokens)

        st.subheader("Decoded Text")
        st.write(decoded_text)

        st.divider()

    if st.button("⬅ Back"):

        st.session_state.page = "home"
        st.rerun()


# =========================================================
# TRAIN PAGE
# =========================================================

def train_page():

    st.title("Train Your Tokenizer")

    uploaded_file = st.file_uploader(
        "Upload dataset (.txt)"
    )

    num_merges = st.slider(
        "Number of merges",
        min_value=10,
        max_value=500,
        value=100
    )

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    if st.button("Train"):

        if uploaded_file is not None:

            text = uploaded_file.read().decode("utf-8")

            custom_tokenizer = Tokenizer()

            with st.spinner("Training tokenizer..."):

                custom_tokenizer.train(
                    text,
                    num_merges=num_merges
                )

            # SAVE TO SESSION
            st.session_state.tokenizer = custom_tokenizer

            st.success("Training complete!")

            # -------------------------------------------------
            # CREATE ZIP DOWNLOAD
            # -------------------------------------------------

            buffer = io.BytesIO()

            with zipfile.ZipFile(buffer, "w") as z:

                # MERGES
                z.writestr(
                    "merges.json",
                    json.dumps([
                        {
                            "pair": list(pair),
                            "id": new_id
                        }
                        for pair, new_id
                        in custom_tokenizer.merge_rules
                    ])
                )

                # VOCAB
                z.writestr(
                    "vocab.json",
                    json.dumps({
                        str(k): list(v)
                        for k, v
                        in custom_tokenizer.vocab.items()
                    })
                )

            buffer.seek(0)

            st.download_button(
                label="⬇ Download Tokenizer Model",
                data=buffer,
                file_name="tokenizer_model.zip",
                mime="application/zip"
            )

        else:
            st.warning("Please upload a dataset file.")

    st.divider()

    # =====================================================
    # USE TRAINED TOKENIZER
    # =====================================================

    if st.session_state.tokenizer is not None:

        custom_tokenizer = st.session_state.tokenizer

        st.title("Use Your Trained Tokenizer")

        st.write(
            "Encode and decode text using your trained tokenizer."
        )

        text = st.text_area(
            "Enter text",
            key="custom_model_text"
        )

        if st.button("Encode with Custom Tokenizer"):

            tokens, token_info = custom_tokenizer.encode(text)

            # -------------------------------------------------
            # TOKEN IDS
            # -------------------------------------------------

            st.subheader("Token IDs")
            st.write(tokens)

            st.divider()

            # -------------------------------------------------
            # TOKEN VISUALIZATION
            # -------------------------------------------------

            rows = []

            for token in token_info:

                token_id = token["token_id"]
                token_text = token["token_text"]

                token_type = (
                    "Base Token"
                    if token_id < 256
                    else "Merged Token"
                )

                rows.append({
                    "Token ID": token_id,
                    "Token Text": repr(token_text),
                    "Type": token_type
                })

            df = pd.DataFrame(rows)

            st.subheader("BPE Token Visualization")

            st.dataframe(
                df,
                use_container_width=True
            )

            st.divider()

            # -------------------------------------------------
            # DECODED TEXT
            # -------------------------------------------------

            decoded_text = custom_tokenizer.decode(tokens)

            st.subheader("Decoded Text")
            st.write(decoded_text)

            st.divider()

    if st.button("⬅ Back to Home"):

        st.session_state.page = "home"
        st.rerun()


# =========================================================
# ROUTER
# =========================================================

def main():

    if st.session_state.page == "home":
        home()

    elif st.session_state.page == "pretrained":
        pretrained_page()

    elif st.session_state.page == "train":
        train_page()


# =========================================================
# RUN APP
# =========================================================

if __name__ == "__main__":
    main()