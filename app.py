"""Gradio app for XSum summarization with a fine-tuned BART model."""

from __future__ import annotations

import time

import gradio as gr

from src.inference import summarize_with_warning # pyrefly: ignore


APP_TITLE = "XSum BART Summarizer"
MODEL_LABEL = "facebook/bart-base fine-tuned on XSum"
HF_MODEL_URL = "https://huggingface.co/your-username/xsum-bart-summarizer"

EXAMPLES = [
    [
        "Officers searched properties in the Waterfront Park and Colonsay View "
        "areas of Edinburgh on Wednesday. Detectives said three firearms, "
        "ammunition and a five-figure sum of money were recovered. A 26-year-old "
        "man who was arrested and charged appeared at Edinburgh Sheriff Court on "
        "Thursday."
    ],
    [
        "Emily Thornberry said Labour would not frustrate Brexit even if it "
        "failed to amend the bill. Ten shadow ministers were among 47 Labour MPs "
        "who voted against triggering Article 50. The shadow foreign secretary "
        "said her party would keep pushing for parliamentary scrutiny while "
        "respecting the referendum result."
    ],
    [
        "The death toll doubled over two days as officials found more than 100 "
        "bodies once waters began receding. Officials said floods in the western "
        "Indian state of Gujarat had destroyed homes, damaged roads and forced "
        "thousands of people to move to safer areas."
    ],
    [
        "The move is in response to an £8m cut in the subsidy received from the "
        "Department of Employment and Learning. Queen's University Belfast said "
        "it would cut jobs and student places, while unions warned the decision "
        "would damage teaching and research."
    ],
    [
        "The 33-year-old has featured only twice for the Foxes this term, having "
        "signed a new one-year deal with the Premier League newcomers in the "
        "summer. Former Blackpool forward Gary Taylor-Fletcher scored three goals "
        "in 23 games for his parent club last season."
    ],
]


def build_app() -> gr.Blocks:
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(
            f"# {APP_TITLE}\n"
            "Paste a news article and generate a concise, XSum-style summary."
        )

        with gr.Row():
            with gr.Column(scale=3):
                article = gr.Textbox(
                    label="Article",
                    lines=16,
                    max_lines=22,
                    placeholder="Paste a BBC-style news article here...",
                )
                warning = gr.Markdown(visible=True)
            with gr.Column(scale=2):
                summary = gr.Textbox(
                    label="Summary",
                    lines=8,
                    interactive=False,
                    buttons=["copy"],
                )

        with gr.Row():
            num_beams = gr.Slider(
                minimum=1,
                maximum=8,
                value=4,
                step=1,
                label="Beams",
            )
            max_length = gr.Slider(
                minimum=24,
                maximum=128,
                value=64,
                step=4,
                label="Max summary tokens",
            )
            length_penalty = gr.Slider(
                minimum=0.5,
                maximum=3.0,
                value=2.0,
                step=0.1,
                label="Length penalty",
            )

        gr.Markdown(
            "Length penalty: values greater than 1.0 encourage longer summaries; "
            "values below 1.0 encourage shorter summaries."
        )

        summarize_button = gr.Button("Summarize", variant="primary")
        summarize_button.click(
            fn=summarize_with_warning,
            inputs=[article, num_beams, length_penalty, max_length],
            outputs=[summary, warning],
            show_progress=True,
        )

        gr.Examples(
            examples=EXAMPLES,
            inputs=[article],
            label="Examples",
        )

        gr.Markdown(
            f"{MODEL_LABEL} · Hugging Face Hub: [{HF_MODEL_URL}]({HF_MODEL_URL})"
        )

    return demo


demo = build_app()


if __name__ == "__main__":
    print("Starting Gradio app on http://127.0.0.1:7860", flush=True)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        prevent_thread_lock=True,
    )
    print("Gradio app is running. Press Ctrl+C to stop.", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        demo.close()
