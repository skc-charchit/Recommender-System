from __future__ import annotations

import gradio as gr

from .service import recommend_movies


def recommend_for_ui(movie_title: str) -> tuple[str, str]:
    if not movie_title or not movie_title.strip():
        return "Please enter a movie title.", ""

    recommendations = recommend_movies(movie_title, top_n=5)
    if not recommendations:
        return f"No match found for '{movie_title}'. Try another movie title.", ""

    markdown = "\n".join(
        f"{index + 1}. **{item['title']}** - {item['genre']} ({item['language']})\n"
        f"   {item['overview'][:180]}..."
        for index, item in enumerate(recommendations)
    )
    return "Here are the top recommendations:", markdown


with gr.Blocks(title="TMDB Movie Recommender") as demo:
    gr.Markdown("# TMDB Movie Recommender")
    with gr.Row():
        movie_input = gr.Textbox(label="Movie title")
        submit_btn = gr.Button("Recommend")
    status = gr.Textbox(label="Status")
    output = gr.Markdown()
    submit_btn.click(fn=recommend_for_ui, inputs=movie_input, outputs=[status, output])
