"""Turns a ticket into the text the model sees.

Shared by training (notebooks/) and serving (app/) so both always build inputs the same way.
The API only sends channel, subject and text, so nothing else may be used here.
"""


def ticket_text(channel: str, subject: str | None, text: str) -> str:
    """Input for the TF-IDF baseline."""
    return f"__ch_{channel} {subject or ''}\n{text}"


def transformer_text(channel: str, subject: str | None, text: str) -> str:
    """Input for the transformer model. notebooks/02_transformer_colab.ipynb has an identical copy."""
    subject = (subject or "").strip()
    head = f"{channel} | {subject}" if subject else channel
    return f"{head}\n{text.strip()}"
