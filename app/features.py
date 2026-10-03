"""Turns a ticket into the text the model sees.

Shared by training (notebooks/) and serving (app/) so both always build inputs the same way.
The API only sends channel, subject and text, so nothing else may be used here.
"""


def ticket_text(channel: str, subject: str | None, text: str) -> str:
    return f"__ch_{channel} {subject or ''}\n{text}"
