"""Request validation that follows the OpenAPI contract exactly.

Done by hand (not pydantic) so error details match the spec format, including `index`
for batch items.
"""

import re

CHANNELS = ("email", "chat", "call_transcript")
NON_WHITESPACE = re.compile(r"\S")


def validate_ticket(obj, require_id: bool):
    """Returns (clean_ticket, issues). issues is a list of {field, issue}."""
    if not isinstance(obj, dict):
        return None, [{"field": "body" if not require_id else "ticket", "issue": "must be a JSON object"}]

    issues = []

    if "ticket_id" in obj:
        tid = obj["ticket_id"]
        if not isinstance(tid, str):
            issues.append({"field": "ticket_id", "issue": "must be a string"})
        elif len(tid) > 64:
            issues.append({"field": "ticket_id", "issue": "must be at most 64 characters"})
    elif require_id:
        issues.append({"field": "ticket_id", "issue": "is required"})

    if "channel" not in obj:
        issues.append({"field": "channel", "issue": "is required"})
    elif obj["channel"] not in CHANNELS or not isinstance(obj["channel"], str):
        issues.append({"field": "channel", "issue": "must be one of email, chat, call_transcript"})

    if "subject" in obj:
        subject = obj["subject"]
        if not isinstance(subject, str):
            issues.append({"field": "subject", "issue": "must be a string"})
        elif len(subject) > 500:
            issues.append({"field": "subject", "issue": "must be at most 500 characters"})

    if "text" not in obj:
        issues.append({"field": "text", "issue": "is required"})
    else:
        text = obj["text"]
        if not isinstance(text, str):
            issues.append({"field": "text", "issue": "must be a string"})
        elif not NON_WHITESPACE.search(text):
            issues.append({"field": "text", "issue": "must contain at least one non-whitespace character"})
        elif len(text) > 10000:
            issues.append({"field": "text", "issue": "must be at most 10000 characters"})

    if issues:
        return None, issues

    clean = {"channel": obj["channel"], "subject": obj.get("subject", ""), "text": obj["text"]}
    if "ticket_id" in obj:
        clean["ticket_id"] = obj["ticket_id"]
    return clean, []


def validate_batch(body, max_items: int):
    """Returns (tickets, details). The whole batch fails if any item fails."""
    if not isinstance(body, dict):
        return None, [{"field": "body", "issue": "must be a JSON object"}]
    if "tickets" not in body:
        return None, [{"field": "tickets", "issue": "is required"}]
    items = body["tickets"]
    if not isinstance(items, list):
        return None, [{"field": "tickets", "issue": "must be an array"}]
    if not 1 <= len(items) <= max_items:
        return None, [{"field": "tickets", "issue": f"must contain between 1 and {max_items} items"}]

    tickets, details, seen = [], [], set()
    for i, item in enumerate(items):
        clean, issues = validate_ticket(item, require_id=True)
        details.extend({"index": i, **issue} for issue in issues)
        if clean is not None:
            if clean["ticket_id"] in seen:
                details.append({"index": i, "field": "ticket_id", "issue": "duplicate of an earlier item"})
            seen.add(clean["ticket_id"])
            tickets.append(clean)
    return (None, details) if details else (tickets, [])
