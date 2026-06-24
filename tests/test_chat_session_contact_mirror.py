"""H4 guard: the agent-api ChatSession model must mirror ai-api's
contact_id column.

The migration that adds chat_sessions.contact_id is owned by ai-api; this
service only has a hand-synced mirror model. If the mirror drifts,
`session.contact_id` would raise AttributeError *only* on the contact-chat
path and likely slip past other tests. This pure-introspection test makes
that drift fail loudly at test time instead.
"""

from src.db.models import ChatSession


def test_chat_session_has_contact_id_column():
    columns = ChatSession.__table__.columns
    assert "contact_id" in columns, (
        "agent-api ChatSession is missing contact_id — it has drifted "
        "from the ai-api model / migration."
    )


def test_contact_id_is_nullable_fk_to_contacts():
    col = ChatSession.__table__.columns["contact_id"]
    assert col.nullable is True, "contact_id must be nullable (unbound sessions)"

    targets = {fk.target_fullname for fk in col.foreign_keys}
    assert "contacts.id" in targets, (
        f"contact_id must be a FK to contacts.id; got {targets}"
    )
