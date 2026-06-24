"""H4 guard: agent-api must mirror ai-api's contacts / contact_events.

The tables + migrations are owned by ai-api; these are hand-synced mirror
models. Drift breaks the contact-chat tools only on that path, so assert the
columns the tools rely on exist. Pure introspection — no DB.
"""

from src.db.models import Contact, ContactEvent


def test_contact_mirror_has_required_columns():
    cols = Contact.__table__.columns
    for name in ("id", "account_id", "first_name", "last_name", "email"):
        assert name in cols, f"Contact mirror missing {name}"


def test_contact_event_mirror_has_required_columns():
    cols = ContactEvent.__table__.columns
    for name in ("id", "contact_id", "account_id", "event_type", "title", "description"):
        assert name in cols, f"ContactEvent mirror missing {name}"


def test_contact_event_scope_columns_are_fks():
    cols = ContactEvent.__table__.columns
    contact_fk = {fk.target_fullname for fk in cols["contact_id"].foreign_keys}
    account_fk = {fk.target_fullname for fk in cols["account_id"].foreign_keys}
    assert "contacts.id" in contact_fk
    assert "accounts.id" in account_fk
