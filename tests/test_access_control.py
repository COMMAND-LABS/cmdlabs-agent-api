"""Tests for agent access control logic."""

from unittest.mock import MagicMock
from types import SimpleNamespace

from src.routers.agents.access import can_access_agent, load_agent_with_access_check


def _mock_db(agent=None, grant_exists=False):
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = agent

    grant_query = MagicMock()
    grant_query.join.return_value.filter.return_value.first.return_value = (
        SimpleNamespace(id=1) if grant_exists else None
    )

    def query_side_effect(model):
        from src.db.models import Agent, AgentAccessGrant
        if model is Agent:
            result = MagicMock()
            result.filter.return_value.first.return_value = agent
            return result
        if model is AgentAccessGrant or (hasattr(model, '__name__') and 'id' in str(model)):
            return grant_query
        return MagicMock()

    db.query.side_effect = query_side_effect
    return db


def test_owner_has_access():
    agent = SimpleNamespace(id=1, account_id=10)
    db = _mock_db(agent=agent)
    assert can_access_agent(db, account_id=10, agent_id=1) is True


def test_no_agent_no_access():
    db = _mock_db(agent=None)
    assert can_access_agent(db, account_id=10, agent_id=999) is False


def test_load_agent_returns_none_for_missing():
    db = _mock_db(agent=None)
    assert load_agent_with_access_check(db, account_id=10, agent_id=999) is None


def test_load_agent_returns_agent_for_owner():
    agent = SimpleNamespace(id=1, account_id=10)
    db = _mock_db(agent=agent)
    result = load_agent_with_access_check(db, account_id=10, agent_id=1)
    assert result is agent
