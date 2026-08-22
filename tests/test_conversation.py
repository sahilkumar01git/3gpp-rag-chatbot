import time

from app.memory.conversation import ConversationStore


def test_append_turn_creates_session_when_none_given():
    store = ConversationStore(max_history_turns=5, ttl_minutes=60)
    session_id = store.append_turn(None, "hi", "hello")
    assert session_id
    assert store.get_history(session_id)[0].user == "hi"


def test_history_is_trimmed_to_max_turns():
    store = ConversationStore(max_history_turns=2, ttl_minutes=60)
    sid = None
    for i in range(5):
        sid = store.append_turn(sid, f"q{i}", f"a{i}")
    history = store.get_history(sid)
    assert len(history) == 2
    assert [t.user for t in history] == ["q3", "q4"]


def test_reset_clears_session():
    store = ConversationStore(max_history_turns=5, ttl_minutes=60)
    sid = store.append_turn(None, "hi", "hello")
    store.reset(sid)
    assert store.get_history(sid) == []


def test_unknown_session_id_returns_empty_history():
    store = ConversationStore(max_history_turns=5, ttl_minutes=60)
    assert store.get_history("does-not-exist") == []


def test_expired_session_is_evicted():
    store = ConversationStore(max_history_turns=5, ttl_minutes=60)
    sid = store.append_turn(None, "hi", "hello")
    # Simulate the session having gone stale.
    store._sessions[sid].last_active = time.time() - 3600 * 3
    assert store.get_history(sid) == []
