def test_conversation_model_columns():
    from db.models import Conversation
    assert Conversation.__tablename__ == "conversations"
    cols = {c.name for c in Conversation.__table__.columns}
    assert {"id", "tenant_id", "user_id", "title", "created_at", "updated_at"} <= cols


def test_conversation_message_model_columns():
    from db.models import ConversationMessage
    assert ConversationMessage.__tablename__ == "conversation_messages"
    cols = {c.name for c in ConversationMessage.__table__.columns}
    assert {"id", "conversation_id", "tenant_id", "role", "content", "metadata", "created_at"} <= cols


def test_conversation_cascades_messages():
    from db.models import Conversation
    rel = Conversation.__mapper__.relationships["messages"]
    assert rel.cascade.delete_orphan
