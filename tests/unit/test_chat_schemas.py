import pytest
from pydantic import ValidationError


def test_message_in_requires_content():
    from api.schemas.chat import MessageIn
    with pytest.raises(ValidationError):
        MessageIn(content="")
    assert MessageIn(content="hello").content == "hello"


def test_conversation_create_title_optional():
    from api.schemas.chat import ConversationCreate
    assert ConversationCreate().title is None
    assert ConversationCreate(title="Invoices June").title == "Invoices June"
