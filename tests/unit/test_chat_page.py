def test_chat_page_exports_render():
    from frontend.pages.chat import render
    assert callable(render)


def test_typing_indicator_markup():
    from frontend.pages.chat import _TYPING_HTML
    assert "typing-dots" in _TYPING_HTML
