def test_theme_exports_inject():
    from frontend.theme import inject_theme, _THEME_CSS
    assert callable(inject_theme)


def test_theme_css_has_glass_and_motion_safety():
    from frontend.theme import _THEME_CSS
    assert "backdrop-filter" in _THEME_CSS
    assert "prefers-reduced-motion" in _THEME_CSS
    assert "typing-dots" in _THEME_CSS
    assert "@keyframes" in _THEME_CSS
