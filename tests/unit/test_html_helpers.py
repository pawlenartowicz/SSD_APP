"""HTML escaping for user text rendered in stage-3 tabs.

Guards against XSS-style issues if a user loads a CSV with HTML in the
text column.
"""

from ssdiff_gui.views.stage3.html_helpers import escape_html


class TestEscapeHtml:
    def test_escapes_angle_brackets(self):
        assert escape_html("<script>") == "&lt;script&gt;"

    def test_escapes_ampersand(self):
        assert escape_html("A & B") == "A &amp; B"

    def test_ampersand_escaped_first(self):
        """& must be escaped BEFORE <, otherwise '<' becomes '&amp;lt;'."""
        result = escape_html("<&>")
        assert result == "&lt;&amp;&gt;"

    def test_preserves_newlines_as_br(self):
        assert escape_html("line1\nline2") == "line1<br/>line2"

    def test_combined(self):
        assert escape_html("<b>A & B</b>\nnext") == (
            "&lt;b&gt;A &amp; B&lt;/b&gt;<br/>next"
        )

    def test_empty(self):
        assert escape_html("") == ""

    def test_plain_text_unchanged(self):
        assert escape_html("hello world") == "hello world"
