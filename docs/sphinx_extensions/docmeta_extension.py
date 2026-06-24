from docutils.parsers.rst import directives
from docutils.parsers.rst import Directive


class DocMetaDirective(Directive):
    has_content = False
    option_spec = {
        "status": directives.unchanged,
        "needs_example": directives.unchanged,
        "needs_advanced_example": directives.unchanged,
        "reviewer": directives.unchanged,
        "last_reviewed": directives.unchanged,
    }

    def run(self):
        return []


def setup(app):
    app.add_directive("docmeta", DocMetaDirective)
