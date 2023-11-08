from docutils import nodes
from docutils.parsers.rst.roles import register_local_role

def paramref_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    # Simply replace :paramref: with :param:
    new_role = nodes.literal(text=text[1:])
    return [new_role], []

def setup(app):
    register_local_role('paramref', paramref_role)

