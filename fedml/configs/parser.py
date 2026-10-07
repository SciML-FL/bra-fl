"""Module to parse the configurations provided by user."""

import yaml
from yaml.constructor import SafeConstructor

class LiteralList(list):
    """Marker subclass — stored as-is, never expanded."""

def construct_literal_list(self, node):
    return LiteralList(self.construct_sequence(node))

SafeConstructor.add_constructor(
    u"tag:yaml.org,2002:python/list",   # or use "!list" if you prefer shorter syntax
    construct_literal_list
)

def construct_yaml_tuple(self, node):
    seq = self.construct_sequence(node)
    # only make "leaf sequences" into tuples, you can add dict 
    # and other types as necessary
    if seq and isinstance(seq[0], (list, tuple)):
        return seq
    return tuple(seq)

SafeConstructor.add_constructor(
    u"tag:yaml.org,2002:python/tuple",
    construct_yaml_tuple
)

# -- Representers: serialize back as plain YAML sequences ---------------------
yaml.add_representer(
    LiteralList,
    lambda dumper, data: dumper.represent_sequence('tag:yaml.org,2002:seq', data)
)
yaml.add_representer(
    tuple,
    lambda dumper, data: dumper.represent_sequence('tag:yaml.org,2002:seq', data)
)

def parse_configs(config_path) -> dict:
    """Load and return user configurations."""
    with open(config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(exc)