from . import markov
from . import weather #i changed
from . import traffic
CONFIG_FORMAT_TO_MODULE_MAP = {
    "markov": markov,
    "weather": weather, #i changed
    "traffic": traffic
}

def parse_args_with_format(format, base_parser, args, namespace):
    return CONFIG_FORMAT_TO_MODULE_MAP[format].parse_args(base_parser, args, namespace)

def registered_formats():
    return CONFIG_FORMAT_TO_MODULE_MAP.keys()
