# PyYAML safe loader

Source: [PyYAML 6.0.3](https://github.com/yaml/pyyaml/tree/6.0.3/lib/yaml), MIT license (see LICENSE).

Only the pure-Python safe-loading pipeline is included. The reader, scanner, parser,
composer, resolver, nodes, events, tokens, and errors are unchanged. `constructor.py`
retains `BaseConstructor`, `SafeConstructor`, their errors and tag registrations;
`loader.py` retains only `SafeLoader`. The small `__init__.py` exposes `safe_load`.
Dumpers, emitters, Python-object constructors, and the native extension are omitted.

Regenerate from the repository root with `python scripts/vendor_yaml.py` after
installing `PyYAML==6.0.3`. Keep this copy private; it does not replace the global
`yaml` module. Benchmark tools and development tests use upstream PyYAML separately.
