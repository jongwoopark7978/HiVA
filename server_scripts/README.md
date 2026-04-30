# Server-specific launch scripts

Root-level training/evaluation helper scripts are grouped here by the server or
runtime context they target.

- `bigbrain/`: scripts configured for `/nfs/bigbrain/...` datasets.
- `bigflow/`: scripts configured for `/nfs/bigflow...` datasets.
- `bigcornea/`: scripts configured for `/nfs/bigcornea/...` datasets.
- `generic/`: scripts that are not tied to one server-specific dataset path.

Each script computes the repository root from its own location and changes into
that root before running, so outputs still go under the repository-level
`outputs/` directory.
