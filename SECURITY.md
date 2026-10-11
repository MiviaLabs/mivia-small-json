# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| latest release | yes |

## Reporting a vulnerability

Use GitHub's **private vulnerability reporting** (Security → Report a
vulnerability). Do not open public issues for security problems.

Include: a description, a reproduction, the affected artifact's sha256 if
model bytes are involved, and the impact. You will get an acknowledgment
within 7 days; fixes ship via GitHub Releases, with a GitHub Security
Advisory (and a CVE where applicable).

## Built-in mitigations, for context

- Bearer-token auth on every route except `/health`; constant-time token
  comparison; admin key separated from API keys.
- Host allow-list (`MIVIA_ALLOWED_HOSTS`) against DNS rebinding; fails
  closed.
- Hard request ceilings: 256 KiB of text and a 4 MiB body per request,
  enforced uniformly on REST and MCP (including the admin key route).
- Bounded inference gate (queue-not-reject) to keep concurrent serving
  inside the 2 GB memory envelope.
- Byte-verified zero retention: metering stores hashed keys and counters
  only; the suite dumps the entire store bytes and asserts no request
  content is present.
- All model artifacts are sha256-verified before load; a corrupt or
  missing artifact degrades honestly instead of silently.
