# Security Policy

aivyx-route picks which model serves a request and decides what counts
as local or cloud. A way to make it send a request to a cloud endpoint
when the operator allowed only local models (or the reverse), or to
steer selection from untrusted input, is in scope.

This is not a bug bounty program.

## Reporting a vulnerability

Report it privately on GitHub: [Report a vulnerability](https://github.com/Aivyx-Agent/aivyx-route/security/advisories/new)
(this repo's **Security** tab). Only the maintainer sees the report
until a fix is published. Please don't open a public issue for a
vulnerability.

We aim to resolve or provide a remediation plan for a confirmed
vulnerability within 90 days of the report, or coordinate a later
disclosure date with the reporter if a fix genuinely needs longer.
Credit is offered in release notes at the reporter's preference.
