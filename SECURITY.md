# Security policy

## What this project is

phonometry is a scientific Python library. It computes acoustic, vibration and
psychoacoustic quantities from arrays and files you hand it, inside your own
process. It runs no service, opens no socket, holds no credentials, and never
executes code derived from what you pass it: it calls its own code and its
declared dependencies, nothing else. That shapes what a vulnerability can be
here, so this policy is written for this library rather than copied from a
generic template.

## Supported versions

Fixes ship forward. There are no backports: a security fix is released as a new
version from `main`, and the remedy for any older version is to upgrade.

| Version | Supported |
| :--- | :--- |
| Latest 3.x release | Yes |
| Earlier 3.x releases | Upgrade to the latest 3.x |
| 2.x and earlier, including `PyOctaveBand` | No |

The `PyOctaveBand` package on PyPI is a stub that only redirects to
`phonometry` and receives no fixes.

Supported interpreters and platforms are whatever the test matrix in
[`.github/workflows/python-app.yml`](.github/workflows/python-app.yml)
exercises, at the time of writing Python 3.13 and 3.14 on Ubuntu, macOS and
Windows. That workflow is the authority, not this paragraph. A report that
reproduces only outside the matrix is handled as an ordinary bug.

## Reporting a vulnerability

Report privately through GitHub, with **Security > Report a vulnerability**, or
directly at
[github.com/jmrplens/phonometry/security/advisories/new](https://github.com/jmrplens/phonometry/security/advisories/new).
That opens a private advisory visible only to you and the maintainers, and keeps
the report, the fix and any CVE in one place.

If private reporting is not available to you, write to `mail@jmrp.io`, the
maintainer address already published in [`CITATION.cff`](CITATION.cff) and
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md), with `phonometry security` in the
subject. Please do not use a public issue, a discussion or a pull request for a
suspected vulnerability.

A useful report states the version and interpreter, a minimal reproduction, the
input that triggers the behaviour, and what an attacker gains. If a standard is
involved, cite the clause or table by number rather than pasting the text.
Standards are copyrighted.

### What to expect

This library is maintained by one person alongside research work, so these are
commitments I can keep rather than an enterprise service level:

| Step | Target |
| :--- | :--- |
| Acknowledgement that the report arrived | 5 working days |
| First assessment: accepted, more information needed, or out of scope | 15 working days |
| Release containing the fix, once accepted | with the next release, sooner if the impact warrants it |

If the acknowledgement window passes in silence, add a comment on the advisory.
It means the notification was missed, not that the report was dismissed.

## Coordinated disclosure

I work to coordinated disclosure. Please give me a chance to publish a fixed
release before going public. In return I keep the advisory updated while it is
open, credit you unless you would rather stay anonymous, and publish the advisory
(with a CVE where GitHub assigns one) once the fixed version is on PyPI.

Ninety days from acknowledgement is a reasonable ceiling. If a fix is going to
take longer than that I will say so on the advisory instead of letting it go
quiet.

## Scope

### Treated as a vulnerability

- **Untrusted input reaching a computation.** Measurement arrays, samples
  decoded elsewhere and passed in, and above all the file readers: the aircraft
  noise module reads ANP CSV exports from a directory you point it at. If a
  crafted file causes anything worse than a clean exception, for example
  unbounded memory growth, a path that escapes the directory it was given, or an
  attempt to execute what was read, that is a vulnerability.
- **Deserialisation.** The library deliberately never unpickles, evaluates or
  imports anything derived from input. A code path that does is a defect worth
  reporting even without a working exploit.
- **Supply chain.** Anything that can place code into a published artefact: the
  release workflow, the contents of the built wheel or sdist, package data
  reaching outside the package, or a dependency declaration that resolves
  somewhere unexpected.
- **Repository automation.** A workflow in this repository that can be made to
  run attacker-controlled code, leak a secret, or escalate its token
  permissions.

### Not treated as a vulnerability

- **A value that disagrees with a standard.** This is the failure mode that
  matters most in this library, and it is a conformance defect, not a security
  issue. Once the disagreement has been established, usually in
  [Standards & conformance](https://github.com/jmrplens/phonometry/discussions/categories/standards-conformance),
  file it with the
  [conformance defect form](https://github.com/jmrplens/phonometry/issues/new?template=02-conformance-defect.yml).
  It is handled with the same seriousness as a security report, in public, where
  the numbers can be checked.
- **A defect in the published standard itself.** Misprints and worked examples
  that contradict their own normative text are recorded in
  [`docs/ERRATA.md`](docs/ERRATA.md) with the evidence and the reading the
  library implements. Those entries are documentation, never security
  advisories.
- **Resource exhaustion you asked for**, such as requesting a transform over an
  array that does not fit in memory. The library computes what it is told to
  compute.
- **Results from inputs outside the validity range of a method**, for instance a
  temperature or a frequency beyond what the standard covers. The guides state
  the ranges; a missing range check is a bug report.
- **Vulnerabilities in numpy, scipy, matplotlib, reportlab or any other
  dependency.** Report those upstream. If phonometry uses the dependency in a way
  that exposes users to it, that use is in scope here.
- **Scanner output with no demonstrated impact**, and hardening suggestions with
  no exploit path. Both are welcome as ordinary issues.

## What already runs

Rather than describe a process I do not have, here is the tooling that runs on
every pull request and on `main`:

- **CodeQL** default setup, weekly and on change, over the Python,
  JavaScript/TypeScript and GitHub Actions code.
- **bandit** over `src/` in the `quality` job. Any finding fails the build,
  including Low severity, so the source stays clean of `assert` in library code,
  unreviewed subprocess use and hardcoded secrets.
- **GitGuardian** secret scanning on every pull request.
- **Dependabot** security and version updates for pip, npm and GitHub Actions,
  grouped weekly.
- **Snyk** through the `snyk` target in the `Makefile`, for an on-demand
  dependency audit.

Workflows declare `contents: read` at the top level and widen permissions only
on the individual jobs that need them.

Integrity of the numbers is gated too, which matters more here than in most
libraries: `docs/CONFORMANCE.md` is regenerated in CI from the library's own
computations against reference values and the build fails if the committed
report drifts, so a change that quietly moves a standards-conformant result
cannot reach a release unnoticed.

## Credit

Reporters are credited in the published advisory and in the changelog entry for
the release that carries the fix, unless they ask not to be.
