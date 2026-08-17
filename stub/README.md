# PyOctaveBand → phonometry

**PyOctaveBand has been renamed to [phonometry](https://pypi.org/project/phonometry/).**

This is a transition package: installing or upgrading `PyOctaveBand` installs
`phonometry` and provides a `pyoctaveband` module that re-exports the full,
unchanged API with a `FutureWarning`, which Python shows by default so the
migration notice actually reaches you. Your existing code keeps working, but
new code should use the new name:

```bash
pip install phonometry
```

I recommend `pip install phonometry[full]` instead, which pulls the optional
matplotlib, numba, reportlab and svglib and so enables every feature: the
`.plot()` figures, the normative PDF fiches of `.report()` (whose figure panel
needs matplotlib as well as reportlab and svglib) and the compiled kernel that
speeds up the `impulse` time weighting, which without numba still works on a
pure-Python kernel that is only slower. One caveat: numba declares
`numpy<2.5`, so `[full]` resolves NumPy below 2.5 while a plain install gets
the newest release; `phonometry[plot,report]` keeps NumPy current and only
gives up the compiled kernel.

```python
import phonometry  # instead of: import pyoctaveband
```

The API is identical — renaming the import is a complete migration.

This package requires `phonometry>=3.0.0,<4`. The 3.x line is the one whose
API is the API of the last release under the old name, so that pin is what
makes the sentence above true: it cannot hand you a release that has retired a
name `pyoctaveband` promises. Later lines are reached by renaming the import,
which is the whole of the migration.

- Documentation: https://jmrplens.github.io/phonometry/
- Repository: https://github.com/jmrplens/phonometry
- Last release under the old name: [`pyoctaveband-v2` branch](https://github.com/jmrplens/phonometry/tree/pyoctaveband-v2)
