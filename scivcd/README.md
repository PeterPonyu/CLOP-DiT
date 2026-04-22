# scivcd

Scientific visualization conflict detection for matplotlib.

See the full documentation at `docs/scivcd/README.md` or the project repository.

## Quickstart

```bash
pip install -e scivcd/
```

```python
import scivcd
scivcd.install()   # patches savefig — checks fire automatically from here on

# or check a specific figure on demand:
report = scivcd.check(fig)
print(report.summary())
```
