# Two-laptop test run

Fill this in during the real test. Do not fill it in beforehand.

| Item | Value |
|---|---|
| Date and time | |
| Laptop 1 (API) person / IP address | |
| Laptop 2 (UI) person | |
| Network (hotspot name) | |
| Trip folder served (`student_test` or `evaluator_test`) | |
| Model shown as final | `random_forest` |

## Checks

- [ ] `http://<laptop-1-ip>:8000/docs` opens on Laptop 2
- [ ] UI shows "Model online" after Connect
- [ ] `python -m pytest api/test_api.py -v` passes on Laptop 1 (paste the last line below)
- [ ] Every trip below loads on the map and shows metrics
- [ ] API terminal shows a `REQUEST` and a `RESPONSE` line for every click (screenshot saved)

| Trip | Loaded? | Right road (with HMM) | Connected route | Time (s) |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

pytest result line:

```
(paste here)
```

## Problems found and how we fixed them

-
