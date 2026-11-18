# Two-laptop deployment

| Laptop | Runs | Needs |
|---|---|---|
| Laptop 1 | API (`api/`), saved models, road network and trip data | Python + `requirements.txt`, `data/` folder |
| Laptop 2 | UI (`ui/index.html`) | Any browser, internet access for map tiles. Python is optional, used only to serve the page |

Both laptops must be on the same Wi-Fi or hotspot.

## Laptop 1: API

1. Install: `pip install -r requirements.txt`
2. Put the trips to serve in `data/<split>/` (for example `data/evaluator_test/`).
3. Choose the default model and trip folder in `deployment/config.json`, or set an environment variable:
   `set PATHSNAP_SPLIT=evaluator_test`
4. Start the API: `deployment\run_api.bat`
5. Find the IP address: `ipconfig` (IPv4 address, for example `192.168.1.20`).
6. If Windows Firewall asks, allow Python on **private networks**. Otherwise allow inbound TCP port 8000.
7. Check it: open `http://<laptop-1-ip>:8000/docs` in a browser.

Every request is logged in this terminal (`REQUEST` / `RESPONSE` lines). Show these logs to the evaluator.

## Laptop 2: UI

1. Copy the `ui/` folder, or clone the repository.
2. Run `deployment\run_ui.bat` and open `http://localhost:5500`, or just open `ui/index.html` directly.
3. Enter `http://<laptop-1-ip>:8000` and press **Connect**.
4. Click any trip. The UI sends the trip id, and the API returns the predicted route and metrics.

## Rules followed

- The UI only sends a trip id. All prediction happens on Laptop 1.
- The reference route is read only on Laptop 1, after prediction, to compute metrics. It is never sent to the UI.
