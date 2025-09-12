# parties247_backend

## CORS configuration

The backend only accepts cross‑origin requests from a small set of known
front‑end deployments. Update `app.py` if additional origins are required.

| Environment | Allowed origin               |
|-------------|------------------------------|
| Development | `http://localhost:3000`      |
| Staging     | `https://staging.parties247.com` |
| Production  | `https://parties247.com`     |

All other origins are blocked, and credentialed cross‑site requests are
disabled by default.
