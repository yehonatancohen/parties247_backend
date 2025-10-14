# Parties247 Backend

Backend service powering the [Parties247](https://parties247.com) website. It exposes a REST API for listing parties and managing homepage carousel items. The service is built with Flask and MongoDB and includes URL validation, rate limiting and JWT‑secured admin endpoints.

## Features
- RESTful endpoints for parties and carousel items
- URL normalization and validation to prevent malicious or duplicate links
- JWT‑based admin authentication
- Rate limiting and CORS restrictions
- Easy configuration through environment variables
- Automated imports for Go-Out nightlife and weekend events

## Getting Started

### Requirements
- Python 3.12+
- A running MongoDB instance

### Installation
```bash
git clone <repo>
cd parties247_backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables
| Variable | Purpose |
| -------- | ------- |
| `MONGODB_URI` | Connection string for MongoDB |
| `FLASK_ENV` | Set to `development` to enable debug mode; defaults to `production` |
| `DEBUG` | Alternative flag to enable debug mode when `FLASK_ENV` is not `production` |
| `PORT` | Port the server listens on (defaults to `3001`) |
| `ADMIN_PASSWORD_HASH` | Bcrypt hash for the admin password |
| `JWT_SECRET_KEY` | Secret used to sign admin JWTs |

### Running Locally
Enable debug mode for local development and start the server:
```bash
export FLASK_ENV=development  # or: export DEBUG=1
python app.py
```

For production deployments, leave debug mode disabled:
```bash
export FLASK_ENV=production
python app.py
```

## CORS Configuration
The backend only accepts cross‑origin requests from a small set of known front‑end deployments. Update `app.py` if additional origins are required.

| Environment | Allowed origin |
|-------------|------------------------------|
| Development | `http://localhost:3000` |
| Staging | `https://parties247-website.vercel.app/` |
| Production | `https://parties247.com` |

## Admin Authentication
Admin endpoints are protected using short‑lived JSON Web Tokens (JWTs).

1. Generate a bcrypt hash for your admin password and store it in the `ADMIN_PASSWORD_HASH` environment variable. For example:
   ```python
   import bcrypt, os
   print(bcrypt.hashpw(b"my-admin-password", bcrypt.gensalt()).decode())
   ```
2. Set a `JWT_SECRET_KEY` environment variable used to sign tokens.
3. Obtain a token by POSTing the password to `/api/admin/login`:
   ```http
   POST /api/admin/login
   {"password": "my-admin-password"}
   ```
   A successful response returns `{ "token": "<jwt>" }` with a 15 minute expiration.
4. Access protected routes by including the token in the `Authorization` header:
   ```http
   Authorization: Bearer <jwt>
   ```
5. To verify that a token is still valid, POST to `/api/admin/verify-token`.

## Automated Go-Out Imports

Authenticated admins can seed two curated carousels directly from Go-Out's official feeds:

- `POST /api/admin/import/nightlife` scrapes the nightlife category and updates the "חיי לילה" carousel.
- `POST /api/admin/import/weekend` scrapes the weekend feed and updates the "סופ״ש" carousel.

Both endpoints fetch the latest events via Go-Out's APIs, append the configured referral code, and ensure the related parties exist before updating their respective carousels.

## API Documentation

- Browse the human-friendly overview at [`/docs`](http://localhost:3001/docs) when the server is running.
- Consume the machine-readable OpenAPI description at [`/openapi.json`](http://localhost:3001/openapi.json). Import this file into tools like Postman, Hoppscotch, or VS Code's REST client for interactive exploration.

## Testing
Run the test suite with:
```bash
pytest
```

## Release History
- 0.1.0 – initial release. See [CHANGELOG](CHANGELOG.md) for details.

## Changelog
This project follows [Semantic Versioning](https://semver.org/). All notable changes are documented in [CHANGELOG.md](CHANGELOG.md).

## Contributing
Pull requests are welcome. Please open an issue to discuss significant changes before submitting a PR.

