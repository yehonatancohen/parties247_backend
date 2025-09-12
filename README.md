# parties247_backend

### Admin authentication

Admin endpoints are now protected using short‑lived JSON Web Tokens (JWTs).

1. Generate a bcrypt hash for your admin password and store it in the
   `ADMIN_PASSWORD_HASH` environment variable. For example:

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

   A successful response returns `{ "token": "<jwt>" }` with a 15 minute
   expiration.

4. Access protected routes by including the token in the `Authorization`
   header:

   ```http
   Authorization: Bearer <jwt>
   ```

5. To verify that a token is still valid, POST to `/api/admin/verify-token`.

The previous `x-admin-secret-key` header is no longer supported.
