# parties247_backend

## Environment Configuration

The application reads environment variables to control its runtime
behavior. Debug mode is disabled by default to ensure production
deployments run safely.

### Local development

Enable debug mode by setting `FLASK_ENV` to `development` or by setting
`DEBUG` to a truthy value. Then run the application:

```bash
export FLASK_ENV=development  # or: export DEBUG=1
python app.py
```

### Production

Leave debug mode off in production. Ensure `FLASK_ENV` is set to
`production` (the default) and avoid setting `DEBUG`:

```bash
export FLASK_ENV=production
python app.py
```

The `PORT` environment variable can be used to change the port from the
default of `3001`.

