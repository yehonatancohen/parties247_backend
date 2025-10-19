import os
import sys
import types
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import app


def test_normalize_url():
    url = 'HTTP://Example.COM:80/path/?utm_source=x&b=2&a=1&fbclid=123'
    assert app.normalize_url(url) == 'http://example.com/path?b=2&a=1'


def test_normalized_or_none_for_dedupe():
    assert app.normalized_or_none_for_dedupe('http://example.com') == 'http://example.com/'
    assert app.normalized_or_none_for_dedupe('not a url') is None


def test_parse_datetime_normalizes_naive_iso():
    dt = app.parse_datetime('2024-05-01T12:00:00')
    assert dt.tzinfo is not None
    assert dt.isoformat().endswith('+00:00')


def test_ensure_index_behaviour():
    coll = app.Collection()
    coll._indexes = {'same': {'key': [('a', 1)], 'unique': True, 'partialFilterExpression': None}}
    app.ensure_index(coll, [('a', 1)], name='same', unique=True)
    assert coll._indexes['same']['key'] == [('a', 1)]  # unchanged

    coll._indexes = {'diff': {'key': [('a', 1)], 'unique': False, 'partialFilterExpression': None}}
    app.ensure_index(coll, [('b', 1)], name='diff', unique=True)
    assert coll._indexes['diff']['key'] == [('b', 1)]
    assert coll._indexes['diff']['unique'] is True


def test_get_region_music_event_age_tags():
    assert app.get_region('באר שבע') == 'דרום'
    assert app.get_region('חיפה') == 'צפון'
    assert app.get_region('תל אביב') == 'מרכז'
    assert app.get_region('לוד') == 'לא ידוע'

    assert app.get_music_type('this is Techno music') == 'טכנו'
    assert app.get_music_type('great trance vibes') == 'טראנס'
    assert app.get_music_type('hip hop night') == 'מיינסטרים'
    assert app.get_music_type('classical') == 'אחר'

    assert app.get_event_type('פסטיבל של קיץ') == 'פסטיבל'
    assert app.get_event_type('מסיבת טבע בחוף') == 'מסיבת טבע'
    assert app.get_event_type('club party at block') == 'מסיבת מועדון'
    assert app.get_event_type('other') == 'אחר'

    assert app.get_age('', 21) == '21+'
    assert app.get_age('', 18) == '18+'
    assert app.get_age('מסיבה לנוער', 0) == 'נוער'
    assert app.get_age('', 1) == '18+'
    assert app.get_age('', 0) == 'כל הגילאים'

    tags = app.get_tags('free alcohol open air', 'תל אביב')
    assert set(tags) == {'אלכוהול חופשי', 'בחוץ', 'תל אביב'}


def test_get_parties_appends_default_ref(monkeypatch):
    docs = [
        {
            '_id': '1',
            'goOutUrl': 'https://example.com/event?foo=1',
            'originalUrl': 'https://example.com/event',
            'date': '2099-01-01T00:00:00',
            'name': 'My Great Party',
        },
        {
            '_id': '2',
            'goOutUrl': 'https://example.com/other?ref=keep',
            'date': '2099-01-02T00:00:00',
            'slug': 'existing-slug',
        },
    ]

    class DummyCursor:
        def __init__(self, items):
            self._items = items

        def sort(self, key, direction):
            return sorted(self._items, key=lambda d: d.get(key))

        def __iter__(self):
            return iter(self._items)

    class DummyCollection:
        def find(self):
            return DummyCursor(list(docs))

    class DummySettings:
        def find_one(self, filter):
            return {'value': 'default-ref'}

    monkeypatch.setattr(app, 'parties_collection', DummyCollection())
    monkeypatch.setattr(app, 'settings_collection', DummySettings())

    payload, status = app.get_parties()
    assert status == 200
    urls = {item['_id']: item.get('goOutUrl') for item in payload}
    assert urls['1'].endswith('ref=default-ref')
    assert urls['2'].count('ref=keep') == 1
    assert 'default-ref' not in urls['2']
    slugs = {item['_id']: item.get('slug') for item in payload}
    assert slugs['1'] == 'my-great-party'
    assert slugs['2'] == 'existing-slug'


def test_protect_decorator():
    flask_mod = sys.modules['flask']
    app.JWT_SECRET = 'secret'

    @app.protect
    def hello():
        return 'ok'

    flask_mod.request.headers = {}
    resp, status = hello()
    assert status == 401

    flask_mod.request.headers = {'Authorization': 'Bearer wrong'}
    resp, status = hello()
    assert status == 401

    flask_mod.request.headers = {'Authorization': 'Bearer token'}
    assert hello() == 'ok'


def test_admin_login(monkeypatch):
    flask_mod = sys.modules['flask']
    bcrypt_mod = sys.modules['bcrypt']

    app.ADMIN_HASH = b'hash'
    app.JWT_SECRET = 'secret'

    def fake_hashpw(pw, salt):
        return salt if pw == b'good' else b'bad'

    bcrypt_mod.hashpw = fake_hashpw

    flask_mod.request.get_json = lambda silent=True: {'password': 'good'}
    res, status = app.admin_login()
    assert status == 200
    assert res['token'] == 'token'

    flask_mod.request.get_json = lambda silent=True: {'password': 'bad'}
    res, status = app.admin_login()
    assert status == 401
    assert res['message'] == 'Invalid credentials.'


def test_api_docs_and_openapi_spec():
    flask_mod = sys.modules['flask']
    flask_mod.request.host_url = 'http://localhost/'
    flask_mod.request.url_root = 'http://localhost/'

    data, status = app.openapi_json()
    assert status == 200
    assert data['info']['title'] == 'Parties247 API'
    assert '/api/parties' in data['paths']
    assert '/api/events/{slug}' in data['paths']
    assert '/api/analytics/events' in data['paths']
    assert '/api/admin/import/nightlife' in data['paths']
    assert '/api/admin/import/weekend' in data['paths']
    assert data['servers'][0]['url'] == 'http://localhost'

    html, status, headers = app.docs_page()
    assert status == 200
    assert headers['Content-Type'].startswith('text/html')
    assert 'Parties247 API' in html
    assert '/openapi.json' in html
