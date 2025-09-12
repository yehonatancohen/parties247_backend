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


def test_protect_decorator():
    flask_mod = sys.modules['flask']

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
