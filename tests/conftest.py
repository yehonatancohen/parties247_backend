import sys, types

# stub bcrypt
sys.modules['bcrypt'] = types.SimpleNamespace(hashpw=lambda pw, salt: salt)

# stub jwt
class _JwtModule(types.SimpleNamespace):
    def encode(self, payload, secret, algorithm="HS256"):
        return "token"
    def decode(self, token, secret, algorithms=None):
        if token != "token":
            raise self.InvalidTokenError("bad token")
        return {}
    class ExpiredSignatureError(Exception):
        pass
    class InvalidTokenError(Exception):
        pass
jwt_module = _JwtModule()
jwt_module.ExpiredSignatureError = jwt_module.ExpiredSignatureError
jwt_module.InvalidTokenError = jwt_module.InvalidTokenError
sys.modules['jwt'] = jwt_module

# stub dotenv
sys.modules['dotenv'] = types.SimpleNamespace(load_dotenv=lambda: None)

# stub flask
class _Flask:
    def __init__(self, name):
        self.name = name
        self.config = {}
        self.logger = types.SimpleNamespace(info=lambda *a, **k: None,
                                            warning=lambda *a, **k: None,
                                            error=lambda *a, **k: None)
    def route(self, *args, **kwargs):
        def decorator(f):
            return f
        return decorator
Flask_request = types.SimpleNamespace(headers={}, get_json=lambda silent=True: {})
def jsonify(obj):
    return obj
def url_for(endpoint, _external=False, **values):
    if endpoint == "openapi_json":
        path = "/openapi.json"
    else:
        path = f"/{endpoint}"
    if _external:
        return f"http://testserver{path}"
    return path
sys.modules['flask'] = types.SimpleNamespace(Flask=_Flask, request=Flask_request, jsonify=jsonify, url_for=url_for)

# stub flask_cors
sys.modules['flask_cors'] = types.SimpleNamespace(CORS=lambda app: None)

# stub flask_limiter
class _Limiter:
    def __init__(self, func, app=None):
        pass
    def limit(self, *args, **kwargs):
        def decorator(f):
            return f
        return decorator
sys.modules['flask_limiter'] = types.SimpleNamespace(Limiter=_Limiter)
sys.modules['flask_limiter.util'] = types.SimpleNamespace(get_remote_address=lambda: '0.0.0.0')

# stub pymongo and related
class _Collection(dict):
    def index_information(self):
        return getattr(self, '_indexes', {})
    def drop_index(self, name):
        self._indexes.pop(name, None)
    def create_index(self, keys, name=None, **kwargs):
        self._indexes = getattr(self, '_indexes', {})
        self._indexes[name] = {'key': list(keys), 'unique': kwargs.get('unique', False), 'partialFilterExpression': kwargs.get('partialFilterExpression')}
    def update_many(self, filter, update):
        pass
    def delete_one(self, filter):
        return types.SimpleNamespace(deleted_count=1)
    def update_one(self, filter, update, **kwargs):
        return types.SimpleNamespace(matched_count=1, upserted_id='id')
    def find_one(self, filter, projection=None):
        return {'_id': 'id'}
    def find(self):
        return []
    def insert_one(self, doc):
        return types.SimpleNamespace(inserted_id='id')
    def sort(self, key, order):
        return []
class _DB:
    def __init__(self):
        self.parties = _Collection()
        self.carousels = _Collection()
        self.sections = _Collection()
    def __getitem__(self, item):
        return self
class _MongoClient:
    def __init__(self, uri):
        self.db = _DB()
    def __getitem__(self, name):
        return self.db
sys.modules['pymongo'] = types.SimpleNamespace(MongoClient=_MongoClient, errors=types.SimpleNamespace(DuplicateKeyError=Exception))
sys.modules['pymongo.collection'] = types.SimpleNamespace(Collection=_Collection)

# stub bson.objectid
class _ObjectId(str):
    def __new__(cls, value):
        return str.__new__(cls, value)


sys.modules['bson'] = types.SimpleNamespace(objectid=types.SimpleNamespace(ObjectId=_ObjectId))
sys.modules['bson.objectid'] = sys.modules['bson'].objectid

# stub requests
sys.modules['requests'] = types.SimpleNamespace(
    get=lambda url, headers=None, timeout=None: types.SimpleNamespace(status_code=200, text='', raise_for_status=lambda: None),
    post=lambda url, json=None, timeout=None: types.SimpleNamespace(status_code=200, text='', raise_for_status=lambda: None),
    exceptions=types.SimpleNamespace(RequestException=Exception),
)

# stub bs4
import re


class _FakeTag:
    def __init__(self, attrs=None, string=None):
        self._attrs = attrs or {}
        self.string = string

    def get(self, key, default=None):
        return self._attrs.get(key, default)


class _FakeSoup:
    def __init__(self, text):
        self._text = text or ""

    def find_all(self, name):
        if name != "a":
            return []
        pattern = re.compile(r'<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
        tags = []
        for href, inner in pattern.findall(self._text):
            tags.append(_FakeTag({"href": href}, inner))
        return tags

    def find(self, name, attrs=None):
        attrs = attrs or {}
        if name == "script" and attrs.get("id") == "__NEXT_DATA__":
            match = re.search(r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>', self._text, re.IGNORECASE | re.DOTALL)
            if match:
                return _FakeTag(string=match.group(1))
        return None


sys.modules['bs4'] = types.SimpleNamespace(BeautifulSoup=lambda text, parser: _FakeSoup(text))

# stub pydantic
class _ValidationError(Exception):
    def __init__(self, errors=None):
        self._errors = errors or []
    def errors(self):
        return self._errors
class _BaseModel:
    def __init__(self, **data):
        for k, v in data.items():
            setattr(self, k, v)
    def dict(self, exclude_unset=False):
        return self.__dict__
sys.modules['pydantic'] = types.SimpleNamespace(BaseModel=_BaseModel, ValidationError=_ValidationError)
