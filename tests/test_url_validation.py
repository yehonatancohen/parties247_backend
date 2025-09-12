import os
import socket
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from app import is_url_allowed


def _mock_getaddrinfo(ip):
    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (ip, port))]
    return fake_getaddrinfo


def test_valid_public_url(monkeypatch):
    monkeypatch.setattr(socket, 'getaddrinfo', _mock_getaddrinfo('93.184.216.34'))
    assert is_url_allowed('http://example.com')


def test_reject_file_scheme():
    assert not is_url_allowed('file:///etc/passwd')


def test_reject_loopback(monkeypatch):
    monkeypatch.setattr(socket, 'getaddrinfo', _mock_getaddrinfo('127.0.0.1'))
    assert not is_url_allowed('http://127.0.0.1')


def test_reject_disallowed_port(monkeypatch):
    monkeypatch.setattr(socket, 'getaddrinfo', _mock_getaddrinfo('93.184.216.34'))
    assert not is_url_allowed('http://example.com:81')
