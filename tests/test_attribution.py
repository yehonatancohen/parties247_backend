import app as backend


def test_whatsapp_code_wins():
    assert backend.classify_channel("google", "google.com", "abc", None) == "whatsapp"


def test_utm_source_and_aliases():
    assert backend.classify_channel("IG", None) == "instagram"
    assert backend.classify_channel("newsletter", None) == "newsletter"


def test_referrer_hosts():
    assert backend.classify_channel(None, "www.google.co.il") == "google"
    assert backend.classify_channel(None, "l.instagram.com") == "instagram"
    assert backend.classify_channel(None, "t.me") == "telegram"
    assert backend.classify_channel(None, "somesite.com") == "referral"


def test_fallback_and_direct():
    assert backend.classify_channel(None, None, None, "https://www.google.com/x") == "google"
    assert backend.classify_channel(None, None, None, "https://www.parties247.co.il/event/a") == "direct"
    assert backend.classify_channel(None, None) == "direct"
