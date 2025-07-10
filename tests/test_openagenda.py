import pytest
from openagenda.scraper import fetch_and_parse_events

def test_fetch_and_parse_events():
    events = fetch_and_parse_events()
    assert isinstance(events, list)
    if events:
        event = events[0]
        assert "title" in event
        assert "description" in event
        assert isinstance(event["title"], str)
