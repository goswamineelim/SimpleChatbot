import pytest
from services.tools.web_search_tool import web_search


def test_empty_query():
    assert "cannot be empty" in web_search("   ")


def test_web_search_success(mocker):
    mock_results = [
        {
            "title": "Kolkata Weather Today",
            "href": "https://weather.example.com/kolkata",
            "body": "Current temperature in Kolkata is 28 degrees Celsius with partly cloudy skies.",
        },
        {
            "title": "Kolkata Forecast",
            "href": "https://meteo.example.com/kolkata",
            "body": "Expect a high of 32C and low of 24C today.",
        },
    ]

    # Mock DDGS.text
    mock_ddgs = mocker.MagicMock()
    mock_ddgs.text.return_value = mock_results
    mocker.patch("services.tools.web_search_tool.DDGS", return_value=mock_ddgs)

    result = web_search("Kolkata weather")

    assert "Kolkata Weather Today" in result
    assert "https://weather.example.com/kolkata" in result
    assert "28 degrees Celsius" in result
    assert "Kolkata Forecast" in result


def test_web_search_no_results(mocker):
    mock_ddgs = mocker.MagicMock()
    mock_ddgs.text.return_value = []
    mocker.patch("services.tools.web_search_tool.DDGS", return_value=mock_ddgs)

    result = web_search("xyznonexistentquery123")
    assert "No web search results found" in result


def test_web_search_exception_handling(mocker):
    mock_ddgs = mocker.MagicMock()
    mock_ddgs.text.side_effect = Exception("Connection timeout")
    mocker.patch("services.tools.web_search_tool.DDGS", return_value=mock_ddgs)

    result = web_search("news")
    assert "Web search failed" in result
    assert "Connection timeout" in result
