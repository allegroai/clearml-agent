import pytest
from furl import furl

from trains_agent.helper.package.translator import RequirementsTranslator


@pytest.mark.parametrize(
    "line",
    (
        furl()
        .set(
            scheme=scheme,
            host=host,
            path=path,
            query=query,
            fragment=fragment,
            port=port,
            username=username,
            password=password,
        )
        .url
        for scheme in ("http", "https", "ftp")
        for host in ("a", "example.com")
        for path in (None, "/", "a", "a/", "a/b", "a/b/", "a b", "a b ")
        for query in (None, "foo", "foo=3", "foo=3&bar")
        for fragment in (None, "foo")
        for port in (None, 1337)
        for username in (None, "", "user")
        for password in (None, "", "password")
    ),
)
def test_supported(line):
    assert "://" in line
    assert RequirementsTranslator.is_supported_link(line)


@pytest.mark.parametrize(
    "line",
    [
        "pytorch",
        "foo",
        "foo1",
        "bar",
        "bar1",
        "foo-bar",
        "foo-bar1",
        "foo-bar-1",
        "foo_bar",
        "foo_bar1",
        "foo_bar_1",
        " https://a",
        " https://a/b",
        " http://a",
        " http://a/b",
        " ftp://a/b",
        "file://a/b",
        "ssh://a/b",
        "foo://a/b",
        "git//a/b",
        "git+https://a/b",
        "https+git://a/b",
        "git+http://a/b",
        "http+git://a/b",
        "",
        " ",
        "-e ",
        "-e x",
        "-e http://a",
        "-e http://a/b",
        "-e https://a",
        "-e https://a/b",
        "-e file://a/b",
    ],
)
def test_not_supported(line):
    assert not RequirementsTranslator.is_supported_link(line)
