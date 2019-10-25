import pytest

from trains_agent.helper.repo import VCS


@pytest.mark.parametrize(
    ["url", "expected"],
    (
        ("a", None),
        ("foo://a/b", None),
        ("foo://a/b/", None),
        ("https://a/b/", None),
        ("https://example.com/a/b", None),
        ("https://example.com/a/b/", None),
        ("ftp://example.com/a/b", None),
        ("ftp://example.com/a/b/", None),
        ("github.com:foo/bar.git", "https://github.com/foo/bar.git"),
        ("git@github.com:foo/bar.git", "https://github.com/foo/bar.git"),
        ("bitbucket.org:foo/bar.git", "https://bitbucket.org/foo/bar.git"),
        ("hg@bitbucket.org:foo/bar.git", "https://bitbucket.org/foo/bar.git"),
        ("ssh://bitbucket.org/foo/bar.git", "https://bitbucket.org/foo/bar.git"),
        ("ssh://git@github.com/foo/bar.git", "https://github.com/foo/bar.git"),
        ("ssh://user@github.com/foo/bar.git", "https://user@github.com/foo/bar.git"),
        ("ssh://git:password@github.com/foo/bar.git", "https://git:password@github.com/foo/bar.git"),
        ("ssh://user:password@github.com/foo/bar.git", "https://user:password@github.com/foo/bar.git"),
        ("ssh://hg@bitbucket.org/foo/bar.git", "https://bitbucket.org/foo/bar.git"),
        ("ssh://user@bitbucket.org/foo/bar.git", "https://user@bitbucket.org/foo/bar.git"),
        ("ssh://hg:password@bitbucket.org/foo/bar.git", "https://hg:password@bitbucket.org/foo/bar.git"),
        ("ssh://user:password@bitbucket.org/foo/bar.git", "https://user:password@bitbucket.org/foo/bar.git"),
    ),
)
def test(url, expected):
    result = VCS.resolve_ssh_url(url)
    expected = expected or url
    assert result == expected
