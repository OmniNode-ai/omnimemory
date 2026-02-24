# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Tests for database URL display utilities.

Validates that safe_db_url_display() correctly strips credentials from
PostgreSQL connection URLs while preserving host, port, and database
information for safe logging and error messages.
"""

from __future__ import annotations

import pytest

from omnimemory.utils.db_url import safe_db_url_display


@pytest.mark.unit
class TestSafeDbUrlDisplay:
    """Tests for safe_db_url_display()."""

    def test_full_url_strips_credentials(self) -> None:
        """Test that user:password are stripped from a full PostgreSQL URL."""
        url = "postgresql://myuser:secret_password@dbhost:5432/mydb"
        result = safe_db_url_display(url)
        assert result == "dbhost:5432/mydb"
        assert "myuser" not in result
        assert "secret_password" not in result

    def test_url_without_password(self) -> None:
        """Test URL with user but no password."""
        url = "postgresql://myuser@dbhost:5432/mydb"
        result = safe_db_url_display(url)
        assert result == "dbhost:5432/mydb"
        assert "myuser" not in result

    def test_url_without_credentials(self) -> None:
        """Test URL with no credentials at all."""
        url = "postgresql://dbhost:5432/mydb"
        result = safe_db_url_display(url)
        assert result == "dbhost:5432/mydb"

    def test_url_without_port(self) -> None:
        """Test URL without explicit port."""
        url = "postgresql://user:pass@dbhost/mydb"
        result = safe_db_url_display(url)
        assert result == "dbhost/mydb"

    def test_url_without_database(self) -> None:
        """Test URL without database name."""
        url = "postgresql://user:pass@dbhost:5432"
        result = safe_db_url_display(url)
        assert result == "dbhost:5432"

    def test_url_host_only(self) -> None:
        """Test URL with only host."""
        url = "postgresql://user:pass@dbhost"
        result = safe_db_url_display(url)
        assert result == "dbhost"

    def test_postgres_scheme(self) -> None:
        """Test URL with 'postgres' scheme (shorter variant)."""
        url = "postgres://user:pass@dbhost:5432/mydb"
        result = safe_db_url_display(url)
        assert result == "dbhost:5432/mydb"

    def test_non_postgres_url_returns_unparseable(self) -> None:
        """Test that non-PostgreSQL URLs return unparseable marker."""
        result = safe_db_url_display("https://example.com/path")
        assert result == "(unparseable URL)"

    def test_mysql_url_returns_unparseable(self) -> None:
        """Test that MySQL URLs return unparseable marker."""
        result = safe_db_url_display("mysql://user:pass@host:3306/db")
        assert result == "(unparseable URL)"

    def test_empty_string_returns_unparseable(self) -> None:
        """Test that empty string returns unparseable marker."""
        result = safe_db_url_display("")
        assert result == "(unparseable URL)"

    def test_garbage_string_returns_unparseable(self) -> None:
        """Test that non-URL garbage returns unparseable marker."""
        result = safe_db_url_display("not a url at all")
        assert result == "(unparseable URL)"

    def test_localhost_url(self) -> None:
        """Test standard localhost development URL."""
        url = "postgresql://omnimemory:devpass@localhost:5432/omnimemory"
        result = safe_db_url_display(url)
        assert result == "localhost:5432/omnimemory"
        assert "devpass" not in result

    def test_ip_address_url(self) -> None:
        """Test URL with IP address host."""
        url = "postgresql://dbadmin:secretpass@203.0.113.5:5436/omninode_bridge"
        result = safe_db_url_display(url)
        assert result == "203.0.113.5:5436/omninode_bridge"
        assert "secretpass" not in result
        assert "dbadmin" not in result

    def test_special_characters_in_password(self) -> None:
        """Test URL with special characters in password."""
        url = "postgresql://user:p%40ss%23word@dbhost:5432/mydb"
        result = safe_db_url_display(url)
        assert result == "dbhost:5432/mydb"
        assert "p%40ss" not in result

    def test_ipv6_host_url(self) -> None:
        """Test URL with IPv6 host preserves brackets for unambiguous output."""
        url = "postgresql://testuser:s3cret@[::1]:5432/mydb"
        result = safe_db_url_display(url)
        assert result == "[::1]:5432/mydb"
        assert "testuser" not in result
        assert "s3cret" not in result

    def test_ipv6_host_without_port(self) -> None:
        """Test IPv6 URL without port."""
        url = "postgresql://user:pass@[::1]/mydb"
        result = safe_db_url_display(url)
        assert result == "[::1]/mydb"

    def test_non_postgres_url_logs_warning(self) -> None:
        """Test that non-PostgreSQL URLs emit a structured warning."""
        from unittest.mock import patch

        with patch("omnimemory.utils.db_url.logger") as mock_logger:
            result = safe_db_url_display("https://example.com/path")
            assert result == "(unparseable URL)"
            mock_logger.warning.assert_called_once()
            assert "scheme" in str(mock_logger.warning.call_args)

    def test_sqlalchemy_asyncpg_dialect(self) -> None:
        """Test SQLAlchemy asyncpg dialect URL."""
        url = "postgresql+asyncpg://appuser:dbpass@db.example.com:5432/myapp"
        result = safe_db_url_display(url)
        assert result == "db.example.com:5432/myapp"
        assert "appuser" not in result
        assert "dbpass" not in result

    def test_sqlalchemy_psycopg2_dialect(self) -> None:
        """Test SQLAlchemy psycopg2 dialect URL."""
        url = "postgresql+psycopg2://appuser:dbpass@localhost:5432/testdb"
        result = safe_db_url_display(url)
        assert result == "localhost:5432/testdb"
        assert "appuser" not in result
        assert "dbpass" not in result

    def test_query_params_stripped(self) -> None:
        """Test that query parameters (which may contain credentials) are stripped."""
        url = "postgresql://appuser:dbpass@db.example.com:5432/myapp?sslpassword=secret&sslmode=require"
        result = safe_db_url_display(url)
        assert result == "db.example.com:5432/myapp"
        assert "sslpassword" not in result
        assert "secret" not in result
        assert "appuser" not in result
        assert "dbpass" not in result

    def test_import_from_utils_package(self) -> None:
        """Test that safe_db_url_display is importable from utils package."""
        from omnimemory.utils import safe_db_url_display as imported_fn

        assert callable(imported_fn)
        result = imported_fn("postgresql://u:p@host:5432/db")
        assert result == "host:5432/db"
