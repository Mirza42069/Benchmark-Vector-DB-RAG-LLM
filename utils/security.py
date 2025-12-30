"""
Security Utilities for RAG System
Provides input validation, sanitization, and secure configuration handling
"""

import os
import sys
import re
import html
import logging
from urllib.parse import quote_plus
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def require_env(key: str, default: Optional[str] = None) -> str:
    """
    Get a required environment variable.
    Exits with error if not found and no default provided.
    
    Args:
        key: Environment variable name
        default: Optional default value
        
    Returns:
        The environment variable value
    """
    value = os.getenv(key, default)
    if value is None:
        logger.error(f"Required environment variable '{key}' is not set")
        print(f"❌ Error: Required environment variable '{key}' is not set")
        sys.exit(1)
    return value


def sanitize_query(query: str, max_length: int = 1000) -> str:
    """
    Sanitize user input query to prevent injection attacks.
    
    Args:
        query: Raw user input
        max_length: Maximum allowed query length
        
    Returns:
        Sanitized query string
    """
    if not query or not isinstance(query, str):
        return ""
    
    # Trim whitespace
    query = query.strip()
    
    # Limit length
    if len(query) > max_length:
        query = query[:max_length]
    
    # Remove null bytes and control characters (except newlines/tabs)
    query = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', query)
    
    return query


def escape_html(text: str) -> str:
    """
    Escape HTML entities to prevent XSS attacks.
    
    Args:
        text: Raw text that may contain HTML
        
    Returns:
        HTML-escaped text
    """
    if not text or not isinstance(text, str):
        return ""
    return html.escape(text, quote=True)


def build_pg_connection_string(
    user: str,
    password: str,
    host: str,
    port: str,
    database: str
) -> str:
    """
    Build a PostgreSQL connection string with properly encoded credentials.
    
    Args:
        user: Database username
        password: Database password
        host: Database host
        port: Database port
        database: Database name
        
    Returns:
        Properly formatted connection string
    """
    # URL-encode credentials to handle special characters
    encoded_user = quote_plus(user)
    encoded_password = quote_plus(password)
    
    return f"postgresql+psycopg://{encoded_user}:{encoded_password}@{host}:{port}/{database}"


def sanitize_error_message(error: Exception, include_type: bool = True) -> str:
    """
    Sanitize error messages to prevent information disclosure.
    Logs the full error internally but returns a safe message.
    
    Args:
        error: The exception that occurred
        include_type: Whether to include the error type in the message
        
    Returns:
        A safe error message for display
    """
    # Log the full error internally
    logger.error(f"Error occurred: {type(error).__name__}: {str(error)}")
    
    # Return a sanitized message
    if include_type:
        return f"An error occurred: {type(error).__name__}"
    return "An unexpected error occurred"


def validate_api_key(api_key: Optional[str], service_name: str = "API") -> bool:
    """
    Validate that an API key is present and has a reasonable format.
    
    Args:
        api_key: The API key to validate
        service_name: Name of the service for error messages
        
    Returns:
        True if valid, exits if invalid
    """
    if not api_key:
        logger.error(f"{service_name} key is not configured")
        print(f"❌ Error: {service_name} key is not set in environment variables")
        sys.exit(1)
    
    # Basic format validation (at least 10 chars, no whitespace)
    if len(api_key) < 10 or ' ' in api_key:
        logger.error(f"{service_name} key appears to be invalid")
        print(f"❌ Error: {service_name} key appears to be invalid")
        sys.exit(1)
    
    return True
