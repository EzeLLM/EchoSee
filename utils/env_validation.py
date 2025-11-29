"""Environment variable validation for EchoSee.

This module validates that all required environment variables are present
before the application starts, providing clear error messages if any are missing.
"""

import os
import sys
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


def validate_required_env_vars() -> Tuple[bool, List[str]]:
    """Validate all required environment variables are present.

    Returns:
        (is_valid, missing_vars) tuple where is_valid is True if all required
        variables are present, and missing_vars is a list of missing variable names.
    """
    required = [
        'OPENAI_API_KEY',
        'TAVILY_API_KEY',
    ]

    # Load config to check provider-specific requirements
    try:
        import yaml
        with open('config.yml', 'r') as f:
            config = yaml.safe_load(f)

        # Check if DeepSeek is being used as a provider
        llm_config = config.get('LLM', {})
        provider = llm_config.get('provider')
        hp_provider = llm_config.get('high_performance_provider')

        if provider == 'deepseek' or hp_provider == 'deepseek':
            required.append('DEEPSEEK_API_KEY')

    except Exception as e:
        logger.warning(f"Could not read config.yml to check provider requirements: {e}")
        logger.info("Proceeding with basic validation...")

    missing = [var for var in required if not os.getenv(var)]
    return len(missing) == 0, missing


def require_env_vars():
    """Check required env vars and exit if any are missing.

    This function should be called early in the application startup process.
    If any required environment variables are missing, it will print a clear
    error message and exit the program with status code 1.
    """
    is_valid, missing = validate_required_env_vars()

    if not is_valid:
        print("=" * 70)
        print("ERROR: Missing required environment variables")
        print("=" * 70)
        for var in missing:
            print(f"  ✗ {var}")
        print()
        print("Please set these variables in your .env file")
        print()
        print("Example .env file:")
        print("  OPENAI_API_KEY=sk-...")
        print("  TAVILY_API_KEY=tvly-...")
        print("  DEEPSEEK_API_KEY=sk-...  # If using DeepSeek provider")
        print("=" * 70)
        sys.exit(1)

    # If validation passes, optionally print success message
    # Commented out to avoid cluttering output during normal operation
    # print("✓ Environment variables validated successfully")


if __name__ == '__main__':
    # Test the validation
    print("Testing environment variable validation...")
    is_valid, missing = validate_required_env_vars()

    if is_valid:
        print("✓ All required environment variables are present")
    else:
        print(f"✗ Missing {len(missing)} required environment variable(s):")
        for var in missing:
            print(f"    - {var}")
