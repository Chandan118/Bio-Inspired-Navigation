#!/usr/bin/env python3
"""
Bio-Inspired Navigation Data Generation - Main Application

This script provides a complete, production-ready implementation for generating
bio-inspired navigation research data and visualizations.

Author: Research Team
Version: 2.0.0
License: MIT
"""

import argparse
import logging
import sys
import traceback
from pathlib import Path

# Ensure the local package can be imported when running from source
sys.path.insert(0, str(Path(__file__).parent))

from bio_nav_data.pipeline import BioNavDataGenerator  # noqa: E402
from bio_nav_data.utils.config import Config  # noqa: E402
from bio_nav_data.utils.logger import get_logger, setup_logger  # noqa: E402


def main() -> int:
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Bio-Inspired Navigation Data Generation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run complete pipeline
  python main.py --config-only      # Show configuration only
  python main.py --validate         # Validate configuration
        """,
    )

    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Show configuration and exit",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate configuration and exit",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Custom output directory",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = getattr(logging, args.log_level.upper())
    setup_logger(level=log_level)
    logger = get_logger("main")

    try:
        # Initialize configuration
        config = Config()
        if args.output_dir:
            config = Config(args.output_dir)

        # Show configuration if requested
        if args.config_only:
            config.print_summary()
            return 0

        # Validate configuration if requested
        if args.validate:
            validation = config.validate_config()
            print("\nConfiguration Validation Results:")
            for key, value in validation.items():
                status = "✓ PASS" if value else "✗ FAIL"
                print(f"  {status}: {key}")

            if all(validation.values()):
                print("\n✅ All validations passed!")
            else:
                print("\n❌ Some validations failed!")
                return 1
            return 0

        # Run the complete pipeline
        generator = BioNavDataGenerator(config)
        generator.run_complete_pipeline()

        return 0

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 1

    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Application failed: %s", exc)
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
