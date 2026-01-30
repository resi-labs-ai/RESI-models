"""Entry point for running validator as a module."""

import asyncio

from .validator import main

if __name__ == "__main__":
    asyncio.run(main())
