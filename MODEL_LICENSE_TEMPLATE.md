# MIT License Template for Subnet 46 Models

**Version 1.0**
**Last Updated:** January 28, 2026

---

## Overview

All models submitted to Bittensor Subnet 46 must be licensed under the **MIT License**. This document provides the standard MIT license text for your reference.

---

## How to Apply the MIT License

When creating your model repository on HuggingFace, select **MIT** from the license dropdown. This is the **only required step** - the validator checks the HuggingFace API for this metadata.

You may optionally include a `LICENSE` file in your repository with the text below.

---

## MIT License Text

```
MIT License

Copyright (c) [YEAR] [YOUR NAME OR ORGANIZATION]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Fill-in Fields

When using the LICENSE file, replace:
- `[YEAR]` - Current year (e.g., 2026)
- `[YOUR NAME OR ORGANIZATION]` - Your name or entity name

---

## Why MIT License?

The MIT License is one of the most permissive open-source licenses:

- **Simple and short** - Easy to understand
- **Permissive** - Allows commercial use, modification, distribution
- **Compatible** - Works with almost all other licenses
- **Standard** - Widely recognized and accepted

This aligns with the open-source ethos of the Bittensor ecosystem.

---

## Validator Verification

The Subnet 46 validator automatically checks your license:

1. Fetches model metadata from HuggingFace API
2. Checks the `license` field in model card
3. Performs case-insensitive match for "mit"
4. Rejects models without MIT license (`LicenseError`)

---

## Questions?

For questions about licensing:
- **Email:** support@resilabs.ai
- **GitHub:** https://github.com/resi-labs-ai/RESI-models/issues

---

**END OF DOCUMENT**
