"""RESI exclusive license constants and helpers."""

from __future__ import annotations

import hashlib

# Canonical URL to the LICENSE file that miners must reference in license_link
EXCLUSIVE_LICENSE_LINK = (
    "https://huggingface.co/resi-ai/model-license/blob/main/LICENSE"
)

EXCLUSIVE_LICENSE_TEXT = """\
PROPRIETARY MODEL LICENSE AGREEMENT: RESI SUBNET 46
1. DEFINITIONS

"The Model" refers to the ONNX model files, weights, parameters, and associated code contained within the specific Hugging Face repository in which this License file resides.
"The Company" refers to Resi Inc.
"The Submitter" refers to the individual or entity that uploaded The Model to this repository and submitted the repository URI to Bittensor Subnet 46.
"Effective Date" refers to the timestamp of the upload or initial commit of The Model to this Hugging Face repository.
"Authorized Resi Gateways" refers to the official Resi Inc. Chutes infrastructure, Resi OpenRouter endpoints, and the Resi Model Portal.
2. GRANT OF EXCLUSIVE LICENSE
By submitting The Model to Resi Subnet 46, the Submitter hereby grants to The Company a worldwide, exclusive, perpetual, irrevocable, and royalty-free license to use, host, store, reproduce, modify, create derivative works, communicate, publish, publicly perform, publicly display, and distribute The Model for any commercial or non-commercial purpose.
3. AUTHORIZED USES
The Company grants third parties and developers the right to build applications, services, or integrations leveraging The Model, provided that:

All inference and access to The Model is routed exclusively through Authorized Resi Gateways.
The application or service does not involve the hosting of the raw weights or the local execution of The Model outside of the Resi Inc. ecosystem.
Authorized Validators of Bittensor Subnet 46 are granted a limited sub-license to download and execute The Model solely for the purpose of network verification and scoring.
4. UNAUTHORIZED USES
The following actions are strictly prohibited and constitute a breach of this License:

Unauthorized Forking: Copying, cloning, or "forking" the raw weights and architecture of The Model to any platform, repository, or local environment not controlled by Resi Inc.
Bypass of Gateways: Accessing or serving The Model via any infrastructure other than Authorized Resi Gateways (e.g., self-hosting for commercial use or providing third-party API access).
Unauthorized Distribution: Sharing, selling, or redistributing the model files or any derivative fine-tunes that utilize the core weights of The Model without express written consent from Resi Inc.
5. CONSIDERATION AND WORK-FOR-HIRE ACKNOWLEDGMENT
The Submitter acknowledges that the opportunity to earn rewards (Alpha) via the Bittensor Network, subject to the terms and conditions of the Resi Subnet 46 codebase, constitutes full and sufficient legal consideration for this grant of rights. To the extent permitted by law, The Model shall be treated as a "work made for hire," and the Submitter waives all moral rights and proprietary claims upon submission.
6. ENFORCEMENT
Resi Inc. reserves the right to initiate legal proceedings, seek injunctive relief, and issue DMCA takedown requests against any party found to be copying, hosting, or utilizing The Model in a manner that bypasses Authorized Resi Gateways.
7. REPRESENTATIONS AND WARRANTIES
The Submitter represents and warrants that they are the sole creator of The Model or have sufficient rights to grant this exclusive license, and that The Model does not infringe upon the intellectual property rights of any third party.
8. TERMINATION
This license is perpetual. Failure to receive rewards due to model performance or network conditions does not terminate the rights granted to The Company under this Agreement.
9. GOVERNING LAW
This License shall be governed by and construed in accordance with the laws of the jurisdiction in which Resi Inc. is incorporated."""


def _normalize(text: str) -> str:
    """Normalize license text for hash comparison."""
    return text.strip().replace("\r\n", "\n")


# SHA-256 hash of the canonical license text (normalized)
EXCLUSIVE_LICENSE_HASH: str = hashlib.sha256(
    _normalize(EXCLUSIVE_LICENSE_TEXT).encode()
).hexdigest()


def compute_license_hash(text: str) -> str:
    """Compute SHA-256 hash of license text (normalized)."""
    return hashlib.sha256(_normalize(text).encode()).hexdigest()
