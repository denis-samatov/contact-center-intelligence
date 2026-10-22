"""Domain model of an insurance contact-center case.

The taxonomy is deliberately small and explicit: each intent has a fixed set of fields, a
subset of which is *required* before a case can be processed without a human. Routing and
recommendations use only the stated intent and *explicitly stated* customer preferences;
no attribute of the customer is inferred (see ``docs/adr/ADR-B4-no-sensitive-inference.md``).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Intent(StrEnum):
    CLAIM_AUTO = "claim_auto"  # road accident under motor insurance (OSAGO/KASKO)
    CLAIM_PROPERTY = "claim_property"  # flood, fire, theft at home
    PAYMENT_STATUS = "payment_status"  # status of an existing claim payout
    POLICY_RENEWAL = "policy_renewal"
    POLICY_CANCELLATION = "policy_cancellation"
    DATA_CHANGE = "data_change"  # change of address / phone / drivers list
    COMPLAINT = "complaint"
    DOCUMENT_REQUEST = "document_request"
    GENERAL_INFO = "general_info"


class Queue(StrEnum):
    CLAIMS = "claims"
    CLAIMS_PRIORITY = "claims_priority"  # reported injuries
    PAYMENTS = "payments"
    SALES_SERVICE = "sales_service"  # renewals, data changes, documents
    RETENTION = "retention"  # cancellations
    COMPLAINTS = "complaints"
    INFO = "info"


# Field names are shared across intents so that extraction has one output schema.
FIELDS: dict[str, str] = {
    "policy_number": "Policy number, e.g. 'XXX 0123456789' (series + 10 digits)",
    "claim_number": "Claim number, e.g. 'U-123456'",
    "incident_date": "Date of the incident, ISO yyyy-mm-dd",
    "incident_place": "City or road where the incident happened",
    "address": "Address of the insured property",
    "damage_type": "flood | fire | theft",
    "injuries": "yes | no - whether anybody was injured",
    "other_party": "yes | no - whether another vehicle was involved",
    "amount": "Amount in roubles (integer)",
    "product": "osago | kasko | property | travel",
    "change_type": "address | phone | drivers",
    "new_value": "New value for a data change",
    "document_type": "policy_copy | payment_certificate | claim_decision",
    "cancellation_reason": "Customer-stated reason, free text",
    "complaint_subject": "delay | amount | service",
    "question_topic": "deductible | coverage | documents | deadlines",
}

REQUIRED: dict[Intent, tuple[str, ...]] = {
    Intent.CLAIM_AUTO: ("policy_number", "incident_date", "incident_place", "injuries", "other_party"),
    Intent.CLAIM_PROPERTY: ("policy_number", "incident_date", "address", "damage_type"),
    Intent.PAYMENT_STATUS: ("claim_number",),
    Intent.POLICY_RENEWAL: ("policy_number", "product"),
    Intent.POLICY_CANCELLATION: ("policy_number",),
    Intent.DATA_CHANGE: ("policy_number", "change_type", "new_value"),
    Intent.COMPLAINT: ("claim_number", "complaint_subject"),
    Intent.DOCUMENT_REQUEST: ("policy_number", "document_type"),
    Intent.GENERAL_INFO: ("product", "question_topic"),
}

OPTIONAL: dict[Intent, tuple[str, ...]] = {
    Intent.CLAIM_AUTO: ("amount",),
    Intent.CLAIM_PROPERTY: ("amount",),
    Intent.POLICY_CANCELLATION: ("cancellation_reason",),
    Intent.COMPLAINT: ("policy_number",),
}


def fields_for(intent: Intent) -> tuple[str, ...]:
    return REQUIRED[intent] + OPTIONAL.get(intent, ())


class Preferences(BaseModel):
    """Only what the customer *said*. Absent means not stated, never guessed."""

    model_config = ConfigDict(extra="forbid")
    channel: str | None = Field(None, description="sms | email | phone")
    callback_window: str | None = Field(None, description="morning | afternoon | evening")


class Contradiction(BaseModel):
    field: str
    values: list[str]


class CaseLabel(BaseModel):
    """Gold annotation of one synthetic call (also the target schema of extraction)."""

    model_config = ConfigDict(extra="forbid")
    intent: Intent
    fields: dict[str, str]
    preferences: Preferences = Preferences()
    contradictions: list[Contradiction] = []
    missing_required: list[str] = []
    queue: Queue
    next_actions: list[str] = []


def route(intent: Intent, fields: dict[str, str]) -> Queue:
    """Deterministic, auditable routing policy (business rule, not a model)."""
    if intent is Intent.CLAIM_AUTO and fields.get("injuries") == "yes":
        return Queue.CLAIMS_PRIORITY
    return {
        Intent.CLAIM_AUTO: Queue.CLAIMS,
        Intent.CLAIM_PROPERTY: Queue.CLAIMS,
        Intent.PAYMENT_STATUS: Queue.PAYMENTS,
        Intent.POLICY_RENEWAL: Queue.SALES_SERVICE,
        Intent.DATA_CHANGE: Queue.SALES_SERVICE,
        Intent.DOCUMENT_REQUEST: Queue.SALES_SERVICE,
        Intent.POLICY_CANCELLATION: Queue.RETENTION,
        Intent.COMPLAINT: Queue.COMPLAINTS,
        Intent.GENERAL_INFO: Queue.INFO,
    }[intent]


def missing_required(intent: Intent, fields: dict[str, str]) -> list[str]:
    return [f for f in REQUIRED[intent] if not fields.get(f)]
