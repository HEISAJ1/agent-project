"""
Week 2: the Sender agent.

Its only job: take a finished summary and email it. No LLM call happens
here at all — sending an email doesn't need "intelligence," it just needs
to reliably execute. Keeping this agent dumb-and-simple on purpose is a
deliberate design choice worth being able to explain: not every step in a
multi-agent system needs to be an LLM call.
"""

import os
import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException
from dotenv import load_dotenv

load_dotenv()

configuration = sib_api_v3_sdk.Configuration()
configuration.api_key["api-key"] = os.environ["BREVO_API_KEY"]

api_instance = sib_api_v3_sdk.TransactionalEmailsApi(
    sib_api_v3_sdk.ApiClient(configuration)
)


def send_email(to_email: str, subject: str, body: str) -> str:
    """
    Sends a plain-text email via Brevo.
    Returns a short status message — never raises, so a failed send doesn't
    crash the whole pipeline; the caller can decide what to do about it.
    """
    email = sib_api_v3_sdk.SendSmtpEmail(
        to=[{"email": to_email}],
        sender={"name": "Agent Project", "email": to_email},
        subject=subject,
        text_content=body,
    )

    try:
        api_instance.send_transac_email(email)
        return f"Email sent successfully to {to_email}."
    except ApiException as e:
        return f"Error: failed to send email ({e})"


if __name__ == "__main__":
    test_to = input("Enter your email to send a test to: ")
    result = send_email(
        to_email=test_to,
        subject="Test email from agent project",
        body="If you're reading this, the Sender agent works.",
    )
    print(result)
