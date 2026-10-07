"""Outgoing email, over SMTP configured from the environment.

    SMTP_HOST, SMTP_PORT (587 = STARTTLS, 465 = SSL), SMTP_USER,
    SMTP_PASSWORD, SMTP_FROM

With no SMTP_HOST (local development) nothing is sent and the message is
logged instead, so the code can be read off the server log.
"""

import logging
import os
import smtplib
from email.message import EmailMessage

from fastapi.concurrency import run_in_threadpool

log = logging.getLogger("heartbeat.mail")


def _send(to: str, subject: str, body: str) -> None:
    host = os.getenv("SMTP_HOST")
    if not host:
        log.warning("SMTP_HOST not set, not sending. To %s: %s\n%s", to, subject, body)
        return

    port = int(os.getenv("SMTP_PORT", "587"))
    msg = EmailMessage()
    msg["From"] = os.getenv("SMTP_FROM") or os.getenv("SMTP_USER", "")
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)

    smtp_cls = smtplib.SMTP_SSL if port == 465 else smtplib.SMTP
    with smtp_cls(host, port, timeout=20) as smtp:
        if port != 465:
            smtp.starttls()
        if os.getenv("SMTP_USER"):
            smtp.login(os.environ["SMTP_USER"], os.getenv("SMTP_PASSWORD", ""))
        smtp.send_message(msg)


async def send_verification_code(to: str, name: str, code: str, minutes: int) -> None:
    body = (
        f"Chào {name},\n\n"
        f"Mã xác thực tài khoản HeartBeat của bạn là: {code}\n\n"
        f"Mã có hiệu lực trong {minutes} phút. Nếu bạn không đăng ký HeartBeat, "
        "hãy bỏ qua email này.\n"
    )
    await run_in_threadpool(_send, to, f"Mã xác thực HeartBeat: {code}", body)
