"""Envío de correo (códigos de recuperación / acceso).

Se configura por entorno (drop-in de systemd, nunca en el repo):
  SMTP_HOST, SMTP_PORT (587), SMTP_USER, SMTP_PASSWORD, SMTP_FROM, SMTP_TLS (starttls|ssl|none)
Sin SMTP_HOST el envío está desactivado y los flujos que lo necesitan lo dicen
(jamás se devuelve el código en la respuesta: eso daba la cuenta a quien supiera el email).
"""
from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage


def mail_enabled() -> bool:
    return bool(os.environ.get("SMTP_HOST"))


def send_mail(to: str, subject: str, text: str) -> bool:
    host = os.environ.get("SMTP_HOST")
    if not host:
        return False
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER", "")
    password = os.environ.get("SMTP_PASSWORD", "")
    sender = os.environ.get("SMTP_FROM", user)
    mode = os.environ.get("SMTP_TLS", "starttls").lower()

    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(text)
    try:
        if mode == "ssl":
            with smtplib.SMTP_SSL(host, port, context=ssl.create_default_context(), timeout=15) as s:
                if user:
                    s.login(user, password)
                s.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=15) as s:
                if mode == "starttls":
                    s.starttls(context=ssl.create_default_context())
                if user:
                    s.login(user, password)
                s.send_message(msg)
        return True
    except Exception as exc:  # noqa: BLE001 — el flujo responde igual; se registra
        print(f"[mailer] fallo enviando a {to.split('@')[-1]}: {type(exc).__name__}")
        return False
