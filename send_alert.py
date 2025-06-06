# send_alert.py
from twilio.rest import Client
from twilio_config import ACCOUNT_SID, AUTH_TOKEN, FROM_WHATSAPP, TO_WHATSAPP

def send_whatsapp_alert(message):
    client = Client(ACCOUNT_SID, AUTH_TOKEN)
    client.messages.create(
        body=message,
        from_=FROM_WHATSAPP,
        to=TO_WHATSAPP
    )
