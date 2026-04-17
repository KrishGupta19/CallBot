import os
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()
client = Client(os.getenv('TWILIO_ACCOUNT_SID'), os.getenv('TWILIO_AUTH_TOKEN'))

call = client.calls.create(
    url=os.getenv('NGROK_URL') + '/incoming-call',
    to='+919871134157',  # <--- Your actual phone number
    from_=os.getenv('TWILIO_PHONE_NUMBER')
)
print('Calling your phone...', call.sid)
