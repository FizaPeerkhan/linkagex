from backend.services.nlp_service import analyze_complaint
from pprint import pprint


test_complaint = """
I received a call from 9988233057 on my number 9989966455.
The caller claimed to be from FedEx Mumbai Andheri East and said
a courier containing illegal items in my name had been intercepted.
He then connected me to a person on Skype who claimed to be a
Mumbai Crime Branch officer.

They showed me a forged FIR and told me that my bank accounts
were under investigation. They asked me to transfer 90% of my
bank balance.

I transferred Rs. 12,49,633 through RTGS from my ICICI Bank
account to Punjab National Bank account 3335002100021930,
IFSC PUNB0436400.

After confirming the transfer, they disconnected the call.
"""


result = analyze_complaint(test_complaint)

pprint(result["incident"])