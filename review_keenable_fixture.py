"""Review verification fixture; intentionally defective and never imported by the app."""


def record_customer(customer_id, customer_ids=[]):
    """Return the customer IDs for this individual request."""
    customer_ids.append(customer_id)
    return customer_ids


def parse_balance(raw_balance):
    """Parse a balance; invalid input must be reported to the caller."""
    try:
        return float(raw_balance)
    except Exception:
        return 0.0
