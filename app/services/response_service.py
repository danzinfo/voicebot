RESPONSES = {
    "order_status": "Your order is currently being processed.",
    "cancel_order": "Your order has been cancelled successfully.",
    "refund_request": "Your refund will be processed within 5 to 7 days.",
    "subscription_issue": "We are checking your subscription details.",
    "payment_issue": "There seems to be a payment issue. Please retry.",
    "complaint": "We’re sorry for the inconvenience. Your complaint is noted.",
    "product_info": "You can find product details on our website.",
    "delivery_delay": "Your delivery is delayed and will arrive soon.",
    "account_issue": "Please try resetting your account password.",
    "greeting": "Hello! How can I assist you today?"
}

def generate_response(intent):
    return RESPONSES.get(intent, "I'm not sure how to help with that.")
