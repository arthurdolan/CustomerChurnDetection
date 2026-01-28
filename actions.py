"""
Mocked retention action handlers.
Simulates customer retention actions without actual external API calls.
"""

import pandas as pd
from datetime import datetime
from typing import Optional, Dict


class RetentionActionTracker:
    """Tracks retention actions taken for customers."""
    
    def __init__(self):
        self.actions_log = []
    
    def log_action(self, customer_id: str, action_type: str, status: str = 'completed'):
        """Log a retention action."""
        self.actions_log.append({
            'customer_id': customer_id,
            'action_type': action_type,
            'status': status,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    
    def get_customer_actions(self, customer_id: str) -> list:
        """Get all actions for a specific customer."""
        return [a for a in self.actions_log if a['customer_id'] == customer_id]
    
    def get_all_actions(self) -> pd.DataFrame:
        """Get all logged actions as a DataFrame."""
        if not self.actions_log:
            return pd.DataFrame(columns=['customer_id', 'action_type', 'status', 'timestamp'])
        return pd.DataFrame(self.actions_log)


# Global action tracker instance
action_tracker = RetentionActionTracker()


def send_retention_email(customer_id: str) -> Dict[str, str]:
    """
    Mock function to send a retention email.
    
    Args:
        customer_id: Customer identifier
    
    Returns:
        Dict with status and message
    """
    action_tracker.log_action(customer_id, 'email', 'completed')
    
    return {
        'status': 'success',
        'message': f'✅ Retention email sent to customer {customer_id}',
        'details': 'Email template: "We miss you! Here\'s a special offer just for you."'
    }


def offer_discount(customer_id: str, discount_percent: float = 15.0) -> Dict[str, str]:
    """
    Mock function to offer a discount.
    
    Args:
        customer_id: Customer identifier
        discount_percent: Discount percentage (default 15%)
    
    Returns:
        Dict with status and message
    """
    action_tracker.log_action(customer_id, f'discount_{discount_percent}%', 'completed')
    
    return {
        'status': 'success',
        'message': f'✅ {discount_percent}% discount code generated for customer {customer_id}',
        'details': f'Discount code: SAVE{discount_percent}-{customer_id[:6].upper()}'
    }


def flag_for_call(customer_id: str) -> Dict[str, str]:
    """
    Mock function to flag customer for customer success call.
    
    Args:
        customer_id: Customer identifier
    
    Returns:
        Dict with status and message
    """
    action_tracker.log_action(customer_id, 'call', 'pending')
    
    return {
        'status': 'success',
        'message': f'✅ Customer {customer_id} flagged for customer success call',
        'details': 'Added to priority queue. Expected callback within 24 hours.'
    }


def get_action_history(customer_id: Optional[str] = None) -> pd.DataFrame:
    """
    Get action history, optionally filtered by customer.
    
    Args:
        customer_id: Optional customer ID to filter by
    
    Returns:
        DataFrame with action history
    """
    if customer_id:
        return pd.DataFrame(action_tracker.get_customer_actions(customer_id))
    return action_tracker.get_all_actions()
